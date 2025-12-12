from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from src.pipeline.HomographyManager import HomographyManager
from src.utils.DataLoader import DataLoader
from src.detectionModel.DetectionModel import DetectionModel


class GazeAnalyser:
	"""Analyseur de regard basé sur détection + homographie.

	Charge le modèle de détection, les affiches et le HomographyManager à l'initialisation,
	puis expose une méthode pour analyser une frame + un point de regard.
	
	Système de vote simple : une affiche est confirmée après plusieurs détections concordantes.
	"""

	# Nombre de votes nécessaires pour confirmer une affiche
	VOTES_TO_CONFIRM = 5  # Augmenté pour plus de robustesse
	# Nombre minimum de matchs pour considérer une correspondance valide
	MIN_MATCHES_THRESHOLD = 15
	# Ratio de padding à ajouter autour du ROI (évite les problèmes de bords)
	ROI_PADDING_RATIO = 0.1
	# Marge acceptable pour les points projetés hors de l'affiche (ratio)
	PROJECTION_MARGIN_RATIO = 0.05

	def __init__(self, config_path: str) -> None:
		self.loader = DataLoader(config_path)
		self.h_manager = HomographyManager()

		# Modèle de détection
		weights_path = self.loader.get_yolo_detection_weights()
		self.det_model = DetectionModel(weights_path)

		# Affiches
		self.posters: List[Tuple[str, np.ndarray]] = self.loader.load_posters()
		if not self.posters:
			raise RuntimeError("No PNG posters found in posters folder")

		# Cache confirmé : Associe les track_id aux index d'affiches CONFIRMÉS
		self.poster_name_to_index: Dict[int, int] = {}

		# Système de vote simple : track_id -> list des votes (poster_index)
		self.vote_cache: Dict[int, List[int]] = defaultdict(list)
		
		# Cache des dernières homographies valides pour chaque track_id
		self.homography_cache: Dict[int, np.ndarray] = {}
		
		# Cache des scores de matchs pour chaque track_id (pour filtrer les mauvais matchs)
		self.match_scores_cache: Dict[int, List[int]] = defaultdict(list)

	def _expand_bbox(self, bbox: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
		"""Agrandit la bounding box avec un padding pour améliorer la détection de features.
		
		Le padding permet de capturer plus de contexte et d'avoir des features
		plus fiables près des bords de l'affiche.
		"""
		x1, y1, x2, y2 = bbox
		h, w = frame_shape[:2]
		
		bbox_width = x2 - x1
		bbox_height = y2 - y1
		
		pad_x = bbox_width * self.ROI_PADDING_RATIO
		pad_y = bbox_height * self.ROI_PADDING_RATIO
		
		# Appliquer le padding en restant dans les limites de l'image
		new_x1 = max(0, x1 - pad_x)
		new_y1 = max(0, y1 - pad_y)
		new_x2 = min(w, x2 + pad_x)
		new_y2 = min(h, y2 + pad_y)
		
		return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

	def _get_best_match(self, roi: np.ndarray) -> Tuple[Optional[int], int]:
		"""Trouve la meilleure affiche candidate par nombre de matchs.
		
		Retourne (index de l'affiche, nombre de matchs) ou (None, 0) si aucune trouvée.
		"""
		roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
		
		# Améliorer le contraste du ROI pour une meilleure détection
		roi_gray = cv2.equalizeHist(roi_gray)
		
		kp1, desc1 = self.h_manager.keypoints_manager.detect_and_describe(roi_gray)
		
		if desc1 is None or len(kp1) < 8:
			return None, 0

		best_idx = -1
		best_matches = 0

		for idx, (_, poster_img) in enumerate(self.posters):
			poster_gray = cv2.cvtColor(poster_img, cv2.COLOR_BGR2GRAY)
			kp2, desc2 = self.h_manager.keypoints_manager.detect_and_describe(poster_gray)
			
			if desc2 is None or len(kp2) < 8:
				continue

			good_matches = self.h_manager.keypoints_manager.match_descriptors(desc1, desc2)
			n_matches = len(good_matches)
			
			if n_matches > best_matches:
				best_matches = n_matches
				best_idx = idx

		# Vérifier qu'on a assez de matchs pour être confiant
		if best_idx >= 0 and best_matches >= self.MIN_MATCHES_THRESHOLD:
			return best_idx, best_matches
		
		return None, best_matches

	def _update_vote(self, track_id: int, poster_idx: int, match_count: int) -> Optional[int]:
		"""Ajoute un vote pondéré par la qualité et retourne l'index confirmé si on atteint le seuil.
		
		Les votes avec plus de matchs ont plus de poids.
		"""
		self.vote_cache[track_id].append(poster_idx)
		self.match_scores_cache[track_id].append(match_count)
		votes = self.vote_cache[track_id]
		
		# Compter les votes pour chaque affiche
		from collections import Counter
		counts = Counter(votes)
		best_idx, best_count = counts.most_common(1)[0]
		
		# Pour confirmer : soit beaucoup de votes concordants, soit majorité claire
		if best_count >= self.VOTES_TO_CONFIRM:
			return best_idx
		
		# Alternative : si on a assez de votes et une majorité claire (>60%)
		total_votes = len(votes)
		if total_votes >= 3 and best_count / total_votes > 0.6:
			return best_idx
		
		return None

	def analyse_gaze_on_frame(
		self,
		frame_undist: np.ndarray,
		gaze_point: Tuple[float, float],
		frame_idx: int,
		subject_idx: int,
		tracks_df: pd.DataFrame,
	) -> Tuple[str, int, Tuple[float, float]]:
		"""Analyse une frame et un point de regard pour retrouver l'affiche correspondante.

		Utilise les détections pré-calculées dans le CSV detection_results
		pour la frame donnée au lieu de relancer YOLO.

		Retourne (best_name, best_index, (px, py)).
		Lève RuntimeError si la projection échoue ou n'est pas valide.
		"""

		gx, gy = gaze_point

		# 1 - Filtrer les détections pour cette frame
		rows = tracks_df[tracks_df["frame"] == frame_idx]
		if rows.empty:
			raise RuntimeError("No posters detected on frame (CSV empty for this frame)")

		bboxes_list = []
		for _, row in rows.iterrows():
			x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
			track_id = int(row["track_id"])
			bboxes_list.append((np.array([x1, y1, x2, y2], dtype=np.float32), track_id))

		bboxes_array = np.stack([bbox for bbox, _ in bboxes_list], axis=0) if bboxes_list else None
		if bboxes_array is None or len(bboxes_array) == 0:
			raise RuntimeError("No posters detected on frame (no boxes in CSV)")

		# 2 - Trouver l'affiche regardée: bbox contenant le point de regard
		#     Utiliser une marge pour être plus tolérant
		looked_bbox: Optional[np.ndarray] = None
		looked_bbox_track_id: int = -1
		min_distance: float = float('inf')
		
		for idx, bbox in enumerate(bboxes_array):
			x1, y1, x2, y2 = bbox
			# Ajouter une petite marge autour de la bbox pour la détection du gaze
			margin = 5  # pixels
			if (x1 - margin) <= gx <= (x2 + margin) and (y1 - margin) <= gy <= (y2 + margin):
				# Calculer la distance au centre de la bbox
				center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
				dist = (gx - center_x)**2 + (gy - center_y)**2
				if dist < min_distance:
					min_distance = dist
					looked_bbox = bbox
					looked_bbox_track_id = bboxes_list[idx][1]

		if looked_bbox is None:
			raise RuntimeError("Gaze point not inside any detected poster bbox")

		# 3 - ROI étendu à partir de la bbox regardée
		expanded_bbox = self._expand_bbox(looked_bbox, frame_undist.shape)
		x1_exp, y1_exp, x2_exp, y2_exp = expanded_bbox.astype(int)
		roi = frame_undist[y1_exp:y2_exp, x1_exp:x2_exp]
		
		if roi.size == 0:
			raise RuntimeError("ROI is empty")

		# 4 - Trouver l'affiche correspondante
		best_H: Optional[np.ndarray] = None
		best_index: int = -1
		match_count: int = 0

		# Vérifier si on connaît déjà cette affiche (confirmée via système de vote)
		if looked_bbox_track_id in self.poster_name_to_index:
			best_index = self.poster_name_to_index[looked_bbox_track_id]
			poster_img = self.posters[best_index][1]
			best_H, _ = self.h_manager.compute_homography_between(roi, poster_img)
			
			# Si l'homographie échoue, utiliser le cache si disponible
			if best_H is None and looked_bbox_track_id in self.homography_cache:
				best_H = self.homography_cache[looked_bbox_track_id]
		else:
			# Affiche non encore confirmée : trouver la meilleure et voter
			candidate_idx, match_count = self._get_best_match(roi)
			
			if candidate_idx is None:
				raise RuntimeError(f"Could not find any matching poster PNG for ROI (best match: {match_count} features)")
			
			# Mettre à jour les votes et vérifier si on peut confirmer
			confirmed_idx = self._update_vote(looked_bbox_track_id, candidate_idx, match_count)
			
			if confirmed_idx is not None:
				# Affiche confirmée ! L'enregistrer
				self.poster_name_to_index[looked_bbox_track_id] = confirmed_idx
				best_index = confirmed_idx
			else:
				# Pas encore confirmé, utiliser le candidat temporairement
				best_index = candidate_idx
			
			poster_img = self.posters[best_index][1]
			best_H, _ = self.h_manager.compute_homography_between(roi, poster_img)

		if best_H is None:
			raise RuntimeError("Homography computation failed - not enough quality matches")
		
		# Stocker l'homographie valide dans le cache
		self.homography_cache[looked_bbox_track_id] = best_H

		best_name = self.posters[best_index][0]
		poster_img = self.posters[best_index][1]
		poster_h, poster_w = poster_img.shape[:2]

		# 5 - Projeter le point de regard sur l'affiche avec validation
		# Calculer le point local par rapport au ROI étendu
		local_point = np.array([gx - x1_exp, gy - y1_exp], dtype=np.float32)
		
		# Utiliser la projection avec vérification des limites
		projected, is_valid = self.h_manager.project_point_with_bounds(
			local_point, best_H, poster_w, poster_h, self.PROJECTION_MARGIN_RATIO
		)
		
		if not is_valid:
			raise RuntimeError(f"Projected point ({projected[0]:.1f}, {projected[1]:.1f}) is outside poster bounds ({poster_w}x{poster_h})")
		
		px, py = float(projected[0]), float(projected[1])

		return best_name, best_index, (px, py)

	def analyse_video_for_subject(
		self,
		subject_idx: int,
		output_csv_path: str,
		start_frame: int = 0,
		end_frame: Optional[int] = None,
	) -> None:
		"""Analyse une vidéo d'un sujet et sauvegarde les résultats dans un CSV.

		Chaque ligne du CSV contient :
		frame_idx, poster_name, poster_index, proj_x, proj_y

		Les frames sans détection d'affiche ou sans regard sur une affiche
		sont simplement ignorées (pas de ligne écrite).
		"""

		# Récupérer capture vidéo et gazes pour le sujet
		cap = self.loader.get_video_capture(subject_idx)
		K, D = self.loader.get_load_camera_params(subject_idx)
		gazes_undist = self.loader.get_undistorted_gazes(subject_idx)

		# Charger le CSV de détections une seule fois
		det_csv_path = self.loader.get_detection_results_path(subject_idx)
		if not os.path.exists(det_csv_path):
			raise RuntimeError(f"Detection results CSV not found: {det_csv_path}")
		tracks_df = pd.read_csv(det_csv_path)

		import csv
		from pathlib import Path

		output_path = Path(output_csv_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if end_frame is None or end_frame > n_frames:
			end_frame = n_frames

		# Réinitialiser les caches pour ce sujet
		self.poster_name_to_index.clear()
		self.vote_cache.clear()
		self.homography_cache.clear()
		self.match_scores_cache.clear()

		with output_path.open("w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				"frame_idx",
				"poster_name",
				"poster_index",
				"proj_x",
				"proj_y",
			])

			frame_range = range(start_frame, end_frame)
			for frame_idx in tqdm(frame_range, desc=f"Analysing subject {subject_idx}", unit="frame"):
				# Traiter seulement toutes les skip_step frames
				if frame_idx % self.loader.skip_step != 0:
					continue

				# Sécurité si pas de gaze dispo pour cette frame
				if frame_idx >= len(gazes_undist):
					continue

				cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
				ret, frame = cap.read()
				if not ret:
					continue

				frame_undist = cv2.undistort(frame, K, D)
				gx, gy = gazes_undist[frame_idx]

				# Essayer d'analyser cette frame, en étant robuste aux erreurs
				try:
					best_name, best_index, (px, py) = self.analyse_gaze_on_frame(
						frame_undist,
						(gx, gy),
						frame_idx,
						subject_idx,
						tracks_df,
					)
				except RuntimeError:
					# Pas de détection, ou regard hors affiche : on skip
					continue

				# On écrit uniquement les frames valides
				writer.writerow([
					frame_idx,
					best_name,
					best_index,
					px,
					py,
				])

		cap.release()


__all__ = ["GazeAnalyser"]

