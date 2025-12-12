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
	
	Système de vote simple : une affiche est confirmée après 3 détections concordantes.
	"""

	# Nombre de votes nécessaires pour confirmer une affiche
	VOTES_TO_CONFIRM = 3

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

	def _get_best_match(self, roi: np.ndarray) -> Optional[int]:
		"""Trouve la meilleure affiche candidate par nombre de matchs.
		
		Retourne l'index de l'affiche ou None si aucune trouvée.
		"""
		roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
		kp1, desc1 = self.h_manager.keypoints_manager.detect_and_describe(roi_gray)
		
		if desc1 is None or len(kp1) < 4:
			return None

		best_idx = -1
		best_matches = 0

		for idx, (_, poster_img) in enumerate(self.posters):
			poster_gray = cv2.cvtColor(poster_img, cv2.COLOR_BGR2GRAY)
			kp2, desc2 = self.h_manager.keypoints_manager.detect_and_describe(poster_gray)
			
			if desc2 is None or len(kp2) < 4:
				continue

			good_matches = self.h_manager.keypoints_manager.match_descriptors(desc1, desc2)
			n_matches = len(good_matches)
			
			if n_matches > best_matches:
				best_matches = n_matches
				best_idx = idx

		return best_idx if best_idx >= 0 else None

	def _update_vote(self, track_id: int, poster_idx: int) -> Optional[int]:
		"""Ajoute un vote et retourne l'index confirmé si on atteint 3 votes identiques."""
		self.vote_cache[track_id].append(poster_idx)
		votes = self.vote_cache[track_id]
		
		# Compter les votes pour chaque affiche
		from collections import Counter
		counts = Counter(votes)
		best_idx, best_count = counts.most_common(1)[0]
		
		if best_count >= self.VOTES_TO_CONFIRM:
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
		looked_bbox: Optional[np.ndarray] = None
		looked_bbox_track_id: int = -1
		for idx, bbox in enumerate(bboxes_array):
			x1, y1, x2, y2 = bbox
			if x1 <= gx <= x2 and y1 <= gy <= y2:
				looked_bbox = bbox
				looked_bbox_track_id = bboxes_list[idx][1]
				break

		if looked_bbox is None:
			raise RuntimeError("Gaze point not inside any detected poster bbox")

		# 3 - ROI à partir de la bbox regardée
		x1_lb, y1_lb, x2_lb, y2_lb = looked_bbox.astype(int)
		roi = frame_undist[y1_lb:y2_lb, x1_lb:x2_lb]

		# 4 - Trouver l'affiche correspondante
		best_H: Optional[np.ndarray] = None
		best_index: int = -1

		# Vérifier si on connaît déjà cette affiche (confirmée via système de vote)
		if looked_bbox_track_id in self.poster_name_to_index:
			best_index = self.poster_name_to_index[looked_bbox_track_id]
			poster_img = self.posters[best_index][1]
			best_H, _ = self.h_manager.compute_homography_between(roi, poster_img)
		else:
			# Affiche non encore confirmée : trouver la meilleure et voter
			candidate_idx = self._get_best_match(roi)
			
			if candidate_idx is None:
				raise RuntimeError("Could not find any matching poster PNG for ROI")
			
			# Mettre à jour les votes et vérifier si on peut confirmer
			confirmed_idx = self._update_vote(looked_bbox_track_id, candidate_idx)
			
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
			raise RuntimeError("Homography computation failed")

		best_name = self.posters[best_index][0]

		# 5 - Projeter le point de regard sur l'affiche
		x1, y1, x2, y2 = looked_bbox
		local_point = np.array([gx - x1, gy - y1], dtype=np.float32)
		projected = self.h_manager.project_point(local_point, best_H)
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

