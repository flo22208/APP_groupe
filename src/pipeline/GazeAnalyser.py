from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import cv2
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from src.utils.DataLoader import DataLoader
from src.detectionModel.DetectionModel import DetectionModel


class PosterTemplate:
	"""Structure pour stocker les données multi-échelles d'une affiche."""
	def __init__(self, name: str, img_orig: np.ndarray):
		self.name = name
		self.img_orig = img_orig
		self.img_orig_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) if img_orig.ndim == 3 else img_orig
		self.img_orig_shape = self.img_orig_gray.shape[:2]
		self.multi_scale: List[Dict] = []


class PosterTracker:
	"""Gère le suivi KLT d'une affiche détectée."""
	
	def __init__(self, poster_index: int, poster_name: str, corners: np.ndarray, 
				 gray_frame: np.ndarray, homography: np.ndarray, scale: float):
		self.poster_index = poster_index
		self.poster_name = poster_name
		self.corners = corners.copy()
		self.original_corners = corners.copy()
		self.homography = homography
		self.scale = scale
		self.frames_tracked = 0
		self.frames_lost = 0
		self.max_frames_lost = 10
		self.min_features = 10
		
		# Initialiser le tracking KLT
		self.features = self._init_klt_tracking(gray_frame, corners)
		self.original_features = self.features.copy() if self.features is not None else None
	
	def _init_klt_tracking(self, gray_frame: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
		"""Initialise le suivi KLT avec les coins détectés."""
		feature_params = dict(
			maxCorners=50,
			qualityLevel=0.01,
			minDistance=7,
			blockSize=7
		)
		
		mask = np.zeros(gray_frame.shape, dtype=np.uint8)
		cv2.fillPoly(mask, [np.int32(corners)], 255)
		
		features = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
		
		if features is None or len(features) < 4:
			features = corners.reshape(-1, 1, 2).astype(np.float32)
		
		return features
	
	def update(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
		"""Met à jour le suivi KLT. Retourne True si le tracking est valide."""
		if self.features is None or len(self.features) < 4:
			self.frames_lost += 1
			return False
		
		lk_params = dict(
			winSize=(21, 21),
			maxLevel=3,
			criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
		)
		
		next_features, status, _ = cv2.calcOpticalFlowPyrLK(
			prev_gray, curr_gray, self.features, None, **lk_params
		)
		
		if next_features is None:
			self.frames_lost += 1
			return self.frames_lost < self.max_frames_lost
		
		good_new = next_features[status == 1]
		good_old = self.features[status == 1]
		
		if len(good_new) < 4:
			self.frames_lost += 1
			return self.frames_lost < self.max_frames_lost
		
		# Estimer la nouvelle position des coins
		new_corners = self._estimate_corners_from_tracking(good_new, good_old)
		
		if new_corners is None:
			self.frames_lost += 1
			return self.frames_lost < self.max_frames_lost
		
		self.features = good_new.reshape(-1, 1, 2)
		self.corners = new_corners
		self.frames_tracked += 1
		self.frames_lost = 0
		
		return True
	
	def _estimate_corners_from_tracking(self, tracked_points: np.ndarray, 
										original_points: np.ndarray) -> Optional[np.ndarray]:
		"""Estime la position des coins à partir des points suivis."""
		if len(tracked_points) < 4 or len(original_points) < 4:
			return None
		
		tracked_pts = tracked_points.reshape(-1, 2).astype(np.float32)
		original_pts = original_points.reshape(-1, 2).astype(np.float32)
		
		if tracked_pts.shape[0] != original_pts.shape[0]:
			return None
		
		M, mask = cv2.findHomography(original_pts, tracked_pts, cv2.RANSAC, 5.0)
		
		if M is None:
			return None
		
		corners = self.original_corners.reshape(-1, 1, 2).astype(np.float32)
		new_corners = cv2.perspectiveTransform(corners, M)
		
		return new_corners.reshape(-1, 2)
	
	def needs_redetection(self) -> bool:
		"""Vérifie si on a besoin d'une nouvelle détection."""
		if self.features is None:
			return True
		return len(self.features) < self.min_features
	
	def is_valid(self) -> bool:
		"""Vérifie si le tracker est toujours valide."""
		return self.frames_lost < self.max_frames_lost


class GazeAnalyser:
	"""Analyseur de regard avec détection ORB multi-échelle et tracking KLT.

	Intègre directement :
	- La détection multi-échelle des points d'intérêt ORB
	- Le tracking KLT pour la persistance temporelle
	- Le matching des affiches
	- La projection sur l'affiche avec homographie

	Les bounding boxes YOLO et leurs track_id sont conservés pour la reconnaissance persistante.
	"""

	# Paramètres ORB
	NB_POI = 2000
	TOP_FILTRAGE = 1000
	
	# Paramètres de matching
	MIN_MATCHES_THRESHOLD = 10
	RATIO_THRESHOLD = 0.7
	BEST_THRESHOLD = 15
	
	# Paramètres de validation
	VOTES_TO_CONFIRM = 5
	PROJECTION_MARGIN_RATIO = 0.05
	
	# Échelles pour la détection multi-échelle
	SCALES = [0.25, 0.5, 0.75, 1.0]

	def __init__(self, config_path: str) -> None:
		self.loader = DataLoader(config_path)

		# Modèle de détection YOLO
		weights_path = self.loader.get_yolo_detection_weights()
		self.det_model = DetectionModel(weights_path)

		# Détecteur ORB
		self.orb = cv2.ORB_create(
			nfeatures=self.NB_POI,
			scaleFactor=1.2,
			nlevels=12
		)
		
		# CLAHE pour amélioration du contraste
		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		
		# Matcher BruteForce
		self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

		# Charger les templates d'affiches avec multi-échelles
		self.poster_templates: List[PosterTemplate] = self._load_poster_templates()
		if not self.poster_templates:
			raise RuntimeError("No PNG posters found in posters folder")

		# Cache confirmé : track_id -> index d'affiche CONFIRMÉ
		self.poster_index_by_track: Dict[int, int] = {}

		# Système de vote : track_id -> liste des votes (poster_index)
		self.vote_cache: Dict[int, List[int]] = defaultdict(list)
		self.match_scores_cache: Dict[int, List[int]] = defaultdict(list)
		
		# Trackers KLT actifs : track_id -> PosterTracker
		self.active_trackers: Dict[int, PosterTracker] = {}
		
		# Frame précédente pour le KLT
		self.prev_gray: Optional[np.ndarray] = None

	def _load_poster_templates(self) -> List[PosterTemplate]:
		"""Charge les templates d'affiches avec plusieurs échelles."""
		posters_raw = self.loader.load_posters()
		templates = []

		for name, img in posters_raw:
			template = PosterTemplate(name, img)
			
			# Amélioration avec CLAHE
			enhanced_orig = self.clahe.apply(template.img_orig_gray)

			# Pour chaque échelle
			for scale in self.SCALES:
				w_scaled = int(enhanced_orig.shape[1] * scale)
				h_scaled = int(enhanced_orig.shape[0] * scale)
				gray_scaled = cv2.resize(enhanced_orig, (w_scaled, h_scaled))

				kp, des = self.orb.detectAndCompute(gray_scaled, None)
				if kp is None or len(kp) == 0:
					continue

				template.multi_scale.append({
					"scale": scale,
					"img": gray_scaled,
					"kp": kp,
					"des": des
				})

			if template.multi_scale:
				templates.append(template)
				total_kp = sum(len(ms["kp"]) for ms in template.multi_scale)
				print(f"> Poster chargé : {name}, {len(template.multi_scale)} échelles, {total_kp} keypoints")

		return templates

	def _detect_and_describe_roi(self, roi_gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
		"""Détecte les keypoints ORB dans une ROI."""
		enhanced = self.clahe.apply(roi_gray)
		kp, des = self.orb.detectAndCompute(enhanced, None)
		return kp if kp else [], des

	def _match_roi_to_poster_multiscale(self, roi_gray: np.ndarray, 
										poster_idx: Optional[int] = None
	) -> Tuple[Optional[int], Optional[np.ndarray], int, float]:
		"""
		Match une ROI contre les templates multi-échelles.
		
		Si poster_idx est fourni, ne teste que cette affiche.
		Retourne (best_poster_idx, best_H, best_inliers, best_scale).
		"""
		kp_roi, des_roi = self._detect_and_describe_roi(roi_gray)
		
		if des_roi is None or len(kp_roi) < 8:
			return None, None, 0, 1.0

		best_poster_idx = None
		best_H = None
		best_inliers = 0
		best_scale = 1.0

		# Liste des templates à tester
		templates_to_test = [self.poster_templates[poster_idx]] if poster_idx is not None else self.poster_templates
		indices_to_test = [poster_idx] if poster_idx is not None else range(len(self.poster_templates))

		for idx, template in zip(indices_to_test, templates_to_test):
			for ms in template.multi_scale:
				kp_t, des_t = ms["kp"], ms["des"]

				if des_t is None or len(des_t) == 0:
					continue

				# KNN matching
				try:
					matches = self.bf_matcher.knnMatch(des_t, des_roi, k=2)
				except cv2.error:
					continue

				good = []
				for m_n in matches:
					if len(m_n) == 2:
						m, n = m_n
						if m.distance < self.RATIO_THRESHOLD * n.distance:
							good.append(m)
					elif len(m_n) == 1:
						good.append(m_n[0])

				if len(good) < self.MIN_MATCHES_THRESHOLD:
					continue

				src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
				dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

				H, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

				if H is None:
					continue

				inliers = int(mask_ransac.sum())

				if inliers > best_inliers and inliers >= self.BEST_THRESHOLD:
					best_inliers = inliers
					best_poster_idx = idx
					best_H = H
					best_scale = ms["scale"]

		return best_poster_idx, best_H, best_inliers, best_scale

	def _update_vote(self, track_id: int, poster_idx: int, match_count: int) -> Optional[int]:
		"""Ajoute un vote et retourne l'index confirmé si le seuil est atteint."""
		self.vote_cache[track_id].append(poster_idx)
		self.match_scores_cache[track_id].append(match_count)
		votes = self.vote_cache[track_id]
		
		counts = Counter(votes)
		best_idx, best_count = counts.most_common(1)[0]
		
		if best_count >= self.VOTES_TO_CONFIRM:
			return best_idx
		
		total_votes = len(votes)
		if total_votes >= 3 and best_count / total_votes > 0.6:
			return best_idx
		
		return None

	def _validate_homography(self, H: np.ndarray, template_shape: Tuple[int, int]) -> bool:
		"""Valide si l'homographie est raisonnable."""
		if H is None:
			return False
		
		h, w = template_shape

		corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

		try:
			transformed = cv2.perspectiveTransform(corners, H)
		except cv2.error:
			return False

		if transformed is None:
			return False

		transformed = transformed.reshape(-1, 2)
		area = cv2.contourArea(transformed)
		original_area = w * h
		
		if area < 0.01 * original_area or area > 10 * original_area:
			return False

		return True

	def _project_gaze_on_poster(self, gaze_local: Tuple[float, float], 
								H: np.ndarray, scale: float,
								poster_shape: Tuple[int, int]) -> Tuple[Optional[float], Optional[float]]:
		"""Projette le point de regard sur l'affiche originale."""
		gx_local, gy_local = gaze_local
		
		# Inverser l'homographie pour projeter frame -> template
		try:
			H_inv = np.linalg.inv(H)
		except np.linalg.LinAlgError:
			return None, None
		
		pt = np.array([[[gx_local, gy_local]]], dtype=np.float32)
		pt_affiche = cv2.perspectiveTransform(pt, H_inv)
		
		if pt_affiche is None:
			return None, None
		
		gx_scaled, gy_scaled = pt_affiche[0, 0]
		
		# Rescaler vers l'image originale
		gx_aff = gx_scaled / scale
		gy_aff = gy_scaled / scale
		
		# Clipping aux dimensions de l'affiche
		h_orig, w_orig = poster_shape
		gx_aff = float(np.clip(gx_aff, 0, w_orig - 1))
		gy_aff = float(np.clip(gy_aff, 0, h_orig - 1))
		
		# Vérifier si le point est dans les limites avec marge
		margin_x = w_orig * self.PROJECTION_MARGIN_RATIO
		margin_y = h_orig * self.PROJECTION_MARGIN_RATIO
		
		if (gx_aff < -margin_x or gx_aff > w_orig + margin_x or
			gy_aff < -margin_y or gy_aff > h_orig + margin_y):
			return None, None
		
		return gx_aff, gy_aff

	def _check_gaze_on_poster(self, gx: float, gy: float, corners: np.ndarray) -> bool:
		"""Vérifie si le point de regard est sur une affiche (polygon test)."""
		if corners is None or len(corners) < 3:
			return False
		result = cv2.pointPolygonTest(corners.astype(np.float32), (float(gx), float(gy)), False)
		return result >= 0

	def analyse_gaze_on_frame(
		self,
		frame_undist: np.ndarray,
		gaze_point: Tuple[float, float],
		frame_idx: int,
		subject_idx: int,
		tracks_df: pd.DataFrame,
	) -> Tuple[str, int, Tuple[float, float]]:
		"""Analyse une frame et un point de regard pour retrouver l'affiche correspondante.

		Utilise les détections YOLO pré-calculées dans le CSV detection_results,
		puis effectue le matching ORB multi-échelle et la projection.

		Retourne (best_name, best_index, (px, py)).
		Lève RuntimeError si la projection échoue ou n'est pas valide.
		"""
		gx, gy = gaze_point
		
		# Convertir en niveaux de gris
		gray_frame = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY) if frame_undist.ndim == 3 else frame_undist

		# 1 - Filtrer les détections YOLO pour cette frame
		rows = tracks_df[tracks_df["frame"] == frame_idx]
		if rows.empty:
			raise RuntimeError("No posters detected on frame (CSV empty for this frame)")

		bboxes_list = []
		for _, row in rows.iterrows():
			x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
			track_id = int(row["track_id"])
			bboxes_list.append((np.array([x1, y1, x2, y2], dtype=np.float32), track_id))

		if not bboxes_list:
			raise RuntimeError("No posters detected on frame (no boxes in CSV)")

		# 2 - Trouver la bbox contenant le point de regard
		looked_bbox: Optional[np.ndarray] = None
		looked_track_id: int = -1
		min_distance: float = float('inf')
		
		for bbox, track_id in bboxes_list:
			x1, y1, x2, y2 = bbox
			if x1 <= gx <= x2 and y1 <= gy <= y2:
				center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
				dist = (gx - center_x)**2 + (gy - center_y)**2
				if dist < min_distance:
					min_distance = dist
					looked_bbox = bbox
					looked_track_id = track_id

		if looked_bbox is None:
			raise RuntimeError("Gaze point not inside any detected poster bbox")

		# 3 - Extraire la ROI
		x1_roi, y1_roi, x2_roi, y2_roi = looked_bbox.astype(int)
		x1_roi = max(0, x1_roi)
		y1_roi = max(0, y1_roi)
		x2_roi = min(frame_undist.shape[1], x2_roi)
		y2_roi = min(frame_undist.shape[0], y2_roi)
		
		roi_gray = gray_frame[y1_roi:y2_roi, x1_roi:x2_roi]
		
		if roi_gray.size == 0:
			raise RuntimeError("ROI is empty")

		# 4 - Utiliser le tracker KLT si disponible et valide
		best_index: int = -1
		best_H: Optional[np.ndarray] = None
		best_scale: float = 1.0
		
		# Vérifier si on a un tracker actif pour ce track_id
		if looked_track_id in self.active_trackers:
			tracker = self.active_trackers[looked_track_id]
			
			# Mettre à jour le tracker KLT
			if self.prev_gray is not None and tracker.is_valid():
				tracker.update(self.prev_gray, gray_frame)
			
			if tracker.is_valid() and not tracker.needs_redetection():
				# Utiliser le tracking existant
				best_index = tracker.poster_index
				best_scale = tracker.scale
				
				# Recalculer l'homographie pour la projection
				poster_idx_confirmed, H, inliers, scale = self._match_roi_to_poster_multiscale(
					roi_gray, poster_idx=best_index
				)
				
				if H is not None:
					best_H = H
					best_scale = scale
				else:
					# Fallback: utiliser l'homographie du tracker (moins précis)
					best_H = tracker.homography

		# 5 - Si pas de tracker valide, faire une détection complète
		if best_H is None:
			# Vérifier si on connaît déjà cette affiche (confirmée via système de vote)
			if looked_track_id in self.poster_index_by_track:
				known_idx = self.poster_index_by_track[looked_track_id]
				poster_idx, H, inliers, scale = self._match_roi_to_poster_multiscale(
					roi_gray, poster_idx=known_idx
				)
				if H is not None:
					best_index = known_idx
					best_H = H
					best_scale = scale
			
			# Si toujours pas d'homographie, recherche complète
			if best_H is None:
				poster_idx, H, inliers, scale = self._match_roi_to_poster_multiscale(roi_gray)
				
				if poster_idx is None or H is None:
					raise RuntimeError(f"Could not find any matching poster for ROI")
				
				# Système de vote pour confirmer l'affiche
				confirmed_idx = self._update_vote(looked_track_id, poster_idx, inliers)
				
				if confirmed_idx is not None:
					self.poster_index_by_track[looked_track_id] = confirmed_idx
					best_index = confirmed_idx
				else:
					best_index = poster_idx
				
				best_H = H
				best_scale = scale

		# 6 - Créer ou mettre à jour le tracker KLT
		if best_H is not None and best_index >= 0:
			template = self.poster_templates[best_index]
			
			# Calculer les coins projetés
			h_scaled = int(template.img_orig_shape[0] * best_scale)
			w_scaled = int(template.img_orig_shape[1] * best_scale)
			corners_template = np.float32([[0, 0], [w_scaled, 0], [w_scaled, h_scaled], [0, h_scaled]]).reshape(-1, 1, 2)
			corners_frame = cv2.perspectiveTransform(corners_template, best_H)
			
			if corners_frame is not None:
				# Ajuster les coordonnées des coins pour la frame complète
				corners_absolute = corners_frame.reshape(-1, 2) + np.array([x1_roi, y1_roi])
				
				# Créer ou remplacer le tracker
				self.active_trackers[looked_track_id] = PosterTracker(
					poster_index=best_index,
					poster_name=template.name,
					corners=corners_absolute,
					gray_frame=gray_frame,
					homography=best_H,
					scale=best_scale
				)

		# 7 - Projeter le point de regard sur l'affiche
		if best_H is None:
			raise RuntimeError("Homography computation failed - not enough quality matches")

		template = self.poster_templates[best_index]
		
		# Point local dans la ROI
		gx_local = gx - x1_roi
		gy_local = gy - y1_roi
		
		px, py = self._project_gaze_on_poster(
			(gx_local, gy_local), best_H, best_scale, template.img_orig_shape
		)
		
		if px is None or py is None:
			raise RuntimeError(f"Projected point is outside poster bounds")

		# Stocker la frame pour le KLT
		self.prev_gray = gray_frame.copy()

		return template.name, best_index, (px, py)

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

