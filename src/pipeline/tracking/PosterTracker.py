from typing import Optional, Tuple
import numpy as np
import cv2


class PosterTracker:
	"""
	Gère le suivi KLT d'une affiche détectée.
	"""
	
	def __init__(self, poster_index: int, poster_name: str, corners: np.ndarray,
				 gray_frame: np.ndarray, homography: np.ndarray, scale: float, track_id: int):
		self.poster_index = poster_index
		self.poster_name = poster_name
		self.track_id = track_id
		self.corners = corners.copy()
		self.original_corners = corners.copy()
		self.homography = homography
		self.scale = scale
		self.frames_tracked = 0
		self.frames_lost = 0
		self.max_frames_lost = 10
		self.min_features = 10
		self.last_area = None
		
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
		
		if new_corners is None or not self._are_corners_valid(new_corners):
			self.frames_lost += 1
			return self.frames_lost < self.max_frames_lost
		
		self.features = good_new.reshape(-1, 1, 2)
		self.corners = new_corners
		self.frames_tracked += 1
		self.frames_lost = 0
		
		return True

	def _are_corners_valid(self, corners: np.ndarray) -> bool:
		"""Vérifie la validité géométrique des coins."""
		# 1. Vérifier la convexité
		if not cv2.isContourConvex(corners.astype(np.int32)):
			return False

		# 2. Vérifier la stabilité de la surface
		current_area = cv2.contourArea(corners)
		if self.last_area is not None:
			area_ratio = current_area / self.last_area if self.last_area > 0 else float('inf')
			# Si la surface change de plus de 50% en une frame, c'est suspect
			if not (0.5 < area_ratio < 1.5):
				return False
		
		self.last_area = current_area
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
	
	def is_lost(self) -> bool:
		"""Vérifie si le tracker est perdu."""
		return self.frames_lost >= self.max_frames_lost
