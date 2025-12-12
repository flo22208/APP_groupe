from typing import List, Optional, Tuple
import cv2
import numpy as np
from src.pipeline.KeypointsManager import KeypointsManager

class HomographyManager:
	def __init__(self, ransac_thresh: float = 5.0, keypoints_manager: Optional[KeypointsManager] = None) -> None:
		self.ransac_thresh = ransac_thresh
		self.keypoints_manager = keypoints_manager or KeypointsManager()

	def estimate_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], int]:
		if len(matches) < 4:
			return None, 0

		src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

		H, mask = cv2.findHomography(
			src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh
		)
		if H is None or mask is None:
			return None, 0

		inliers = int(mask.ravel().sum())
		return H, inliers

	# ---- High-level helper using KeypointsManager ----
	def compute_homography_between(
		self,
		image1: np.ndarray,
		image2: np.ndarray,
		mask1: Optional[np.ndarray] = None,
		mask2: Optional[np.ndarray] = None,
	) -> Tuple[Optional[np.ndarray], int]:
		"""Detect, match and estimate homography between two images.

		Returns (H, num_inliers).
		"""

		kp1, desc1 = self.keypoints_manager.detect_and_describe(image1, mask1)
		kp2, desc2 = self.keypoints_manager.detect_and_describe(image2, mask2)

		if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
			return None, 0

		good_matches = self.keypoints_manager.match_descriptors(desc1, desc2)
		H, inliers = self.estimate_homography(kp1, kp2, good_matches)

		return H, inliers
	
	def find_best_match(
		self,
		image1: np.ndarray,
		image_list: List[np.ndarray],
		mask1: Optional[np.ndarray] = None,
		mask_list: Optional[List[Optional[np.ndarray]]] = None,
	) -> Tuple[Optional[np.ndarray], int, int]:
		"""Find the best matching image from a list based on number of KNN matches.

		Détecte les keypoints de image1 une seule fois, puis compare avec chaque
		image de la liste en comptant le nombre de bons matchs (KNN + ratio test).

		Returns (best_H, best_matches_count, best_index).
		"""

		# Détecter les keypoints de l'image source une seule fois
		kp1, desc1 = self.keypoints_manager.detect_and_describe(image1, mask1)
		if desc1 is None or len(kp1) < 4:
			return None, 0, -1

		best_H: Optional[np.ndarray] = None
		best_matches_count = 0
		best_index = -1
		best_kp2 = None
		best_matches = None

		# Comparer avec chaque image de la liste
		for idx, image2 in enumerate(image_list):
			mask2 = mask_list[idx] if mask_list is not None else None
			kp2, desc2 = self.keypoints_manager.detect_and_describe(image2, mask2)

			if desc2 is None or len(kp2) < 4:
				continue

			# Matcher avec KNN
			good_matches = self.keypoints_manager.match_descriptors(desc1, desc2)
			n_matches = len(good_matches)

			if n_matches > best_matches_count:
				best_matches_count = n_matches
				best_index = idx
				best_kp2 = kp2
				best_matches = good_matches

		# Calculer l'homographie seulement pour la meilleure affiche
		if best_index >= 0 and best_matches is not None and len(best_matches) >= 4:
			best_H, _ = self.estimate_homography(kp1, best_kp2, best_matches)

		return best_H, best_matches_count, best_index

	def project_point(self, point: np.ndarray, H: np.ndarray) -> np.ndarray:
		"""
		Project a 2D point using a homography matrix.

		point : np.ndarray
			A 2D point represented as a numpy array of shape (2,) containing [x, y] coordinates.
		H : np.ndarray
			A 3x3 homography matrix used for the projection transformation.
		"""
		point_homogeneous = np.array([point[0], point[1], 1.0]).reshape(3, 1)
		projected_point_homogeneous = H @ point_homogeneous
		projected_point_homogeneous /= projected_point_homogeneous[2, 0]
		return projected_point_homogeneous[:2, 0]
	
__all__ = ["HomographyManager"]

