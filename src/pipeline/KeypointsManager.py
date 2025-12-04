import cv2
import numpy as np
from typing import List, Optional, Tuple


class KeypointsManager:
	def __init__(self, nfeatures: int = 1000, scale_factor: float = 1.5, nlevels: int = 8, edge_threshold: int = 31, patch_size: int = 31, ratio_thresh: float = 0.75) -> None:
		self.keypoint_detector = cv2.ORB_create(
			nfeatures=nfeatures,
			scaleFactor=scale_factor,
			nlevels=nlevels,
			edgeThreshold=edge_threshold,
			patchSize=patch_size,
		)
		self.ratio_thresh = ratio_thresh

	def detect_and_describe(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
		if mask is not None and mask.ndim == 3:
			mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

		keypoints, descriptors = self.keypoint_detector.detectAndCompute(gray, mask)
		return keypoints, descriptors

	def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
		matches_knn = bf.knnMatch(desc1, desc2, k=2)
		good_matches: List[cv2.DMatch] = []
		for m, n in matches_knn:
			if m.distance < self.ratio_thresh * n.distance:
				good_matches.append(m)
		return good_matches
	
__all__ = ["KeypointsManager"]