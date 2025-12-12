import cv2
import numpy as np
from typing import List, Optional, Tuple


class KeypointsManager:
	def __init__(self, nfeatures: int = 2000, scale_factor: float = 1.2, nlevels: int = 12, edge_threshold: int = 15, patch_size: int = 31, ratio_thresh: float = 0.70) -> None:
		"""Gestionnaire de keypoints avec paramètres optimisés pour la détection d'affiches.
		
		Paramètres ajustés pour :
		- Plus de features (2000 vs 1000) pour avoir plus de correspondances
		- Scale factor plus fin (1.2 vs 1.5) pour mieux gérer les changements d'échelle
		- Plus de niveaux de pyramide (12 vs 8) pour robustesse multi-échelle
		- Edge threshold plus bas (15 vs 31) pour détecter plus de features près des bords
		- Ratio test plus strict (0.70 vs 0.75) pour des matchs de meilleure qualité
		"""
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
		"""Match des descripteurs avec ratio test de Lowe.
		
		Utilise un matcher BF avec vérification croisée implicite via le ratio test.
		"""
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
		try:
			matches_knn = bf.knnMatch(desc1, desc2, k=2)
		except cv2.error:
			return []
		
		good_matches: List[cv2.DMatch] = []
		for match_pair in matches_knn:
			# Vérifier qu'on a bien 2 matchs pour le ratio test
			if len(match_pair) < 2:
				continue
			m, n = match_pair
			if m.distance < self.ratio_thresh * n.distance:
				good_matches.append(m)
		return good_matches
	
__all__ = ["KeypointsManager"]