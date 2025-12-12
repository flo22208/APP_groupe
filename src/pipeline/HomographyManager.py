from typing import List, Optional, Tuple
import cv2
import numpy as np
from src.pipeline.KeypointsManager import KeypointsManager

class HomographyManager:
	# Seuils de qualité pour une homographie fiable
	MIN_INLIERS = 10  # Minimum d'inliers pour considérer l'homographie valide
	MIN_INLIER_RATIO = 0.3  # Au moins 30% des matchs doivent être des inliers
	
	def __init__(self, ransac_thresh: float = 3.0, keypoints_manager: Optional[KeypointsManager] = None) -> None:
		"""Gestionnaire d'homographie avec seuil RANSAC réduit pour plus de précision."""
		self.ransac_thresh = ransac_thresh
		self.keypoints_manager = keypoints_manager or KeypointsManager()

	def estimate_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], int]:
		"""Estime l'homographie avec validation de qualité.
		
		Retourne None si l'homographie n'est pas assez fiable (trop peu d'inliers
		ou ratio d'inliers trop faible).
		"""
		if len(matches) < 8:  # Augmenté de 4 à 8 pour plus de robustesse
			return None, 0

		src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

		H, mask = cv2.findHomography(
			src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh
		)
		if H is None or mask is None:
			return None, 0

		inliers = int(mask.ravel().sum())
		inlier_ratio = inliers / len(matches)
		
		# Vérifier la qualité de l'homographie
		if inliers < self.MIN_INLIERS or inlier_ratio < self.MIN_INLIER_RATIO:
			return None, 0
		
		# Vérifier que l'homographie est géométriquement valide
		if not self._is_valid_homography(H):
			return None, 0
		
		return H, inliers
	
	def _is_valid_homography(self, H: np.ndarray) -> bool:
		"""Vérifie que l'homographie est géométriquement plausible.
		
		Rejecte les homographies avec :
		- Déterminant trop petit (dégénérée) ou négatif (retournement)
		- Transformation trop extrême (scale, shear excessifs)
		"""
		if H is None:
			return False
		
		# Vérifier le déterminant
		det = np.linalg.det(H)
		if det < 0.01 or det > 100:  # Éviter les transformations dégénérées
			return False
		
		# Vérifier la condition number (stabilité numérique)
		try:
			cond = np.linalg.cond(H)
			if cond > 1e6:  # Matrice mal conditionnée
				return False
		except np.linalg.LinAlgError:
			return False
		
		# Vérifier que les coins d'un carré unitaire se transforment correctement
		# (pas d'inversion, pas de distorsion excessive)
		corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32).reshape(-1, 1, 2)
		transformed = cv2.perspectiveTransform(corners, H)
		
		if transformed is None:
			return False
		
		# Vérifier l'aire du quadrilatère transformé (doit être positive = pas de retournement)
		pts = transformed.reshape(-1, 2)
		area = 0.5 * abs(
			(pts[0, 0] - pts[2, 0]) * (pts[1, 1] - pts[3, 1]) -
			(pts[1, 0] - pts[3, 0]) * (pts[0, 1] - pts[2, 1])
		)
		
		# L'aire doit être raisonnable (ni trop petite ni trop grande)
		if area < 0.001 or area > 1000:
			return False
		
		return True

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
		
		# Éviter la division par zéro
		w = projected_point_homogeneous[2, 0]
		if abs(w) < 1e-10:
			return np.array([np.nan, np.nan])
		
		projected_point_homogeneous /= w
		return projected_point_homogeneous[:2, 0]
	
	def project_point_with_bounds(self, point: np.ndarray, H: np.ndarray, 
									 target_width: int, target_height: int,
									 margin_ratio: float = 0.05) -> Tuple[np.ndarray, bool]:
		"""
		Projecte un point et vérifie s'il est dans les limites de l'image cible.
		
		Retourne (point_projeté, est_valide).
		Si le point est légèrement hors limites (dans la marge), il est clippé.
		Si le point est trop loin hors limites, est_valide est False.
		"""
		projected = self.project_point(point, H)
		
		if np.any(np.isnan(projected)):
			return projected, False
		
		px, py = projected
		
		# Marge acceptable (5% par défaut)
		margin_x = target_width * margin_ratio
		margin_y = target_height * margin_ratio
		
		# Vérifier si le point est dans les limites avec marge
		if (px < -margin_x or px > target_width + margin_x or
			py < -margin_y or py > target_height + margin_y):
			return projected, False
		
		# Clipper le point aux dimensions de l'image
		px_clipped = np.clip(px, 0, target_width - 1)
		py_clipped = np.clip(py, 0, target_height - 1)
		
		return np.array([px_clipped, py_clipped]), True
	
__all__ = ["HomographyManager"]

