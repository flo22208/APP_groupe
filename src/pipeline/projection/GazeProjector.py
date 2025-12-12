from typing import Optional, Tuple
import numpy as np
import cv2

class GazeProjector:
    """Projete le regard sur une affiche."""

    PROJECTION_MARGIN_RATIO = 0.05

    def project(self, gaze_point: Tuple[float, float], tracker) -> Optional[Tuple[float, float]]:
        """Vérifie si le regard est sur l'affiche et le projette si c'est le cas."""
        
        gx, gy = gaze_point
        
        # 1. Vérifier si le regard tombe dans les coins courants
        if not self._is_gaze_on_poster(gx, gy, tracker.corners):
            return None

        # 2. Projeter le regard
        # L'homographie H dans le tracker mappe les points du template vers la ROI.
        # Pour projeter le regard de la frame complète vers le template, il faut ajuster le point de regard.
        # Le point de regard (gx, gy) est dans le repère de la frame complète.
        # L'homographie a été calculée pour une ROI. Il faut translater le point de regard.
        
        # On ne peut pas simplement utiliser la H du tracker car elle a été calculée sur une ROI.
        # Il faudrait recalculer une homographie sur la frame entière ou stocker la translation de la ROI.
        # Pour l'instant, on suppose que la H du tracker est suffisante si le regard est DANS les coins.
        
        # Le plus simple est de recalculer une homographie à la volée pour la projection.
        # Mais cela va à l'encontre de la séparation TRACK/DETECT.
        
        # Solution: On doit inverser la transformation.
        # La H stockée mappe template -> ROI.
        # On a besoin de frame -> template.
        # La H du tracker est H(template -> ROI).
        # Les coins du tracker sont dans le repère de la frame.
        # Les coins originaux du tracker sont dans le repère de la ROI.
        
        # On peut estimer une nouvelle homographie à partir des coins actuels et des coins du template.
        h_template, w_template = tracker.poster_template.img_orig_shape
        
        # Coins du template original (non-scalé)
        template_corners = np.float32([
            [0, 0],
            [w_template, 0],
            [w_template, h_template],
            [0, h_template]
        ])

        # Homographie de la frame complète vers le template
        try:
            H_frame_to_template, _ = cv2.findHomography(tracker.corners, template_corners, cv2.RANSAC, 5.0)
        except cv2.error:
            return None

        if H_frame_to_template is None:
            return None

        pt = np.array([[[gx, gy]]], dtype=np.float32)
        pt_affiche = cv2.perspectiveTransform(pt, H_frame_to_template)

        if pt_affiche is None:
            return None
            
        px, py = pt_affiche[0, 0]

        # Clipping et validation
        margin_x = w_template * self.PROJECTION_MARGIN_RATIO
        margin_y = h_template * self.PROJECTION_MARGIN_RATIO
        
        if (px < -margin_x or px > w_template + margin_x or
            py < -margin_y or py > h_template + margin_y):
            return None

        px = float(np.clip(px, 0, w_template - 1))
        py = float(np.clip(py, 0, h_template - 1))

        return px, py

    def _is_gaze_on_poster(self, gx: float, gy: float, corners: np.ndarray) -> bool:
        """Vérifie si le point de regard est sur une affiche (polygon test)."""
        if corners is None or len(corners) < 3:
            return False
        result = cv2.pointPolygonTest(corners.astype(np.float32), (float(gx), float(gy)), False)
        return result >= 0

    def _project_gaze_on_poster(self, gaze_local: Tuple[float, float], 
                                H: np.ndarray, scale: float,
                                poster_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Projette le point de regard sur l'affiche originale."""
        gx_local, gy_local = gaze_local
        
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        
        pt = np.array([[[gx_local, gy_local]]], dtype=np.float32)
        pt_affiche = cv2.perspectiveTransform(pt, H_inv)
        
        if pt_affiche is None:
            return None
        
        gx_scaled, gy_scaled = pt_affiche[0, 0]
        
        gx_aff = gx_scaled / scale
        gy_aff = gy_scaled / scale
        
        h_orig, w_orig = poster_shape
        gx_aff = float(np.clip(gx_aff, 0, w_orig - 1))
        gy_aff = float(np.clip(gy_aff, 0, h_orig - 1))
        
        margin_x = w_orig * self.PROJECTION_MARGIN_RATIO
        margin_y = h_orig * self.PROJECTION_MARGIN_RATIO
        
        if (gx_aff < -margin_x or gx_aff > w_orig + margin_x or
            gy_aff < -margin_y or gy_aff > h_orig + margin_y):
            return None
        
        return gx_aff, gy_aff
