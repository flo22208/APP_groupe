from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from src.detectionModel.DetectionModel import DetectionModel
from src.pipeline.tracking.PosterTracker import PosterTracker

class PosterTemplate:
    """Structure pour stocker les données multi-échelles d'une affiche."""
    def __init__(self, name: str, img_orig: np.ndarray):
        self.name = name
        self.img_orig = img_orig
        self.img_orig_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) if img_orig.ndim == 3 else img_orig
        self.img_orig_shape = self.img_orig_gray.shape[:2]
        self.multi_scale: List[Dict] = []

class PosterDetector:
    """Détecte les affiches dans une image en utilisant YOLO et ORB."""

    # Paramètres ORB
    NB_POI = 2000
    
    # Paramètres de matching
    MIN_MATCHES_THRESHOLD = 10
    RATIO_THRESHOLD = 0.7
    BEST_THRESHOLD = 15
    
    # Échelles pour la détection multi-échelle
    SCALES = [0.25, 0.5, 0.75, 1.0]

    def __init__(self, yolo_weights_path: str, poster_templates: List[PosterTemplate]):
        self.det_model = DetectionModel(yolo_weights_path)
        self.poster_templates = poster_templates
        self.orb = cv2.ORB_create(nfeatures=self.NB_POI, scaleFactor=1.2, nlevels=12)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, frame: np.ndarray, gray_frame: np.ndarray, gaze_point: Tuple[float, float]) -> List[PosterTracker]:
        """
        Exécute la détection YOLO, trouve l'affiche regardée, et initialise un tracker.
        """
        # 1. Détection YOLO rapide
        xyxy = self.det_model.predict(frame)
        
        # Skip immédiat si pas de détections
        if xyxy is None or len(xyxy) == 0:
            return []

        # 2. Trouver la bbox contenant le point de regard
        gx, gy = gaze_point
        looked_det = None
        min_distance = float('inf')

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                dist = (gx - center_x)**2 + (gy - center_y)**2
                if dist < min_distance:
                    min_distance = dist
                    looked_det = (x1, y1, x2, y2, i)
        
        if looked_det is None:
            return []

        # 3. Pour la bbox regardée, extraire la ROI et matcher
        x1, y1, x2, y2, track_id = looked_det
        roi = gray_frame[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return []

        poster_idx, H, inliers, scale = self._match_roi_to_poster_multiscale(roi)

        if poster_idx is not None and H is not None:
            # 4. Initialiser un tracker KLT
            template = self.poster_templates[poster_idx]
            h_scaled, w_scaled = (int(template.img_orig_shape[0] * scale), int(template.img_orig_shape[1] * scale))
            
            corners_template = np.float32([[0, 0], [w_scaled, 0], [w_scaled, h_scaled], [0, h_scaled]]).reshape(-1, 1, 2)
            corners_frame = cv2.perspectiveTransform(corners_template, H)

            if corners_frame is not None:
                corners_absolute = corners_frame.reshape(-1, 2) + np.array([x1, y1])
                tracker = PosterTracker(
                    poster_index=poster_idx,
                    poster_name=template.name,
                    corners=corners_absolute,
                    gray_frame=gray_frame,
                    homography=H,
                    scale=scale,
                    track_id=track_id
                )
                return [tracker]
        
        return []

    def _detect_and_describe_roi(self, roi_gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """Détecte les keypoints ORB dans une ROI."""
        enhanced = self.clahe.apply(roi_gray)
        kp, des = self.orb.detectAndCompute(enhanced, None)
        return kp if kp else [], des

    def _match_roi_to_poster_multiscale(self, roi_gray: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray], int, float]:
        """Match une ROI contre les templates multi-échelles."""
        kp_roi, des_roi = self._detect_and_describe_roi(roi_gray)
        
        if des_roi is None or len(kp_roi) < 8:
            return None, None, 0, 1.0

        best_poster_idx = None
        best_H = None
        best_inliers = 0
        best_scale = 1.0

        for idx, template in enumerate(self.poster_templates):
            for ms in template.multi_scale:
                kp_t, des_t = ms["kp"], ms["des"]

                if des_t is None or len(des_t) == 0:
                    continue

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
