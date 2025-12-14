from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import csv
from pathlib import Path

from src.utils.DataLoader import DataLoader
from src.pipeline.detection.PosterDetector import PosterDetector, PosterTemplate
from src.pipeline.tracking.PosterTracker import PosterTracker
from src.pipeline.projection.GazeProjector import GazeProjector


class GazeAnalyser:
    """
    Analyseur de regard qui orchestre la détection, le suivi et la projection.
    """

    def __init__(self, config_path: str) -> None:
        self.loader = DataLoader(config_path)

        # Charger les templates d'affiches
        self.poster_templates: List[PosterTemplate] = self._load_poster_templates()
        if not self.poster_templates:
            raise RuntimeError("No PNG posters found in posters folder")

        # Initialiser les modules du pipeline
        weights_path = self.loader.get_yolo_detection_weights()
        self.detector = PosterDetector(weights_path, self.poster_templates)
        self.projector = GazeProjector()

        # État du pipeline
        self.active_trackers: Dict[int, PosterTracker] = {}
        self.prev_gray: Optional[np.ndarray] = None

    def _load_poster_templates(self) -> List[PosterTemplate]:
        """Charge les templates d'affiches avec plusieurs échelles."""
        posters_raw = self.loader.load_posters()
        templates = []
        
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=12)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for name, img in posters_raw:
            template = PosterTemplate(name, img)
            enhanced_orig = clahe.apply(template.img_orig_gray)

            for scale in PosterDetector.SCALES:
                w_scaled = int(enhanced_orig.shape[1] * scale)
                h_scaled = int(enhanced_orig.shape[0] * scale)
                gray_scaled = cv2.resize(enhanced_orig, (w_scaled, h_scaled))

                kp, des = orb.detectAndCompute(gray_scaled, None)
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

    def analyse_video_for_subject(
        self,
        subject_idx: int,
        output_csv_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        visualize: bool = False,
        always_detect: bool = False,
    ) -> None:
        """
        Analyse la vidéo d'un sujet, applique le pipeline DETECT/TRACK et sauvegarde les projections.
        
        Args:
            always_detect: Si True, désactive le tracking et fait une détection à chaque frame.
        """
        # Paramètres de détection périodique
        DETECT_EVERY_N_FRAMES = 1 if always_detect else 60  # Détection à chaque frame si always_detect
        last_detection_frame = -DETECT_EVERY_N_FRAMES  # Pour forcer la détection dès le début
        
        # 1. Chargement des données
        cap = self.loader.get_video_capture(subject_idx)
        K, D = self.loader.get_load_camera_params(subject_idx)
        gazes_undist = self.loader.get_undistorted_gazes(subject_idx)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None or end_frame > n_frames:
            end_frame = n_frames

        # 2. Préparation du fichier de sortie et de la visualisation
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if visualize:
            cv2.namedWindow("Pipeline Visualization", cv2.WINDOW_NORMAL)

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "poster_name", "poster_index", "proj_x", "proj_y"])

            # 3. Boucle principale sur les frames
            frame_range = range(start_frame, end_frame)
            for frame_idx in tqdm(frame_range, desc=f"Analysing subject {subject_idx}", unit="frame"):
                
                if frame_idx >= len(gazes_undist):
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_undist = cv2.undistort(frame, K, D)
                gray_frame = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
                gaze_point = gazes_undist[frame_idx]
                
                vis_frame = frame_undist.copy() if visualize else None
                pipeline_status = "TRACK"
                
                # Vérifier si détection périodique obligatoire
                frames_since_detection = frame_idx - last_detection_frame
                need_periodic_detection = frames_since_detection >= DETECT_EVERY_N_FRAMES

                # 4. Phase TRACK
                if self.prev_gray is not None and not need_periodic_detection:
                    lost_trackers = []
                    for track_id, tracker in self.active_trackers.items():
                        if not tracker.update(self.prev_gray, gray_frame) or tracker.is_lost():
                            lost_trackers.append(track_id)
                    
                    for track_id in lost_trackers:
                        del self.active_trackers[track_id]

                # 5. Phase DETECT (si nécessaire ou périodique)
                if not self.active_trackers or need_periodic_detection:
                    pipeline_status = "DETECT"
                    last_detection_frame = frame_idx
                    self.active_trackers.clear()  # Reset trackers lors d'une nouvelle détection
                    new_trackers = self.detector.detect(frame_undist, gray_frame, gaze_point)
                    for tracker in new_trackers:
                        # Assigner le template au tracker pour un accès futur
                        tracker.poster_template = self.poster_templates[tracker.poster_index]
                        self.active_trackers[tracker.track_id] = tracker
                
                # 6. Projection du regard et visualisation
                projected_gaze_info = None
                for tracker in self.active_trackers.values():
                    projected_gaze = self.projector.project(gaze_point, tracker)
                    
                    if projected_gaze:
                        px, py = projected_gaze
                        writer.writerow([
                            frame_idx,
                            tracker.poster_name,
                            tracker.poster_index,
                            px,
                            py,
                        ])
                        projected_gaze_info = (tracker.poster_name, px, py)
                        # On ne projette que sur la première affiche trouvée
                        break
                
                if visualize and vis_frame is not None:
                    # Dessiner le point de regard
                    cv2.circle(vis_frame, (int(gaze_point[0]), int(gaze_point[1])), 10, (255, 0, 0), 2)
                    
                    # Afficher le statut
                    cv2.putText(vis_frame, f"Status: {pipeline_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Dessiner les trackers actifs
                    for tracker in self.active_trackers.values():
                        cv2.polylines(vis_frame, [tracker.corners.astype(np.int32)], True, (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"ID: {tracker.track_id}", (int(tracker.corners[0][0]), int(tracker.corners[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Afficher les infos de projection
                    if projected_gaze_info:
                        name, px, py = projected_gaze_info
                        cv2.putText(vis_frame, f"On {name}: ({px:.2f}, {py:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imshow("Pipeline Visualization", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.prev_gray = gray_frame.copy()

        cap.release()
        if visualize:
            cv2.destroyAllWindows()
        self.active_trackers.clear()
        self.prev_gray = None

