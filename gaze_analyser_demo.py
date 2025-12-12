"""
Demo temps réel du GazeAnalyser avec analyse_gaze_on_frame.

Affiche la vidéo avec :
- Les bounding boxes des affiches détectées
- Le point de regard
- L'affiche détectée et la projection du regard sur celle-ci
"""

import cv2
import numpy as np
import pandas as pd
import os

from src.pipeline.GazeAnalyser import GazeAnalyser
from src.utils.DataLoader import DataLoader


def draw_poster_thumbnail(frame: np.ndarray, poster_img: np.ndarray, 
                          proj_point: tuple, poster_name: str,
                          position: tuple = (10, 10), max_height: int = 200) -> np.ndarray:
    """Dessine une miniature de l'affiche avec le point de projection."""
    h, w = poster_img.shape[:2]
    scale = max_height / h
    new_w, new_h = int(w * scale), int(h * scale)
    
    thumbnail = cv2.resize(poster_img, (new_w, new_h))
    
    # Dessiner le point de projection sur la miniature
    px, py = proj_point
    px_scaled = int(px * scale)
    py_scaled = int(py * scale)
    cv2.circle(thumbnail, (px_scaled, py_scaled), radius=6, color=(0, 0, 255), thickness=-1)
    cv2.circle(thumbnail, (px_scaled, py_scaled), radius=8, color=(255, 255, 255), thickness=2)
    
    # Position où placer la miniature
    x_pos, y_pos = position
    
    # S'assurer que la miniature tient dans la frame
    if y_pos + new_h > frame.shape[0]:
        new_h = frame.shape[0] - y_pos
        thumbnail = thumbnail[:new_h, :]
    if x_pos + new_w > frame.shape[1]:
        new_w = frame.shape[1] - x_pos
        thumbnail = thumbnail[:, :new_w]
    
    # Dessiner un fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_pos - 5, y_pos - 25), (x_pos + new_w + 5, y_pos + new_h + 5), 
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Placer la miniature
    frame[y_pos:y_pos + new_h, x_pos:x_pos + new_w] = thumbnail
    
    # Ajouter le nom de l'affiche
    cv2.putText(frame, poster_name[:30], (x_pos, y_pos - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_info_panel(frame: np.ndarray, info: dict, position: tuple = (10, 10)) -> np.ndarray:
    """Dessine un panneau d'informations sur la frame."""
    x, y = position
    line_height = 25
    
    # Fond semi-transparent
    overlay = frame.copy()
    panel_height = len(info) * line_height + 20
    panel_width = 350
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Texte
    y_offset = y + 20
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
    
    return frame


def main():
    config_path = "config.json"
    
    # Paramètres
    subject_idx = 1  # Sujet à analyser
    delay_ms = 1  # Délai entre frames (1 = le plus rapide possible)
    
    print("Initialisation du GazeAnalyser...")
    analyser = GazeAnalyser(config_path)
    loader = DataLoader(config_path)
    
    # Charger les données du sujet
    print(f"Chargement des données pour le sujet {subject_idx}...")
    cap = loader.get_video_capture(subject_idx)
    K, D = loader.get_load_camera_params(subject_idx)
    gazes_undist = loader.get_undistorted_gazes(subject_idx)
    
    # Charger le CSV de détections
    det_csv_path = loader.get_detection_results_path(subject_idx)
    if not os.path.exists(det_csv_path):
        raise RuntimeError(f"Detection results CSV not found: {det_csv_path}. "
                          f"Run generate_tracking_files.py first.")
    tracks_df = pd.read_csv(det_csv_path)
    
    # Créer un dictionnaire des affiches pour l'affichage
    posters_dict = {t.name: t.img_orig for t in analyser.poster_templates}
    
    # Nombre de frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, len(gazes_undist))
    
    print(f"Démarrage de l'analyse en temps réel ({num_frames} frames)...")
    print("Contrôles: 'q' ou ESC pour quitter, SPACE pour pause")
    
    paused = False
    frame_idx = 0
    
    # Stats
    total_analysed = 0
    total_success = 0
    
    while frame_idx < num_frames:
        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx += 1
                continue
            
            # Undistort frame
            frame_undist = cv2.undistort(frame, K, D)
            display_frame = frame_undist.copy()
            
            # Coordonnées du regard
            gx, gy = gazes_undist[frame_idx]
            
            # Dessiner le point de regard
            cv2.circle(display_frame, (int(gx), int(gy)), radius=10, 
                      color=(0, 255, 0), thickness=2)
            cv2.circle(display_frame, (int(gx), int(gy)), radius=3, 
                      color=(0, 255, 0), thickness=-1)
            
            # Dessiner les bounding boxes des détections
            frame_detections = tracks_df[tracks_df["frame"] == frame_idx]
            for _, row in frame_detections.iterrows():
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                track_id = int(row["track_id"])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, f"ID:{track_id}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Analyser le regard sur cette frame
            poster_name = None
            proj_point = None
            error_msg = None
            
            total_analysed += 1
            
            try:
                poster_name, poster_idx, proj_point = analyser.analyse_gaze_on_frame(
                    frame_undist,
                    (gx, gy),
                    frame_idx,
                    subject_idx,
                    tracks_df
                )
                total_success += 1
                
                # Dessiner la miniature de l'affiche avec la projection
                poster_img = posters_dict.get(poster_name)
                if poster_img is not None:
                    # Position en bas à droite
                    thumb_x = display_frame.shape[1] - 320
                    thumb_y = display_frame.shape[0] - 250
                    display_frame = draw_poster_thumbnail(
                        display_frame, poster_img, proj_point, poster_name,
                        position=(thumb_x, thumb_y), max_height=200
                    )
                    
            except RuntimeError as e:
                error_msg = str(e)[:50]
            
            # Panneau d'informations
            info = {
                "Frame": f"{frame_idx}/{num_frames}",
                "Gaze": f"({int(gx)}, {int(gy)})",
                "Detections": len(frame_detections),
                "Success rate": f"{total_success}/{total_analysed} ({100*total_success/max(1,total_analysed):.1f}%)"
            }
            if poster_name:
                info["Poster"] = poster_name[:25]
                info["Projection"] = f"({proj_point[0]:.1f}, {proj_point[1]:.1f})"
            if error_msg:
                info["Status"] = error_msg
            
            display_frame = draw_info_panel(display_frame, info)
            
            cv2.imshow("GazeAnalyser Real-time Demo", display_frame)
            frame_idx += 1
        
        # Gestion des touches
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q') or key == 27:  # q ou ESC pour quitter
            break
        elif key == ord(' '):  # SPACE pour pause
            paused = not paused
            if paused:
                print("Pause - Appuyez sur SPACE pour reprendre")
        elif key == ord('n') and paused:  # N pour frame suivante en pause
            paused = False
            # Sera remis en pause après une frame
        elif key == ord('r'):  # R pour reset
            frame_idx = 0
            analyser.poster_index_by_track.clear()
            analyser.vote_cache.clear()
            analyser.match_scores_cache.clear()
            analyser.active_trackers.clear()
            analyser.prev_gray = None
            total_analysed = 0
            total_success = 0
            print("Reset effectué")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nAnalyse terminée!")
    print(f"Frames analysées: {total_analysed}")
    print(f"Projections réussies: {total_success}")
    print(f"Taux de succès: {100*total_success/max(1,total_analysed):.1f}%")


if __name__ == "__main__":
    main()
