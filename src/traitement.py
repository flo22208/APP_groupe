### importation bibliothèques
import cv2
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd

### path video, scene_camera.json, gaze.csv
path_video = r"C:\Users\flori\Documents\Travail\APP\APP\AcquisitionsEyeTracker\sujet1_f-42e0d11a\e0b2c246_0.0-138.011.mp4"
path_scene_camera = "C:/Users/flori/Documents/Travail/APP/APP/AcquisitionsEyeTracker/sujet1_f-42e0d11a/scene_camera.json"
path_gaze_csv = "C:/Users/flori/Documents/Travail/APP/APP/AcquisitionsEyeTracker/sujet1_f-42e0d11a/gaze.csv"
path_affiche_Anning = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/Anning.png"
path_affiche_Barres = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/Barres.png"
path_affiche_Bell = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/Bell.png"
path_affiche_Bunten_Berry = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/Bunten-Berry.png"
path_affiche_Franklin = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/Franklin.png"
path_affiche_Gautier = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/Gautier.png"
path_affiche_johnson = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/johnson.png"
path_affiche_Noether = "C:/Users/flori/Documents/Travail/APP/APP/Affiches/Noether.png"

### importation 
print("Importation des données...")
scene_camera = json.load(open(path_scene_camera))
gaze_csv = pd.read_csv(path_gaze_csv)
video = cv2.VideoCapture(path_video)
if not video.isOpened():
    raise FileNotFoundError(f"Cannot open video file: {path_video}")
anning_image = cv2.imread(path_affiche_Anning)
barres_image = cv2.imread(path_affiche_Barres)
bell_image = cv2.imread(path_affiche_Bell)
bunten_berry_image = cv2.imread(path_affiche_Bunten_Berry)
franklin_image = cv2.imread(path_affiche_Franklin)
gautier_image = cv2.imread(path_affiche_Gautier)
johnson_image = cv2.imread(path_affiche_johnson)
noether_image = cv2.imread(path_affiche_Noether)

##importation de la colone gaze x [px]
gaze_x = gaze_csv['gaze x [px]'].to_numpy()
##importation de la colone gaze y [px]
gaze_y = gaze_csv['gaze y [px]'].to_numpy()

distortion_coefficients = np.array(scene_camera['distortion_coefficients'])
camera_matrix = np.array(scene_camera['camera_matrix'])

def affichage_disdorded_video():
    ### affichage des gaze_x et gaze_y sur la vidéo pour chaque frame
    frame_index = 0
    print("Affichage des données de regard sur la vidéo...")
    while True:
        ret, frame = video.read()

        # Stop if no frame was read
        if not ret or frame is None:
            print(f"No frame returned at index {frame_index}. Stopping playback.")
            break

        h, w = frame.shape[:2]

        # Stop if no gaze data for this frame
        if frame_index >= len(gaze_x) or frame_index >= len(gaze_y):
            print(f"No gaze data for frame {frame_index}. Stopping playback.")
            break

        # Safely get gaze coordinates
        gx = gaze_x[frame_index]
        gy = gaze_y[frame_index]


        # Handle missing/NaN gaze values
        if pd.isna(gx) or pd.isna(gy):
            annotated = frame.copy()
            print(f"Frame {frame_index}: gaze is NaN, skipping annotation.")
        else:
            try:
                x = int(round(float(gx)))
                y = int(round(float(gy)))
            except Exception as e:
                print(f"Frame {frame_index}: invalid gaze value ({gx}, {gy}) -> {e}")
                annotated = frame.copy()
            else:
                # clamp coordinates to frame bounds
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                annotated = frame.copy()
                cv2.circle(annotated, (x, y), 10, (0, 255, 0), -1)
                print(f"Frame {frame_index}: Gaze at ({x}, {y})")

        cv2.imshow('Annotated Video', annotated)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_index += 1
    video.release()
    cv2.destroyAllWindows()

def affchiage_undistorded_video():
    ### affichage des gaze_x et gaze_y sur la vidéo undistorded pour chaque frame
    frame_index = 0
    print("Affichage des données de regard sur la vidéo undistorded...")
    while True:
        ret, frame = video.read()

        # Stop if no frame was read
        if not ret or frame is None:
            print(f"No frame returned at index {frame_index}. Stopping playback.")
            break

        # Undistort the frame
        undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)

        h, w = undistorted_frame.shape[:2]

        # Stop if no gaze data for this frame
        if frame_index >= len(gaze_x) or frame_index >= len(gaze_y):
            print(f"No gaze data for frame {frame_index}. Stopping playback.")
            break

        # Safely get gaze coordinates
        gx = gaze_x[frame_index]
        gy = gaze_y[frame_index]

        # Handle missing/NaN gaze values
        if pd.isna(gx) or pd.isna(gy):
            annotated = undistorted_frame.copy()
            print(f"Frame {frame_index}: gaze is NaN, skipping annotation.")
        else:
            try:
                pts = np.array([[[float(gx), float(gy)]]], dtype=np.float32)  # shape (1,1,2)
                # Undistort the point; P=camera_matrix returns pixel coords in the undistorted image
                undist_pt = cv2.undistortPoints(pts, camera_matrix, distortion_coefficients, P=camera_matrix)
                x = int(round(undist_pt[0, 0, 0]))
                y = int(round(undist_pt[0, 0, 1]))
            except Exception as e:
                print(f"Frame {frame_index}: invalid gaze value ({gx}, {gy}) -> {e}")
                annotated = undistorted_frame.copy()
            else:
                # clamp coordinates to frame bounds
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                annotated = undistorted_frame.copy()
                cv2.circle(annotated, (x, y), 10, (0, 255, 0), -1)
                print(f"Frame {frame_index}: Undistorted gaze at ({x}, {y})")

        cv2.imshow('Annotated Undistorted Video', annotated)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_index += 1
    video.release()
    cv2.destroyAllWindows()


affchiage_undistorded_video()

def list_x_y_undistorded():
    ### liste des gaze_x et gaze_y undistorded
    undistorted_gaze_x = []
    undistorted_gaze_y = []
    print("Calcul des coordonnées de regard undistorded...")
    for i in range(len(gaze_x)):
        gx = gaze_x[i]
        gy = gaze_y[i]

        if pd.isna(gx) or pd.isna(gy):
            undistorted_gaze_x.append(np.nan)
            undistorted_gaze_y.append(np.nan)
            print(f"Index {i}: gaze is NaN, skipping undistortion.")
        else:
            try:
                pts = np.array([[[float(gx), float(gy)]]], dtype=np.float32)  # shape (1,1,2)
                undist_pt = cv2.undistortPoints(pts, camera_matrix, distortion_coefficients, P=camera_matrix)
                x = undist_pt[0, 0, 0]
                y = undist_pt[0, 0, 1]
            except Exception as e:
                print(f"Index {i}: invalid gaze value ({gx}, {gy}) -> {e}")
                undistorted_gaze_x.append(np.nan)
                undistorted_gaze_y.append(np.nan)
            else:
                undistorted_gaze_x.append(x)
                undistorted_gaze_y.append(y)
                print(f"Index {i}: Undistorted gaze at ({x}, {y})")
    return undistorted_gaze_x, undistorted_gaze_y


#list_gx_undist, list_gy_undist = list_x_y_undistorded()

