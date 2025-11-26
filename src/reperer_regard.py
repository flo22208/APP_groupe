import cv2
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import sys


# Recupération des données
def retrieve_subject_data(num_sujet):
    '''
    Fonction permettant de récupérer les données associées à un sujet
    
    Args: int : le numéro du sujet
    
    Return: - string : le chemin vers la vidéo
            - string : le chemin vers le fichier gaze
            - string : le chemin vers les paramètres de la caméra
    '''
    sujet_ids = ["sujet1_f-42e0d11a", "sujet2_f-835bf855", 
                 "sujet3_m-84ce1158", "sujet4_m-fee537df", 
                 "sujet5_m-671cf44e", "sujet6_m-0b355b51"]
    video_names = ["e0b2c246_0.0-138.011", "b7bd6c34_0.0-271.583",
                   "422f10f2_0.0-247.734", "2fb8301a_0.0-71.632",
                   "585d8df7_0.0-229.268", "429d311a_0.0-267.743"]
    
    sujet_id = sujet_ids[num_sujet-1]
    video_name = video_names[num_sujet-1]
    
    video_file = f"AcquisitionsEyeTracker/{sujet_id}/{video_name}.mp4"
    gaze_file = f"AcquisitionsEyeTracker/{sujet_id}/gaze.csv"
    camera_parameters_file =  f"AcquisitionsEyeTracker/{sujet_id}/scene_camera.json"
    
    return video_file, gaze_file, camera_parameters_file



def affichage_video_undistorted(video_file, gaze_file, camera_parameters_file):
    # Ouverture des fichiers
    scene_camera = json.load(open(camera_parameters_file))
    gaze_csv = pd.read_csv(gaze_file)
    video = cv2.VideoCapture(video_file)

    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_file}")

    # Récupération de la colone gaze x [px]
    gaze_x = gaze_csv['gaze x [px]'].to_numpy()
    # Récupération de la colone gaze y [px]
    gaze_y = gaze_csv['gaze y [px]'].to_numpy()

    # Récupération des paramètres de la caméra
    distortion_coefficients = np.array(scene_camera['distortion_coefficients'])
    camera_matrix = np.array(scene_camera['camera_matrix'])
    
    frame_index = 0
    while not(cv2.waitKey(30) & 0xFF == ord('q')):
        ret, frame = video.read()
        if not(ret):
            print("Error: Could not read the frame.")
            break
        
        # Undistort the frame
        undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)
        h, w = undistorted_frame.shape[:2]
        
        # Safely get gaze coordinates
        
        gx = gaze_x[frame_index]
        gy = gaze_y[frame_index]
        pts = np.array([[[float(gx), float(gy)]]], dtype=np.float32)  # shape (1,1,2)
        # Undistort the point; P=camera_matrix returns pixel coords in the undistorted image
        undist_pt = cv2.undistortPoints(pts, camera_matrix, distortion_coefficients, P=camera_matrix)
        x = int(round(undist_pt[0, 0, 0]))
        y = int(round(undist_pt[0, 0, 1]))

        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        annotated = undistorted_frame.copy()
        cv2.circle(annotated, (x, y), 10, (0, 255, 0), -1)

        cv2.imshow('Annotated Undistorted Video', annotated)
        frame_index += 1
        
    video.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) == 2:
        sujet = int(sys.argv[1])
    else:
        sujet = 2
    video_file, gaze_file, camera_parameters_file = retrieve_subject_data(sujet)
    affichage_video_undistorted(video_file, gaze_file, camera_parameters_file)

if __name__ == '__main__':
    main()
    

