import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import os
import csv
from src.utils.data_loader import DataLoader

NB_POI = 2000
TOP_FILTRAGE = 1000
subject_idx = 2
poster_folder = "posters/"
DETECT_EVERY_N_FRAMES = 5
MIN_MATCHES = 5
RATIO_THRESHOLD = 0.7
BEST_THRESHOLD = 5
FILTRAGE = 0
ROI = 1



# CHARGEMENT DES TEMPLATES
def load_poster_templates(folder, scales=[0.25, 0.5, 0.75, 1.0]):
    orb = cv2.ORB_create(nfeatures=NB_POI)
    templates = {}

    for path in Path(folder).glob("*"):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img = cv2.imread(str(path))
        if img is None:
            print("Impossible de charger :", path)
            continue

        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Pour chaque échelle
        multi_scale_data = []
        for scale in scales:
            w_scaled = int(gray_orig.shape[1] * scale)
            h_scaled = int(gray_orig.shape[0] * scale)
            gray = cv2.resize(gray_orig, (w_scaled, h_scaled))

            kp, des = orb.detectAndCompute(gray, None)
            if kp is None or len(kp) == 0:
                continue

            multi_scale_data.append({
                "scale": scale,
                "img": gray,
                "kp": kp,
                "des": des
            })

        templates[path.name] = {
            "name": path.name,
            "img_orig_shape": gray_orig.shape[:2],
            "multi_scale": multi_scale_data
        }

        total_kp = sum(len(ms["kp"]) for ms in multi_scale_data)
        print(f"> Poster chargé : {path.name}, total keypoints toutes échelles : {total_kp}")

    return templates, orb


# VALIDATION HOMOGRAPHIE

def validate_homography(M, template_shape, frame_shape):
    if M is None:
        return False
    
    h, w = template_shape[:2]
    frame_h, frame_w = frame_shape[:2]

    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)

    try:
        transformed = cv2.perspectiveTransform(corners, M)
    except:
        return False

    transformed = transformed.reshape(-1, 2)
    area = cv2.contourArea(transformed)
    original_area = w * h
    
    if area < 0.01 * original_area or area > 10 * original_area:
        return False

    return True



# MATCHING AFFICHES avec ROI dynamique


def filter_best_keypoints(kp, des, top_n=500):
    if len(kp) <= top_n:
        return kp, des
    # Trier par réponse
    kp_des = sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)
    kp, des = zip(*kp_des[:top_n])
    return list(kp), np.array(des)

def match_posters_in_frame(frame, templates, orb, gaze_x=None, gaze_y=None,
                           roi_width=900, roi_height=None,
                           min_matches=10, ratio_threshold=0.7, best_threshold=15,
                           last_detected=[]):
    h, w = frame.shape[:2]


    # ROI autour du regard
    x1_roi = max(0, gaze_x - roi_width // 2)
    x2_roi = min(w, gaze_x + roi_width // 2)
    y1_roi = 0 if roi_height is None else max(0, gaze_y - roi_height // 2)
    y2_roi = h if roi_height is None else min(h, gaze_y + roi_height // 2)

    # Si on a déjà détecté une affiche, adapter la ROI autour de ses coins
    if len(last_detected) > 0:
        poster = last_detected[0]
        corners = poster["corners"]
        x_min = int(np.min(corners[:,0]))
        x_max = int(np.max(corners[:,0]))
        y_min = int(np.min(corners[:,1]))
        y_max = int(np.max(corners[:,1]))

        padding = 20
        x1_roi = max(0, x_min - padding)
        x2_roi = min(w, x_max + padding)
        y1_roi = max(0, y_min - padding)
        y2_roi = min(h, y_max + padding)

    roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]

    # Calcul des POI dans la ROI
    kp_roi, des_roi = orb.detectAndCompute(roi, None)
    if des_roi is None or len(kp_roi) == 0:
        return [], frame.copy()
    kp_roi, des_roi = filter_best_keypoints(kp_roi, des_roi, top_n=TOP_FILTRAGE)
    img_kp_roi = cv2.drawKeypoints(roi, kp_roi, None, color=(0,255,0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("ROI Keypoints", img_kp_roi)
    cv2.waitKey(1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    best_poster = None
    best_score = 0
    detected_posters = []

    annotated = frame.copy()
    cv2.rectangle(annotated, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 255, 0), 2)

    for poster in templates.values():
        best_scale_score = 0
        best_scale_data = None
        best_H = None
        best_inliers = 0

        # Tester toutes les échelles
        for ms in poster["multi_scale"]:
            kp_t, des_t = ms["kp"], ms["des"]

            if des_t is None or len(des_t) == 0:
                continue

            # KNN matching
            matches = bf.knnMatch(des_t, des_roi, k=2)
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio_threshold * n.distance:
                        good.append(m)
                elif len(m_n) == 1:
                    good.append(m_n[0])

            if len(good) < min_matches:
                continue

            src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Ajouter offset ROI
            dst_pts[:, :, 0] += x1_roi
            dst_pts[:, :, 1] += y1_roi

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                continue

            inliers = mask.sum()

            if inliers > best_scale_score:
                best_scale_score = inliers
                best_scale_data = ms
                best_H = H
                best_inliers = inliers

        # Garder le meilleur parmi toutes les échelles
        if best_scale_score > best_score and best_inliers >= best_threshold:
            best_score = best_scale_score
            best_poster = poster.copy()
            best_poster["H"] = best_H
            best_poster["inliers"] = best_inliers
            best_poster["best_scale"] = best_scale_data 

    if best_poster is None:
        return [], annotated

    # Dessiner l’affiche détectée
    h_t, w_t = best_poster["best_scale"]["img"].shape[:2]
    corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, best_poster["H"])
    best_poster["corners"] = projected.reshape(4, 2)

    pts = best_poster["corners"].astype(int)
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)

    detected_posters.append(best_poster)
    return detected_posters, annotated


# MATCHING AFFICHES SANS ROI
def match_posters_in_frame_no_roi(frame, templates, orb,
                                  min_matches=5, ratio_threshold=0.7, best_threshold=5):
    """
    Détecte les affiches dans l'image complète sans ROI,
    en testant plusieurs échelles pour chaque template.
    """
    h, w = frame.shape[:2]

    # Calcul des features sur l'image entière
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    if des_frame is None or len(kp_frame) == 0:
        return [], frame.copy()
    if FILTRAGE:
        kp_frame, des_frame = filter_best_keypoints(kp_frame, des_frame, top_n=NB_POI)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    best_poster = None
    best_score = 0
    detected_posters = []

    annotated = frame.copy()

    for poster in templates.values():
        best_scale_score = 0
        best_scale_data = None
        best_H = None
        best_inliers = 0

        # Tester toutes les échelles
        for ms in poster["multi_scale"]:
            kp_t, des_t = ms["kp"], ms["des"]

            if des_t is None or len(des_t) == 0:
                continue

            # KNN matching
            matches = bf.knnMatch(des_t, des_frame, k=2)
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio_threshold * n.distance:
                        good.append(m)
                elif len(m_n) == 1:
                    good.append(m_n[0])

            if len(good) < min_matches:
                continue

            src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                continue

            inliers = mask.sum()

            if inliers > best_scale_score:
                best_scale_score = inliers
                best_scale_data = ms
                best_H = H
                best_inliers = inliers

        # Garder le meilleur parmi toutes les échelles
        if best_scale_score > best_score and best_inliers >= best_threshold:
            best_score = best_scale_score
            best_poster = poster.copy()
            best_poster["H"] = best_H
            best_poster["inliers"] = best_inliers
            best_poster["best_scale"] = best_scale_data  # <-- stocker le dict complet

    if best_poster is None:
        return [], annotated

    # Dessiner l'affiche détectée
    h_t, w_t = best_poster["best_scale"]["img"].shape[:2]
    corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, best_poster["H"])
    best_poster["corners"] = projected.reshape(4, 2)

    pts = best_poster["corners"].astype(int)
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)

    detected_posters.append(best_poster)
    return detected_posters, annotated





# GAZE SUR L’AFFICHE
def check_gaze_on_poster(x, y, corners):
    result = cv2.pointPolygonTest(corners.reshape(-1,2), (float(x), float(y)), False)
    return result >= 0



# PROGRAMME PRINCIPAL

print("Chargement templates…")
templates, orb = load_poster_templates(poster_folder)

# DataLoader
loader = DataLoader("config.json")
gazes = loader.get_undistorted_gazes(subject_idx)
cap = loader.get_video_capture(subject_idx)
K, D = loader.get_load_camera_params(subject_idx)

delay_ms = 1
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames = min(num_frames, len(gazes))

frame_index = 0
last_detected = []
gaze_stats = {}


os.makedirs("gaze_points", exist_ok=True)

csv_files = {}
file_handles = {}

for poster_name in templates.keys():
    fh = open(f"gaze_points/{poster_name}.csv", "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["x_aff", "y_aff"])  
    csv_files[poster_name] = writer
    file_handles[poster_name] = fh

print("Détection démarrée…")

while True:
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if idx >= num_frames:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame_und = cv2.undistort(frame, K, D)
    x, y = map(int, gazes[idx])

    if frame_index % DETECT_EVERY_N_FRAMES == 0:
        if ROI : 
            detected, annotated = match_posters_in_frame(
                frame_und, templates, orb,
                gaze_x=x, gaze_y=y,
                min_matches=MIN_MATCHES,
                ratio_threshold=RATIO_THRESHOLD,
                best_threshold=BEST_THRESHOLD
            )
        else : 
            detected, annotated = match_posters_in_frame_no_roi(
            frame_und, templates, orb,
            min_matches=MIN_MATCHES,
            ratio_threshold=RATIO_THRESHOLD,
            best_threshold=BEST_THRESHOLD
        )

        last_detected = detected
    else:
        annotated = frame_und.copy()

    gaze_color = (0, 255, 0)


    # Enregistrement des points transformés → CSV
    
    for poster in last_detected:
        if check_gaze_on_poster(x, y, poster["corners"]):
            gaze_color = (0, 0, 255)

            # Dimensions de l'image à l'échelle utilisée pour la détection
            h_scaled, w_scaled = poster["best_scale"]["img"].shape[:2]
            scale = poster["best_scale"]["scale"]
            
            # Points sources (coins du template à l'échelle réduite)
            src_corners = np.float32([[0, 0], [w_scaled, 0], [w_scaled, h_scaled], [0, h_scaled]])
            
            # Points destinations (coins projetés dans la frame)
            dst_corners = poster["corners"].astype(np.float32)
            
            # Calculer l'homographie inverse (frame -> template à l'échelle réduite)
            H_inv, _ = cv2.findHomography(dst_corners, src_corners)
            
            if H_inv is not None:
                pt = np.array([[[x, y]]], dtype=np.float32)
                pt_affiche = cv2.perspectiveTransform(pt, H_inv)
                gx_aff_scaled, gy_aff_scaled = pt_affiche[0, 0]
                
                # Rescaler vers l'image originale du poster
                gx_aff = gx_aff_scaled / scale
                gy_aff = gy_aff_scaled / scale
                
                # Dimensions de l'image originale pour le clipping
                h_orig, w_orig = poster["img_orig_shape"]
                gx_aff = float(np.clip(gx_aff, 0, w_orig - 1))
                gy_aff = float(np.clip(gy_aff, 0, h_orig - 1))

                # Sauvegarde CSV
                csv_files[poster["name"]].writerow([gx_aff, gy_aff])
                gaze_stats[poster["name"]] = gaze_stats.get(poster["name"], 0) + 1
            break


    cv2.circle(annotated, (x, y), 10, gaze_color, -1)

    # Affichage stats en overlay
    y0 = 30
    for name, count in gaze_stats.items():
        cv2.putText(annotated, f"{name}: {count}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2)
        y0 += 30

    cv2.namedWindow("Poster detection + gaze", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Poster detection + gaze", 1280, 720)  # largeur x hauteur
    cv2.imshow("Poster detection + gaze", annotated)
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord("q"):
        break

    frame_index += 1

# Fermeture des fichiers CSV
for fh in file_handles.values():
    fh.close()

cap.release()
cv2.destroyAllWindows()
