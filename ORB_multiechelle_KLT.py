import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import os
import csv
from src.utils.data_loader import DataLoader

# PARAMÈTRES GLOBAUX

NB_POI = 2000
TOP_FILTRAGE = 1000
subject_idx = 2
poster_folder = "posters/"
DETECT_EVERY_N_FRAMES_NO_TRACK = 5  
DETECT_EVERY_N_FRAMES_TRACKING = 30  
MIN_MATCHES = 10
RATIO_THRESHOLD = 0.7
BEST_THRESHOLD = 15
FILTRAGE = 0
ROI = 1

# CHARGEMENT DES TEMPLATES MULTI-ÉCHELLES
def load_poster_templates(folder, scales=[0.25, 0.5, 0.75, 1.0]):
    """Charge les templates d'affiches avec plusieurs échelles"""
    orb = cv2.ORB_create(nfeatures=NB_POI, scaleFactor=1.2, nlevels=12)
    templates = {}

    for path in Path(folder).glob("*"):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img = cv2.imread(str(path))
        if img is None:
            print(f"Impossible de charger : {path}")
            continue

        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Amélioration avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_orig = clahe.apply(gray_orig)

        # Pour chaque échelle
        multi_scale_data = []
        for scale in scales:
            w_scaled = int(enhanced_orig.shape[1] * scale)
            h_scaled = int(enhanced_orig.shape[0] * scale)
            gray = cv2.resize(enhanced_orig, (w_scaled, h_scaled))

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
            "img_orig": gray_orig,
            "img_orig_shape": gray_orig.shape[:2],
            "multi_scale": multi_scale_data
        }

        total_kp = sum(len(ms["kp"]) for ms in multi_scale_data)
        print(f"> Poster chargé : {path.name}, {len(multi_scale_data)} échelles, {total_kp} keypoints")

    return templates, orb

# FILTRAGE DES KEYPOINTS

def filter_best_keypoints(kp, des, top_n=500):
    """Garde les N meilleurs keypoints selon leur réponse"""
    if len(kp) <= top_n:
        return kp, des
    kp_des = sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)
    kp, des = zip(*kp_des[:top_n])
    return list(kp), np.array(des)

# VALIDATION HOMOGRAPHIE
def validate_homography(M, template_shape, frame_shape):
    """Valide si l'homographie est raisonnable"""
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

# PHASE DETECT - DÉTECTION MULTI-ÉCHELLES
def detect_posters_multiscale(frame, templates, orb, gaze_x=None, gaze_y=None,
                              roi_width=900, roi_height=None, min_matches=10, 
                              ratio_threshold=0.7, best_threshold=15, 
                              last_detected=[], annotated=None):

    """Détecte les affiches avec approche multi-échelles"""
    h, w = frame.shape[:2]
    
    # Amélioration de la frame
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_frame = clahe.apply(frame)

    # Définition de la ROI
    if ROI and gaze_x is not None and gaze_y is not None:
        if len(last_detected) > 0:
                poster = last_detected[0]
                corners = poster["corners"]
                x_min = int(np.min(corners[:,0])); x_max = int(np.max(corners[:,0]))
                y_min = int(np.min(corners[:,1])); y_max = int(np.max(corners[:,1]))
                padding = 20
                x1_roi = max(0, x_min - padding)
                x2_roi = min(w, x_max + padding)
                y1_roi = max(0, y_min - padding)
                y2_roi = min(h, y_max + padding)
        else:
            roi_w = 900 if ROI else w
            x1_roi = max(0, x - roi_w // 2)
            x2_roi = min(w, x + roi_w // 2)
            if roi_height is None:
                y1_roi, y2_roi = 0, h
            else:
                y1_roi = max(0, y - roi_height // 2)
                y2_roi = min(h, y + roi_height // 2)

        if annotated is not None:
            cv2.rectangle(annotated, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 255), 2)
            cv2.putText(annotated, "ROI", (x1_roi+5, max(15, y1_roi+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        
        # Extraire la ROI
        roi = enhanced_frame[y1_roi:y2_roi, x1_roi:x2_roi]
        
        # Calcul des features dans la ROI
        kp_roi, des_roi = orb.detectAndCompute(roi, None)
        
        if des_roi is None or len(kp_roi) == 0:
            return []
        
        if FILTRAGE:
            kp_roi, des_roi = filter_best_keypoints(kp_roi, des_roi, top_n=TOP_FILTRAGE)
        
        # Ajuster les coordonnées des keypoints pour la frame complète
        kp_frame = []
        for kp in kp_roi:
            new_kp = cv2.KeyPoint(kp.pt[0] + x1_roi, kp.pt[1] + y1_roi, 
                                  kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            kp_frame.append(new_kp)
        des_frame = des_roi
    else:
        x1_roi, y1_roi = 0, 0
        
        # Calcul des features sur toute la frame
        kp_frame, des_frame = orb.detectAndCompute(enhanced_frame, None)
    
    if des_frame is None or len(kp_frame) == 0:
        return []
    
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    best_poster = None
    best_score = 0
    detected_posters = []

    # Tester chaque template avec toutes ses échelles
    for poster in templates.values():
        best_scale_score = 0
        best_scale_data = None
        best_H = None
        best_inliers = 0

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
            
            # Si on utilise la ROI, les keypoints de la frame sont déjà dans le bon référentiel
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                continue

            inliers = mask_ransac.sum()

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
        return []

    # Calculer les coins projetés
    h_t, w_t = best_poster["best_scale"]["img"].shape[:2]
    corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, best_poster["H"])
    best_poster["corners"] = projected.reshape(4, 2)

    detected_posters.append(best_poster)
    print(f"Affiche détectée: {best_poster['name']} - {best_poster['inliers']} inliers (échelle {best_poster['best_scale']['scale']})")
    
    return detected_posters

# PHASE TRACK - SUIVI KLT
def init_klt_tracking(gray_frame, corners):
    """Initialise le suivi KLT avec les coins détectés"""
    feature_params = dict(maxCorners=50,
                         qualityLevel=0.01,
                         minDistance=7,
                         blockSize=7)
    
    mask = np.zeros(gray_frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(corners)], 255)
    
    features = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
    
    if features is None or len(features) < 4:
        features = corners.reshape(-1, 1, 2).astype(np.float32)
    
    return features

def track_klt(prev_gray, curr_gray, prev_features):
    """Suit les points avec KLT"""
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    next_features, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_features, None, **lk_params)
    
    if next_features is None:
        return None, None
    
    good_new = next_features[status == 1]
    good_old = prev_features[status == 1]
    
    return good_new, good_old

def estimate_corners_from_tracking(tracked_points, original_points, original_corners):
    """Estime la position des coins à partir des points suivis"""
    if len(tracked_points) < 4 or len(original_points) < 4:
        return None
    
    tracked_pts = tracked_points.reshape(-1, 2).astype(np.float32)
    original_pts = original_points.reshape(-1, 2).astype(np.float32)
    
    if tracked_pts.shape[0] != original_pts.shape[0]:
        return None
    
    M, mask = cv2.findHomography(original_pts, tracked_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        return None
    
    corners = original_corners.reshape(-1, 1, 2).astype(np.float32)
    new_corners = cv2.perspectiveTransform(corners, M)
    
    return new_corners.reshape(-1, 2)

class PosterTracker:
    """Gère le suivi d'une affiche avec KLT"""
    def __init__(self, poster_data, gray_frame):
        self.name = poster_data['name']
        self.corners = poster_data['corners']
        self.poster_data = poster_data  # Contient img_orig_shape, best_scale, etc.
        self.features = init_klt_tracking(gray_frame, self.corners)
        self.original_features = self.features.copy()
        self.original_corners = self.corners.copy()
        self.frames_tracked = 0
        self.frames_lost = 0
        self.max_frames_lost = 10
        self.min_features = 10  # Seuil minimum de features
    
    def update(self, prev_gray, curr_gray):
        """Met à jour le suivi KLT"""
        if self.features is None or len(self.features) < 4:
            self.frames_lost += 1
            return False
        
        new_features, old_features = track_klt(prev_gray, curr_gray, self.features)
        
        if new_features is None or len(new_features) < 4:
            self.frames_lost += 1
            return self.frames_lost < self.max_frames_lost
        
        new_corners = estimate_corners_from_tracking(
            new_features, self.original_features, self.original_corners)
        
        if new_corners is None:
            self.frames_lost += 1
            return self.frames_lost < self.max_frames_lost
        
        self.features = new_features.reshape(-1, 1, 2)
        self.corners = new_corners
        self.frames_tracked += 1
        self.frames_lost = 0
        
        return True
    
    def needs_redetection(self):
        """Vérifie si on a besoin d'une nouvelle détection"""
        if self.features is None:
            return True
        return len(self.features) < self.min_features
    
    def is_valid(self):
        """Vérifie si le tracker est toujours valide"""
        return self.frames_lost < self.max_frames_lost

# UTILITAIRES
def check_gaze_on_poster(x, y, corners):
    """Vérifie si le point de regard est sur une affiche"""
    if corners is None or len(corners) < 3:
        return False
    result = cv2.pointPolygonTest(corners.astype(np.float32), 
                                   (float(x), float(y)), False)
    return result >= 0

def compute_gaze_on_poster(x, y, poster_data):
    """Calcule les coordonnées du regard dans le repère de l'affiche originale"""
    # Dimensions de l'image à l'échelle utilisée pour la détection
    h_scaled, w_scaled = poster_data["best_scale"]["img"].shape[:2]
    scale = poster_data["best_scale"]["scale"]
    
    # Points sources (coins du template à l'échelle réduite)
    src_corners = np.float32([[0, 0], [w_scaled, 0], [w_scaled, h_scaled], [0, h_scaled]])
    
    # Points destinations (coins projetés dans la frame)
    dst_corners = poster_data["corners"].astype(np.float32)
    
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
        h_orig, w_orig = poster_data["img_orig_shape"]
        gx_aff = float(np.clip(gx_aff, 0, w_orig - 1))
        gy_aff = float(np.clip(gy_aff, 0, h_orig - 1))
        
        return gx_aff, gy_aff
    
    return None, None

# PROGRAMME PRINCIPAL
if __name__ == '__main__':
    print("Chargement des templates multi-échelles...")
    templates, orb = load_poster_templates(poster_folder)
    
    if len(templates) == 0:
        print(f"ERREUR: Aucun template trouvé dans {poster_folder}")
        exit(1)
    
    # DataLoader
    print("Chargement des données eye-tracker...")
    loader = DataLoader("config.json")
    gazes = loader.get_undistorted_gazes(subject_idx)
    cap = loader.get_video_capture(subject_idx)
    K, D = loader.get_load_camera_params(subject_idx)
    
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, len(gazes))
    
    # Initialisation des CSV
    os.makedirs("gaze_points", exist_ok=True)
    csv_files = {}
    file_handles = {}
    
    for poster_name in templates.keys():
        fh = open(f"gaze_points/{poster_name}.csv", "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["x_aff", "y_aff"])
        csv_files[poster_name] = writer
        file_handles[poster_name] = fh
    
    # Variables de suivi
    frame_index = 0
    active_trackers = []
    last_detected = []
    gaze_stats = {}
    prev_gray = None
    last_detection_frame = -1  
    
    print(f"\nDétection et suivi avec KLT multi-échelles...\n")
    
    while True:
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if idx >= num_frames:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_und = cv2.undistort(frame, K, D)
        gray_frame = cv2.cvtColor(frame_und, cv2.COLOR_BGR2GRAY)
        h, w = frame_und.shape[:2]
        
        x, y = map(int, gazes[idx])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        # Vérifier si on a besoin d'une nouvelle détection
        need_detection = False
        detection_reason = ""
        
        # Déterminer l'intervalle de détection selon l'état du tracking
        has_valid_trackers = len(active_trackers) > 0 and any(t.is_valid() for t in active_trackers)
        
        if has_valid_trackers:
            detect_interval = DETECT_EVERY_N_FRAMES_TRACKING
        else:
            detect_interval = DETECT_EVERY_N_FRAMES_NO_TRACK
        
        # Détection périodique adaptative
        frames_since_detection = frame_index - last_detection_frame
        if frames_since_detection >= detect_interval:
            need_detection = True
            if has_valid_trackers:
                detection_reason = f"périodique ({detect_interval} frames)"
            else:
                detection_reason = f"recherche active ({detect_interval} frames)"
        
        # Détection si trackers ont trop peu de features
        if not need_detection and len(active_trackers) > 0:
            for tracker in active_trackers:
                if tracker.needs_redetection():
                    need_detection = True
                    detection_reason = f"< {tracker.min_features} features sur {tracker.name}"
                    break

        annotated = frame_und.copy()
        
        # PHASE DETECT: détection multi-échelles
        if need_detection:
            last_detection_frame = frame_index
            print(f"\n--- Frame {frame_index}: DETECTION ({detection_reason}) ---")
            detected = detect_posters_multiscale(
                gray_frame, templates, orb,
                gaze_x=x, gaze_y=y,
                roi_width=w if not ROI else 900,
                min_matches=MIN_MATCHES,
                ratio_threshold=RATIO_THRESHOLD,
                best_threshold=BEST_THRESHOLD,
                last_detected=last_detected,
                annotated=annotated
            )

            
            # Créer de nouveaux trackers
            active_trackers = []
            for poster in detected:
                tracker = PosterTracker(poster, gray_frame)
                active_trackers.append(tracker)
                print(f"Nouveau tracker créé pour: {poster['name']}")
            
            last_detected = detected
        
        # PHASE TRACK: suivi KLT
        elif prev_gray is not None and len(active_trackers) > 0:
            new_trackers = []
            for tracker in active_trackers:
                if tracker.update(prev_gray, gray_frame):
                    new_trackers.append(tracker)
                else:
                    print(f"Tracker perdu: {tracker.name}")
            active_trackers = new_trackers
        
        # Visualisation
        gaze_color = (0, 255, 0)
        
        # Dessiner les affiches suivies
        for tracker in active_trackers:
            if tracker.is_valid():
                corners_int = np.int32(tracker.corners)
                
                # Couleur selon la durée du suivi
                if tracker.frames_tracked < 10:
                    color = (0, 255, 255)  # Jaune: nouveau
                else:
                    color = (0, 255, 0)  # Vert: stable
                
                cv2.polylines(annotated, [corners_int], True, color, 3)
                
                # Afficher les features suivis
                if tracker.features is not None:
                    for pt in tracker.features:
                        cv2.circle(annotated, tuple(pt[0].astype(int)), 3, (255, 0, 0), -1)
                
                # Alerte si peu de features
                if tracker.needs_redetection():
                    color = (0, 165, 255)  # Orange: attention
                
                # Label avec nombre de features
                center = tracker.corners.mean(axis=0).astype(int)
                n_features = len(tracker.features) if tracker.features is not None else 0
                label = f"{tracker.name} (T:{tracker.frames_tracked}, F:{n_features})"
                cv2.putText(annotated, label, tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Vérifier si le regard est sur l'affiche
                if check_gaze_on_poster(x, y, tracker.corners):
                    gaze_color = (0, 0, 255)
                    
                    # Calculer les coordonnées dans le repère de l'affiche
                    gx_aff, gy_aff = compute_gaze_on_poster(x, y, tracker.poster_data)
                    
                    if gx_aff is not None:
                        csv_files[tracker.name].writerow([gx_aff, gy_aff])
                        gaze_stats[tracker.name] = gaze_stats.get(tracker.name, 0) + 1
                    break
        
        # Dessiner le point de regard
        cv2.circle(annotated, (x, y), 10, gaze_color, -1)
        
        # Afficher les statistiques
        y_offset = 30
        
        # Statut de détection
        frames_since = frame_index - last_detection_frame
        has_trackers = len(active_trackers) > 0
        interval = DETECT_EVERY_N_FRAMES_TRACKING if has_trackers else DETECT_EVERY_N_FRAMES_NO_TRACK
        next_detect = max(0, interval - frames_since)
        
        status_text = f"Frame: {frame_index} | Trackers: {len(active_trackers)} | Next detect: {next_detect}"
        cv2.putText(annotated, status_text, 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        for name, count in gaze_stats.items():
            text = f"{name}: {count} frames"
            cv2.putText(annotated, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        cv2.namedWindow("Detect & Track Multi-Scale (KLT)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detect & Track Multi-Scale (KLT)", 1280, 720)
        cv2.imshow("Detect & Track Multi-Scale (KLT)", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        prev_gray = gray_frame.copy()
        frame_index += 1
    
    # Fermeture
    for fh in file_handles.values():
        fh.close()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n=== Statistiques de fixation sur les affiches ===")
    for name, count in sorted(gaze_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {count} frames ({count/frame_index*100:.2f}%)")
    
    print("\nFichiers CSV sauvegardés dans le dossier 'gaze_points/'")