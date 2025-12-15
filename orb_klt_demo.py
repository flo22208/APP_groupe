import cv2
import numpy as np
from pathlib import Path
from ORB_multiechelle_KRT import load_poster_templates, detect_posters_multiscale, NB_POI
import matplotlib.pyplot as plt

# === PARAMÈTRES DEMO ===
POSTER_FOLDER = "data/Affiches"
POSTER_NAME = None  # Laisser None pour prendre le premier poster trouvé
VIDEO_PATH = r"data\AcquisitionsEyeTracker\sujet2_f-835bf855\b7bd6c34_0.0-271.583.mp4"  # À adapter selon vos vidéos
FRAME_NUMBER = 100  # Numéro de la frame à extraire

# === CHARGEMENT DES AFFICHES (multi-échelle) ===
print("Chargement des templates multi-échelles...")
templates, orb = load_poster_templates(POSTER_FOLDER)

if len(templates) == 0:
    print(f"ERREUR: Aucun template trouvé dans {POSTER_FOLDER}")
    exit(1)

if POSTER_NAME is None:
    POSTER_NAME = list(templates.keys())[0]
poster_data = templates[POSTER_NAME]


# === EXTRACTION DE LA FRAME DEPUIS LA VIDÉO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERREUR: Impossible d'ouvrir la vidéo {VIDEO_PATH}")
    exit(1)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if FRAME_NUMBER >= total_frames:
    print(f"ERREUR: La vidéo ne contient que {total_frames} frames.")
    exit(1)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
ret, frame = cap.read()
cap.release()
if not ret:
    print(f"ERREUR: Impossible de lire la frame {FRAME_NUMBER} de la vidéo.")
    exit(1)
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# === DÉTECTION MULTI-ÉCHELLE ===
detected = detect_posters_multiscale(
    gray_frame, templates, orb,
    gaze_x=None, gaze_y=None,
    roi_width=gray_frame.shape[1],
    min_matches=10,
    ratio_threshold=0.7,
    best_threshold=15,
    last_detected=[],
    annotated=None
)

if not detected:
    print("Aucune affiche détectée dans la frame.")
    exit(0)

best_poster = detected[0]

# === AFFICHAGE DES POINTS D'INTÉRÊT ET MATCHING ===
# On prend la meilleure échelle trouvée
scale_data = best_poster["best_scale"]
template_img = scale_data["img"]
template_kp = scale_data["kp"]
template_des = scale_data["des"]


# Détection des points d'intérêt sur la frame UNIQUEMENT dans la ROI (comme ORB_multiechelle_KRT.py)
ROI = True
roi_width = 900
roi_height = None
gaze_x = gray_frame.shape[1] // 2
gaze_y = gray_frame.shape[0] // 2

if ROI:
    w = gray_frame.shape[1]
    h = gray_frame.shape[0]
    x1_roi = max(0, gaze_x - roi_width // 2)
    x2_roi = min(w, gaze_x + roi_width // 2)
    if roi_height is None:
        y1_roi, y2_roi = 0, h
    else:
        y1_roi = max(0, gaze_y - roi_height // 2)
        y2_roi = min(h, gaze_y + roi_height // 2)

    roi = gray_frame[y1_roi:y2_roi, x1_roi:x2_roi]
    kp_roi, des_roi = orb.detectAndCompute(roi, None)
    # Ajuster les coordonnées des keypoints pour la frame complète
    frame_kp = []
    for kp in kp_roi:
        new_kp = cv2.KeyPoint(kp.pt[0] + x1_roi, kp.pt[1] + y1_roi, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        frame_kp.append(new_kp)
    frame_des = des_roi
else:
    frame_kp, frame_des = orb.detectAndCompute(gray_frame, None)

# Matching (même méthode que dans detect_posters_multiscale)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(template_des, frame_des, k=2)
good = []
for m_n in matches:
    if len(m_n) == 2:
        m, n = m_n
        if m.distance < 0.7 * n.distance:
            good.append(m)
    elif len(m_n) == 1:
        good.append(m_n[0])


# Affichage des points d'intérêt sur le template et la frame (keypoints dans la ROI)
img_template_kp = cv2.drawKeypoints(template_img, template_kp, None, color=(0,255,0), flags=0)
img_frame_kp = gray_frame.copy()
img_frame_kp = cv2.cvtColor(img_frame_kp, cv2.COLOR_GRAY2BGR)
img_frame_kp = cv2.drawKeypoints(img_frame_kp, frame_kp, None, color=(0,255,0), flags=0)
# Dessiner la ROI sur l'image de la frame
cv2.rectangle(img_frame_kp, (x1_roi, y1_roi), (x2_roi, y2_roi), (255,0,255), 2)


# Affichage avec matplotlib
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(img_frame_kp, cmap='gray')
ax.set_title("Points d'intérêt - Frame")
ax.axis('off')

# Affichage séparé pour le matching
img_matches = cv2.drawMatches(
    template_img, template_kp,
    gray_frame, frame_kp,
    good, None,
    matchColor=(0,255,0),
    singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(12, 6))
plt.imshow(img_matches[..., ::-1])  # BGR to RGB
plt.title("Matching ORB multi-échelle")
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Nombre de points d'intérêt sur le template: {len(template_kp)}")
print(f"Nombre de points d'intérêt sur la frame: {len(frame_kp)}")
print(f"Nombre de bons matching: {len(good)}")
