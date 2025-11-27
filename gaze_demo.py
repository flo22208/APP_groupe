from src.utils.DataLoader import DataLoader
import numpy as np
import cv2


# Initialise le DataLoader (à lancer depuis le dossier APP)
loader = DataLoader("config.json")

# Sujet (0 = premier)
subject_idx = 0

# Gaze en pixels undistordus (même espace que la vidéo undistordue)
gazes = loader.get_undistorted_gazes(subject_idx)

# Vidéo distordue (pas de remap)
cap = loader.get_video_capture(subject_idx)

# Paramètres de la caméra
K, D = loader.get_load_camera_params(subject_idx)

# fps = cap.get(cv2.CAP_PROP_FPS)
# delay_ms = int(1000 / fps)
delay_ms = 1

# Nombre de frames à afficher (borne par la longueur de la gaze)
num_frames_to_show = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames_to_show = min(num_frames_to_show, len(gazes))

while True:
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if frame_idx >= num_frames_to_show:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame_undistorted = cv2.undistort(frame, K, D)

    # Coordonnées de gaze pour cette frame
    x = int(gazes[frame_idx, 0])
    y = int(gazes[frame_idx, 1])

    # Dessine un point rouge
    cv2.circle(frame, (x, y), radius=8, color=(0, 0, 255), thickness=-1)

    cv2.imshow("Gaze realtime demo", frame)

    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord("q") or key == 27:  # 'q' or ESC to quit
        break

cap.release()
cv2.destroyAllWindows()