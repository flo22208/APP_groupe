import os
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils.DataLoader import DataLoader
from src.pipeline.HomographyManager import HomographyManager
from src.detectionModel.DetectionModel import DetectionModel

def main():
	# ------------------ Chargement données ------------------
	loader = DataLoader("config.json")
	h_manager = HomographyManager()
	subject_idx = 0

	# ------------------ Récupération vidéo et paramètres ------------------
	cap = loader.get_video_capture(subject_idx)
	K, D = loader.get_load_camera_params(subject_idx)
	gazes_undist = loader.get_undistorted_gazes(subject_idx)

	# ------------------ Chargement du modèle de détection ------------------
	weights_path = loader.get_yolo_detection_weights()
	det_model = DetectionModel(weights_path)

	# ------------------ Chargement des affiches ------------------
	posters = loader.load_posters()
	if not posters:
		raise RuntimeError("No PNG posters found in posters folder")

	# ------------------ Récupération d'une frame au milieu ------------------
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if n_frames == 0:
		raise RuntimeError("Video has 0 frame")
	target_frame_idx = n_frames // 2

	cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
	ret, frame = cap.read()
	if not ret:
		raise RuntimeError("Cannot read target frame")
	frame_undist = cv2.undistort(frame, K, D)

	# ------------------ Traitement ------------------

	# 1- Récupération du point de regard sur la frame cible
	gx, gy = gazes_undist[target_frame_idx]

	# 2 - Détection des affiches sur la frame (xyxy directement)
	bboxes_array = det_model.predict(frame_undist)
	if bboxes_array is None or len(bboxes_array) == 0:
		raise RuntimeError("No posters detected on target frame (model output empty)")

	# 3 - Trouver l'affiche regardée: bbox contenant le point de regard
	looked_bbox: Optional[np.ndarray] = None
	for bbox in bboxes_array:
		x1, y1, x2, y2 = bbox
		if x1 <= gx <= x2 and y1 <= gy <= y2:
			looked_bbox = bbox
			break

	if looked_bbox is None:
		raise RuntimeError("Gaze point not inside any detected poster bbox")

	# ROI à partir de la bbox regardée
	x1_lb, y1_lb, x2_lb, y2_lb = looked_bbox.astype(int)
	roi = frame_undist[y1_lb:y2_lb, x1_lb:x2_lb]

	# 4 - Trouver la meilleure affiche PNG via homographie (utilise find_best_match)
	poster_imgs = [img for _, img in posters]
	best_H, best_inliers, best_index = h_manager.find_best_match(roi, poster_imgs)
	if best_H is None or best_index < 0:
		raise RuntimeError("Could not find a matching poster PNG for ROI")

	best_name = posters[best_index][0]
	poster_img = posters[best_index][1]

	print("Best poster match:", best_name)
	print("Number of inliers:", best_inliers)

	# Homographie ROI -> poster original
	H_roi_to_poster = best_H

	# Projeter le point de regard sur l'affiche
	x1, y1, x2, y2 = looked_bbox
	local_point = np.array([gx - x1, gy - y1], dtype=np.float32)
	projected = h_manager.project_point(local_point, H_roi_to_poster)
	px, py = int(projected[0]), int(projected[1])

	# --------------------- Visualisation ------------------
	# Image de la vidéo avec bboxes et point de regard
	vis_frame = frame_undist.copy()
	for bbox in bboxes_array:
		x1b, y1b, x2b, y2b = bbox
		color = (0, 255, 0) if np.array_equal(bbox, looked_bbox) else (255, 0, 0)
		cv2.rectangle(vis_frame, (int(x1b), int(y1b)), (int(x2b), int(y2b)), color, 2)
	# point regard
	cv2.circle(vis_frame, (int(gx), int(gy)), 8, (0, 255, 0), -1)

	# Image d'affiche avec point projeté
	vis_poster = poster_img.copy()
	cv2.circle(vis_poster, (px, py), 12, (0, 255, 0), -1)

	# Conversion BGR -> RGB pour matplotlib
	vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
	vis_poster_rgb = cv2.cvtColor(vis_poster, cv2.COLOR_BGR2RGB)

	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.imshow(vis_frame_rgb)
	plt.title("Frame undistorted with gaze & posters")
	plt.axis("off")

	plt.subplot(1, 2, 2)
	plt.imshow(vis_poster_rgb)
	plt.title("Matched poster with projected gaze")
	plt.axis("off")

	plt.tight_layout()
	plt.show()

	cap.release()


if __name__ == "__main__":
	main()

