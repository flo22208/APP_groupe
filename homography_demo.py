import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.DataLoader import DataLoader
from src.pipeline.HomographyManager import HomographyManager
from src.detectionModel.DetectionModel import DetectionModel


def boxes_to_bboxes(boxes) -> List[Tuple[int, int, int, int]]:
	"""Convertit les boxes YOLO (results.boxes) en bboxes (x1, y1, x2, y2)."""
	bboxes: List[Tuple[int, int, int, int]] = []
	if boxes is None or len(boxes) == 0:
		return bboxes

	xyxy = boxes.xyxy.cpu().numpy()
	for x1, y1, x2, y2 in xyxy:
		bboxes.append((int(x1), int(y1), int(x2), int(y2)))
	return bboxes


def crop_with_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
	x1, y1, x2, y2 = bbox
	return image[y1:y2, x1:x2]


def find_best_poster_for_roi(
	roi: np.ndarray,
	posters: List[Tuple[str, np.ndarray]],
	h_manager: HomographyManager,
	min_inliers: int = 10,
) -> Tuple[Optional[str], Optional[np.ndarray], int]:
	"""Trouve le poster le plus cohérent avec la ROI via HomographyManager.find_best_match.

	Retourne (nom_poster, H_roi_vers_poster_redimensionne, nb_inliers).
	"""
	if not posters:
		return None, None, 0

	# Redimensionne tous les posters à la taille de la ROI pour une comparaison cohérente
	roi_h, roi_w = roi.shape[:2]
	poster_imgs_resized: List[np.ndarray] = []
	for _, poster in posters:
		poster_resized = cv2.resize(poster, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
		poster_imgs_resized.append(poster_resized)

	best_H, best_inliers, best_index = h_manager.find_best_match(
		roi,
		poster_imgs_resized,
	)

	if best_H is None or best_inliers < min_inliers or best_index < 0:
		return None, None, 0

	best_name = posters[best_index][0]
	return best_name, best_H, best_inliers


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

	# 2 - Détection des affiches sur la frame
	boxes = det_model.predict(frame_undist)
	bboxes = boxes_to_bboxes(boxes)
	if len(bboxes) == 0:
		raise RuntimeError("No posters detected on target frame (model output empty)")

	# 3 - Trouver l'affiche regardée: bbox contenant le point de regard
	looked_bbox: Optional[Tuple[int, int, int, int]] = None
	for bbox in bboxes:
		x1, y1, x2, y2 = bbox
		if x1 <= gx <= x2 and y1 <= gy <= y2:
			looked_bbox = bbox
			break

	if looked_bbox is None:
		raise RuntimeError("Gaze point not inside any detected poster bbox")

	roi = crop_with_bbox(frame_undist, looked_bbox)

	# 4 - Trouver la meilleure affiche PNG via homographie
	best_name, best_H, best_inliers = find_best_poster_for_roi(roi, posters, h_manager)
	if best_name is None or best_H is None:
		raise RuntimeError("Could not find a matching poster PNG for ROI")

	print("Best poster match:", best_name)
	print("Number of inliers:", best_inliers)

	# Charger l'image PNG correspondante (taille d'origine)
	poster_path = os.path.join(loader.get_posters_path(), best_name)
	poster_img = cv2.imread(poster_path)
	if poster_img is None:
		raise RuntimeError(f"Cannot read poster image: {poster_path}")

	# Recalculer homographie entre ROI et poster redimensionné à la taille de la ROI
	poster_resized = cv2.resize(poster_img, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
	H_roi_to_poster, inliers2 = h_manager.compute_homography_between(roi, poster_resized)
	if H_roi_to_poster is None:
		raise RuntimeError("Failed to recompute homography between ROI and poster")

	# Projeter le point de regard sur l'affiche
	x1, y1, x2, y2 = looked_bbox
	local_point = np.array([gx - x1, gy - y1], dtype=np.float32)
	projected = h_manager.project_point(local_point, H_roi_to_poster)
	px, py = int(projected[0]), int(projected[1])
 
	# --------------------- Visualisation ------------------
	vis_frame = frame_undist.copy()
	for bbox in bboxes:
		x1b, y1b, x2b, y2b = bbox
		color = (0, 255, 0) if bbox == looked_bbox else (255, 0, 0)
		cv2.rectangle(vis_frame, (x1b, y1b), (x2b, y2b), color, 2)
	# point regard
	cv2.circle(vis_frame, (int(gx), int(gy)), 8, (0, 0, 255), -1)

	vis_poster = poster_resized.copy()
	cv2.circle(vis_poster, (px, py), 12, (0, 0, 255), -1)

	cv2.imshow("Frame undistorted with gaze & posters", vis_frame)
	cv2.imshow("Matched poster with projected gaze", vis_poster)
	print("Press any key on image windows to close...")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cap.release()


if __name__ == "__main__":
	main()

