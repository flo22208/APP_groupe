from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import os

from src.pipeline.HomographyManager import HomographyManager
from src.utils.DataLoader import DataLoader
from src.detectionModel.DetectionModel import DetectionModel


class GazeAnalyser:
	"""Analyseur de regard basé sur détection + homographie.

	Charge le modèle de détection, les affiches et le HomographyManager à l'initialisation,
	puis expose une méthode pour analyser une frame + un point de regard.
	"""

	def __init__(self, config_path: str) -> None:
		self.loader = DataLoader(config_path)
		self.h_manager = HomographyManager()

		# Modèle de détection
		weights_path = self.loader.get_yolo_detection_weights()
		self.det_model = DetectionModel(weights_path)

		# Affiches
		self.posters: List[Tuple[str, np.ndarray]] = self.loader.load_posters()
		if not self.posters:
			raise RuntimeError("No PNG posters found in posters folder")

        # Cache : Associe les id des affiches détectées aux index dans self.posters
		self.poster_name_to_index = {}

	def analyse_gaze_on_frame(
		self,
		frame_undist: np.ndarray,
		gaze_point: Tuple[float, float],
		frame_idx: int,
		subject_idx: int,
	):
		"""Analyse une frame et un point de regard pour retrouver l'affiche correspondante.

		Utilise les détections pré-calculées dans le CSV detection_results
		pour la frame donnée au lieu de relancer YOLO.

		Retourne (best_name, best_index, (px, py)).
		"""

		gx, gy = gaze_point

		# 1 - Charger les détections depuis le CSV du sujet et filtrer sur frame_idx
		det_csv_path = self.loader.get_detection_results_path(subject_idx)
		import pandas as pd

		if not os.path.exists(det_csv_path):
			raise RuntimeError(f"Detection results CSV not found: {det_csv_path}")

		tracks = pd.read_csv(det_csv_path)
		rows = tracks[tracks["frame"] == frame_idx]
		if rows.empty:
			raise RuntimeError("No posters detected on frame (CSV empty for this frame)")

		bboxes_list = []
		for _, row in rows.iterrows():
			x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
			track_id = int(row["track_id"])
			bboxes_list.append((np.array([x1, y1, x2, y2], dtype=np.float32), track_id))

		bboxes_array = np.stack([bbox for bbox, _ in bboxes_list], axis=0) if bboxes_list else None
		if bboxes_array is None or len(bboxes_array) == 0:
			raise RuntimeError("No posters detected on frame (no boxes in CSV)")

		# 2 - Trouver l'affiche regardée: bbox contenant le point de regard
		looked_bbox: Optional[np.ndarray] = None
		for idx, bbox in enumerate(bboxes_array):
			x1, y1, x2, y2 = bbox
			if x1 <= gx <= x2 and y1 <= gy <= y2:
				looked_bbox = bbox
				looked_bbox_track_id = bboxes_list[idx][1]
				break

		if looked_bbox is None:
			raise RuntimeError("Gaze point not inside any detected poster bbox")

		# 3 - ROI à partir de la bbox regardée
		x1_lb, y1_lb, x2_lb, y2_lb = looked_bbox.astype(int)
		roi = frame_undist[y1_lb:y2_lb, x1_lb:x2_lb]

		# 4 - Trouver la meilleure affiche PNG via homographie (utilise find_best_match) si affiche non vue auparavant
		if looked_bbox_track_id not in self.poster_name_to_index:
			poster_imgs = [img for _, img in self.posters]
			best_H, best_inliers, best_index = self.h_manager.find_best_match(roi, poster_imgs)
			if best_index >= 0:
				self.poster_name_to_index[looked_bbox_track_id] = best_index
			else:
				raise RuntimeError("Could not find a matching poster PNG for ROI")
		else:
			best_index = self.poster_name_to_index[looked_bbox_track_id]
			best_H, _ = self.h_manager.compute_homography_between(roi, self.posters[best_index][1])
		
		if best_H is None:
			raise RuntimeError("Homography computation failed")

		best_name = self.posters[best_index][0]

		# 5 - Projeter le point de regard sur l'affiche
		H_roi_to_poster = best_H
		x1, y1, x2, y2 = looked_bbox
		local_point = np.array([gx - x1, gy - y1], dtype=np.float32)
		projected = self.h_manager.project_point(local_point, H_roi_to_poster)
		px, py = float(projected[0]), float(projected[1])

		return best_name, best_index, (px, py)

	def analyse_video_for_subject(
		self,
		subject_idx: int,
		output_csv_path: str,
		start_frame: int = 0,
		end_frame: Optional[int] = None,
	) -> None:
		"""Analyse une vidéo d'un sujet et sauvegarde les résultats dans un CSV.

		Chaque ligne du CSV contient :
		frame_idx, poster_name, poster_index, proj_x, proj_y

		Les frames sans détection d'affiche ou sans regard sur une affiche
		sont simplement ignorées (pas de ligne écrite).
		"""

		# Récupérer capture vidéo et gazes pour le sujet
		cap = self.loader.get_video_capture(subject_idx)
		K, D = self.loader.get_load_camera_params(subject_idx)
		gazes_undist = self.loader.get_undistorted_gazes(subject_idx)

		import csv
		from pathlib import Path

		output_path = Path(output_csv_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if end_frame is None or end_frame > n_frames:
			end_frame = n_frames

		with output_path.open("w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				"frame_idx",
				"poster_name",
				"poster_index",
				"proj_x",
				"proj_y",
			])

			frame_range = range(start_frame, end_frame)
			for frame_idx in tqdm(frame_range, desc=f"Analysing subject {subject_idx}", unit="frame"):
				# Sécurité si pas de gaze dispo pour cette frame
				if frame_idx >= len(gazes_undist):
					continue

				cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
				ret, frame = cap.read()
				if not ret:
					continue

				frame_undist = cv2.undistort(frame, K, D)
				gx, gy = gazes_undist[frame_idx]

				# Essayer d'analyser cette frame, en étant robuste aux erreurs
				try:
					best_name, best_index, (px, py) = self.analyse_gaze_on_frame(
						frame_undist,
						(gx, gy),
						frame_idx,
						subject_idx,
					)
				except RuntimeError:
					# Pas de détection, ou regard hors affiche : on skip
					continue

				# On écrit uniquement les frames valides
				writer.writerow([
					frame_idx,
					best_name,
					best_index,
					px,
					py,
				])

		cap.release()


__all__ = ["GazeAnalyser"]

