import os
import time

import cv2
import numpy as np
import pandas as pd

from src.utils.DataLoader import DataLoader
from src.detectionModel.DetectionModel import DetectionModel


def get_video_path(loader: DataLoader, subject_idx: int) -> str:
	subject_path = loader.get_subject_path(subject_idx)
	mp4_files = [f for f in os.listdir(subject_path) if f.lower().endswith(".mp4")]
	if len(mp4_files) == 0:
		raise RuntimeError(f"No .mp4 file found in {subject_path}")
	if len(mp4_files) > 1:
		raise RuntimeError(f"Multiple .mp4 files found in {subject_path}: {mp4_files}")
	return os.path.join(subject_path, mp4_files[0])


def main():
	loader = DataLoader("config.json")
	subject_idx = 0

	video_path = get_video_path(loader, subject_idx)
	K, D = loader.get_load_camera_params(subject_idx)
	detection_results_csv = loader.get_detection_results_path(subject_idx)

	if False:
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise RuntimeError(f"Cannot open video: {video_path}")

		tracks = pd.read_csv(detection_results_csv)
		while True:
			frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
			ret, frame = cap.read()
			if not ret:
				break

			# Undistort frame before drawing boxes
			frame_undist = cv2.undistort(frame, K, D)

			rows = tracks[tracks["frame"] == frame_idx]
			for _, row in rows.iterrows():
				x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
				track_id = int(row["track_id"]) if row["track_id"] != -1 else None
				cls_id = int(row["class_id"]) if row["class_id"] != -1 else None
				conf = float(row["conf"]) if row["conf"] >= 0 else None

				label_parts = []
				if track_id is not None:
					label_parts.append(f"id:{track_id}")
				if cls_id is not None:
					label_parts.append(f"cls:{cls_id}")
				if conf is not None:
					label_parts.append(f"{conf:.2f}")
				label = " ".join(label_parts)

				color = (0, 255, 0)
				cv2.rectangle(frame_undist, (x1, y1), (x2, y2), color, 2)
				if label:
					cv2.putText(
						frame_undist,
						label,
						(x1, max(y1 - 5, 0)),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5,
						color,
						1,
						cv2.LINE_AA,
					)

			cv2.imshow("Detection tracking demo (from CSV, undistorted)", frame_undist)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break

		cap.release()
		cv2.destroyAllWindows()
	else:
		weights_path = loader.get_yolo_detection_weights()
		det_model = DetectionModel(weights_path)

		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise RuntimeError(f"Cannot open video: {video_path}")

		frame_idx = -1

		while True:
			ret, frame = cap.read()
			if not ret:
				break
			
			frame_idx += 1
			if frame_idx % 5 == 0:
				continue

			frame_undist = cv2.undistort(frame, K, D)

			# Run detection on undistorted frame
			start_time = time.perf_counter()
			boxes_xyxy = det_model.predict(frame_undist, conf_threshold=0.3)
			inference_time = (time.perf_counter() - start_time) * 1000  # ms

			for x1, y1, x2, y2 in boxes_xyxy:
				cv2.rectangle(
					frame_undist,
					(int(x1), int(y1)),
					(int(x2), int(y2)),
					(0, 255, 0),
					2,
				)

			# Display inference time
			cv2.putText(
				frame_undist,
				f"Inference: {inference_time:.1f} ms",
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)

			cv2.imshow("Detection demo (undistorted)", frame_undist)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break

		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

