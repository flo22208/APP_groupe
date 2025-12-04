import cv2
import json
import os
import numpy as np
import pandas as pd

def load_camera_params(json_path : str):
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float32)
    D = np.array(data["distortion_coefficients"], dtype=np.float32)
    return K, D

class DataLoader:
	def __init__(self, config_path: str):
		with open(config_path, "r") as f:
			config = json.load(f)

		self.acquisition_root = config["paths"]["acquisition_data"]
		self.subjects = config["subjects"]
		self.camera_params_filename = config["files"]["camera_params"]
		self.gaze_filename = config["files"]["gaze"]
		self.world_timestamps_filename = config["files"]["world_timestamps"]
		self.yolo_detection_weights = config["paths"]["yolo_detection_weights"]
		self.detection_results_filename = config["files"]["detection_results"]
		self.posters_root = config["paths"].get("posters", "data/Affiches/")

	def get_subject_path(self, subject_idx: int) -> str:
		subject_name = self.subjects[subject_idx]
		return os.path.join(self.acquisition_root, subject_name)

	def get_load_camera_params(self, subject_idx: int):
		json_path = os.path.join(self.get_subject_path(subject_idx), self.camera_params_filename)
		return load_camera_params(json_path)
	
	def get_gazes(self, subject_idx: int, method: str = "loop") -> np.ndarray:
		subject_path = self.get_subject_path(subject_idx)
		gaze_path = os.path.join(subject_path, self.gaze_filename)
		world_ts_path = os.path.join(subject_path, self.world_timestamps_filename)

		gaze_df = pd.read_csv(gaze_path)
		world_df = pd.read_csv(world_ts_path)

		gaze_ts = gaze_df["timestamp [ns]"].to_numpy(dtype=np.int64)
		world_ts = world_df["timestamp [ns]"].to_numpy(dtype=np.int64)
		gaze_x = gaze_df["gaze x [px]"].to_numpy(dtype=np.float32)
		gaze_y = gaze_df["gaze y [px]"].to_numpy(dtype=np.float32)

		gaze_per_frame = np.empty((len(world_ts), 2), dtype=np.float32)

		if method == "searchsorted":
			# For each world timestamp, find index of closest gaze timestamp
			idx = np.searchsorted(gaze_ts, world_ts, side="left")

			# Adjust indices to the nearer neighbour (previous or next gaze sample)
			idx_closest = idx.copy()
			# Where idx > 0 and (idx == len(gaze_ts) or previous is closer than next)
			mask = (idx > 0) & ((idx == len(gaze_ts)) | (np.abs(world_ts - gaze_ts[idx - 1]) <= np.abs(world_ts - gaze_ts[np.minimum(idx, len(gaze_ts) - 1)])))
			idx_closest[mask] = idx_closest[mask] - 1

			gaze_per_frame[:, 0] = gaze_x[idx_closest]
			gaze_per_frame[:, 1] = gaze_y[idx_closest]
		elif method == "loop":
			# Récupération du timestamp pour cette frame 
			for i in range(len(world_ts)):
				ts = world_ts[i]

				gi = np.argmin(np.abs(gaze_ts - ts))

				# Récupération des coordonnées du regard
				gx = gaze_x[gi]
				gy = gaze_y[gi]
				gaze_per_frame[i, 0] = gx
				gaze_per_frame[i, 1] = gy

		return gaze_per_frame

	def get_undistorted_gazes(self, subject_idx: int):
		gazes = self.get_gazes(subject_idx)
		K, D = self.get_load_camera_params(subject_idx)

		# Undistort gaze points
		gazes_undistorted = cv2.undistortPoints(
			gazes.reshape(-1, 1, 2),
			cameraMatrix=K,
			distCoeffs=D,
			P=K
		).reshape(-1, 2)

		return gazes_undistorted
	
	def get_yolo_detection_weights(self) -> str:
		return self.yolo_detection_weights
	
	def get_video_path(self, subject_idx: int) -> str:
		subject_path = self.get_subject_path(subject_idx)
		# Find the single .mp4 file in the subject directory
		mp4_files = [f for f in os.listdir(subject_path) if f.lower().endswith(".mp4")]
		if len(mp4_files) == 0:
			raise RuntimeError(f"No .mp4 file found in {subject_path}")
		if len(mp4_files) > 1:
			raise RuntimeError(f"Multiple .mp4 files found in {subject_path}: {mp4_files}")
		return os.path.join(subject_path, mp4_files[0])
	
	def get_video_capture(self, subject_idx: int):
		video_path = self.get_video_path(subject_idx)
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise RuntimeError(f"Cannot open video: {video_path}")
		return cap
	
	def get_detection_results_path(self, subject_idx: int) -> str:
		subject_path = self.get_subject_path(subject_idx)
		return os.path.join(subject_path, self.detection_results_filename)

	def get_posters_path(self) -> str:
		"""Return the root folder where poster PNGs are stored."""
		return self.posters_root

	def load_posters(self):
		"""Load all PNG posters from the posters root folder.

		Returns a list of (filename, image).
		"""
		folder = self.get_posters_path()
		if not os.path.isdir(folder):
			return []

		paths = [
			os.path.join(folder, f)
			for f in os.listdir(folder)
			if f.lower().endswith(".png")
		]
		posters = []
		for p in paths:
			img = cv2.imread(p)
			if img is None:
				continue
			posters.append((os.path.basename(p), img))
		return posters
	