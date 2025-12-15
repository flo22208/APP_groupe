import cv2
import json
import os
import numpy as np
import pandas as pd


def load_camera_params(json_path: str):
    """Charge les paramètres de caméra depuis un fichier JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float32)
    D = np.array(data["distortion_coefficients"], dtype=np.float32)
    return K, D



def load_gaze_offset(json_path: str):
    """Load gaze offset from info.json"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data.get("gaze_offset", [0.0, 0.0]), dtype=np.float32)


class DataLoader:
    """
    Chargeur de données centralisé pour le pipeline de traitement des regards.
    Gère les chemins vers les données d'acquisition, les paramètres de caméra, les offsets, etc.
    """
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.acquisition_root = config["paths"]["acquisition_data"]
        self.yolo_detection_weights = config["paths"]["yolo_detection_weights"]
        self.posters_root = config["paths"].get("posters", "data/Affiches/")

        self.camera_params_filename = config["files"]["camera_params"]
        self.gaze_filename = config["files"]["gaze"]
        self.world_timestamps_filename = config["files"]["world_timestamps"]
        self.detection_results_filename = config["files"]["detection_results"]
        self.info_filename = config["files"].get("info", "info.json")

        self.skip_step = config.get("settings", {}).get("skip_step", 3)
        self.start_frame = config.get("settings", {}).get("start_frame", 0)
        self.end_frame = config.get("settings", {}).get("end_frame", None)
        self.detection_confidence = config.get("settings", {}).get("detection_confidence", 0.3)
        self.tracking_iou = config.get("settings", {}).get("tracking_iou", 0.5)

        self.projections_dir = config.get("output", {}).get("projections_dir", "data/")
        self.projections_prefix = config.get("output", {}).get("projections_prefix", "gaze_projections_subject")

        self.subjects = config["subjects"]

    def get_subject_path(self, subject_idx: int) -> str:
        subject_name = self.subjects[subject_idx]
        return os.path.join(self.acquisition_root, subject_name)

    def get_load_camera_params(self, subject_idx: int):
        json_path = os.path.join(self.get_subject_path(subject_idx), self.camera_params_filename)
        return load_camera_params(json_path)

    def get_gaze_offset(self, subject_idx: int) -> np.ndarray:
        info_path = os.path.join(self.get_subject_path(subject_idx), self.info_filename)
        return load_gaze_offset(info_path)

    def get_gazes(self, subject_idx: int, method: str = "loop", apply_offset: bool = True) -> np.ndarray:
        subject_path = self.get_subject_path(subject_idx)
        gaze_path = os.path.join(subject_path, self.gaze_filename)
        world_ts_path = os.path.join(subject_path, self.world_timestamps_filename)

        gaze_df = pd.read_csv(gaze_path)
        world_df = pd.read_csv(world_ts_path)

        gaze_ts = gaze_df["timestamp [ns]"].to_numpy(dtype=np.int64)
        world_ts = world_df["timestamp [ns]"].to_numpy(dtype=np.int64)
        gaze_x = gaze_df["gaze x [px]"].to_numpy(dtype=np.float32)
        gaze_y = gaze_df["gaze y [px]"].to_numpy(dtype=np.float32)

        # Appliquer l'offset de calibration si demandé
        if apply_offset:
            offset = self.get_gaze_offset(subject_idx)
            gaze_x = gaze_x - offset[0]
            gaze_y = gaze_y - offset[1]

        gaze_per_frame = np.empty((len(world_ts), 2), dtype=np.float32)

        if method == "searchsorted":
            idx = np.searchsorted(gaze_ts, world_ts, side="left")
            idx_closest = idx.copy()
            mask = (idx > 0) & ((idx == len(gaze_ts)) | (np.abs(world_ts - gaze_ts[idx - 1]) <= np.abs(world_ts - gaze_ts[np.minimum(idx, len(gaze_ts) - 1)])))
            idx_closest[mask] = idx_closest[mask] - 1
            gaze_per_frame[:, 0] = gaze_x[idx_closest]
            gaze_per_frame[:, 1] = gaze_y[idx_closest]
        elif method == "loop":
            for i in range(len(world_ts)):
                ts = world_ts[i]
                gi = np.argmin(np.abs(gaze_ts - ts))
                gx = gaze_x[gi]
                gy = gaze_y[gi]
                gaze_per_frame[i, 0] = gx
                gaze_per_frame[i, 1] = gy

        return gaze_per_frame

    def get_undistorted_gazes(self, subject_idx: int, apply_offset: bool = True):
        gazes = self.get_gazes(subject_idx, apply_offset=apply_offset)
        K, D = self.get_load_camera_params(subject_idx)
        gazes_undistorted = cv2.undistortPoints(
            gazes.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=D,
            P=K
        ).reshape(-1, 2)
        return gazes_undistorted

    # =========================================================================
    # CHEMINS ET ACCÈS AUX DONNÉES
    # =========================================================================

    def get_subject_path(self, subject_idx: int) -> str:
        """Retourne le chemin complet vers le dossier d'un sujet."""
        subject_name = self.subjects[subject_idx]
        return os.path.join(self.acquisition_root, subject_name)

    def get_video_path(self, subject_idx: int) -> str:
        """Retourne le chemin vers la vidéo d'un sujet (fichier .mp4)."""
        subject_path = self.get_subject_path(subject_idx)
        mp4_files = [f for f in os.listdir(subject_path) if f.lower().endswith(".mp4")]
        if len(mp4_files) == 0:
            raise RuntimeError(f"Aucun fichier .mp4 trouvé dans {subject_path}")
        if len(mp4_files) > 1:
            raise RuntimeError(f"Plusieurs fichiers .mp4 trouvés dans {subject_path}: {mp4_files}")
        return os.path.join(subject_path, mp4_files[0])

    def get_video_capture(self, subject_idx: int) -> cv2.VideoCapture:
        """Retourne un VideoCapture ouvert pour la vidéo d'un sujet."""
        video_path = self.get_video_path(subject_idx)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")
        return cap

    def get_detection_results_path(self, subject_idx: int) -> str:
        """Retourne le chemin vers le fichier detection_results.csv d'un sujet."""
        subject_path = self.get_subject_path(subject_idx)
        return os.path.join(subject_path, self.detection_results_filename)

    def get_projection_output_path(self, subject_idx: int) -> str:
        """Retourne le chemin de sortie pour les projections de regard d'un sujet."""
        return os.path.join(self.projections_dir, f"{self.projections_prefix}{subject_idx}.csv")

    def get_yolo_detection_weights(self) -> str:
        """Retourne le chemin vers les poids du modèle YOLO."""
        return self.yolo_detection_weights

    def get_posters_path(self) -> str:
        """Retourne le chemin vers le dossier des affiches."""
        return self.posters_root

    # =========================================================================
    # CHARGEMENT DES DONNÉES
    # =========================================================================

    def get_load_camera_params(self, subject_idx: int):
        """Charge les paramètres de caméra (matrice K et coefficients de distorsion D)."""
        json_path = os.path.join(self.get_subject_path(subject_idx), self.camera_params_filename)
        return load_camera_params(json_path)

    def get_gazes(self, subject_idx: int) -> np.ndarray:
        """
        Charge les points de regard pour chaque frame de la vidéo.
        
        Associe chaque timestamp vidéo au point de regard le plus proche
        dans les données eye-tracker.
        
        Returns:
            np.ndarray: Array de shape (n_frames, 2) avec les coordonnées (x, y) du regard
        """
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

        # Pour chaque frame, trouver le point de regard le plus proche en temps
        for i in range(len(world_ts)):
            ts = world_ts[i]
            gi = np.argmin(np.abs(gaze_ts - ts))
            gaze_per_frame[i, 0] = gaze_x[gi]
            gaze_per_frame[i, 1] = gaze_y[gi]

        return gaze_per_frame

    def get_undistorted_gazes(self, subject_idx: int) -> np.ndarray:
        """
        Charge les points de regard et corrige la distorsion de la caméra.
        
        Returns:
            np.ndarray: Points de regard corrigés, shape (n_frames, 2)
        """
        gazes = self.get_gazes(subject_idx)
        K, D = self.get_load_camera_params(subject_idx)

        # Corriger la distorsion des points de regard
        gazes_undistorted = cv2.undistortPoints(
            gazes.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=D,
            P=K
        ).reshape(-1, 2)

        return gazes_undistorted

    def load_posters(self):
        """
        Charge toutes les affiches PNG depuis le dossier des affiches.
        
        Returns:
            List[Tuple[str, np.ndarray]]: Liste de (nom_fichier, image)
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


__all__ = ["DataLoader", "load_camera_params"]
	