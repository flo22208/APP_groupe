import os

from src.utils.DataLoader import DataLoader
from src.detectionModel.DetectionModel import DetectionModel
import cv2
import csv
from pathlib import Path
from tqdm import tqdm


def main():
    # Charge la configuration et les chemins des sujets
    loader = DataLoader("config.json")

    # Charge le modèle de détection/track
    weights_path = loader.get_yolo_detection_weights()
    model = DetectionModel(weights_path)

    # Parcourt tous les sujets définis dans la config
    num_subjects = len(loader.subjects)

    for subject_idx in range(1,num_subjects):
        subject_path = loader.get_subject_path(subject_idx)
        subject_name = loader.subjects[subject_idx]

        print(f"Processing subject {subject_idx}: {subject_name}")

        # Récupère le chemin de la vidéo et les paramètres de caméra
        video_path = loader.get_video_path(subject_idx)
        K, D = loader.get_load_camera_params(subject_idx)

        # Fichier de sortie de tracking, à côté des autres CSV
        output_csv = loader.get_detection_results_path(subject_idx)

        print(f"  Video:  {video_path}")
        print(f"  Output: {output_csv}")

        # Ouverture vidéo et écriture CSV personnalisée
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Could not open video {video_path}, skipping.")
            continue

        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "track_id",
                "class_id",
                "conf",
                "x1",
                "y1",
                "x2",
                "y2",
            ])

            frame_idx = 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in tqdm(range(total_frames), desc=f"Tracking subject {subject_idx}", unit="frame"):
                ret, frame = cap.read()
                if not ret:
                    break

                # Traiter seulement toutes les skip_step frames (0, 3, 6, ...)
                if frame_idx % loader.skip_step != 0:
                    frame_idx += 1
                    continue

                # Détordre la frame avant détection/tracking
                frame_undist = cv2.undistort(frame, K, D)

                # Utiliser le modèle YOLO en mode tracking sur une frame unique
                results = model.model.track(
                    source=frame_undist,
                    conf=0.3,
                    iou=0.5,
                    persist=True,
                    verbose=False,
                    device=model.device if model.device is not None else None,
                )

                # Les résultats peuvent être une liste même pour une seule frame
                if not results:
                    frame_idx += 1
                    continue

                r = results[0]
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    frame_idx += 1
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    track_id = int(ids[i]) if ids is not None else -1
                    cls_id = int(cls_ids[i]) if cls_ids is not None else -1
                    conf = float(confs[i]) if confs is not None else -1.0

                    writer.writerow([
                        frame_idx,
                        track_id,
                        cls_id,
                        conf,
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ])

                frame_idx += 1

        cap.release()

        print(f"  Done.\n")


if __name__ == "__main__":
    main()
