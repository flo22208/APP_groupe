from ultralytics import YOLO
import numpy as np
import csv
from pathlib import Path


class DetectionModel:
    def __init__(self, weights_path: str, device: str = "cuda"):
        self.weights_path = weights_path
        self.model = YOLO(weights_path)
        self.device = device
        # Déplace le modèle sur le device demandé si possible
        try:
            self.model.to(device)
        except Exception:
            # Si le device n'est pas dispo, on laisse le défaut ultralytics
            print(f"Warning: could not move model to device '{device}'. Using default device.")
            # Propager l'exception si besoin
            print('Exception details:', Exception)
            self.device = None

    def predict(self, image, conf_threshold: float = 0.25)-> np.ndarray:
        """Retourne les boxes YOLO pour une image.
        Retourne un tableau numpy de shape (N, 4) avec les coordonnées (x1, y1, x2, y2).
        """
        results = self.model(
            image,
            conf=conf_threshold,
            verbose=False,
            device=self.device if self.device is not None else None,
        )
        return results[0].boxes.xyxy.cpu().numpy()

    def track(self, source, conf_threshold: float = 0.3, iou: float = 0.5, persist: bool = True, show: bool = False):
        """Effectue du tracking multi-objet sur une source vidéo.

        `source` peut être un chemin vidéo, un index de webcam, etc.
        Cette méthode encapsule `YOLO.track` avec des paramètres par défaut
        adaptés à votre cas d'usage.
        """
        results = self.model.track(
            source=source,
            conf=conf_threshold,
            iou=iou,
            persist=persist,
            show=show,
            device=self.device if self.device is not None else None,
        )
        return results

    def track_and_save(
            self,
            source,
            output_path: str,
            conf_threshold: float = 0.3,
            iou: float = 0.5,
            persist: bool = True,
        ):
        """Track sur une vidéo et sauvegarde les résultats dans un fichier CSV.

        Chaque ligne contient : frame, track_id, class_id, conf, x1, y1, x2, y2.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        results = self.model.track(
            source=source,
            conf=conf_threshold,
            iou=iou,
            persist=persist,
            show=False,
            device=self.device if self.device is not None else None,
        )

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

            for r in results:
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