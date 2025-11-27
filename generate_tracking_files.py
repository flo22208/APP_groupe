import os

from src.utils.DataLoader import DataLoader
from src.detectionModel.DetectionModel import DetectionModel

def main():
    # Charge la configuration et les chemins des sujets
    loader = DataLoader("config.json")

    # Charge le modèle de détection/track
    weights_path = loader.get_yolo_detection_weights()
    model = DetectionModel(weights_path)

    # Parcourt tous les sujets définis dans la config
    num_subjects = len(loader.subjects)

    for subject_idx in range(num_subjects):
        subject_path = loader.get_subject_path(subject_idx)
        subject_name = loader.subjects[subject_idx]

        print(f"Processing subject {subject_idx}: {subject_name}")

        # Récupère le chemin de la vidéo
        video_path = loader.get_video_path(subject_idx)

        # Fichier de sortie de tracking, à côté des autres CSV
        output_csv = loader.get_detection_results_path(subject_idx)

        print(f"  Video:  {video_path}")
        print(f"  Output: {output_csv}")

        # Lance le tracking et sauvegarde les résultats
        model.track_and_save(
            source=video_path,
            output_path=output_csv,
            conf_threshold=0.3,
            iou=0.5,
            persist=True,
        )

        print(f"  Done.\n")


if __name__ == "__main__":
    main()
