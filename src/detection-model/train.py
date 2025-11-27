import os
import sys

# Directory management 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(project_root)
os.chdir(project_root)
sys.path.insert(0, project_root)

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Entraînement
model.train(
    data="data/datasets/posters-detection/posters-detection.yaml",          # fichier YAML du dataset
    epochs=20,                      # nombre d'époques
    imgsz=640,                      # taille d'image
    batch=8,                        # batch size
    lr0=1e-3,                       # learning rate initial
    project="src/detection-model/", # dossier de sortie
    name="poster_detection",        # nom de l'expérience
    exist_ok=True                   # écrase si le dossier existe
)
