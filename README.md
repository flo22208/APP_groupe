# Pipeline de Traitement des Données de Regard (APP Project)

## Description

Ce projet analyse les données eye-tracking pour déterminer quelles affiches sont regardées par les sujets. Le pipeline complet inclut :

1. **Tracking** : Détection et tracking des affiches dans les vidéos (YOLO)
2. **Projection** : Projection des points de regard sur les affiches (Homographie)

## Installation

Créez un environnement Python (exemple avec conda) :
```bash
conda create -n APP python=3.8 -y
conda activate APP
```

Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation du Pipeline

### Commande principale

```bash
# Traite tous les sujets (tracking + projection)
python run_pipeline.py

# Traite uniquement le sujet 1
python run_pipeline.py --subject 1

# Traite les sujets 0, 1 et 2
python run_pipeline.py --subject 0 1 2

# Génère uniquement les fichiers de tracking
python run_pipeline.py --tracking-only

# Génère uniquement les projections de regard
python run_pipeline.py --projection-only

# Force la régénération des fichiers existants
python run_pipeline.py --force

# Liste les sujets disponibles
python run_pipeline.py --list-subjects

# Utilise un fichier de configuration différent
python run_pipeline.py --config other_config.json
```

### Configuration

Tous les paramètres sont centralisés dans `config.json` :

```json
{
    "paths": {
        "acquisition_data": "data/AcquisitionsEyeTracker/",
        "yolo_detection_weights": "src/detectionModel/run/weights/best.pt",
        "posters": "data/Affiches/"
    },
    "output": {
        "projections_dir": "data/",
        "projections_prefix": "gaze_projections_subject"
    },
    "settings": {
        "skip_step": 3,
        "start_frame": 50,
        "end_frame": null
    },
    "subjects": [
        "sujet1_f-42e0d11a/",
        "sujet2_f-835bf855/",
        ...
    ]
}
```

## Structure du Projet

```
├── run_pipeline.py          # Script principal du pipeline
├── config.json              # Configuration centralisée
├── requirements.txt         # Dépendances Python
│
├── data/
│   ├── AcquisitionsEyeTracker/    # Données brutes des sujets
│   │   └── sujetX_*/
│   │       ├── *.mp4              # Vidéo de la scène
│   │       ├── gaze.csv           # Données de regard
│   │       ├── scene_camera.json  # Paramètres de caméra
│   │       └── detection_results.csv  # (généré) Détections YOLO
│   │
│   ├── Affiches/                  # Images de référence des affiches
│   └── gaze_projections_subject*.csv  # (généré) Projections de regard
│
└── src/
    ├── pipeline/
    │   ├── GazeAnalyser.py        # Analyse et projection des regards
    │   ├── HomographyManager.py   # Calcul des homographies
    │   └── KeypointsManager.py    # Détection de features (ORB)
    │
    ├── detectionModel/
    │   └── DetectionModel.py      # Wrapper YOLO
    │
    └── utils/
        └── DataLoader.py          # Chargement des données
```

## Étapes du Pipeline

### Étape 1 : Tracking (YOLO)

Génère un fichier `detection_results.csv` pour chaque sujet contenant les bounding boxes des affiches détectées à chaque frame.

```
frame, track_id, class_id, conf, x1, y1, x2, y2
0, 1, 0, 0.95, 100, 200, 400, 600
3, 1, 0, 0.93, 102, 198, 402, 598
...
```

### Étape 2 : Projection des regards

Génère un fichier `gaze_projections_subjectX.csv` contenant les coordonnées du regard projeté sur chaque affiche.

```
frame_idx, poster_name, poster_index, proj_x, proj_y
50, affiche1.png, 0, 234.5, 567.8
53, affiche1.png, 0, 240.2, 570.1
...
```

## Scripts de Démonstration

- `gaze_demo.py` : Visualisation des données de regard
- `gaze_projection_demo.py` : Démonstration de la projection
- `heatmap_demo.py` : Génération de heatmaps
- `homography_demo.py` : Test des homographies
- `detection_model_demo.py` : Test du modèle de détection

## Références

- [Automatic Detection and Rectification of Paper Receipts on Smartphones](https://arxiv.org/pdf/2303.05763)