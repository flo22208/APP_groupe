#!/usr/bin/env python3
"""
Pipeline principal de traitement des données de regard.

Ce script permet de traiter les données eye-tracking pour :
1. Générer les fichiers de détection/tracking (detection_results.csv)
2. Projeter les points de regard sur les affiches (gaze_projections.csv)

Usage:
    python run_pipeline.py                    # Traite tous les sujets
    python run_pipeline.py --subject 1        # Traite uniquement le sujet 1
    python run_pipeline.py --tracking-only    # Génère uniquement les fichiers de tracking
    python run_pipeline.py --projection-only  # Génère uniquement les projections de regard
    python run_pipeline.py --config other.json # Utilise un fichier de config différent
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class PipelineLogger:
    """Logger simple pour afficher les étapes du pipeline de manière claire."""
    
    COLORS = {
        "HEADER": "\033[95m",
        "BLUE": "\033[94m",
        "CYAN": "\033[96m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BOLD": "\033[1m",
        "END": "\033[0m",
    }
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.step_count = 0
        self.start_time = None
    
    def _color(self, text: str, color: str) -> str:
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['END']}"
        return text
    
    def header(self, text: str):
        """Affiche un header principal."""
        line = "=" * 70
        print(f"\n{self._color(line, 'HEADER')}")
        print(f"{self._color(f'  {text}', 'BOLD')}")
        print(f"{self._color(line, 'HEADER')}\n")
    
    def step(self, text: str):
        """Affiche une nouvelle étape."""
        self.step_count += 1
        print(f"\n{self._color(f'[Étape {self.step_count}]', 'CYAN')} {self._color(text, 'BOLD')}")
        print("-" * 50)
    
    def info(self, text: str):
        """Affiche une information."""
        print(f"  {self._color('ℹ', 'BLUE')} {text}")
    
    def success(self, text: str):
        """Affiche un succès."""
        print(f"  {self._color('✓', 'GREEN')} {text}")
    
    def warning(self, text: str):
        """Affiche un avertissement."""
        print(f"  {self._color('⚠', 'YELLOW')} {text}")
    
    def error(self, text: str):
        """Affiche une erreur."""
        print(f"  {self._color('✗', 'RED')} {text}")
    
    def progress(self, current: int, total: int, item: str = ""):
        """Affiche une progression."""
        percent = (current / total) * 100 if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * current // total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        suffix = f" - {item}" if item else ""
        print(f"\r  [{bar}] {percent:5.1f}% ({current}/{total}){suffix}", end="", flush=True)
        if current == total:
            print()
    
    def start_timer(self):
        """Démarre le chronomètre."""
        self.start_time = time.time()
    
    def elapsed(self) -> str:
        """Retourne le temps écoulé."""
        if self.start_time is None:
            return "N/A"
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m {seconds}s"


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> dict:
    """Charge la configuration depuis un fichier JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: dict, logger: PipelineLogger) -> bool:
    """Valide que la configuration contient tous les champs requis."""
    required_paths = ["acquisition_data", "yolo_detection_weights", "posters"]
    required_files = ["camera_params", "gaze", "world_timestamps", "detection_results"]
    
    valid = True
    
    # Vérifier les chemins
    for path_key in required_paths:
        if path_key not in config.get("paths", {}):
            logger.error(f"Chemin manquant dans config: paths.{path_key}")
            valid = False
    
    # Vérifier les fichiers
    for file_key in required_files:
        if file_key not in config.get("files", {}):
            logger.error(f"Fichier manquant dans config: files.{file_key}")
            valid = False
    
    # Vérifier les sujets
    if "subjects" not in config or len(config["subjects"]) == 0:
        logger.error("Aucun sujet défini dans config.subjects")
        valid = False
    
    return valid


# =============================================================================
# ÉTAPE 1: GÉNÉRATION DES FICHIERS DE TRACKING
# =============================================================================

def run_tracking_for_subject(
    subject_idx: int,
    config: dict,
    logger: PipelineLogger,
    force: bool = False,
    show: bool = False
) -> bool:
    """
    Génère le fichier detection_results.csv pour un sujet donné.
    
    Cette étape utilise YOLO pour détecter et tracker les affiches dans la vidéo.
    """
    import cv2
    import csv
    from tqdm import tqdm
    
    # Import des modules du projet
    from src.utils.DataLoader import DataLoader, load_camera_params
    from src.detectionModel.DetectionModel import DetectionModel
    
    subject_name = config["subjects"][subject_idx]
    subject_path = os.path.join(config["paths"]["acquisition_data"], subject_name)
    
    logger.info(f"Sujet: {subject_name}")
    
    # Vérifier si le fichier existe déjà
    output_csv = os.path.join(subject_path, config["files"]["detection_results"])
    if os.path.exists(output_csv) and not force:
        logger.warning(f"Fichier déjà existant: {output_csv}")
        logger.info("Utilisez --force pour régénérer")
        return True
    
    # Trouver la vidéo
    mp4_files = [f for f in os.listdir(subject_path) if f.lower().endswith(".mp4")]
    if len(mp4_files) == 0:
        logger.error(f"Aucune vidéo .mp4 trouvée dans {subject_path}")
        return False
    video_path = os.path.join(subject_path, mp4_files[0])
    
    logger.info(f"Vidéo: {mp4_files[0]}")
    
    # Charger les paramètres de caméra
    camera_json = os.path.join(subject_path, config["files"]["camera_params"])
    K, D = load_camera_params(camera_json)
    logger.info(f"Paramètres caméra chargés")
    
    # Charger le modèle de détection
    weights_path = config["paths"]["yolo_detection_weights"]
    logger.info(f"Chargement du modèle YOLO...")
    model = DetectionModel(weights_path)
    logger.success(f"Modèle chargé (device: {model.device or 'cpu'})")
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Impossible d'ouvrir la vidéo: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_step = config.get("settings", {}).get("skip_step", 3)
    
    logger.info(f"Frames totales: {total_frames} ({fps:.1f} FPS)")
    logger.info(f"Skip step: {skip_step} (traitement 1 frame sur {skip_step})")
    
    # Créer le fichier de sortie
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class_id", "conf", "x1", "y1", "x2", "y2"])
        
        detections_count = 0
        
        for frame_idx in tqdm(range(total_frames), desc="  Tracking", unit="frame", ncols=80):
            ret, frame = cap.read()
            if not ret:
                break

            # Traiter seulement toutes les skip_step frames
            if frame_idx % skip_step != 0:
                continue

            # Corriger la distorsion
            frame_undist = cv2.undistort(frame, K, D)

            # Tracking avec YOLO (sans affichage natif YOLO)
            results = model.model.track(
                source=frame_undist,
                conf=0.3,
                iou=0.5,
                persist=True,
                verbose=False,
                device=model.device,
                show=False
            )

            if not results:
                continue

            r = results[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            # Dessiner les boxes sur la frame pour affichage
            if show:
                display_frame = frame_undist.copy()
                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    color = (0, 255, 0)
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID:{int(ids[i]) if ids is not None else -1}"
                    cv2.putText(display_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imshow("Tracking", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Arrêt de l'affichage demandé par l'utilisateur.")
                    break

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                track_id = int(ids[i]) if ids is not None else -1
                cls_id = int(cls_ids[i]) if cls_ids is not None else -1
                conf = float(confs[i]) if confs is not None else -1.0

                writer.writerow([frame_idx, track_id, cls_id, conf, float(x1), float(y1), float(x2), float(y2)])
                detections_count += 1
    
    cap.release()
    if show:
        cv2.destroyAllWindows()
    
    logger.success(f"Fichier généré: {output_csv}")
    logger.info(f"Total détections: {detections_count}")
    
    return True


# =============================================================================
# ÉTAPE 2: PROJECTION DES REGARDS SUR LES AFFICHES
# =============================================================================

def run_projection_for_subject(
    subject_idx: int,
    config: dict,
    logger: PipelineLogger,
    force: bool = False,
    no_cache: bool = False,
    show: bool = False
) -> bool:
    """
    Génère le fichier de projections de regard pour un sujet donné.
    
    Cette étape projette chaque point de regard sur l'affiche correspondante
    en utilisant une homographie calculée à partir des features détectées.
    
    Args:
        no_cache: Si True, désactive le cache de vote par track_id et détecte l'affiche à chaque frame
        show: Si True, affiche les détections en temps réel
    """
    from src.pipeline.GazeAnalyser import GazeAnalyser
    
    subject_name = config["subjects"][subject_idx]
    
    logger.info(f"Sujet: {subject_name}")
    
    # Définir le chemin de sortie
    output_dir = config.get("output", {}).get("projections_dir", "data/")
    output_csv = os.path.join(output_dir, f"gaze_projections_subject{subject_idx}.csv")
    
    if os.path.exists(output_csv) and not force:
        logger.warning(f"Fichier déjà existant: {output_csv}")
        logger.info("Utilisez --force pour régénérer")
        return True
    
    # Vérifier que le fichier de détection existe
    subject_path = os.path.join(config["paths"]["acquisition_data"], subject_name)
    det_csv = os.path.join(subject_path, config["files"]["detection_results"])
    if not os.path.exists(det_csv):
        logger.error(f"Fichier de détection manquant: {det_csv}")
        logger.info("Exécutez d'abord l'étape de tracking (--tracking-only)")
        return False
    
    logger.info(f"Fichier détection: {config['files']['detection_results']}")
    logger.info(f"Chargement du GazeAnalyser...")
    
    # Créer le GazeAnalyser (charge le modèle et les affiches)
    # Note: on crée un fichier config temporaire car GazeAnalyser attend un chemin
    gaze_analyser = GazeAnalyser("config.json")
    
    logger.success(f"GazeAnalyser chargé ({len(gaze_analyser.posters)} affiches)")
    
    # Paramètres d'analyse
    start_frame = config.get("settings", {}).get("start_frame", 0)
    end_frame = config.get("settings", {}).get("end_frame", None)
    
    logger.info(f"Frame de départ: {start_frame}")
    if end_frame:
        logger.info(f"Frame de fin: {end_frame}")
    
    if no_cache:
        logger.info(f"Mode sans cache: détection à chaque frame (plus lent mais plus précis)")
    else:
        logger.info(f"Mode avec cache: vote par track_id (plus rapide)")
    
    logger.info(f"Analyse en cours...")
    
    if show:
        logger.info(f"Visualisation activée (appuyez sur 'q' pour quitter)")
    
    # Lancer l'analyse
    gaze_analyser.analyse_video_for_subject(
        subject_idx=subject_idx,
        output_csv_path=output_csv,
        start_frame=start_frame,
        end_frame=end_frame,
        use_cache=not no_cache,
        show=show
    )
    
    logger.success(f"Fichier généré: {output_csv}")
    
    # Statistiques basiques
    if os.path.exists(output_csv):
        import pandas as pd
        df = pd.read_csv(output_csv)
        logger.info(f"Points projetés: {len(df)}")
        if len(df) > 0:
            posters = df["poster_name"].unique()
            logger.info(f"Affiches détectées: {len(posters)}")
            for poster in posters:
                count = len(df[df["poster_name"] == poster])
                logger.info(f"  - {poster}: {count} points")
    
    return True


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(
    config_path: str,
    subject_indices: Optional[List[int]] = None,
    run_tracking: bool = True,
    run_projection: bool = True,
    force: bool = False,
    no_cache: bool = False,
    show: bool = False
):
    """
    Exécute le pipeline complet de traitement des données de regard.
    
    Étapes:
    1. Tracking: Détection et tracking des affiches dans les vidéos (YOLO)
    2. Projection: Projection des points de regard sur les affiches (Homographie)
    
    Args:
        no_cache: Si True, désactive le système de vote par track_id
        show: Si True, affiche les détections en temps réel
    """
    logger = PipelineLogger()
    logger.start_timer()
    
    # Header
    logger.header("PIPELINE DE TRAITEMENT DES DONNÉES DE REGARD")
    
    # Étape 0: Chargement de la configuration
    logger.step("Chargement de la configuration")
    
    if not os.path.exists(config_path):
        logger.error(f"Fichier de configuration introuvable: {config_path}")
        return False
    
    config = load_config(config_path)
    logger.success(f"Configuration chargée: {config_path}")
    
    if not validate_config(config, logger):
        logger.error("Configuration invalide")
        return False
    
    logger.success("Configuration validée")
    
    # Afficher les paramètres
    logger.info(f"Dossier données: {config['paths']['acquisition_data']}")
    logger.info(f"Dossier affiches: {config['paths']['posters']}")
    logger.info(f"Poids YOLO: {config['paths']['yolo_detection_weights']}")
    logger.info(f"Nombre de sujets: {len(config['subjects'])}")
    
    # Déterminer les sujets à traiter
    all_subjects = list(range(len(config["subjects"])))
    subjects_to_process = subject_indices if subject_indices else all_subjects
    
    logger.info(f"Sujets à traiter: {subjects_to_process}")
    
    success_count = 0
    total_count = len(subjects_to_process)
    
    # Étape 1: Tracking (optionnel)
    if run_tracking:
        logger.step("Génération des fichiers de tracking (YOLO)")

        for i, subject_idx in enumerate(subjects_to_process):
            logger.info(f"\n--- Sujet {subject_idx} ({i+1}/{total_count}) ---")
            try:
                if run_tracking_for_subject(subject_idx, config, logger, force, show):
                    success_count += 1
            except Exception as e:
                logger.error(f"Erreur: {str(e)}")

        logger.info(f"\nTracking terminé: {success_count}/{total_count} sujets")
    
    # Étape 2: Projection (optionnel)
    if run_projection:
        logger.step("Projection des regards sur les affiches")
        
        success_count = 0
        for i, subject_idx in enumerate(subjects_to_process):
            logger.info(f"\n--- Sujet {subject_idx} ({i+1}/{total_count}) ---")
            try:
                if run_projection_for_subject(subject_idx, config, logger, force, no_cache, show):
                    success_count += 1
            except Exception as e:
                logger.error(f"Erreur: {str(e)}")
        
        logger.info(f"\nProjection terminée: {success_count}/{total_count} sujets")
    
    # Résumé final
    logger.header("PIPELINE TERMINÉ")
    logger.info(f"Temps total: {logger.elapsed()}")
    logger.info(f"Sujets traités: {total_count}")
    
    return True


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline de traitement des données de regard eye-tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python run_pipeline.py                     # Traite tous les sujets (tracking + projection)
  python run_pipeline.py --subject 1         # Traite uniquement le sujet 1
  python run_pipeline.py --subject 1 2 3     # Traite les sujets 1, 2 et 3
  python run_pipeline.py --tracking-only     # Génère uniquement les fichiers de tracking
  python run_pipeline.py --no-cache          # Détecte l'affiche à chaque frame (sans cache de vote)
  python run_pipeline.py --projection-only   # Génère uniquement les projections de regard
  python run_pipeline.py --force             # Force la régénération des fichiers existants
  python run_pipeline.py --config other.json # Utilise un autre fichier de configuration
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.json",
        help="Chemin vers le fichier de configuration (défaut: config.json)"
    )
    
    parser.add_argument(
        "--subject", "-s",
        type=int,
        nargs="+",
        default=None,
        help="Indice(s) du/des sujet(s) à traiter (défaut: tous)"
    )
    
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Exécute uniquement l'étape de tracking (génération des detection_results.csv)"
    )
    
    parser.add_argument(
        "--projection-only",
        action="store_true",
        help="Exécute uniquement l'étape de projection (génération des gaze_projections.csv)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force la régénération des fichiers même s'ils existent déjà"
    )
    
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="Affiche la liste des sujets disponibles et quitte"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Désactive le cache de vote par track_id (détecte l'affiche à chaque frame, plus lent mais plus précis)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche les détections en temps réel (appuyez sur 'q' pour quitter)"
    )
    
    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Mode liste des sujets
    if args.list_subjects:
        if not os.path.exists(args.config):
            print(f"Erreur: fichier de configuration introuvable: {args.config}")
            sys.exit(1)
        
        config = load_config(args.config)
        print("\nSujets disponibles:")
        print("-" * 40)
        for i, subject in enumerate(config.get("subjects", [])):
            print(f"  {i}: {subject}")
        print()
        sys.exit(0)
    
    # Déterminer les étapes à exécuter
    run_tracking = True
    run_projection = True
    
    if args.tracking_only:
        run_projection = False
    elif args.projection_only:
        run_tracking = False
    
    # Exécuter le pipeline
    success = run_pipeline(
        config_path=args.config,
        subject_indices=args.subject,
        run_tracking=run_tracking,
        run_projection=run_projection,
        force=args.force,
        no_cache=args.no_cache,
        show=args.show
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
