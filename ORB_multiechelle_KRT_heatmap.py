import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import math

# ==========================
# PARAMÈTRES
# ==========================
GAUSSIAN_SIGMA = 50
HEATMAP_ALPHA = 0.6
COLORMAP = cv2.COLORMAP_JET

AFFICHES_DIR = "data/Affiches"
GAZE_DIR = "gaze_points"

# ==========================
# ARGUMENTS CLI
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=int, default=None,
                    help="Filtrer par sujet (ex: --subject 2)")
args = parser.parse_args()

SUBJECT_FILTER = args.subject

# ==========================
# FONCTIONS
# ==========================
def generate_heatmap(points, shape):
    heatmap = np.zeros(shape, dtype=np.float32)

    # Décalage en pixels
    DECALAGE_X = -90  # vers la gauche
    DECALAGE_Y = 90   # vers le bas
    for x, y in points:
        ix = int(x) + DECALAGE_X
        iy = int(y) + DECALAGE_Y
        if 0 <= ix < shape[1] and 0 <= iy < shape[0]:
            heatmap[iy, ix] += 1

    heatmap = gaussian_filter(heatmap, sigma=GAUSSIAN_SIGMA)
    heatmap /= heatmap.max() if heatmap.max() > 0 else 1
    return heatmap


def overlay_heatmap(image, heatmap):
    heatmap_inv = 1.0 - heatmap  # inversion logique
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_inv), COLORMAP
    )
    return cv2.addWeighted(image, 1 - HEATMAP_ALPHA, heatmap_color, HEATMAP_ALPHA, 0)

# ==========================
# CHARGEMENT DES DONNÉES
# ==========================
heatmap_results = []

for csv_file in Path(GAZE_DIR).glob("*.csv"):
    poster_name = csv_file.stem
    poster_path = Path(AFFICHES_DIR) / poster_name

    if not poster_path.exists():
        continue

    poster_img = cv2.imread(str(poster_path))
    poster_img = cv2.cvtColor(poster_img, cv2.COLOR_BGR2RGB)
    h, w = poster_img.shape[:2]

    df = pd.read_csv(csv_file)

    if SUBJECT_FILTER is not None:
        df = df[df["subject_id"] == SUBJECT_FILTER]

    if len(df) == 0:
        continue

    points = df[["x_aff", "y_aff"]].values
    heatmap = generate_heatmap(points, (h, w))
    overlay = overlay_heatmap(poster_img, heatmap)

    heatmap_results.append({
        "name": poster_name,
        "image": overlay,
        "count": len(points)
    })

# ==========================
# AFFICHAGE EN SUBFIGURES
# ==========================
if len(heatmap_results) == 0:
    print("❌ Aucune heatmap à afficher.")
    exit()

ROWS = 2
COLS = 4

fig, axes = plt.subplots(
    ROWS,
    COLS,
    figsize=(5 * COLS, 6 * ROWS)
)

axes = axes.flatten()

for ax, result in zip(axes, heatmap_results):
    ax.imshow(result["image"])
    ax.set_title(f"{result['name']}\n{result['count']} points", fontsize=10)
    ax.axis("off")

# Désactiver axes inutilisés
for ax in axes[len(heatmap_results):]:
    ax.axis("off")

subject_txt = f"Sujet {SUBJECT_FILTER}" if SUBJECT_FILTER is not None else "Tous les sujets"
fig.suptitle(f"Cartes de chaleur par affiche — {subject_txt}", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
