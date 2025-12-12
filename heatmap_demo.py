import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.DataLoader import DataLoader


def load_gaze_projections(csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
	"""Load projected gaze points from CSV and group them by poster name."""
	points_by_poster: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

	with open(csv_path, newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			name = row["poster_name"]
			px = float(row["proj_x"])
			py = float(row["proj_y"])
			points_by_poster[name].append((px, py))

	return points_by_poster


def compute_heatmap(points: List[Tuple[float, float]], width: int, height: int, sigma: float = 50.0) -> np.ndarray:
	"""Compute a simple heatmap (2D density) from projected points.

	The result is a 2D array of shape (height, width) normalized to [0, 1].
	"""
	if not points:
		return np.zeros((height, width), dtype=np.float32)

	heatmap = np.zeros((height, width), dtype=np.float32)

	for x, y in points:
		x_int = int(round(x))
		y_int = int(round(y))
		if 0 <= x_int < width and 0 <= y_int < height:
			heatmap[y_int, x_int] += 1.0

	# Optionally apply Gaussian blur for smoother heatmap
	if sigma > 0:
		heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)

	# Normalize to [0, 1]
	if heatmap.max() > 0:
		heatmap /= heatmap.max()

	return heatmap


def main() -> None:
	config_path = "config.json"
	csv_path = "data/gaze_projections_subject1.csv"

	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_gaze_data.py first.")

	loader = DataLoader(config_path)
	posters = loader.load_posters()
	posters_dict = {name: img for name, img in posters}

	points_by_poster = load_gaze_projections(csv_path)

	if not points_by_poster:
		raise RuntimeError("No gaze projections found in CSV.")

	# Prendre jusqu'à 8 affiches et les afficher en 2x4
	selected_items = list(points_by_poster.items())[:8]
	rows, cols = 2, 4

	plt.figure(figsize=(cols * 4, rows * 4))
	plt.suptitle("Heatmaps de regard pour les affiches", fontsize=16)

	for idx, (poster_name, points) in enumerate(selected_items, start=1):
		poster_img = posters_dict.get(poster_name)
		if poster_img is None:
			continue

		height, width = poster_img.shape[:2]
		# Diviser la taille par 2 pour le calcul de la heatmap
		half_width, half_height = width // 2, height // 2
		scaled_points = [(x / 2, y / 2) for x, y in points]
		heatmap = compute_heatmap(scaled_points, half_width, half_height)
		poster_rgb = cv2.cvtColor(poster_img, cv2.COLOR_BGR2RGB)
		poster_rgb = cv2.resize(poster_rgb, (half_width, half_height))

		plt.subplot(rows, cols, idx)
		plt.imshow(poster_rgb)
		plt.imshow(heatmap, cmap="jet", alpha=0.5, interpolation="bilinear")
		plt.title(poster_name)
		plt.axis("off")

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()


if __name__ == "__main__":
	main()

