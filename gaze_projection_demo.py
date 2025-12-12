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
	plt.suptitle("Projections de regard sur les affiches", fontsize=16)

	for idx, (poster_name, points) in enumerate(selected_items, start=1):
		poster_img = posters_dict.get(poster_name)
		if poster_img is None:
			continue

		height, width = poster_img.shape[:2]
		# Diviser la taille par 2 pour l'affichage
		half_width, half_height = width // 2, height // 2
		scaled_points = [(x / 2, y / 2) for x, y in points]
		
		poster_rgb = cv2.cvtColor(poster_img, cv2.COLOR_BGR2RGB)
		poster_rgb = cv2.resize(poster_rgb, (half_width, half_height))

		plt.subplot(rows, cols, idx)
		plt.imshow(poster_rgb)
		
		# Afficher les points de regard
		if scaled_points:
			xs, ys = zip(*scaled_points)
			plt.scatter(xs, ys, c='red', s=5, alpha=0.5, marker='o')
		
		plt.title(poster_name)
		plt.axis("off")

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()


if __name__ == "__main__":
	main()
