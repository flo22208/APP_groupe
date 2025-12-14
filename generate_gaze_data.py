from src.pipeline.GazeAnalyser import GazeAnalyser


def main() -> None:
	config_path = "config.json"
	subject_idx = 1
	output_csv = "data/gaze_projections_subject1.csv"

	gaze_analyser = GazeAnalyser(config_path)
	gaze_analyser.analyse_video_for_subject(subject_idx, output_csv, start_frame=50)


if __name__ == "__main__":
	main()

