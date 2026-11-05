"""Run image inference with a locally trained checkpoint."""
import argparse

from project_utils import existing_file, output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image file or image directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device")
    parser.add_argument("--project", default="runs/predict")
    args = parser.parse_args()
    weights = existing_file(args.weights, "Checkpoint")
    from pathlib import Path
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Image source does not exist: {args.source}")
    from ultralytics import YOLO
    options = dict(source=args.source, imgsz=args.imgsz, conf=args.conf,
                   save=True, project=str(output_directory(args.project)))
    if args.device is not None:
        options["device"] = args.device
    YOLO(str(weights), task="detect").predict(**options)


if __name__ == "__main__":
    main()
