"""Evaluate a locally trained checkpoint on the specified dataset split."""
import argparse
import json

from project_utils import dataset_file, existing_file, output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to the checkpoint produced by training")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="LMSFE-YOLO")
    args = parser.parse_args()
    weights, data = existing_file(args.weights, "Checkpoint"), dataset_file(args.data)
    from ultralytics import YOLO

    options = dict(data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                   conf=args.conf, iou=args.iou, max_det=args.max_det, workers=args.workers,
                   save_json=args.save_json, project=str(output_directory(args.project)),
                   name=args.name, half=False)
    if args.device is not None:
        options["device"] = args.device
    metrics = YOLO(str(weights), task="detect").val(**options)
    summary = {key: float(value) for key, value in metrics.results_dict.items()}
    target = metrics.save_dir / "metrics.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Metrics saved to {target}")


if __name__ == "__main__":
    main()
