"""Train LMSFE-YOLO or a released ablation configuration."""
import argparse

from project_utils import ROOT, dataset_file, existing_file, output_directory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ultralytics/cfg/LMSFE-YOLO/LMSFE-YOLO.yaml")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--config", default="ultralytics/cfg/training.yaml")
    parser.add_argument("--weights", help="Optional local initialization checkpoint; omitted for training from scratch")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help="Device such as 0 or cpu; omitted for automatic selection")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="LMSFE-YOLO")
    return parser.parse_args()


def main():
    args = parse_args()
    model_file = existing_file(args.model, "Model configuration")
    data_file = dataset_file(args.data)
    config_file = existing_file(args.config, "Training configuration")
    weights = existing_file(args.weights, "Initialization checkpoint") if args.weights else None
    import yaml
    from ultralytics import YOLO

    options = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    for key in ("epochs", "batch", "imgsz", "seed", "workers", "device"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    options.update(data=str(data_file), pretrained=False,
                   project=str(output_directory(args.project)), name=args.name)
    model = YOLO(str(model_file), task="detect")
    if weights:
        model.load(str(weights))
    model.train(**options)


if __name__ == "__main__":
    main()
