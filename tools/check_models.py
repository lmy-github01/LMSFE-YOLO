"""Check construction and inference for the released model configurations."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Check one model; omit to check every released YAML")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    import torch
    from ultralytics.nn.tasks import DetectionModel

    torch.set_num_threads(args.threads)
    paths = [Path(args.model)] if args.model else sorted((ROOT / "ultralytics/cfg/LMSFE-YOLO").rglob("*.yaml"))
    records = []
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        model = DetectionModel(str(path), verbose=False).eval()
        with torch.inference_mode():
            prediction = model(torch.zeros(1, 3, args.imgsz, args.imgsz))[0]
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite prediction: {path.name}")
        record = {"model": str(path.relative_to(ROOT)), "parameters": sum(p.numel() for p in model.parameters()),
                  "stride": model.stride.tolist(), "output_shape": list(prediction.shape)}
        records.append(record)
        print(json.dumps(record), flush=True)
        del model
    if args.output:
        report = Path(args.output)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
