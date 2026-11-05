"""Convert official VisDrone detection annotations into YOLO label files."""
import argparse
from pathlib import Path


def convert_split(split, overwrite=False):
    from PIL import Image

    annotations, images, labels = split / "annotations", split / "images", split / "labels"
    if not annotations.is_dir() or not images.is_dir():
        raise FileNotFoundError(f"Expected annotations/ and images/ inside {split}")
    labels.mkdir(exist_ok=True)
    converted = 0
    for annotation in sorted(annotations.glob("*.txt")):
        target = labels / annotation.name
        if target.exists() and not overwrite:
            raise FileExistsError(f"Label already exists: {target}; use --overwrite explicitly")
        image_path = images / (annotation.stem + ".jpg")
        with Image.open(image_path) as image:
            width, height = image.size
        rows = []
        for line_number, line in enumerate(annotation.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                raise ValueError(f"Invalid annotation at {annotation}:{line_number}")
            left, top, box_width, box_height = map(float, fields[:4])
            score, category = int(fields[4]), int(fields[5])
            # Official categories 1-10 map to zero-based IDs; 0 and 11 are not target classes.
            if score == 0 or not 1 <= category <= 10 or box_width <= 0 or box_height <= 0:
                continue
            center_x = (left + box_width / 2) / width
            center_y = (top + box_height / 2) / height
            rows.append(f"{category - 1} {center_x:.6f} {center_y:.6f} {box_width / width:.6f} {box_height / height:.6f}")
        target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        converted += 1
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["VisDrone2019-DET-train", "VisDrone2019-DET-val"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in args.splits:
        count = convert_split(args.root / name, args.overwrite)
        print(f"{name}: converted {count} annotation files")


if __name__ == "__main__":
    main()
