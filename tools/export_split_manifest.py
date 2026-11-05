"""Record the files and hashes in prepared YOLO train/val/test splits without copying images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def image_files(value, root):
    values = value if isinstance(value, list) else [value]
    files = set()
    for entry in values:
        path = Path(entry).expanduser()
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            files.update(p.resolve() for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == '.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                item = Path(line.strip()).expanduser()
                files.add((item if item.is_absolute() else path.parent / item).resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.add(path.resolve())
        else:
            raise FileNotFoundError(f'Split entry does not exist: {path}')
    return sorted(files)


def export(data, output):
    config = yaml.safe_load(data.read_text(encoding='utf-8-sig'))
    root = Path(config.get('path', data.parent)).expanduser()
    root = (data.parent / root).resolve() if not root.is_absolute() else root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing snapshot: {output}')
    snapshots = {}
    for split in ('train', 'val', 'test'):
        if not config.get(split):
            continue
        rows = []
        for image in image_files(config[split], root):
            if not image.is_file():
                raise FileNotFoundError(image)
            relative = image.relative_to(root)
            parts = list(relative.parts)
            label = None
            if 'images' in parts:
                parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
                label = (root / Path(*parts)).with_suffix('.txt')
            rows.append({'image': relative.as_posix(), 'image_sha256': digest(image),
                         'label': label.relative_to(root).as_posix() if label and label.is_file() else '',
                         'label_sha256': digest(label) if label and label.is_file() else ''})
        if not rows:
            raise ValueError(f'Empty image split: {split}')
        snapshots[split] = rows
    if not snapshots:
        raise ValueError('No train/val/test splits were defined')
    output.mkdir(parents=True)
    for split, rows in snapshots.items():
        with (output / f'{split}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {'dataset_yaml_sha256': digest(data), 'names': config.get('names'),
                'counts': {key: len(rows) for key, rows in snapshots.items()},
                'scope': 'Snapshot of supplied local data; historical experiment identity is not inferred.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.data.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == '__main__':
    main()
