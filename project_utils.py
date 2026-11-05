"""Shared path handling for the command-line entry points."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def existing_file(value, description):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def dataset_file(value):
    """Resolve a prepared dataset config without downloading data."""
    path = existing_file(value, "Dataset configuration")
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError("Dataset YAML must define train, val, and names")
    if "download" in config:
        raise ValueError("Use a prepared dataset YAML without a download section")
    dataset_root = Path(config.get("path", path.parent)).expanduser()
    if not dataset_root.is_absolute():
        raise ValueError(f"Set an absolute dataset root in the 'path' field of {path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    return path


def output_directory(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
