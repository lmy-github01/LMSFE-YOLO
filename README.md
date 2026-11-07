# LMSFE-YOLO

Implementation of **LMSFE-YOLO: A parameter-efficient edge-enhanced detector for UAV small object detection**.

## Contents

- `ultralytics/`: model implementation and configuration files.
- `train.py`, `val.py`, `predict.py`: training, evaluation and inference entry points.
- `tools/`: dataset preparation and software utilities.

## Installation

Use Python 3.10 and install a matching PyTorch/torchvision build for your CUDA environment, then install this modified local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

Confirm that imports use this checkout:

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

## Dataset preparation

Obtain VisDrone from the [official dataset repository](https://github.com/VisDrone/VisDrone-Dataset) and follow the provider's access and use conditions. Set `path` in `ultralytics/cfg/datasets/VisDrone.yaml` to your absolute dataset directory. The entry points use prepared datasets and do not automatically download images.

If starting with the official detection annotations:

```bash
python tools/prepare_visdrone.py --root /absolute/path/to/VisDrone
```

For other datasets, supply the corresponding prepared dataset YAML and class mapping through `--data`.

## Training, evaluation and inference

Run from the repository root:

```bash
python train.py --data ultralytics/cfg/datasets/VisDrone.yaml --device 0
python val.py --weights runs/train/LMSFE-YOLO/weights/best.pt --data ultralytics/cfg/datasets/VisDrone.yaml --device 0
python predict.py --weights runs/train/LMSFE-YOLO/weights/best.pt --source /path/to/image.jpg --device 0
```

Use the actual output directory printed by the trainer, since repeated run names may receive a numeric suffix. Training starts from random initialization unless a local checkpoint is supplied with `--weights`. The training settings are in `ultralytics/cfg/training.yaml`; model and component configurations are in `ultralytics/cfg/LMSFE-YOLO/`. Each entry point provides `--help` for its options.

## License

The Ultralytics-derived software retains the AGPL-3.0 license. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Third-party dataset images and annotations remain subject to their providers' terms.

## Additional documentation

- [Dataset sources, experimental subsets, and category mappings](docs/DATASETS.md).
- [Environment, training settings, and execution instructions](docs/REPRODUCIBILITY.md).
- [Code and data availability](docs/DATA_AVAILABILITY.md).
- [CSV result-file descriptions](ultralytics/results/README.md).

The TinyPerson dataset configuration is available at `ultralytics/cfg/datasets/TinyPerson.yaml`. Set its `path` to the prepared dataset root. It uses two classes, `0: earth_person` and `1: sea_person`, and loads the evaluation subset through `val`.

## Citation and archiving

Software citation information is provided in [CITATION.cff](CITATION.cff), and Zenodo archive metadata are provided in [.zenodo.json](.zenodo.json). Cite the specific archived release used for an analysis.

Version `v1.0.0` is archived in Zenodo: [10.5281/zenodo.24027516](https://doi.org/10.5281/zenodo.24027516).

