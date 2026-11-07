# Environment and execution

## Environment

The manuscript reports Python 3.10.14, PyTorch 2.2.2, CUDA 12.1, and cuDNN 8.9.2, with an NVIDIA GeForce RTX 3090 GPU. The repository dependency list is provided in `requirements.txt`.

Install the local modified package from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

For a CUDA GPU, select matching PyTorch and torchvision builds using the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

## Training settings

The released entry point reads `ultralytics/cfg/training.yaml`. Its current defaults include 200 epochs, batch size 8, image size 640, SGD, initial learning rate 0.01, momentum 0.937, and weight decay 0.0005. Read that file for the remaining settings. A local initialization checkpoint can be passed through `--weights`.

## Execution sequence

1. Obtain the benchmark from its provider and prepare its annotations.
2. Set an absolute dataset root and the correct category mapping in the dataset YAML.
3. Choose the main model or an ablation YAML and run `train.py`.
4. Evaluate the corresponding checkpoint with `val.py` on the specified subset.
5. Retain the metric summary and any prediction outputs alongside the configuration and checkpoint identifier.

Use the actual run directory printed by the trainer; repeated run names may receive a numeric suffix. All command-line entry points provide `--help`.

## Model configurations

The main configurations are `YOLO11s.yaml`, `baseline_P2_P4.yaml`, `LCSM.yaml`, `MSEE.yaml`, `HE-PAN.yaml`, `LCSM_MSEE.yaml`, `LCSM_HE-PAN.yaml`, `MSEE_HE-PAN.yaml`, and `LMSFE-YOLO.yaml`. Component-level variants are stored in `ultralytics/cfg/LMSFE-YOLO/ablations/`.

Model-construction utilities are provided in `tools/check_models.py`. These utilities check software execution; benchmark metrics require the relevant trained checkpoint and evaluation data.
