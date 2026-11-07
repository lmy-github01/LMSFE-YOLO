# Experimental result files

The CSV files in this directory contain per-epoch training and validation logs for the configurations listed below.

| Configuration | File | Data rows |
|---|---|---:|
| P2-P4-Baseline | [01__P2-P4-Baseline__200epochs.csv](01__P2-P4-Baseline__200epochs.csv) | 200 |
| LCSM | [02__LCSM__200epochs.csv](02__LCSM__200epochs.csv) | 200 |
| MSEE | [03__MSEE__200epochs.csv](03__MSEE__200epochs.csv) | 200 |
| HE-PAN | [04__HE-PAN__200epochs.csv](04__HE-PAN__200epochs.csv) | 200 |
| LCSM_MSEE | [05__LCSM_MSEE__200epochs.csv](05__LCSM_MSEE__200epochs.csv) | 200 |
| LCSM_HE-PAN | [06__LCSM_HE-PAN__200epochs.csv](06__LCSM_HE-PAN__200epochs.csv) | 200 |
| MSEE_HE-PAN | [07__MSEE_HE-PAN__200epochs.csv](07__MSEE_HE-PAN__200epochs.csv) | 200 |
| LMSFE-YOLO | [08__LMSFE-YOLO__200epochs.csv](08__LMSFE-YOLO__200epochs.csv) | 200 |

The column names in each CSV identify the logged epoch, losses, validation metrics and learning rates. Epoch-log metrics should be interpreted using their corresponding training configuration and checkpoint-selection procedure.

For independent evaluation of a trained checkpoint, use `val.py` with the matching dataset configuration and evaluation split.

