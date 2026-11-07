# Dataset sources and experimental use

This study uses VisDrone2019-DET, the original AI-TOD release, TinyPerson, and RS-STOD. All four datasets were downloaded from their official providers. No additional modifications were made to the downloaded datasets; annotation-format conversion for use with YOLO is described separately below.

## Sources and experimental subsets

| Dataset | Official source | Detection categories | Training images | Evaluation images and subset |
|---|---|---:|---:|---|
| VisDrone2019-DET | [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset) | 10 | 6,471 | 548, official validation subset |
| AI-TOD (original release) | [AI-TOD dataset](https://github.com/jwwangchn/AI-TOD) | 8 | 11,214 | 2,804, validation subset |
| TinyPerson | [TinyPerson dataset](https://github.com/ucas-vg/PointTinyBenchmark/tree/master/dataset) | 2 | 794 | 816, evaluation subset loaded through the YOLO `val` entry |
| RS-STOD | [RS-STOD provider repository](https://github.com/lixinghua5540/FBVF-YOLO) | 5 | 1,852 | 463, validation subset |

## Data preparation and category definitions

### VisDrone2019-DET

The official partition was used. Evaluation was conducted on the official 548-image validation subset. No additional image selection, cropping, or tiling was applied. The original annotations were converted to normalized YOLO bounding-box format. Apart from this format conversion, no manual changes were made to category definitions or bounding-box annotations.

The ten target categories map to YOLO IDs 0–9 in this order: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor.

### AI-TOD

The experiments used the original AI-TOD release, comprising eight object categories, with 11,214 training images and 2,804 validation images. The official provider gives the dataset access and construction instructions, including the required xView-derived material. Follow those instructions and the applicable dataset terms when obtaining the data.

### TinyPerson

The experiments used a two-class TinyPerson configuration with 794 training images and 816 evaluation images. The TinyPerson YOLO configuration specifies `nc: 2` and the following category order:

| YOLO class ID | Category |
|---:|---|
| 0 | `earth_person` |
| 1 | `sea_person` |

The training and evaluation data are stored under `train/images` and `val/images`, with corresponding YOLO-format label files. The evaluation subset is designated by the `val` entry in the TinyPerson configuration. The experiment uses the two categories shown above separately.

### RS-STOD

The experiments used five object categories, with 1,852 training images and 463 validation images. Dataset access information is provided by the official FBVF-YOLO/RS-STOD repository linked above.

## Data access

Images and original annotations can be obtained through the provider links in the table, following the respective access procedures and dataset terms. For TinyPerson, the provider also supplies [annotation and task documentation](https://github.com/ucas-vg/PointTinyBenchmark/blob/master/dataset/TinyPerson_annotation.md).

Third-party dataset terms remain those of their providers. Dataset licenses and the software license of this study's code are recorded separately. 
