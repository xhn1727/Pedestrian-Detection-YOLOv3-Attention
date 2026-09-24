# Pedestrian Detection with Attention-Enhanced YOLOv3

A PyTorch implementation of YOLOv3 for pedestrian detection on the CityPersons dataset, with experiments on custom anchor generation and multi-scale attention mechanisms including ECA, SE, and CBAM.

This project investigates how attention mechanisms interact with the multi-scale feature pyramid of a one-stage detector. Rather than assuming that attention necessarily improves detection performance, the experiments compare several attention-enhanced architectures with a YOLOv3 baseline and analyze their effects on detection accuracy and stability.

## Overview

Pedestrian detection in urban environments is challenging due to large scale variations, occlusion, dense pedestrian distributions, and complex backgrounds.

This project builds a pedestrian detector based on **YOLOv3 with a Darknet-53 backbone** and focuses on two aspects:

1. Adapting YOLOv3 to the geometric characteristics of pedestrians in the CityPersons dataset through dataset-specific anchor generation.
2. Investigating different strategies for incorporating attention mechanisms into YOLOv3's multi-scale feature hierarchy.

The implemented model variants include:

- **YOLOv3** — baseline model
- **YOLOv3 + CBAM** — CBAM applied to multi-scale output features
- **YOLOv3 + Residual CBAM** — CBAM incorporated into residual blocks
- **YOLOv3 + ECA/SE/CBAM** — heterogeneous attention mechanisms applied to different feature scales

The experiments show that attention mechanisms are highly sensitive to their placement within the detection architecture. In this project, the attention-enhanced variants did not consistently outperform the baseline YOLOv3, highlighting the importance of structural compatibility between attention modules and multi-scale detection features.

## Model Architecture

The baseline detector follows the standard YOLOv3 architecture with a **Darknet-53 backbone** and three detection scales:

| Feature Map | Primary Target Scale |
| --- | --- |
| 52 × 52 | Small pedestrians |
| 26 × 26 | Medium pedestrians |
| 13 × 13 | Large pedestrians |

To investigate scale-specific feature enhancement, a heterogeneous attention architecture was implemented:

- **ECA (Efficient Channel Attention)** on the 52 × 52 feature map
- **SE (Squeeze-and-Excitation)** on the 26 × 26 feature map
- **CBAM (Convolutional Block Attention Module)** on the 13 × 13 feature map

This design was motivated by the different representation requirements of pedestrian targets across feature scales.

## Dataset

Experiments were conducted on the **CityPersons** pedestrian detection dataset.

The preprocessing pipeline focuses exclusively on the `pedestrian` category and includes:

- filtering images without valid pedestrian instances
- filtering heavily occluded pedestrian annotations based on visibility
- removing very small targets unsuitable for the selected input resolution
- clipping bounding boxes to valid image boundaries
- removing highly overlapping duplicate annotations using NMS
- resizing images to **416 × 416**
- applying pedestrian-oriented data augmentation

After preprocessing, the dataset contained **1,544 images**, divided into:

| Split | Images |
| --- | ---: |
| Training | 1,234 |
| Validation | 155 |
| Test | 155 |

The dataset itself is **not included in this repository**.

## Data Augmentation

The training pipeline uses lightweight transformations designed to preserve pedestrian geometry:

- brightness, contrast, and saturation perturbations
- small translations and scale changes
- horizontal flipping
- mild image blur
- normalization

Aggressive cropping and large rotations were intentionally avoided because pedestrian targets typically exhibit strong vertical structure.

## Custom Anchor Generation

Instead of directly using the default COCO anchors, this project generates anchors from the pedestrian bounding boxes in the CityPersons training set using **K-Means clustering (`k = 9`)**.

The resulting anchors at an input resolution of 416 × 416 are:

```text
Small:
(11.20, 54.62)
(14.20, 69.29)
(17.66, 86.16)

Medium:
(21.87, 106.65)
(26.33, 128.45)
(32.10, 156.53)

Large:
(38.87, 189.57)
(46.49, 226.86)
(62.91, 306.77)
```

The large height-to-width ratios reflect the characteristic geometry of pedestrian bounding boxes.

An ablation experiment demonstrated the importance of dataset-specific anchors:

| Anchor Configuration | mAP@0.5 | Mean IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| COCO default anchors | 0.0156 | 0.3851 | 0.0379 | 0.1010 |
| CityPersons clustered anchors | 0.7026 | 0.6207 | 0.6256 | 0.8062 |

## Training

The baseline training configuration used:

- **Input size:** 416 × 416
- **Optimizer:** Adam
- **Batch size:** 4
- **Maximum epochs:** 50
- **Backbone initialization:** Darknet-53 pretrained weights
- **Hardware acceleration:** Apple Metal Performance Shaders (MPS)

Training and validation losses were monitored throughout training. The baseline model reached its lowest validation loss at approximately epoch 23, while later epochs showed signs of overfitting.

Pretrained weights and trained checkpoints are not included in this repository.

## Evaluation

The evaluation pipeline computes pedestrian detection metrics including:

- mAP@0.5
- Precision
- Recall
- Mean IoU
- True Positives (TP)
- False Positives (FP)

The effects of confidence and NMS thresholds were also evaluated using multiple combinations.

For the baseline model, the configuration used for subsequent model comparisons was:

```text
Confidence threshold = 0.20
NMS threshold        = 0.45
```

Under this configuration, the baseline YOLOv3 achieved:

| Metric | Result |
| --- | ---: |
| mAP@0.5 | 0.7026 |
| Mean IoU | 0.6207 |
| Precision | 0.6256 |
| Recall | 0.8062 |
| True Positives | 391 |
| False Positives | 234 |

## Attention Experiments

Three attention-based modifications were evaluated against the baseline:

### Multi-scale CBAM

CBAM modules were applied to the output features of the three detection scales to investigate whether uniform attention enhancement could improve multi-scale feature representations.

### Residual CBAM

CBAM modules were incorporated into residual blocks of the Darknet-53 backbone to study the effect of attention during feature extraction.

### Heterogeneous ECA-SE-CBAM

Different attention mechanisms were assigned to different feature scales:

```text
52 × 52  →  ECA
26 × 26  →  SE
13 × 13  →  CBAM
```

The goal was to provide scale-specific attention rather than applying a single attention mechanism uniformly across the feature pyramid.

## Experimental Findings

The experiments showed that introducing attention mechanisms did **not** consistently improve the baseline detector.

In particular, the results suggest that attention mechanisms in a multi-scale one-stage detector are sensitive to:

- insertion location
- feature-map scale
- interaction with the feature pyramid
- changes in feature distributions
- bounding-box regression stability

Some attention configurations produced unstable behaviors such as confidence fluctuations, redundant predictions, and bounding-box localization shifts.

These results indicate that attention modules should not be treated as universally beneficial plug-and-play components. Their effectiveness depends strongly on how they interact with the surrounding detection architecture.

## Repository Structure

```text
.
├── models/
│   ├── darknet_yolov3.py
│   ├── darknet_yolov3_cbam.py
│   ├── darknet_yolov3_cbam_residual.py
│   └── darknet_yolov3_eca_se_cbam.py
│
├── tools/
│   ├── generate_anchors_citypersons.py
│   ├── pth_model_match.py
│   └── transfer_into_pytorch.py
│
├── dataset.py
├── loss_yolov3.py
├── train_yolov3.py
├── evaluate_yolov3.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Dataset Setup

The CityPersons dataset is not distributed with this repository.

After obtaining the dataset, place it under the project directory or modify `DATA_DIR` in the relevant scripts:

```python
DATA_DIR = "CityPersons"
```

The dataset loader expects the CityPersons image and annotation directories, including `leftImg8bit` and `gtBboxCityPersons`.

## Generate Custom Anchors

Dataset-specific anchors can be generated with:

```bash
python tools/generate_anchors_citypersons.py
```

## Training

Configure the dataset path in `train_yolov3.py` and run:

```bash
python train_yolov3.py
```

Model checkpoints and training logs are written to:

```text
checkpoints/
```

## Evaluation

Set the checkpoint path in `evaluate_yolov3.py`:

```python
CHECKPOINT_PATH = "checkpoints/model.pth"
```

Then run:

```bash
python evaluate_yolov3.py
```

Evaluation outputs are written to:

```text
results/
```

## Notes

- The CityPersons dataset is not included.
- Trained model checkpoints are not included.
- Darknet-53 pretrained weights are not included.
- Dataset paths and checkpoint paths should be configured locally before training or evaluation.

## Acknowledgments

This project uses YOLOv3 as the baseline object detection architecture and experiments with ECA, SE, and CBAM attention mechanisms for multi-scale pedestrian detection.