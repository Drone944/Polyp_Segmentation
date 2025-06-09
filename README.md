# Polyp Segmentation with U-Net

This repository implements a U-Net-based semantic segmentation pipeline to detect polyps in colonoscopy images using the Kvasir-SEG dataset.

## Features
- U-Net model architecture
- BCE + Dice loss
- W&B logging and hyperparameter sweeps
- Augmentations using Albumentations
- Validation with IoU and Dice metrics

## Setup
```bash
pip install -r requirements.txt
```

## Run Training
```bash
python train.py
```

## Run Sweeps
```bash
python sweep.py
```

## Repository Structure
```
Polyp_Segmentation/
│
├── README.md
├── requirements.txt
│
├── train.py
├── sweep.py
├── inference.py
│
├── models/
│   └── unet.py
│
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── augmentations.py
│
└── data/
    ├── images/
    │   ├── img1.png
    │   └── ...
    └── masks/
        ├── img1.png
        └── ...
```

## Acknowledgements
- [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [Weights & Biases](https://wandb.ai)
