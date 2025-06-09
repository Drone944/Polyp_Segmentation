# Polyp Segmentation with U-Net

This repository implements a U-Net-based segmentation pipeline to detect polyps in colonoscopy images using the Kvasir-SEG dataset.

## Features
- U-Net model architecture
- BCE + Dice loss
- W&B logging and hyperparameter sweeps
- Augmentations using Albumentations
- Validation with IoU and Dice metrics

## Setup
1.  **Clone the repository**
    ```
    git clone https://github.com/your-username/Polyp_Segmentation.git
    cd Polyp_Segmentation
    ```
    
2.  **Install dependencies**
    ```
    pip install -r requirements.txt
    ```
    
3.  **Setup Weights & Biases**
    -   Create a free W&B account: [https://wandb.ai](https://wandb.ai)
        
    -   Go to your W&B settings and copy your **API Key**
        
    -   Create a `.env` file in the project root:
        
        ```
        WANDB_API_KEY=your_api_key_here
        ```


## Run Training
```
python3 train.py
```
You can configure training by editing the `config` dictionary in `train.py`:

```
config = {
    "use_pretrained": True,
    "encoder": "efficientnet-b4",
    "encoder_weights": "imagenet",
    "use_amp": True,
    "epochs": 20,
    "batch_size": 8,
    ...
}
```
-   If `use_pretrained` is `True`, the model will use a pretrained EfficientNet encoder via `segmentation_models.pytorch`.
    
-   If `use_pretrained` is `False`, it will fallback to the custom U-Net defined in `models/unet.py`.

## Run Sweeps
```
python3 sweep.py
```
This script will:
-   Initialize a W&B sweep  
-   Launch training jobs with different hyperparameter combination.

You can modify sweep configurations in `sweep.py`:

```
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/dice_coeff", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [8, 16]},
        "encoder": {"values": ["resnet34", "efficientnet-b0", "efficientnet-b4"]},
        ...
    }
}
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
├── data/
    ├── images/
    └── masks/
```

## Acknowledgements
- [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [Weights & Biases](https://wandb.ai)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
