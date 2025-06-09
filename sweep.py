import wandb
from train import train

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/val_iou',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [10, 20]
        },
        'batch_size': {
            'values': [8, 16]
        },
        'use_amp': {
            'values': [False, True]
        },
        'encoder': {
            'values': ['efficientnet-b0', 'efficientnet-b4']
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        }
    }
}

def sweep_train():
    config_defaults = {
        "use_pretrained": True,
        "encoder_weights": "imagenet",
        "architecture": "Unet",
        "dataset": "Kvasir-SEG",
        "image_dir": "./data/images",
        "mask_dir": "./data/masks",
    }

    config = {**config_defaults, **wandb.config}

    train(config)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="Polyp_Segmentation", entity="manasvarmak-amrita-vishwa-vidyapeetham")
    wandb.agent(sweep_id, function=sweep_train)
