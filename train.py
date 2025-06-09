import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from dotenv import load_dotenv
import wandb
from models.unet import Unet
from utils.losses import DiceLoss
from utils.metrics import compute_metrics
from utils.augmentations import get_train_transform, get_val_transform
from utils.dataset import SegmentationDataset


def train(config):
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(
        entity="manasvarmak-amrita-vishwa-vidyapeetham",
        project="Polyp_Segmentation",
        name="unet-test-2",
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.get("use_pretrained", True):
        print("[INFO] Using pretrained UNet from segmentation_models_pytorch.")
        model = smp.Unet(
            encoder_name=config.get("encoder", "resnet34"),
            encoder_weights=config.get("encoder_weights", "imagenet"),
            in_channels=3,
            classes=1,
        )
    else:
        print("[INFO] Using custom UNet.")
        model = Unet()

    model = model.to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    image_dir = config.get("image_dir", "./data/images")
    mask_dir = config.get("mask_dir", "./data/masks")

    image_files = os.listdir(image_dir)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = SegmentationDataset(image_dir, mask_dir, train_files, transform=train_transform)
    val_dataset = SegmentationDataset(image_dir, mask_dir, val_files, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 16), shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 16), shuffle=False, pin_memory=True)

    epochs = config.get("epochs", 20)
    use_amp = config.get("use_amp", False)
    scaler = torch.amp.GradScaler() if use_amp else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).unsqueeze(1).float()
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    b_loss = bce_loss(outputs, masks)
                    d_loss = dice_loss(outputs, masks)
                    loss = b_loss + d_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                b_loss = bce_loss(outputs, masks)
                d_loss = dice_loss(outputs, masks)
                loss = b_loss + d_loss
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        wandb.log({
            "train/loss": avg_loss,
            "train/BCE": b_loss.item(),
            "train/Dice": d_loss.item(),
            "train/LR": scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        model.eval()
        val_dice, val_iou = 0, 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).unsqueeze(1).float()

                outputs = model(images)
                dice, iou = compute_metrics(outputs, masks)

                val_dice += dice
                val_iou += iou

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_iou)

        wandb.log({"val/dice_coeff": val_dice, "val/iou": val_iou})

        print(f"Validation - Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

    torch.save(model.state_dict(), './saved_models/model_1.pth')

    # Log example predictions to wandb
    pred_mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()
    gt_mask = masks[0].squeeze().cpu().numpy()
    input_img = images[0].permute(1, 2, 0).cpu().numpy()

    wandb.log({
        "example": [
            wandb.Image(input_img, caption="Input"),
            wandb.Image(gt_mask, caption="Ground Truth"),
            wandb.Image(pred_mask, caption="Prediction"),
        ]
    })

    run.finish()


if __name__ == "__main__":
    config = {
        "use_pretrained": True,
        "encoder": "efficientnet-b4",
        "encoder_weights": "imagenet",
        "use_amp": False,
        "epochs": 20,
        "batch_size": 4,
        "architecture": "Unet",
        "dataset": "Kvasir-SEG",
        "image_dir": "./data/images",
        "mask_dir": "./data/masks"
    }
    train(config)
