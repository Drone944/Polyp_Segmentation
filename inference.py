import torch
import cv2
import numpy as np
from model import Unet
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize, Resize, Compose
import matplotlib.pyplot as plt
import argparse

def preprocess_image(image):
    transform = Compose([
        Resize(256, 256),
        Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ])
    transformed = transform(image=image)['image']
    return transformed.unsqueeze(0).cuda()

def infer_and_overlay(model, image_path, threshold=0.5):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    input_tensor = preprocess_image(image_rgb)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > threshold).astype(np.uint8)
        pred_mask_resized = cv2.resize(pred_mask, (w, h))

    overlay = image.copy()
    overlay[pred_mask_resized == 1] = [0, 255, 0]

    return image, pred_mask_resized, overlay

def visualize(image, mask, overlay):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default="weights/unet_weights.pth", help="Path to model weights")
    args = parser.parse_args()

    model = Unet().cuda()
    model.load_state_dict(torch.load(args.weights))
    
    image, mask, overlay = infer_and_overlay(model, args.image)
    visualize(image, mask, overlay)