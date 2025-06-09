import torch

def compute_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)

    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (preds_bin.sum() + targets.sum() - intersection + smooth)

    return dice.item(), iou.item()