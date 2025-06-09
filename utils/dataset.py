from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class SegmentationDataset(Dataset):
  def __init__(self, image_dir, mask_dir, image_list, transform=None):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.image_list = image_list

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    img_path = os.path.join(self.image_dir, self.image_list[idx])
    mask_path = os.path.join(self.mask_dir, self.image_list[idx])
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if self.transform:
      augmented = self.transform(image=np.array(image), mask=np.array(mask))
      image = augmented['image']
      mask = augmented['mask'].float() / 255.0

    return image, mask