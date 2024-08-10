import logging
import os

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from config import dataset_root

# Ensure that PIL does not fail on image load errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    def __init__(self, file_path, img_dir=dataset_root, transform=None, sampling=False):
        self.img_dir = img_dir
        self.transform = transform
        self.sampling = sampling
        self.image_labels = pd.read_csv(file_path, sep=' ', header=None, names=['image_path', 'label'])

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_rel_path = self.image_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_rel_path.lstrip('/'))  # Ensure no leading slash in relative path
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            logger.error(f"Failed to load image {img_path} at index {idx}: {e}")
            return None, None, idx  # Return None if image loading fails
        if not self.sampling:
            label = int(self.image_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if not self.sampling:
            return image, label
        return image