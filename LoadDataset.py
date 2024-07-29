import logging
import os

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import dataset_root

# Ensure that PIL does not fail on image load errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    def __init__(self, file_path, folder_path=dataset_root, transform=None):
        self.image_labels = pd.read_csv(file_path, sep=' ', header=None, names=['image_path', 'label'])
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_rel_path = self.image_labels.iloc[idx, 0]
        img_path = os.path.join(self.folder_path, img_rel_path)
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            logger.error(f"Failed to load image {img_path} at index {idx}: {e}")
            return None  # Return None if image loading fails
        label = int(self.image_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        logger.info(f"Successfully loaded image {img_path} at index {idx}")
        return image, label