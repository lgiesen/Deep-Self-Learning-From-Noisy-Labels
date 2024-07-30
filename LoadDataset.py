import io
import logging
import tarfile

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from config import shared_folder_path as TAR_PATH

# Ensure that PIL does not fail on image load errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    def __init__(self, file_path, tar_path=TAR_PATH, transform=None):
        self.image_labels = pd.read_csv(file_path, sep=' ', header=None, names=['image_path', 'label'])
        self.tar_path = tar_path
        self.transform = transform
        self.tar = tarfile.open(tar_path, 'r')

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_rel_path = self.image_labels.iloc[idx, 0]
        try:
            img_file = self.tar.extractfile(img_rel_path)
            if img_file is None:
                raise FileNotFoundError(f"{img_rel_path} not found in tar archive")
            image = Image.open(io.BytesIO(img_file.read())).convert('RGB')
        except (IOError, OSError) as e:
            logger.error(f"Failed to load image {img_rel_path} at index {idx}: {e}")
            return None  # Return None if image loading fails
        label = int(self.image_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        logger.info(f"Successfully loaded image {img_rel_path} at index {idx}")
        return image, label
