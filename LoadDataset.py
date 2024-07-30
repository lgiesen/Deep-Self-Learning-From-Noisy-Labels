import io
import logging
import os
import tarfile

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from config import shared_folder_path

# Ensure that PIL does not fail on image load errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    def __init__(self, file_path, tar_dir=shared_folder_path, transform=None):
        self.image_labels = pd.read_csv(file_path, sep=' ', header=None, names=['image_path', 'label'])
        self.tar_dir = tar_dir
        self.transform = transform
        self.tar_files = {}
        self._load_tar_files()

    def _load_tar_files(self):
        for i in range(10):
            tar_path = os.path.join(self.tar_dir, f"{i}.tar")
            if os.path.exists(tar_path):
                self.tar_files[i] = tar_path
            else:
                logger.error(f"Tar file {tar_path} does not exist")

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_rel_path = self.image_labels.iloc[idx, 0]
        tar_index = int(img_rel_path.split('/')[0])  # Assuming the first part of the path indicates the tar file index
        try:
            tar_path = self.tar_files.get(tar_index)
            if tar_path is None:
                raise FileNotFoundError(f"Tar file for index {tar_index} not found")

            with tarfile.open(tar_path, 'r') as tar:
                img_file = tar.extractfile(img_rel_path)
                if img_file is None:
                    raise FileNotFoundError(f"{img_rel_path} not found in tar archive {tar_index}")
                image = Image.open(io.BytesIO(img_file.read())).convert('RGB')
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to load image {img_rel_path} at index {idx}: {e}")
            return None, None  # Return None if image loading fails

        label = int(self.image_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        logger.info(f"Successfully loaded image {img_rel_path} at index {idx}")
        return image, label