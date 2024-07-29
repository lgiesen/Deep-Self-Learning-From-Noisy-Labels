import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config import dataset_root


class CustomImageDataset(Dataset):
    def __init__(self, file_path, folder_path=dataset_root, transform=None):
        self.image_labels = pd.read_csv(file_path, 
                                        sep=' ', 
                                        header=None, 
                                        names=['image_path', 'label'])
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_rel_path = self.image_labels.iloc[idx, 0]
        img_path = os.path.join(self.folder_path, img_rel_path)
        image = Image.open(img_path).convert('RGB')
        label = int(self.image_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label
