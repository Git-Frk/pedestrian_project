import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class CropsDataset(Dataset):
    def __init__(self, label_file, image_directory, transform=None):
        self.annotations = pd.read_csv(label_file)
        self.root_directory = image_directory
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_directory, self.annotations.iloc[index, 0])
        image = io.imread(image_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)

        return image, y_label
