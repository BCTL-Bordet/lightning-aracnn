import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple
from torch import nn
import albumentations as a
from albumentations.pytorch.functional import img_to_tensor


class CSVDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_folder: str,
        annot_folder: str,
        dataset_folder: str,
        dataset_name: str,
        mean: Tuple[float, float, float],
        stdv: Tuple[float, float, float],
        transforms = [],
        stain_augm = None,
    ):
        super().__init__()
        self.image_root = os.path.join(root, image_folder)
        self.annot_root = os.path.join(root, annot_folder)
        self.dataset_path = os.path.join(root, dataset_folder, dataset_name)

        self.dataset = pd.read_csv(self.dataset_path)
        self.samples = self.dataset["sample"]
        self.targets = self.dataset["class"]

        self.transforms = a.Compose(transforms)
        self.stain_augm = stain_augm   

    def __getitem__(self, idx: int) -> any:
        ## GET IMAGE
        # read sample path from .csv dataset
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root, sample)
        mask_path = os.path.join(self.annot_root, sample)
        
        # read image from filesyste
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        # apply transformation
        res = self.transforms(image=image, mask=mask)
        image = res['image']
        mask = res['mask']

        if self.stain_augm:
            image = self.stain_augm(image=image, mask=mask)
            
        image = img_to_tensor(image)
        
        # GET CLASS
        cls = self.targets[idx]

        return image, cls

    def __len__(
        self,
    ) -> int:
        return self.dataset.shape[0]
