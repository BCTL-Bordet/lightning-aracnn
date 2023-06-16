import os
import torch
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple
from torch import nn
import albumentations as a
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize
from lightning_aracnn import utils
from tqdm import tqdm
from joblib import Parallel, delayed

log = utils.get_pylogger(__name__)


class CSVDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_folder: str,
        annot_folder: str,
        dataset_folder: str,
        dataset_version: str,
        stage: str,
        mean: Tuple[float, float, float],
        stdv: Tuple[float, float, float],
        augmentations=[],
        stain_augmentation=None,
    ):
        super().__init__()
        self.image_root = os.path.join(root, image_folder)
        self.annot_root = os.path.join(root, annot_folder)
        self.dataset_path = os.path.join(
            root, dataset_folder, dataset_version, f"{stage}.csv"
        )

        self.dataset = pd.read_csv(self.dataset_path)
        self.samples = self.dataset["sample"]
        self.targets = self.dataset["class"]

        self.augment = a.Compose(augmentations)
        self.augment_stain = stain_augmentation
        self.normalize = a.Compose([Normalize(mean, stdv), ToTensorV2()])

    def __getitem__(self, idx: int) -> any:
        ## GET CLASS
        cls = self.targets[idx]

        ## GET IMAGE
        # read sample path from .csv dataset
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root, sample)
        # mask_path = os.path.join(self.annot_root, sample)

        # read image from filesyste
        image = np.array(Image.open(image_path))
        # mask = np.array(Image.open(mask_path))

        # apply augmentations
        res = self.augment(image=image)
        image = res["image"]
        # res = self.augment(image=image, mask=mask)
        # image, mask = res["image"], res["mask"]

        # TODO: apply stain augmentations
        # if self.augment_stain:
        # image = self.augment_stain(image, mask)

        # apply normalization
        image = self.normalize(image=image)["image"]

        return image, cls

    def __len__(
        self,
    ) -> int:
        return self.dataset.shape[0]


class WeightedCSVDataset(CSVDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = self.dataset["weight"]

    def get_weights(self):
        return self.weights


class MultiLabelCSVDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_folder: str,
        annot_folder: str,
        dataset_folder: str,
        dataset_version: str,
        stage: str,
        mean: Tuple[float, float, float],
        stdv: Tuple[float, float, float],
        augmentations=[],
        stain_augmentation=None,
    ):
        super().__init__()
        self.image_root = os.path.join(root, image_folder)
        self.annot_root = os.path.join(root, annot_folder)
        self.dataset_path = os.path.join(
            root, dataset_folder, dataset_version, f"{stage}.csv"
        )

        self.dataset = pd.read_csv(self.dataset_path)
        self.samples = self.dataset["sample"]
        self.targets = self.dataset.drop(columns=["sample"])

        self.augment = a.Compose(augmentations)
        self.augment_stain = stain_augmentation
        self.normalize = a.Compose([Normalize(mean, stdv), ToTensorV2()])

    def __getitem__(self, idx: int) -> any:
        ## GET CLASS
        cls = torch.tensor(self.targets.iloc[idx]).float()

        ## GET IMAGE
        # read sample path from .csv dataset
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root, sample)
        # mask_path = os.path.join(self.annot_root, sample)

        # read image from filesyste
        image = np.array(Image.open(image_path))

        # apply augmentations
        res = self.augment(image=image)
        image = res["image"]
        # res = self.augment(image=image, mask=mask)
        # image, mask = res["image"], res["mask"]

        # TODO: apply stain augmentations
        # if self.augment_stain:
        # image = self.augment_stain(image, mask)

        # apply normalization
        image = self.normalize(image=image)["image"]

        return image, cls

    def __len__(
        self,
    ) -> int:
        return self.dataset.shape[0]
