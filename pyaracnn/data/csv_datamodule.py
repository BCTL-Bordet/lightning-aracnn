import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from typing import Any, Dict, Optional, Tuple

from pyaracnn.utils.data_utils import get_splits


class CSVDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        train_val_test_split: Tuple[float, float, float],
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = False,
        persistent_workers=False,
    ):
        super().__init__()

        self.dataset = dataset
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.datasets = {
            "train": None,
            "val": None,
            "test": None,
        }

    def setup(self, stage: Optional[str] = None):
        if not all(self.datasets.values()):
            train, val, test = get_splits(self.train_val_test_split, self.dataset)

            self.datasets["train"] = Subset(self.dataset, train)
            self.datasets["val"] = Subset(self.dataset, val)
            self.datasets["test"] = Subset(self.dataset, test)

    def stage_dataloader(self, stage: str):
        # sampler = DistributedSampler(self.datasets[stage])
        sampler = None
        return DataLoader(
            dataset=self.datasets[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            sampler=sampler,
        )

    def train_dataloader(self):
        return self.stage_dataloader("train")

    def val_dataloader(self):
        return self.stage_dataloader("val")

    def test_dataloader(self):
        return self.stage_dataloader("test")


class CSVDataModuleNoSplit(LightningDataModule):
    def __init__(
        self,
        dataset_train: Dataset,
        dataset_val: Dataset,
        dataset_test: Dataset,
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = False,
        persistent_workers=False,
    ):
        super().__init__()

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.datasets = {
            "train": None,
            "val": None,
            "test": None,
        }

    def setup(self, stage: Optional[str] = None):
        if not all(self.datasets.values()):

            self.datasets["train"] = self.dataset_train
            self.datasets["val"] = self.dataset_val
            self.datasets["test"] = self.dataset_test

    def stage_dataloader(self, stage: str):
        # sampler = DistributedSampler(self.datasets[stage])
        sampler = None
        return DataLoader(
            dataset=self.datasets[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            sampler=sampler,
        )

    def train_dataloader(self):
        return self.stage_dataloader("train")

    def val_dataloader(self):
        return self.stage_dataloader("val")

    def test_dataloader(self):
        return self.stage_dataloader("test")
