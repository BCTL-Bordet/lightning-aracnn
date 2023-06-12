import torch
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from typing import Any, Dict, Optional, Tuple

from lightning_aracnn.utils.data_utils import get_splits
from lightning_aracnn.data.components.csv_dataset import CSVDataset, WeightedCSVDataset

class CSVDataModule(L.LightningDataModule):
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


class CSVDataModuleNoSplit(L.LightningDataModule):
    def __init__(
        self,
        augmentations: dict,
        dataset_kwargs: dict,
        dataloader_kwargs: dict,
        stain_augmentation: object = None,
    ):
        super().__init__()
        
        self.augmentations = augmentations
        self.stain_augmentation = stain_augmentation
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.datasets = {
            "train": None,
            "val": None,
            "test": None,
        }
        
        
        self.dataset_class = CSVDataset

    def setup(self, stage: Optional[str] = None):
        if not all(self.datasets.values()):
            self.datasets["train"] = self.dataset_class(
                stage="train",
                augmentations=self.augmentations,
                stain_augmentation=self.stain_augmentation,
                **self.dataset_kwargs,
            )
            self.datasets["val"] = self.dataset_class(
                stage="val",
                **self.dataset_kwargs,
            )
            self.datasets["test"] = self.dataset_class(
                stage="test",
                **self.dataset_kwargs,
            )
            

    def stage_dataloader(self, stage: str):
        # sampler = DistributedSampler(self.datasets[stage])
        return DataLoader(
            dataset=self.datasets[stage],
            shuffle=False,
            # sampler=sampler,
            **self.dataloader_kwargs
        )

    def train_dataloader(self):
        return self.stage_dataloader("train")

    def val_dataloader(self):
        return self.stage_dataloader("val")

    def test_dataloader(self):
        return self.stage_dataloader("test")


class WeightedCSVDataModuleNoSplit(CSVDataModuleNoSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset_class = WeightedCSVDataset
    
    def stage_dataloader(self, stage: str):
        if stage == 'train':
            sampler = WeightedRandomSampler(
                self.datasets[stage].get_weights(), 
                len(self.datasets[stage].get_weights()),
            )
        else:
            sampler = None
            
        return DataLoader(
            dataset=self.datasets[stage],
            shuffle=False,
            sampler=sampler,
            **self.dataloader_kwargs,
        )