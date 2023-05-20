from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split


def get_splits(
    splits: Tuple[float, float, float],
    dataset: Dataset,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    # scale splits by total (ensure sum(splits) == 1)

    splits = np.array(splits)
    splits = np.around(splits / splits.sum(), 1)

    train_test = np.around([splits[0], splits[1:].sum()], 1)
    val_test = np.around(splits[1:] / splits[1:].sum(), 1)

    indices = range(len(dataset))

    train, tmp = train_test_split(
        indices,
        stratify=dataset.targets,
        train_size=train_test[0],
        random_state=seed,
    )

    val, test = train_test_split(
        tmp,
        stratify=dataset.targets[tmp],
        train_size=val_test[0],
        random_state=seed,
    )

    return train, val, test
