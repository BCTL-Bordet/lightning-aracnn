from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from lightning_aracnn.model.components.aracnn import ARACNN
from lightning_aracnn.model.lit_aracnn import LitARACNN

if __name__ == "__main__":
    # net = ARACNN().cuda()
    # print(net)
    # print(summary(net, (3, 224, 224)))
    model = LitARACNN()
    var_drop = model.variational_dropout(torch.zeros((1, 3, 224, 224)))
    print(var_drop)
    print(var_drop.shape)
    # Shannon entropy
    # h = entropy(var_drop, axis=0)
    