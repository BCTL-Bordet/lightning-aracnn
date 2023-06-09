from typing import Any, List
import lightning as L
import torch.nn as nn
import torch
import pyrootutils
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from torchmetrics import MeanMetric, ClasswiseWrapper, MetricCollection, ConfusionMatrix

from scipy.stats import entropy
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from lightning_aracnn.utils.io_utils import confusion_matrix_to_image


class LitARACNN(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        aux_loss_weight: float,
        main_loss_weight: float,
        num_inferences: int = 10,
        num_classes: int = 11,
        class_names: List[str] = [],
        original_aracnn: bool = True,
    ):
        super().__init__()

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

        # self.aux_loss_weight = torch.tensor(aux_loss_weight)
        self.register_buffer("aux_loss_weight", torch.tensor(aux_loss_weight))
        # self.main_loss_weight = torch.tensor(main_loss_weight)
        self.register_buffer("main_loss_weight", torch.tensor(main_loss_weight))

        self.n_inference = num_inferences
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.original_aracnn = original_aracnn

        self.criterion = torch.nn.NLLLoss()

        # Metric dict
        self.metrics = nn.ModuleDict(
            {
                f"{stage}_metrics": nn.ModuleDict(
                    {
                        "clf": MetricCollection(
                            {
                                "mc_acc": ClasswiseWrapper(
                                    MulticlassAccuracy(num_classes=num_classes, average=None),
                                    labels=self.class_names,
                                ),
                                "mc_f1": ClasswiseWrapper(
                                    MulticlassF1Score(num_classes=num_classes, average=None),
                                    labels=self.class_names,
                                ),
                                "acc": MulticlassAccuracy(num_classes=num_classes),
                                "f1": MulticlassF1Score(num_classes=num_classes),
                            }
                        ),
                        "loss": MetricCollection(
                            {
                                "loss": MeanMetric(),
                            },
                        ),
                        "confusion": MetricCollection(
                            {
                                "confusion": MulticlassConfusionMatrix(
                                    num_classes=num_classes,
                                )
                            },
                        ),
                    }
                )
                for stage in ["train", "val", "test"]
            }
        )

    def forward(self, x: any):
        return self.net(x)

    def reset_metrics(self, stage: str):
        for metric in self.metrics[f"{stage}_metrics"].values():
            metric.reset()

    def on_train_start(self):
        self.reset_metrics("train")
        self.reset_metrics("val")
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

    def model_step_with_aux(self, batch: Any, stage: str):
        x, y = batch
        probs, probs_aux = self(x)

        # loss
        aux_loss = self.criterion(torch.log(probs_aux), y)
        main_loss = self.criterion(torch.log(probs), y)
        loss = self.aux_loss_weight * aux_loss + self.main_loss_weight * main_loss

        preds = torch.argmax(probs, dim=1)

        out = {"loss": loss, "probs": probs, "preds": preds, "target": y}
        self.update_metrics(out, stage)

        return out

    def model_step_classic(self, batch: Any, stage: str):
        x, y = batch
        probs = self(x)

        # loss
        loss = self.criterion(torch.log(probs), y)

        preds = torch.argmax(probs, dim=1)

        out = {"loss": loss, "probs": probs, "preds": preds, "target": y}
        self.update_metrics(out, stage)

        return out

    def model_step(self, batch: Any, stage: str):
        if self.original_aracnn:
            return self.model_step_with_aux(batch, stage)
        else:
            return self.model_step_classic(batch, stage)

    def on_model_epoch_end(self, stage: str) -> None:
        # log clf and loss metrics
        for kind in ["clf", "loss"]:
            # compute metrics
            res = self.metrics[f"{stage}_metrics"][kind].compute()

            # get unique metric groups
            names = res.keys()
            names = [name.split("_")[0] for name in names]
            # uniq_names, idxs = np.unique(np.array(names), return_inverse=True)

            for name, value in res.items():
                self.log(f"{stage}/{name}", value)

            # write groups to same tensorboard plot
            # for name in names:
            # metric_group = {k: v for k, v in res.items() if k.startswith(name)}
            # self.logger.experiment.add_scalars(
            # f"{stage}/{name}",
            # metric_group,
            # self.current_epoch,
            # )

        # log confusion matrix
        # get data
        conf_matrix = self.metrics[f"{stage}_metrics"]["confusion"].compute()["confusion"].cpu()
        # generate image
        conf_matrix_image = confusion_matrix_to_image(conf_matrix, self.class_names)
        # write image
        self.logger.experiment.add_image(
            f"{stage}/conf_matrix",
            img_tensor=conf_matrix_image,
            global_step=self.global_step,
            dataformats="HWC",
        )

        # log loss to progressbar
        # self.log(
        #     f"{stage}/{name}",
        #     self.metrics[f"{stage}_metrics"]["loss"].compute()["loss"],
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        self.reset_metrics(stage)

    def update_metrics(self, outputs: List[Any], stage: str):
        # update metrics

        self.metrics[f"{stage}_metrics"]["clf"].update(
            outputs["preds"],
            outputs["target"],
        )

        self.metrics[f"{stage}_metrics"]["confusion"].update(
            outputs["preds"],
            outputs["target"],
        )

        self.metrics[f"{stage}_metrics"]["loss"].update(
            outputs["loss"],
        )

    def training_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, "test")

    def training_step_end(self, outputs: List[Any]):
        print("ole end")
        self.model_step_end(outputs, stage="train")

    def validation_step_end(self, outputs: List[Any]):
        self.model_step_end(outputs, stage="val")

    def test_step_end(self, outputs: List[Any]):
        self.model_step_end(outputs, stage="test")

    def on_train_epoch_end(self) -> None:
        self.on_model_epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self.on_model_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self.on_model_epoch_end("test")

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def variational_dropout(self, batch: Any):
        # enable dropout for variational dropout
        self.net.aux_head.layers["dropout"].train(True)

        trials = []

        # repeat inference n_inference times
        for _ in range(self.n_inference):
            with torch.no_grad():
                probs, _ = self(batch)

                trials.append(probs.unsqueeze(0))

        trials = torch.vstack(trials).detach().cpu()

        # compute entropies across repetitions
        h = entropy(trials, axis=0)
        return trials, h
