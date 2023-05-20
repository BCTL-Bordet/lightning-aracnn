from typing import Any, List
import lightning.pytorch as pl
import torch.nn as nn
import torch
import pyrootutils

from scipy.stats import entropy
from pyaracnn.utils.io_utils import confusion_matrix_to_image
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class LitARACNN(pl.LightningModule):
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
        self.class_names = class_names
        self.original_aracnn = original_aracnn

        self.criterion = torch.nn.NLLLoss()

        self.metrics = nn.ModuleDict(
            {
                f"{stage}_dict": nn.ModuleDict(
                    {
                        "acc": clf.Accuracy(
                            task="multiclass",
                            num_classes=num_classes,
                        ),
                        "f1": clf.F1Score(
                            task="multiclass",
                            num_classes=num_classes,
                        ),
                        "auroc": clf.AUROC(
                            task="multiclass",
                            num_classes=num_classes,
                        ),
                        "confusion": clf.ConfusionMatrix(
                            task="multiclass",
                            num_classes=num_classes,
                        ),
                        "loss": MeanMetric(),
                    }
                )
                for stage in ["train", "val", "test"]
            }
        )


    def forward(self, x: any):
        return self.net(x)

    def reset_metrics(self, stage: str):
        for metric in self.metrics[f"{stage}_dict"].values():
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
        self.log_metrics(out, stage)

        return out

    def model_step_classic(self, batch: Any, stage: str):
        x, y = batch
        probs = self(x)

        # loss
        loss = self.criterion(torch.log(probs), y)

        preds = torch.argmax(probs, dim=1)

        out = {"loss": loss, "probs": probs, "preds": preds, "target": y}
        self.log_metrics(out, stage)

        return out

    def model_step(self, batch: Any, stage: str):
        if self.original_aracnn:
            return self.model_step_with_aux(batch, stage)
        else:
            return self.model_step_classic(batch, stage)

    def on_model_epoch_end(self, stage: str) -> None:
        # log metrics
        conf_matrix = self.metrics[f"{stage}_dict"]["confusion"].compute().cpu()
        conf_matrix_image = confusion_matrix_to_image(conf_matrix, self.class_names)

        self.logger.experiment.add_image(
            f"confusion_matrix_{stage}",
            img_tensor=conf_matrix_image,
            global_step=self.global_step,
            dataformats="HWC",
        )

        for name, metric in self.metrics[f"{stage}_dict"].items():
            if name != "confusion":
                self.log(
                    f"{stage}/{name}",
                    metric.compute(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        self.reset_metrics(stage)

    def log_metrics(self, outputs: List[Any], stage: str):
        # update metrics
        self.metrics[stage + "_dict"]["acc"].update(
            outputs["preds"],
            outputs["target"],
        )
        self.metrics[stage + "_dict"]["auroc"].update(
            outputs["probs"],
            outputs["target"],
        )
        self.metrics[stage + "_dict"]["confusion"].update(
            outputs["preds"],
            outputs["target"],
        )
        self.metrics[stage + "_dict"]["loss"].update(
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
                out, _ = self(batch)
                trials.append(out.unsqueeze(0))

        trials = torch.vstack(trials)
        # compute entropies across repetitions
        h = entropy(trials, axis=0)
        return trials, h
