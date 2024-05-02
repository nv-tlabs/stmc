import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


class ProgressLogger(Callback):
    def __init__(self, precision: int = 2):
        self.precision = precision

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training done")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        metrics_str = []

        losses_dict = trainer.callback_metrics

        for metric_name in losses_dict:
            res = metric_name.split("_")
            if len(res) != 2:
                continue
            # no epoch anymore
            # split, name, epoch_or_split = metric_name.split("_")
            split, name = metric_name.split("_")

            metric = losses_dict[metric_name].item()
            metric = metric_format.format(metric)

            if split == "train":
                mname = name
            else:
                mname = f"v_{name}"

            metric = f"{mname} {metric}"
            metrics_str.append(metric)

        if len(metrics_str) == 0:
            return

        line = line + ": " + "  ".join(metrics_str)
        logger.info(line)
