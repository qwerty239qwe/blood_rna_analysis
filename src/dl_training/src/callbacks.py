import os
from typing import Any, Dict, Optional, Union
import psutil
import torch
from pytorch_lightning import LightningModule, Trainer, Callback


class CPUMonitor(Callback):
    def __init__(self):
        self.counter = 0
        self.collections = []

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        if not trainer.logger:
            raise MisconfigurationException(
                "Cannot use CPUMonitor callback with Trainer that has no logger."
            )

    def _log_cpu_metric_dict(
        self, hook: str, trainer: LightningModule
    ) -> None:
        """Gets CPU metrics.
        
        Args:
            hook (str): the hook that this was called from (ex. 'on_train_batch_start')
        """
        metrics = {
            f"cpu_metrics/vm_percent/{hook}": torch.tensor(psutil.virtual_memory().percent, dtype=torch.float32),
        }
        for k, v in metrics.items():
            trainer.logger.experiment.add_scalar(k, v, global_step=self.counter)
        self.counter += 1

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self._log_cpu_metric_dict("on_batch_start", pl_module)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self._log_cpu_metric_dict("on_batch_start", trainer)
        
    def on_validation_batch_end(self, trainer, module, outputs, ...):
        vacc = outputs['val_acc'] # you can access them here
        self.collection.append(vacc) # track them