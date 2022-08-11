import os
import math
import torch
from torch import optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_seq_data = batch["data"].float()
        self.curr_device = real_seq_data.device

        results = self.forward(real_seq_data)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        self.log("training_loss", train_loss['loss'])
        #self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True, on_step=False, on_epoch=True, logger=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_seq_data = batch["data"].float()
        self.curr_device = real_seq_data.device

        results = self.forward(real_seq_data)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        self.log("val_loss", val_loss['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True, on_step=False, on_epoch=True, logger=True)
        return val_loss
    
    def on_validation_end(self) -> None:
        self.sample_seq_data()
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/lr": self.params['LR'], "hp/weight_decay": self.params['weight_decay']})
    
    def sample_seq_data(self):
        # Get sample reconstruction image            
        test_input = next(iter(self.trainer.datamodule.test_dataloader()))["data"].float()
        test_input = test_input.to(self.curr_device)
        recons = self.model.generate(test_input, 
                                     #labels = test_label
                                    )
        try:
            samples = self.model.sample(8,
                                        self.curr_device,
                                       # labels = test_label
                                       )
        except:
            pass

    def configure_optimizers(self):
        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims