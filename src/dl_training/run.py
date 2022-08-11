import os
from re import I
import yaml
import argparse
import numpy as np
from src.pathlib import Path
from src.model import *
from src.data import VAEDataset
from src.experiment import VAEXperiment
from src.callbacks import *
from src.util import *
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
#from pytorch_lightning.strategies.ddp import DDPStrategy

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help ='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    config = load_config(args.filename)

    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                   name=config['model_params']['name'],
                                   log_graph=False,
                                   default_hp_metric=False,
                                   flush_secs=5,)

    model = VAE(**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])
    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

    data.setup()
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         CPUMonitor(),
                         LearningRateMonitor("epoch"),
                         ModelCheckpoint(save_top_k=2, 
                                         dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                         monitor= "val_loss",
                                         save_last= True),
                     ],
                     accelerator="auto",
                     devices=1 if torch.cuda.is_available() else None,
                     log_every_n_steps=1,
                     #strategy=DDPPlugin(find_unused_parameters=False),
                     **config['trainer_params'])

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)
    tb_logger.save()