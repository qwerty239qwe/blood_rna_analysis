import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import pandas as pd
from scipy.stats import gmean
import numpy as np
from pathlib import Path

from typing import Union, List
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class RNAseqDataset(Dataset):
    split_name = {"train": "training_ds", "valid": "validation_ds", "test": "testing_ds", "all": "all_ds", "cv_fold": "train_and_valid"}
    
    def __init__(self, data_dir_path, split="train"):
        self.split = split
        self.df = pd.read_csv(Path(data_dir_path) / f"{self.split_name[split]}.csv", index_col=0)
        self.size_factor = self.estimate_size_factor(self.df)
        self.df = self.df.div(pd.DataFrame({c: [v] for c, v in self.size_factor.items()}).values[0], axis=1)
        self.df = np.log2(self.df + 1)

    @staticmethod
    def estimate_size_factor(df):
        ref = df.apply(gmean, axis=1)
        size_factors = {}
        for c in df.columns:
            pre = df[c] / ref
            pre[np.isinf(pre)] = np.nan
            size_factors[c]= np.nanmedian(pre)
        return size_factors
        
    def __len__(self):
        return (self.df.shape[1])
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"data": torch.from_numpy(self.df.iloc[:, idx].values)}
    
    
class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage = None):
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            data_cv = RNAseqDataset(self.data_dir, "cv_fold")
            train_size = int(len(data_cv) * 0.8)
            self.train_dataset, self.val_dataset = random_split(data_cv, [train_size, len(data_cv) - train_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = RNAseqDataset(self.data_dir, "test")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )