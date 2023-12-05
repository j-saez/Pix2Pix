import torch
import torch.utils
import torch.utils.data
from typing import List
import pytorch_lightning as pl
from utils.configuration              import Configuration
from datasets.classes.pix2pix_dataset import Pix2PixDataset
from utils.tools                   import download_dataset

class Pix2PixDataModule(pl.LightningDataModule):

    def __init__(self, config: Configuration) -> None:
        super().__init__()

        self.config = config
        if config.dataparams.prepare_data_per_node:
            self.prepare_data()

        return

    def prepare_data(self) -> None:
        """
        Downloads the data so we have it to disc
        """
        # this is for single gpu
        download_dataset(self.config.dataparams.dataset_name)
        return

    def setup(self, stage: str) -> None:
        """
        Loads the data downloaded in prepate_data as a pytorch dataset class object
        """
        # this is for multiple gpu as it is called in every gpu on the system.

        if stage == 'fit' or stage == None:
            self.train_dataset = Pix2PixDataset( self.config, 'train')
            self.val_dataset = Pix2PixDataset( self.config, 'val')

        return

    def train_dataloader(self,):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.hyperparams.batch_size,
            num_workers=self.config.general.num_workers,
            shuffle=True,)

    def val_dataloader(self,):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.hyperparams.batch_size,
            num_workers=self.config.general.num_workers,
            shuffle=False,)

    def get_fixed_data(self, total_imgs: int) -> List[torch.Tensor]:
        self.setup(stage='fit')
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=total_imgs,
            num_workers=self.config.general.num_workers,
            shuffle=True)
        train_dataiter = iter(dataloader)
        train_source_imgs, train_target_imgs = next(train_dataiter)

        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=total_imgs,
            num_workers=self.config.general.num_workers,
            shuffle=True)
        test_dataiter = iter(dataloader)
        test_source_imgs, test_target_imgs = next(test_dataiter)

        return [train_source_imgs, train_target_imgs, test_source_imgs, test_target_imgs]
