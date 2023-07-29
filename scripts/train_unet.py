from pathlib import Path
import pytorch_lightning as pl
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule

import torch
from torch.nn import functional as F

import numpy as np
import pytorch_lightning as pl
from torch import nn
from torchmetrics.metric import Metric

import fastmri
from fastmri.models.unet import ConvBlock, TransposeConvBlock
from fastmri import evaluate
from fastmri.pl_modules import MriModule
from fastmri.pl_modules.mri_module import DistributedMetricSum

def train():

    mask_types = [
        "random",
        "equispaced",
        "equispaced_fraction",
        "magic",
        "magic_fraction"
    ]

    mask_type = mask_types[0]

    # Number of center lines to use in mask
    center_fractions = [0.09]

    # acceleration rates to use for masks
    accelerations = [4]

    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )

    # Data specific Parameters
    data_path = Path('data/')
    test_path = Path('data/singlecoil_test')
    challenge = "singlecoil"
    test_split = "test"

    sample_rate = 0.5
    val_sample_rate = None
    test_sample_rate = None
    volume_sample_rate = None
    val_volume_sample_rate = None
    test_volume_sample_rate = None
    use_dataset_cache_file = True
    combine_train_val = False

    # data loader arguments
    batch_size = 32
    num_workers = 60

    train_transform = UnetDataTransform(challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(challenge, mask_func=mask)
    test_transform = UnetDataTransform(challenge)

    data_module = FastMriDataModule(
        data_path=data_path,
        challenge=challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=test_split,
        test_path=test_path,
        sample_rate=sample_rate,
        distributed_sampler=None,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    ##############################
    # UNet Model Hyperparameters #
    ##############################
    in_chans=1          # number of input channels to U-Net
    out_chans=1         # number of output chanenls to U-Net
    chans=32            # number of top-level U-Net channels
    num_pool_layers=4   # number of U-Net pooling layers
    drop_prob=0.4       # dropout probability
    lr=0.001            # RMSProp learning rate
    lr_step_size=40     # epoch at which to decrease learning rate
    lr_gamma=0.1        # extent to which to decrease learning rate
    weight_decay=0.0    # weight decay regularization strength

    unet_model = UnetModule(
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob,
        lr=lr,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        weight_decay=weight_decay,
    )

    from pytorch_lightning.loggers import TensorBoardLogger

    trainer_config = dict(
        accelerator = "gpu",
        devices=1,                      # number of gpus to use
        deterministic=False,            # makes things slower, but deterministic
        default_root_dir='../logs',     # directory for logs and checkpoints
        max_epochs=50,                  # max number of epochs
        logger = TensorBoardLogger('logs/', name='unet')
    )

    trainer = pl.Trainer(**trainer_config)
    trainer.fit(unet_model, datamodule=data_module)

if __name__=="__main__":
    train()