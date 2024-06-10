
import pathlib
import random
from typing import Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.utils
import torch.utils.data
from arguments import prepare_args
from segmentation_dataset import (SegmentationDataset, get_preprocessing,
                                  get_training_augmentation,
                                  get_validation_augmentation)
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from setup import setup
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import save, write_logs


def train(trainer: TrainEpoch,
          train_dataloader: DataLoader,
          validator: ValidEpoch,
          validation_dataloader: DataLoader,
          tb_writer: SummaryWriter,
          checkpoint_dir: pathlib.Path,
          last_epoch: int,
          max_epoch: int):
    """train the model and also validate for max_epoch epochs

    Args:
        trainer (TrainEpoch): _description_
        train_dataloader (DataLoader): _description_
        validator (ValidEpoch): _description_
        validation_dataloader (DataLoader): _description_
        tb_writer (SummaryWriter): _description_
        checkpoint_dir (pathlib.Path): _description_
        last_epoch (int): _description_
        max_epoch (int): _description_
    """
    for epoch in range(last_epoch+1, max_epoch):
        print("EPOCH:", epoch)
        train_logs = trainer.run(train_dataloader)
        train_logs["lr"] = trainer.optimizer.param_groups[0]['lr']
        valid_logs = validator.run(validation_dataloader)
        write_logs(train_logs, valid_logs, tb_writer, epoch)
        save(trainer.model, trainer.optimizer, valid_logs, checkpoint_dir, epoch)



      
def prepare_train_and_validation_datasets(args, n_splits=5)->Tuple[SegmentationDataset,SegmentationDataset]:
    full_train_dataset = SegmentationDataset(name=args.dataset, mode="train")
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder,args.encoder_pretrained_source)
    preprocessing = get_preprocessing(preprocessing_fn)
    augmentation = get_training_augmentation()
    validation_augmentation = get_validation_augmentation()
    kfold = KFold(n_splits=n_splits, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_dataset)):
        train_dataset = SegmentationDataset(name=args.dataset, mode="train", augmentation=augmentation, preprocessing=preprocessing, filter_idx_list=train_ids)
        validation_dataset = SegmentationDataset(name=args.dataset, mode="train", augmentation=validation_augmentation, preprocessing=preprocessing, filter_idx_list=val_ids)
        return train_dataset, validation_dataset
    