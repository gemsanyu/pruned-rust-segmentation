
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
from segmentation_models_pytorch.utils.base import Loss
from segmentation_models_pytorch.metrics import iou_score, f1_score
from setup import setup
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import save, write_logs


def train(model, 
          optimizer: torch.optim.Optimizer,
          loss_func: Loss,
          dataloader: torch.utils.data.DataLoader,
          mode:str):
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
    model.train()
    losses = []
    ious = []
    f1s = []
    for _, batch in enumerate(dataloader):
        x, y = batch
        y = y.long()
        optimizer.zero_grad()
        prediction = model.forward(x)
        num_class = prediction.shape[1]
        loss = loss_func(prediction, y)
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
        pred_ = torch.argmax(prediction.detach(), dim=1, keepdim=True).detach()
        
        tp, fp, fn, tn = smp.metrics.get_stats(pred_, y, mode=mode, num_classes=num_class)
        iou = iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        f1 = f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        ious += [iou.item()]
        f1s += [f1.item()]
    logs = {
        "loss": np.asanyarray(losses).mean(),
        "iou_score": np.asanyarray(ious).mean(),
        "f1_score": np.asanyarray(f1s).mean()
    }
    optimizer.zero_grad()
    return logs

@torch.no_grad()
def validate(model, 
          loss_func: Loss,
          dataloader: torch.utils.data.DataLoader,
          mode:str):
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
    model.eval()
    losses = []
    ious = []
    f1s = []
    for _, batch in enumerate(dataloader):
        x, y = batch
        y = y.long()
        prediction = model.forward(x)
        num_class = prediction.shape[1]
        loss = loss_func(prediction, y)
        losses += [loss.item()]
        pred_ = torch.argmax(prediction.detach(), dim=1, keepdim=True).detach()
        
        tp, fp, fn, tn = smp.metrics.get_stats(pred_, y, mode=mode, num_classes=num_class)
        iou = iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        f1 = f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        ious += [iou.item()]
        f1s += [f1.item()]
    logs = {
        "loss": np.asanyarray(losses).mean(),
        "iou_score": np.asanyarray(ious).mean(),
        "f1_score": np.asanyarray(f1s).mean()
    }
    return logs

      
def prepare_train_and_validation_datasets(args, num_images=None, n_splits=5)->Tuple[SegmentationDataset,SegmentationDataset]:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder,args.encoder_pretrained_source)
    preprocessing = get_preprocessing(preprocessing_fn)
    augmentation = get_training_augmentation()
    validation_augmentation = get_validation_augmentation()
    try:
        validation_dataset = SegmentationDataset(name=args.dataset, mode="valid", augmentation=validation_augmentation, preprocessing=preprocessing)
        train_dataset = SegmentationDataset(name=args.dataset, mode="train", num_images=num_images, augmentation=augmentation, preprocessing=preprocessing)
    except FileNotFoundError:
        full_train_dataset = SegmentationDataset(name=args.dataset, mode="train")
        kfold = KFold(n_splits=n_splits, shuffle=True)
        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_dataset)):
            train_dataset = SegmentationDataset(name=args.dataset, mode="train", num_images=num_images, augmentation=augmentation, preprocessing=preprocessing, filter_idx_list=train_ids)
            validation_dataset = SegmentationDataset(name=args.dataset, mode="train", augmentation=validation_augmentation, preprocessing=preprocessing, filter_idx_list=val_ids)
    return train_dataset, validation_dataset
    