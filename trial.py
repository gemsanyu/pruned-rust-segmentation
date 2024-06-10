import pathlib
import random
from typing import Tuple

import numpy as np
import torch
import torch.utils
import torch.utils.data
from arguments import prepare_args
from torch.optim import AdamW
from segmentation_models_pytorch.utils import losses
from segmentation_models_pytorch.utils.metrics import (Accuracy, Fscore, IoU,
                                                       Precision, Recall)
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from setup import setup
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train import prepare_train_and_validation_datasets
from utils import save, write_logs
from setup import setup_model


def run(args, params):
    model = setup_model(args)
    optimizer = AdamW(model.parameters(), lr=params["lr"])
    train_dataset, validation_dataset = prepare_train_and_validation_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loss = losses.JaccardLoss()
    metrics = [IoU()]
    trainer = TrainEpoch(model, loss, metrics, optimizer, args.device, verbose=True)
    validator = ValidEpoch(model, loss, metrics, device=args.device, verbose=True)
    
    valid_logs = {}
    for epoch in range(args.max_epoch):
        train_logs = trainer.run(train_dataloader)
        valid_logs = validator.run(validation_dataloader)
        valid_logs["default"] = valid_logs["iou_score"]
    

if __name__ == "__main__":
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # default params, otw updated by nni for trial
    params = {
        "batch_size": 4,
        "lr": 3e-4
    }
    run(args, params)
        