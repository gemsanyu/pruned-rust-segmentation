import random
import ssl

import numpy as np
import torch
import torch.utils
import torch.utils.data
from arguments import prepare_args
from segmentation_models_pytorch.utils import losses
from segmentation_models_pytorch.utils.metrics import Accuracy, Fscore, IoU, Precision, Recall
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader

from setup import setup
from train import prepare_train_and_validation_datasets, train


def run(args):
    """main function to run the training, calling setup and preparing the train/validation epoch

    Args:
        args (_type_): _description_
    """
    model, optimizer, tb_writer, checkpoint_dir, last_epoch = setup(args)
    train_dataset, validation_dataset = prepare_train_and_validation_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loss = losses.JaccardLoss()
    metrics = [IoU(), Accuracy(), Precision(), Recall(), Fscore()]
    trainer = TrainEpoch(model, loss, metrics, optimizer, args.device, verbose=True)
    validator = ValidEpoch(model, loss, metrics, device=args.device, verbose=True)
    train(trainer, train_dataloader, validator, validation_dataloader, tb_writer, checkpoint_dir, last_epoch, args.max_epoch)
   
if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    run(args)