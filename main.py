import random
import ssl

import numpy as np
import torch
import torch.utils
import torch.utils.data
from arguments import prepare_args
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_loss import CustomLoss
from setup import setup, NUM_CLASSES_DICT
from train import prepare_train_and_validation_datasets, train, validate
from utils import write_logs, save



def run(args):
    """main function to run the training, calling setup and preparing the train/validation epoch

    Args:
        args (_type_): _description_
    """
    model, optimizer, tb_writer, checkpoint_dir, last_epoch = setup(args)
    num_class = NUM_CLASSES_DICT[args.dataset]
    mode = "binary" if num_class==1 else "multiclass"
    train_dataset, validation_dataset = prepare_train_and_validation_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loss_func = CustomLoss(num_class)
    for epoch in tqdm(range(last_epoch+1, args.max_epoch)):
        train_logs = train(model, optimizer, loss_func, train_dataloader, mode)
        validation_logs = validate(model, loss_func, validation_dataloader, mode)
        write_logs(train_logs, validation_logs, tb_writer, epoch)
        save(model, optimizer, validation_logs, checkpoint_dir, epoch)
    
if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    run(args)