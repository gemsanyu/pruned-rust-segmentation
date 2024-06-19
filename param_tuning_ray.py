import os
import random
import tempfile
from typing import Dict, Tuple
from functools import partial
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from arguments import prepare_args
from custom_loss import CustomLoss
from filelock import FileLock
from ray import train as train_ray
from ray import tune
from ray.tune.search.bohb import TuneBOHB
from ray.train import Checkpoint
from ray.tune.schedulers import HyperBandForBOHB
from setup import NUM_CLASSES_DICT, setup_model, setup_optimizer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from train import prepare_train_and_validation_datasets, train, validate


def run(args, params):
    cwd = pathlib.Path(".").home()/"pruned-rust-segmentation"
    os.chdir(cwd.absolute())
    model = setup_model(args)
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = setup_optimizer(model, params["optimizer_name"], params["lr"], params["momentum"])
    train_dataset, validation_dataset = prepare_train_and_validation_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], num_workers=2, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, pin_memory=True)
    num_class = NUM_CLASSES_DICT[args.dataset]
    mode= "binary" if num_class==1 else "multiclass"
    loss_func = CustomLoss(num_class)
    if train_ray.get_checkpoint():
        loaded_checkpoint = train_ray.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    
    
    for epoch in tqdm(range(args.max_epoch)):
        train_logs = train(model, optimizer, loss_func, train_dataloader, mode, device)
        validation_logs = validate(model, loss_func, validation_dataloader, mode, device)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train_ray.report(
                validation_logs,
                checkpoint=checkpoint,
            )

if __name__ == "__main__":
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    run_func = partial(run, args)
    
    params = {
        "batch_size": tune.choice([2, 4, 8]),
        "lr": tune.loguniform(1e-5, 1e-1),
        "momentum":tune.loguniform(0.5, 0.99),
        "optimizer_name":tune.choice(["sgd","rmsprop"]),
    }
    metric="validation_iou_score"
    scheduler = HyperBandForBOHB(time_attr="training_iteration",
                                 metric=metric,
                                 mode="max",
                                 max_t=args.max_epoch)
    search_algo = TuneBOHB(metric=metric, mode="max")
    max_concurrent = 6
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run_func),
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=100,
            max_concurrent_trials=max_concurrent,
            search_alg=search_algo
        ),
        param_space=params,
    )
    tuner.fit()
        