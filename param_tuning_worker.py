import random
from typing import Tuple

import nni
import numpy as np
import torch
import torch.utils
import torch.utils.data
from arguments import prepare_args
from torch.utils.data import DataLoader
from train import prepare_train_and_validation_datasets, train, validate
from setup import setup_model, setup_optimizer, NUM_CLASSES_DICT
from custom_loss import CustomLoss
from tqdm import tqdm


def run(args, params):
    print(params)
    model = setup_model(args)
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = setup_optimizer(model, params["optimizer_name"], params["lr"], params["momentum"])
    train_dataset, validation_dataset = prepare_train_and_validation_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], num_workers=1, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, pin_memory=True)
    num_class = NUM_CLASSES_DICT[args.dataset]
    mode= "binary" if num_class==1 else "multiclass"
    loss_func = CustomLoss(num_class)
    for epoch in tqdm(range(args.max_epoch)):
        train_logs = train(model, optimizer, loss_func, train_dataloader, mode, device)
        validation_logs = validate(model, loss_func, validation_dataloader, mode, device)
        validation_logs["default"] = validation_logs["iou_score"]
        print(validation_logs)
        nni.report_intermediate_result(validation_logs)
    nni.report_final_result(validation_logs)
    

if __name__ == "__main__":
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # default params, otw updated by nni for trial
    params = {
        "batch_size": 4,
        "lr": 3e-4,
        "momentum":0.5,
        "optimizer_name":"sgd",
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    run(args, params)
        