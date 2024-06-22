import random
import ssl
import time

import nni
import nni.compression
import numpy as np
import torch
import torch.utils
import torch.utils.data
from arguments import prepare_args
from custom_loss import CustomLoss
from nni.compression.pruning import AGPPruner, LinearPruner, TaylorPruner
from nni.compression.speedup import ModelSpeedup
from nni.compression.utils import auto_set_denpendency_group_ids
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from setup import NUM_CLASSES_DICT, setup_pruning
from torch.nn import Conv2d, Linear, Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import prepare_train_and_validation_datasets, validate
from utils import write_logs


def calculate_sparsity(model:Module, op_types_str):
    op_types_dict = {
        "Linear":Linear, 
        "Conv2d":Conv2d
    }
    op_types = [op_types_dict[op_type_str] for op_type_str in op_types_str]
    sparsity = 0
    module_count = 0

    for module in model.modules():
        is_considered_op_type=False
        for op_type in op_types:
            if isinstance(module, op_type):
                is_considered_op_type = True
                break
        if not is_considered_op_type:
            continue
        module_count += 1
        module_sparsity = float(torch.sum(module.weight==0))/module.weight.nelement()
        sparsity += module_sparsity
    sparsity /= module_count
    return sparsity 

def get_sample_input(validation_dataloader, device):
    sample_input = None
    for _, batch in enumerate(validation_dataloader):
        x,y = batch
        sample_input = x
        # sample_input = sample_input[None,:,:,:]
        break
    sample_input = sample_input.to(device)
    return sample_input

def run(args):
    """main function to run the training, calling setup and preparing the train/validation epoch

    Args:
        args (_type_): _description_
    """
    model, optimizer, scheduler, tb_writer, checkpoint_dir, last_epoch = setup_pruning(args)
    train_dataset, validation_dataset = prepare_train_and_validation_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, pin_memory=True)
    num_class = NUM_CLASSES_DICT[args.dataset]
    mode = "binary" if num_class==1 else "multiclass"
    loss_func = CustomLoss(num_class, args.loss_combination)
    device = torch.device(args.device)
    sample_input = get_sample_input(validation_dataloader, device)
    model = model.to(device)
    pruned_op_types = ['Conv2d']
    
    def training_step(batch, model, *args, **kwargs):
        x,y = batch
        x, y = x.to(device), y.to(device)
        prediction = model.forward(x)
        loss = loss_func(prediction, y)
        return loss
        
    def training(model, optimizer, training_step, lr_scheduler, max_steps, max_epochs):
        for epoch in tqdm(range(max_epochs)):
            print(epoch)
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                loss = training_step(batch, model)
                loss.backward()
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()
                    
            validation_logs = validate(model, loss_func, validation_dataloader, mode, device)
            validation_logs["sparsity"] = calculate_sparsity(model, pruned_op_types)
            write_logs({}, validation_logs, tb_writer, epoch)
    
    
    
    config_list = [{
        'op_types': pruned_op_types,
        'sparse_ratio': args.sparsity
    }]
    total_training_steps = len(train_dataloader)*args.max_epoch
    total_times = 10
    total_times = int(total_times*0.8)
    training_steps = int(total_training_steps/total_times)
    # 80% initial -> scheduled pruning. 20% final fine-tuning
    config_list = auto_set_denpendency_group_ids(model, config_list, sample_input)
    evaluator = nni.compression.TorchEvaluator(training, optimizer, training_step, scheduler)
    sub_pruner = TaylorPruner(model, config_list, evaluator, training_steps=training_steps)
    scheduled_pruner = AGPPruner(sub_pruner, interval_steps=training_steps, total_times=total_times)
    _, masks = scheduled_pruner.compress(max_steps=None, max_epochs=args.max_epoch)
    
    model.zero_grad()
    model.eval()
    model = model.to(torch.device("cpu"))
    checkpoint_path = checkpoint_dir/(f"pruned_model-{str(args.sparsity)}.pth")
    mask_path = checkpoint_dir/(f"pruned_mask-{str(args.sparsity)}.pth")
    torch.save(model, checkpoint_path.absolute())
    torch.save(masks, mask_path.absolute())
    
    
if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    run(args)