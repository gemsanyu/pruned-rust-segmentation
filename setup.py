import pathlib
import os
from typing import Tuple

import nni
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import Optimizer

ARCH_CLASS_DICT = {
    "fpn":smp.FPN,
    "unet":smp.Unet,
    "unet++":smp.UnetPlusPlus,
    "manet":smp.MAnet,
    "linknet":smp.Linknet,
    "pspnet":smp.PSPNet,
    "pan":smp.PAN,
    "deeplabv3":smp.DeepLabV3,
    "deeplabv3+":smp.DeepLabV3Plus}



def prepare_tb_writer(args)->SummaryWriter:
    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/(args.title+str(args.sparsity))
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=model_summary_dir.absolute())
    return tb_writer


def setup_model(args)->SegmentationModel:
    Arch_Class = ARCH_CLASS_DICT[args.arch]
    model = Arch_Class(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_pretrained_source,
        classes=1,
        activation="sigmoid"
    )
    return model

def setup(args, load_best:bool=False)->Tuple[SegmentationModel, Optimizer, SummaryWriter, pathlib.Path, int]:
    model = setup_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tb_writer = prepare_tb_writer(args)
    
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path("")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/"checkpoint.pt"
    if load_best:
        checkpoint_path = checkpoint_dir/"best_checkpoint.pt"
    
    checkpoint = None
    last_epoch = 0
    if os.path.exists(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute())
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = checkpoint["epoch"]
    
    return model, optimizer, tb_writer, checkpoint_dir, last_epoch


def setup_pruning(args, load_best:bool=False)->Tuple[SegmentationModel, Optimizer, SummaryWriter, pathlib.Path, int]:
    
    
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path("")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/"checkpoint_pruning.pt"
    if load_best:
        checkpoint_path = checkpoint_dir/"best_checkpoint_pruning.pt"
        
    model = setup_model(args)
    optimizer = nni.trace(torch.optim.AdamW)(model.parameters(), lr=args.lr)
        
    checkpoint = None
    last_epoch = 0
    if os.path.exists(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute())
        Arch_Class = ARCH_CLASS_DICT[args.arch]
        model: Arch_Class = checkpoint["pruned_model"]
        optimizer = nni.trace(torch.optim.AdamW)(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = checkpoint["epoch"]
    else:
        initial_model_checkpoint_path = checkpoint_dir/"best_checkpoint.pt"
        initial_checkpoint = torch.load(initial_model_checkpoint_path.absolute())
        model.load_state_dict(initial_checkpoint["model_state_dict"])
    tb_writer = prepare_tb_writer(args)
    
    
    return model, optimizer, tb_writer, checkpoint_dir, last_epoch
