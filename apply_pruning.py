import pathlib
import random
import ssl

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from arguments import prepare_args
from nni.compression.speedup import ModelSpeedup
from segmentation_dataset import (SegmentationDataset, get_preprocessing,
                                  get_validation_augmentation)
from segmentation_models_pytorch.utils import losses
from segmentation_models_pytorch.utils.metrics import (Accuracy, Fscore, IoU,
                                                       Precision, Recall)
from segmentation_models_pytorch.utils.train import ValidEpoch
from setup import setup
from torch.utils.data import DataLoader, Dataset


def get_test_dataset(args)->Dataset:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder,args.encoder_pretrained_source)
    preprocessing = get_preprocessing(preprocessing_fn)
    validation_augmentation = get_validation_augmentation()
    test_dataset = SegmentationDataset(name=args.dataset, mode="test", augmentation=validation_augmentation, preprocessing=preprocessing)
    return test_dataset
    
def test(args):
    """main function to run the training, calling setup and preparing the train/validation epoch

    Args:
        args (_type_): _description_
    """
    test_dataset = get_test_dataset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    for _, batch in enumerate(test_dataloader):
        sample_iput, y = batch
        break
    loss = losses.JaccardLoss()
    metrics = [IoU(), Accuracy(), Precision(), Recall(), Fscore()]
    
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path("")/checkpoint_root/args.title
    checkpoint_path = checkpoint_dir/(f"pruned_model-{str(args.sparsity)}.pth")
    mask_path = checkpoint_dir/(f"pruned_mask-{str(args.sparsity)}.pth")
    model: smp.FPN = torch.load(checkpoint_path.absolute(), map_location=torch.device(args.device))
    masks = torch.load(mask_path.absolute(), map_location=torch.device(args.device))
    sample_iput = sample_iput.to(torch.device(args.device))
    ModelSpeedup(model, sample_iput, masks, batch_size=1).speedup_model()
    # print(model)
    # print(masks)
    # tester = ValidEpoch(model, loss, metrics, device=args.device, verbose=True)
    # test_log = tester.run(test_dataloader)


if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    test(args)