import pathlib
import random
import ssl

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from arguments import prepare_args
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
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path("")/checkpoint_root/args.title
    checkpoint_path = checkpoint_dir/(f"pruned_model-{str(args.sparsity)}.pth")
    model = torch.load(checkpoint_path.absolute(), map_location=torch.device(args.device))
    
    test_dataset = get_test_dataset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loss = losses.JaccardLoss()
    metrics = [IoU(), Accuracy(), Precision(), Recall(), Fscore()]
    tester = ValidEpoch(model, loss, metrics, device=args.device, verbose=True)
    test_log = tester.run(test_dataloader)
    test_log["arch"] = args.arch
    test_log["encoder"] = args.encoder
    test_log["title"] = args.title
    test_log["sparsity"] = args.sparsity
    result_dir = pathlib.Path(".")/"results"/args.title
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir/f"result_{str(args.sparsity)}.csv"
    test_df = pd.DataFrame([test_log])
    test_df.to_csv(result_file.absolute())

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    test(args)