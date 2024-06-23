import pathlib
import random
import ssl

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from arguments import prepare_args
from custom_loss import CustomLoss
from segmentation_dataset import (SegmentationDataset, get_preprocessing,
                                  get_validation_augmentation)
from setup import NUM_CLASSES_DICT, setup_model
from torch.utils.data import DataLoader, Dataset
from train import validate


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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/"best_checkpoint.pt"
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path.absolute(), map_location=device)
    model = setup_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    num_class = NUM_CLASSES_DICT[args.dataset]
    mode = "binary" if num_class==1 else "multiclass"
    loss_func = CustomLoss(num_class, args.loss_combination)
    test_dataset = get_test_dataset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    test_log = validate(model, loss_func, test_dataloader, mode, device)
    test_log["arch"] = args.arch
    test_log["encoder"] = args.encoder
    test_log["title"] = args.title
    test_log["sparsity"] = 0
    test_log["dataset"] = args.dataset
    result_dir = pathlib.Path(".")/"results"/args.title
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir/"result.csv"
    test_df = pd.DataFrame([test_log])
    test_df.to_csv(result_file.absolute())

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    test(args)
