import pathlib
import os

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from segmentation_models_pytorch.base import SegmentationModel
from torch.optim import Optimizer
import torch




def write_logs(train_logs: dict, valid_logs: dict, tb_writer:SummaryWriter, epoch:int):
    """write loss/metrics and other logs to track into tensorboard

    Args:
        train_logs (dict): _description_
        valid_logs (dict): _description_
        tb_writer (SummaryWriter): _description_
        epoch (int): _description_
        mode (str, optional): _description_. Defaults to "train".
    """
    train_keys = train_logs.keys()
    valid_keys = valid_logs.keys()
    for t_key, t_value in train_logs.items():
        if t_key in valid_keys:
            v_value = valid_logs[t_key]
            value_dict = {"train":t_value, "valid":v_value}
            tb_writer.add_scalars(t_key, value_dict, epoch)
        else:
            tb_writer.add_scalar(t_key, t_value, epoch)
    
    for v_key in valid_keys:
        if v_key in train_keys:
            continue
        v_value = valid_logs[v_key]
        tb_writer.add_scalar(v_key, v_value, epoch)

def save(model:SegmentationModel,
         optimizer:Optimizer,
         valid_logs:dict,
         checkpoint_dir:pathlib.Path,
         epoch:int):
    checkpoint_path = checkpoint_dir/"checkpoint.pt"
    result_path = checkpoint_dir/"result.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":epoch
    }
    result = valid_logs
    torch.save(checkpoint, checkpoint_path.absolute())
    torch.save(result, result_path.absolute())
    
    best_checkpoint_path = checkpoint_dir/"best_checkpoint.pt"
    best_result_path = checkpoint_dir/"best_result.pt"
    best_result, best_iou_score, best_accuracy = None, None, None
    if os.path.exists(best_result_path.absolute()):
        best_result = torch.load(best_result_path.absolute())
        best_iou_score = best_result["iou_score"]
        best_accuracy = best_result["accuracy"]
    if (best_result is None) or (best_iou_score < result["iou_score"]) or (best_iou_score==result["iou_score"] and best_accuracy<result["accuracy"]):
        torch.save(checkpoint, best_checkpoint_path.absolute())
        torch.save(result, best_result_path.absolute())
    

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    # plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()