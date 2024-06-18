import os
import pathlib
import multiprocessing as mp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image, ImageDraw
from torchvision import datasets, models
import segmentation_models_pytorch as smp

def convert(image_dir: pathlib.Path,
            mask_dir: pathlib.Path,
            new_img_dir: pathlib.Path,
            new_mask_dir: pathlib.Path,
            img_filename: str,
            unique_colors: np.ndarray,
            unique_ids: np.ndarray,
            mode: str):
    img_id = img_filename.split(".")[0]
    mask_path = mask_dir/(img_id+".png")
    img_path = image_dir/img_filename
    new_img_path = new_img_dir/img_filename
    mask = cv2.imread(mask_path.absolute())
    img =  cv2.imread(img_path.absolute())
    x,y,z = img.shape
    mx,my,mz = mask.shape
    if mx!=x or my!=y:
        return 
    mask = np.reshape(mask, [x*y,z])
    is_color = np.equal(mask[:,np.newaxis,:], unique_colors[np.newaxis,:,:])
    is_color = np.all(is_color, axis=-1)
    new_mask = np.sum(unique_ids*is_color, axis=-1, keepdims=True)
    new_mask = np.reshape(new_mask, [x,y,1])
    new_mask_path = new_mask_dir/(img_id+".png")
    if mode=="train":
        img = cv2.resize(img, [512,512], interpolation=cv2.INTER_NEAREST_EXACT)
        new_mask = cv2.resize(new_mask, [512,512], interpolation=cv2.INTER_NEAREST_EXACT)
    cv2.imwrite(str(new_img_path.absolute()), img)
    cv2.imwrite(str(new_mask_path.absolute()), new_mask)

def get_unique_colors(ref_mask_name:str, mask_dir: pathlib.Path):
    ref_mask_path = mask_dir/ref_mask_name
    ref_mask = cv2.imread(ref_mask_path.absolute())
    x,y,z = ref_mask.shape
    ref_mask = np.reshape(ref_mask, [x*y,z])
    unique_colors = np.unique(ref_mask, axis=0)
    return unique_colors


def run(mode, ref_mask_name):
    dataset_dir = pathlib.Path("")/"CCSC_original"/mode
    image_dir = dataset_dir/"images"
    image_filenames = os.listdir(image_dir.absolute())
    mask_dir = dataset_dir/"masks"
    
    new_dataset_dir = pathlib.Path("")/"CCSC"
    new_dir = new_dataset_dir/mode
    new_img_dir = new_dir/"images"
    new_mask_dir = new_dir/"masks"
    new_img_dir.mkdir(parents=True, exist_ok=True)
    new_mask_dir.mkdir(parents=True, exist_ok=True)
    
    unique_colors = get_unique_colors(ref_mask_name, mask_dir)
    unique_ids = np.arange(len(unique_colors))

    args = [(image_dir, mask_dir, new_img_dir, new_mask_dir, img_filename, unique_colors, unique_ids, mode) for img_filename in image_filenames]

    # with mp.Pool(4) as pool:
    #     pool.starmap(convert, args)
    for arg in args:
        convert(*arg)
    
    
if __name__ == "__main__":
    run("train", "355.png")
    # new_mask(mode="valid")
    run("test", "14.png")