import os
import pathlib
import random

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

def get_mask(img, annotations, metadata):
    black_img = np.zeros_like(img)
    v = Visualizer(black_img, metadata=metadata, scale=1)
    for a in annotations:
        segment = np.asanyarray(a["segmentation"])
        _, n = segment.shape
        segment = segment.reshape([int(n/2),2])
        v.draw_polygon(segment, "white")
    v = v.get_output()
    mask = v.get_image()
    mask[mask>0] = 1
    return mask

def convert(mode="train"):
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
    
    
    for img_filename in image_filenames:
        img_id = img_filename.split(".")[0]
        mask_path = mask_dir/(img_id+".png")
        img_path = image_dir/img_filename
        new_img_path = new_img_dir/img_filename
        mask = cv2.imread(mask_path.absolute())
        img =  cv2.imread(img_path.absolute())
        x,y,z = mask.shape
        mask = np.reshape(mask, [x*y,z])
        unique_colors = np.unique(mask, axis=0)
        unique_ids = np.arange(len(unique_colors))
        unique_ids = np.tile(unique_ids, [x*y,1])
        is_color = np.equal(mask[:,np.newaxis,:], unique_colors[np.newaxis,:,:])
        is_color = np.all(is_color, axis=-1)
        new_mask = np.sum(unique_ids*is_color, axis=-1, keepdims=True)
        new_mask = np.reshape(new_mask, [x,y,1])
        new_mask_path = new_mask_dir/(img_filename)
        cv2.imwrite(str(new_img_path.absolute()), img)
        cv2.imwrite(str(new_mask_path.absolute()), new_mask)

def run():
    convert(mode="train")
    # convert(mode="valid")
    convert(mode="test")
    
if __name__ == "__main__":
    run()