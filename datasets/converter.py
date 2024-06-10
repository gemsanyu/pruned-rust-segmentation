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
    dataset_dir = pathlib.Path("")/"NEA-Dataset-coco"/mode
    annotation_file_path = dataset_dir/"_annotations.coco.json"
    dataset_name = "ndea_dataset_"+mode
    register_coco_instances(dataset_name, {}, annotation_file_path.absolute(), dataset_dir.absolute())
    ndea_metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    new_dataset_dir = pathlib.Path("")/"NEA-Dataset-semantic"
    new_dir = new_dataset_dir/mode
    new_img_dir = new_dir/"images"
    new_mask_dir = new_dir/"masks"
    new_img_dir.mkdir(parents=True, exist_ok=True)
    new_mask_dir.mkdir(parents=True, exist_ok=True)
    
    for dict in dataset_dicts:
        img = cv2.imread(dict["file_name"])
        original_path = pathlib.Path(dict["file_name"])
        file_name = original_path.name
        new_img_path = new_img_dir/file_name
        new_mask_path = new_mask_dir/file_name
        annotations = dict["annotations"]
        mask = get_mask(img, annotations, ndea_metadata)
        cv2.imwrite(str(new_img_path.absolute()), img)
        cv2.imwrite(str(new_mask_path.absolute()), mask)

def run():
    convert(mode="train")
    convert(mode="test")
    
if __name__ == "__main__":
    run()