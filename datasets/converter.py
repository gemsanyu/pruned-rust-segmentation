import pathlib
import multiprocessing as mp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
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

def get_mask(img, annotations, metadata):
    h,w,_ = img.shape
    mask = np.zeros([h,w], dtype=int)
    for a in annotations:
        black_img = np.zeros_like(img)
        v = Visualizer(black_img, metadata=metadata, scale=1)
        segment = np.asanyarray(a["segmentation"])
        _, n = segment.shape
        segment = segment.reshape([int(n/2),2])
        v.draw_polygon(segment, "white")
        v = v.get_output()
        label_mask = v.get_image()
        label_mask = np.any(np.greater(label_mask, 0), axis=-1)
        mask[label_mask] = int(a["category_id"])
    return mask

def convert(dataset_dict, new_img_dir, new_mask_dir, metadata, mode):
    img = cv2.imread(dataset_dict["file_name"])
    original_path = pathlib.Path(dataset_dict["file_name"])
    file_name = original_path.stem
    new_img_path = new_img_dir/(file_name+".jpg")
    new_mask_path = new_mask_dir/(file_name+".png")
    annotations = dataset_dict["annotations"]
    mask = get_mask(img, annotations, metadata)
    cv2.imwrite(str(new_img_path.absolute()), img)
    cv2.imwrite(str(new_mask_path.absolute()), mask)
    

def run(mode, orig_name, new_name):
    dataset_dir = pathlib.Path("")/orig_name/mode
    annotation_file_path = dataset_dir/"_annotations.coco.json"
    dataset_name = "ndea_dataset_"+mode
    register_coco_instances(dataset_name, {}, annotation_file_path.absolute(), dataset_dir.absolute())
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    new_dataset_dir = pathlib.Path("")/new_name
    new_dir = new_dataset_dir/mode
    new_img_dir = new_dir/"images"
    new_mask_dir = new_dir/"masks"
    new_img_dir.mkdir(parents=True, exist_ok=True)
    new_mask_dir.mkdir(parents=True, exist_ok=True)
    
    args = [(dataset_dict, new_img_dir, new_mask_dir, metadata, mode) for dataset_dict in dataset_dicts]
    with mp.Pool(4) as pool:
        pool.starmap(convert, args)
    # for dataset_dict in dataset_dicts:
    #     convert(dataset_dict, new_img_dir, new_mask_dir, metadata)
    
if __name__ == "__main__":
    orig_name = "CIS_Coco"
    new_name = "CIS"
    run("train", orig_name, new_name)
    run("valid", orig_name, new_name)
    run("test", orig_name, new_name)
    