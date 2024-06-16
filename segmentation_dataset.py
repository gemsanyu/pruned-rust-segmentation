import os
import pathlib
from typing import Tuple, List, Dict

import numpy as np
import cv2
import albumentations as albu
from torch.utils.data import Dataset
from keras.utils import to_categorical

IMG_EXTENSIONS = [".jpg",".jpeg",".png"]
MASK_EXTENSIONS = [".tif", ".png"]


def read_img(img_dir: pathlib.Path, 
             img_id: str, 
             extensions: List[str],
             imread_flag: int = None):
    for ext in extensions:
        img_path = img_dir/(img_id+ext)
        if not os.path.exists(img_path.absolute()):
            continue
        if imread_flag:
            img = cv2.imread(str(img_path.absolute()), imread_flag)
        else:
            img = cv2.imread(str(img_path.absolute()))
    return img
    

class SegmentationDataset(Dataset):
    def __init__(self, 
                 name:str="NEA-Dataset-semantic",
                 mode:str="train",
                 num_images:int=None,
                 augmentation:albu.Compose=None,
                 preprocessing:albu.Compose=None,
                 filter_idx_list:List[int]=None):
        super().__init__()
        self.augmentation:albu.Compose = augmentation
        self.preprocessing:albu.Compose = preprocessing
        self.data_dir = pathlib.Path("")/"datasets"/name/mode
        self.images_dir = self.data_dir/"images"
        self.masks_dir = self.data_dir/"masks"
        self.image_names = [filepath.name for filepath in self.images_dir.iterdir() if filepath.is_file()]
        if filter_idx_list is not None:
            self.image_names = [self.image_names[idx] for idx in filter_idx_list]
        self.img_ids = [img_name.split(".")[0] for img_name in self.image_names]
        self.num_images = num_images or len(self.img_ids)
        self.img_dict: Dict[str, np.ndarray] = {}
        self.mask_dict: Dict[str, np.ndarray] = {}
        self.num_classes = 1
        
        for img_id in self.img_ids:
            img = read_img(self.images_dir, img_id, IMG_EXTENSIONS)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img_dict[img_id] = img
            mask = read_img(self.masks_dir, img_id, MASK_EXTENSIONS, cv2.IMREAD_UNCHANGED)
            self.num_classes = max(self.num_classes, int(np.max(mask)+1))
            self.mask_dict[img_id] = mask[:,:,np.newaxis]
            
    def __len__(self):
        return self.num_images
        # return len(self.img_ids)
        
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        # so that we can set number of dataset size as much as we want?
        index = index % len(self.img_ids) 
        img_id = self.img_ids[index]
        image, mask = self.img_dict[img_id], self.mask_dict[img_id]
        if self.num_classes > 1:
            # # one hot encode the mask
            mask = to_categorical(mask, self.num_classes)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        # albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightnessContrast(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.Sharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.RandomBrightnessContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(1280, 1280),
    ]
    return albu.Compose(test_transform)




def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)