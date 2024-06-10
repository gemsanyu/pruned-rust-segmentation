import pathlib
from typing import Tuple, List

import numpy as np
import cv2
import albumentations as albu
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, 
                 name:str="NEA-Dataset-semantic",
                 mode:str="train",
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
        image_paths = [self.images_dir/image_name for image_name in self.image_names]
        mask_paths = [self.masks_dir/image_name for image_name in self.image_names]
        self.images: List[cv2.typing.MatLike] = [cv2.imread(str(img_path.absolute())) for img_path in image_paths]
        self.masks: List[cv2.typing.MatLike] = [cv2.imread(str(mask_path.absolute()), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis] for mask_path in mask_paths]
        self.images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in self.images]
        self.images = np.stack(self.images)
        self.masks = np.stack(self.masks)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = self.images[index], self.masks[index]
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
        albu.PadIfNeeded(736, 1280),
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