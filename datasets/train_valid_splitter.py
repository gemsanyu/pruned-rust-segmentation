import random
import os
import shutil
import pathlib


def run():
    dataset_dir = pathlib.Path("")/"CCSC"/"train"
    current_images_dir = dataset_dir/"images"
    current_mask_dir = dataset_dir/"masks"
    
    target_dataset_dir = pathlib.Path("")/"CCSC"/"valid"
    target_images_dir = target_dataset_dir/"images"
    target_mask_dir = target_dataset_dir/"masks"
    
    image_names = [filepath.stem for filepath in current_images_dir.iterdir() if filepath.is_file()]
    num_images = len(image_names)
    num_valid_images = int(num_images/5)
    valid_image_names = random.sample(image_names, num_valid_images)
    
    for img_name in valid_image_names:
        img_path = current_images_dir/(img_name+".jpeg")
        mask_path = current_mask_dir/(img_name+".png")
        
        target_img_path = target_images_dir/(img_name+".jpeg")
        target_mask_path = target_mask_dir/(img_name+".png")
        
        shutil.move(img_path.absolute(), target_img_path.absolute())
        shutil.move(mask_path.absolute(), target_mask_path.absolute())
        
    
if __name__ == "__main__":
    run()