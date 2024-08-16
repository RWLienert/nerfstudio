import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

def create_dir(folder_path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

def compute_error(rgb_path, gt_rgb_path, error_path):
    rgb_images = set(file for file in os.listdir(rgb_path) if file.endswith('.jpg'))
    gt_rgb_images = set(file for file in os.listdir(gt_rgb_path) if file.endswith('.jpg'))

    common_images = rgb_images & gt_rgb_images
    
    for image_file_name in common_images:
        rgb_image_path = os.path.join(rgb_path, image_file_name)
        gt_rgb_image_path = os.path.join(gt_rgb_path, image_file_name)
        error_image_path = os.path.join(error_path, image_file_name)
        
        print(f"Processing {image_file_name}")
        print(f"RGB image path: {rgb_image_path}")
        print(f"GT-RGB image path: {gt_rgb_image_path}")

        rgb_image = np.array(Image.open(rgb_image_path))
        gt_rgb_image = np.array(Image.open(gt_rgb_image_path))
        
        if rgb_image.shape == gt_rgb_image.shape:
            error_image = np.abs(rgb_image.astype(np.int32) - gt_rgb_image.astype(np.int32))
            error_image = np.clip(error_image, 0, 255).astype(np.uint8)
            Image.fromarray(error_image).save(error_image_path)
            print(f"Saved error image to {error_image_path}")
        else:
            print(f"Error: Image shapes do not match for {image_file_name}")
    
def main(rgb_path, gt_rgb_path):
    error_path = Path(rgb_path).parent / 'error'
    create_dir(error_path)
    compute_error(rgb_path, gt_rgb_path, error_path)


if __name__ == "__main__":
    rgb_path = "/home/riley/nerfstudio/renders/train/rgb"
    gt_rgb_path = "/home/riley/nerfstudio/renders/train/gt-rgb"
    main(rgb_path, gt_rgb_path)
