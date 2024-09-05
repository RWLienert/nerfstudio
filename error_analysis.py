import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import subprocess

def create_dir(folder_path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

def copy_images(src_folder, dest_folder):
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        if os.path.isfile(src_path) and item.endswith('.jpg'):
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")

def highlight_errors_with_colour(rgb_image, gt_rgb_image, threshold=70):
    # Compute the absolute difference between the images
    error_image = np.abs(rgb_image.astype(np.int32) - gt_rgb_image.astype(np.int32))
    
    # Compute the magnitude of the error (Euclidean distance in RGB space)
    error_magnitude = np.linalg.norm(error_image, axis=-1)
    
    # Scale the error magnitude with sensitivity factor for less significant thresholding
    scaled_error = np.clip(error_magnitude / threshold, 0, 1)
    
    # Create a mask where the error magnitude is greater than the scaled threshold
    mask = scaled_error > 0
    
    # Create a gradient from the original color to darker shades in the error areas
    gradient = 1 - scaled_error
    gradient = np.stack([gradient] * 3, axis=-1)
    
    # Blend the original image with the gradient in error regions
    result_image = rgb_image.copy()
    result_image[mask] = (rgb_image[mask] * gradient[mask]).astype(np.uint8)
    
    return result_image

def compute_error(rgb_path, gt_rgb_path, error_folder):
    rgb_images = set(file for file in os.listdir(rgb_path) if file.endswith('.jpg'))
    gt_rgb_images = set(file for file in os.listdir(gt_rgb_path) if file.endswith('.jpg'))

    common_images = rgb_images & gt_rgb_images
    
    for image_file_name in common_images:
        rgb_image_path = os.path.join(rgb_path, image_file_name)
        gt_rgb_image_path = os.path.join(gt_rgb_path, image_file_name)
        error_image_path = os.path.join(error_folder, image_file_name)
        
        print(f"Processing {image_file_name}")
        print(f"RGB image path: {rgb_image_path}")
        print(f"GT-RGB image path: {gt_rgb_image_path}")

        rgb_image = np.array(Image.open(rgb_image_path))
        gt_rgb_image = np.array(Image.open(gt_rgb_image_path))
        
        if rgb_image.shape == gt_rgb_image.shape:
            error_image = highlight_errors_with_colour(rgb_image, gt_rgb_image)
            Image.fromarray(error_image).save(error_image_path)
        else:
            print(f"Error: Image shapes do not match for {image_file_name}")
    
def main(rgb_train_path, gt_rgb_train_path, rgb_test_path, gt_rgb_test_path, config_path): 
    ## Create folders
    combined_folder = Path("/home/riley/nerfstudio/renders/combined")
    combined_rgb_path = combined_folder / "rgb"
    combined_gt_rgb_path = combined_folder / "gt-rgb"
    create_dir(combined_rgb_path)
    create_dir(combined_gt_rgb_path)
    
    # Copy images from train and test folders into the combined folder
    copy_images(rgb_train_path, combined_rgb_path)
    copy_images(gt_rgb_train_path, combined_gt_rgb_path)
    copy_images(rgb_test_path, combined_rgb_path)
    copy_images(gt_rgb_test_path, combined_gt_rgb_path)
    
    # Extract the name from the config path
    name = config_path.split('/outputs/')[1].split('/')[0]
    print(f"Extracted name: {name}")
    
    # Define the error path structure by keeping the 'nerfstudio' part in the path
    base_path = config_path.split('/nerfstudio/')[0] + '/nerfstudio/data/nerfstudio/'
    error_folder = Path(base_path) / f"{name}_error"
    create_dir(error_folder)
    print(f"Base path: {base_path}")
    print(f"Error folder: {error_folder}")
    
    # Compute the error images and save them to the 'images' folder
    compute_error(combined_rgb_path, combined_gt_rgb_path, error_folder)


if __name__ == "__main__":
    config_path = input("Enter the path to the pretrained model checkpoint: ")
    rgb_train_path = "/home/riley/nerfstudio/renders/train/rgb"
    gt_rgb_train_path = "/home/riley/nerfstudio/renders/train/gt-rgb"
    rgb_test_path = "/home/riley/nerfstudio/renders/test/rgb"
    gt_rgb_test_path = "/home/riley/nerfstudio/renders/test/gt-rgb"
    main(rgb_train_path, gt_rgb_train_path, rgb_test_path, gt_rgb_test_path, config_path)
