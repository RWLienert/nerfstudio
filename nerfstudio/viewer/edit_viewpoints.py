import os
import subprocess
import json
import platform
from pathlib import Path
from datetime import datetime

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

from nerfstudio.cameras import camera_utils

# Enable user to add/remove images
def open_file_explorer(path: Path) -> None:
    """Opens the file explorer at the given path based on the OS, including WSL support."""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    elif platform.system() == "Linux":
        # Check if running in WSL by looking for 'WSL' in the environment variables
        if 'WSL_DISTRO_NAME' in os.environ:
            # Convert the WSL path to a Windows path
            windows_path = path.as_posix().replace("/", "\\").replace("mnt\\c\\", "C:\\")
            subprocess.Popen(["explorer.exe", windows_path])
        else:
            subprocess.Popen(["xdg-open", path])
    else:
        raise OSError(f"Unsupported OS: {platform.system()}")


# Run colmap again to recalculate camera poses
def generate_colmap(data_path: Path, config_path: Path) -> None:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = str(data_path).replace("/images", f"_{current_time}")
    
    print("Calculating new camera positions: ")
    
    process_data = [
        "ns-process-data",
        "images",
        "--data",
        str(data_path),
        "--output-dir",
        new_path
    ]
    
    subprocess.run(process_data, check=True)
    
    print("Training model on new data: ")
    
    train_data = [
        "ns-train",
        "nerfacto",
        "--data",
        new_path
    ]
    
    subprocess.run(train_data, check=True)
    
    print("Previous model config path: ", config_path)
    print("To view the difference between models run: ")
    print("ns-viewer --load-config {larger_model_config} {smaller_model_config} ")


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Converts a 3x3 rotation matrix to a quaternion.
    """
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    return quat_wxyz


def normalize_quaternion(quaternion):
    """
    Normalizes a quaternion.
    """
    norm = np.linalg.norm(quaternion)
    if norm > 0:
        return quaternion / norm
    return quaternion  # Avoid division by zero


def quaternion_distance(quat1, quat2):
    """
    Calculates the angular distance between two quaternions.
    Returns the angular difference in radians.
    """
    dot_product = np.dot(quat1, quat2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot_product))


def process_poses(file_paths_subset, file_paths, positions, quaternions):
    VISER_NERFSTUDIO_SCALE_RATIO = 10
    poses = []
    
    for file_path in file_paths_subset:
        matrix = file_paths[file_path]["transform_matrix"]
        poses.append(matrix)

    # Convert poses to torch tensors and auto-orient them
    poses = torch.from_numpy(np.array(poses).astype(np.float32))

    # Apply auto-orient and center transformation to poses
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        poses,
        method="up",
        center_method="poses",
    )
    # Scale the poses
    scale_factor = 1.0
    scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    poses[:, :3, 3] *= scale_factor  # Scale the translation part of the pose

    # Extract camera-to-world matrices and store positions and rotations
    camera_to_worlds = poses[:, :3, :4]
    c2w = camera_to_worlds.cpu().numpy()
    for i in range(c2w.shape[0]):
        camera_position = c2w[i, :3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
        positions.append(camera_position)

        rotation_matrix = c2w[i, :3, :3]
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        quaternions.append(quaternion)


def calculate_averages(positions, quaternions):
    # Calculate average position
    if positions:
        positions_array = np.array(positions)
        positions_array = np.round(positions_array, 3)
        
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0

        count = len(positions_array)
        for position in positions_array:
            sum_x += position[0]
            sum_y += position[1]
            sum_z += position[2]

        average_x = sum_x / count
        average_y = sum_y / count
        average_z = sum_z / count
        
        avg_position = [average_x, average_y, average_z]
    else:
        avg_position = None

    # Average quaternions using weighted sum (and normalize result)
    if quaternions:
        quaternions = [normalize_quaternion(q) for q in quaternions]
        avg_quaternion = normalize_quaternion(np.mean(quaternions, axis=0))
    else:
        avg_quaternion = None
    
    return avg_position, avg_quaternion


def compare_with_current(avg_position, avg_quaternion, current_position, current_quaternion, set_name):
    if avg_position is not None and avg_quaternion is not None:
        # Calculate position distance (Euclidean distance)
        position_distance = np.linalg.norm(current_position - avg_position)

        # Calculate quaternion distance (angular difference in radians)
        rotation_distance = quaternion_distance(current_quaternion, avg_quaternion)

        print(f"Average Position ({set_name}): {avg_position}")
        print(f"Average Quaternion ({set_name}): {avg_quaternion}")
        print(f"Current Camera Position: {current_position}")
        print(f"Current Camera Rotation (quaternion): {current_quaternion}")
        print(f"Distance to average position ({set_name}): {position_distance}")
        print(f"Rotation difference ({set_name}, in radians): {rotation_distance}")
        print("")


def return_placement_error(path: Path, path_edited: Path, current_position: np.ndarray, current_quaternion: np.ndarray):
    # Construct the paths for the 'transforms.json' files
    json_path = os.path.join(os.path.dirname(path), 'transforms.json')
    json_path_edited = os.path.join(os.path.dirname(path_edited), 'transforms.json')

    # Load the JSON data from both files
    with open(json_path, 'r') as file:
        data = json.load(file)
    with open(json_path_edited, 'r') as file_edited:
        data_edited = json.load(file_edited)

    # Extract file paths from the 'frames' key
    if 'frames' not in data or 'frames' not in data_edited:
        print("Error: JSON files do not contain the 'frames' key.")
        return

    file_paths = {entry['file_path']: entry for entry in data['frames']}
    file_paths_edited = {entry['file_path']: entry for entry in data_edited['frames']}

    # Find file paths present in one but not the other
    only_in_first = set(file_paths) - set(file_paths_edited)
    only_in_edited = set(file_paths_edited) - set(file_paths)

    # Variables to accumulate positions and quaternions for both datasets
    positions_first, quaternions_first = [], []
    positions_edited, quaternions_edited = [], []

    # Process frames from 'only_in_first'
    if only_in_first:
        process_poses(only_in_first, file_paths, positions_first, quaternions_first)

    # Process frames from 'only_in_edited'
    if only_in_edited:
        process_poses(only_in_edited, file_paths_edited, positions_edited, quaternions_edited)
    
    # Calculate averages for both sets
    avg_position_first, avg_quaternion_first = calculate_averages(positions_first, quaternions_first)
    avg_position_edited, avg_quaternion_edited = calculate_averages(positions_edited, quaternions_edited)

    # Compare current camera with the first set
    compare_with_current(avg_position_first, avg_quaternion_first, current_position, current_quaternion, "removed cameras")

    # Compare current camera with the edited set
    compare_with_current(avg_position_edited, avg_quaternion_edited, current_position, current_quaternion, "removed cameras")
