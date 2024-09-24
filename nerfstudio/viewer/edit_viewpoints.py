import os
import subprocess
import platform
from pathlib import Path
from datetime import datetime

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
def generate_colmap(path: Path) -> None:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = str(path).replace("/images", f"_{current_time}")
    
    print("Calculating new camera positions: ")
    
    process_data = [
        "ns-process-data",
        "images",
        "--data",
        str(path),
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
    
    subprocess.Popen(train_data)
