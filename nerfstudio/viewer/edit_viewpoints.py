import os
import subprocess
import platform
from pathlib import Path

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
