"""
Path utilities for satellite and radar files
"""
import os


def satellite_file_paths(root_directory, timestamp, channel_number):
    """
    Matches real satellite files like:
    satellite_de/2023/01/01/003000_CH7.npz
    """
    year = timestamp.strftime("%Y")
    month = timestamp.strftime("%m")
    day = timestamp.strftime("%d")
    time_str = timestamp.strftime("%H%M%S")
    filename = f"{time_str}_CH{channel_number}.npz"
    path = os.path.join(
        root_directory,
        year,
        month,
        day,
        filename,
    )
    return path, None  # second value kept for compatibility

def radar_file_path(root_directory, timestamp):
    """
    ✅ CORRECT: Matches real radar files like:
    radar_de/2201/01/220101_1550.npy
    (WITH /DD/ subdirectory!)
    """
    yymm = timestamp.strftime("%y%m")
    dd = timestamp.strftime("%d")
    filename = f"{timestamp.strftime('%y%m%d_%H%M')}.npy"
    return os.path.join(root_directory, yymm, dd, filename)