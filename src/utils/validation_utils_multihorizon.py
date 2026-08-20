import os
from utils.path_utils import satellite_file_paths, radar_file_path


def is_valid_sample(
    satellite_root,
    radar_root,
    satellite_times,
    radar_times,          
    channel_numbers,
    check_radar_history=True,
    debug=False,
):
    for channel_number in channel_numbers:
        for timestamp in satellite_times:
            numpy_path, numpy_zip_path = satellite_file_paths(
                satellite_root,
                timestamp,
                channel_number,
            )
            exists_npy = (
                numpy_path is not None
                and isinstance(numpy_path, str)
                and os.path.exists(numpy_path)
            )
            exists_npz = (
                numpy_zip_path is not None
                and isinstance(numpy_zip_path, str)
                and os.path.exists(numpy_zip_path)
            )
            if not (exists_npy or exists_npz):
                if debug:
                    print(f"  MISSING SATELLITE: Ch{channel_number} at {timestamp}")
                    print(f"    Checked: {numpy_path}")
                return False

    # ----------------------------------------------------------------
    # Check radar history — unchanged from original
    # ----------------------------------------------------------------
    if check_radar_history:
        for timestamp in satellite_times:
            radar_hist_path = radar_file_path(radar_root, timestamp)
            if not os.path.exists(radar_hist_path):
                if debug:
                    print(f"  MISSING RADAR HISTORY at {timestamp}")
                    print(f"    Checked: {radar_hist_path}")
                return False

    for radar_time in radar_times:
        radar_path = radar_file_path(radar_root, radar_time)
        if not os.path.exists(radar_path):
            if debug:
                print(f"  MISSING RADAR TARGET at {radar_time}")
                print(f"    Checked: {radar_path}")
            return False

    return True
