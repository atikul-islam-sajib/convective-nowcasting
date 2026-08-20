import os
import numpy as np


def load_satellite_channel(numpy_path, numpy_zip_path, nodata_value, fill_value):

    if numpy_path is None:
        raise FileNotFoundError("Satellite path is None")

    if not os.path.exists(numpy_path):
        raise FileNotFoundError(f"Satellite file missing: {numpy_path}")

    if numpy_path.endswith(".npz"):
        with np.load(numpy_path) as archive:
            keys = list(archive.keys())
            if len(keys) == 0:
                raise ValueError(f"No arrays found in {numpy_path}")
            array = archive[keys[0]]
    else:
        array = np.load(numpy_path)

    array = array.astype("float32")
    array[array == nodata_value] = fill_value

    return array


def load_radar_frame(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Radar file missing: {path}")
    array = np.load(path).astype("float32")
    array[array > 150.0] = np.nan
    return array