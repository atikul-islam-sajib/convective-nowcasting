import numpy as np


def mask_and_fill_nans(array: np.ndarray):
    valid_mask = ~np.isnan(array)
    array_filled = np.nan_to_num(array, nan=0.0)
    return array_filled, valid_mask
