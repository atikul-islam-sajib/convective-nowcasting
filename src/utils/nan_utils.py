import numpy as np


def mask_and_fill_nans(array: np.ndarray):
    """
    Replace NaNs with zero for numerical safety and return a validity mask.

    Parameters
    ----------
    array : np.ndarray
        Input array AFTER all spatial transforms.

    Returns
    -------
    array_filled : np.ndarray
        Array with NaNs replaced by zero.
    valid_mask : np.ndarray (bool)
        Boolean mask where True indicates originally valid values.
    """
    valid_mask = ~np.isnan(array)
    array_filled = np.nan_to_num(array, nan=0.0)
    return array_filled, valid_mask