import numpy as np


def compute_metadata(array, percentiles):
    valid_mask = np.isfinite(array)
    if valid_mask.sum() == 0:
        return None

    valid_values = array[valid_mask]

    metadata = {
        "valid_cells": int(valid_mask.sum()),
        "max(x)": float(valid_values.max()),
        "sum(x)": float(valid_values.sum()),
        "sum(x**2)": float((valid_values ** 2).sum()),
        "log(sum(x))": float(np.log1p(valid_values.sum())),
        "log(sum(x**2))": float(np.log1p((valid_values ** 2).sum())),
    }

    for percentile in percentiles:
        metadata[f"p{percentile}(x)"] = float(
            np.percentile(valid_values, percentile)
        )

    return metadata