import numpy as np
from scipy.ndimage import label
from skimage.morphology import closing, remove_small_objects, disk


def detect_storms_two_level(
    rain_mm_h,
    operational_thr,
    extreme_thr,
    min_pixels=10,
    disk_size=4,
    min_area_km2=10.0,
    max_area_km2=100.0,
    pixel_area_km2=1.0,
):
    valid = np.isfinite(rain_mm_h)
    if valid.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0

    mask_operational = valid & (rain_mm_h >= operational_thr)
    if mask_operational.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0

    mask_operational = closing(mask_operational, disk(disk_size))
    labeled_op, n_candidates = label(mask_operational)

    final_operational_mask = np.zeros_like(mask_operational, dtype=bool)
    n_operational_storms = 0

    for region_id in range(1, n_candidates + 1):
        region = labeled_op == region_id
        area_km2 = region.sum() * pixel_area_km2
        if min_area_km2 <= area_km2 <= max_area_km2:
            final_operational_mask[region] = True
            n_operational_storms += 1

    mask_extreme = final_operational_mask & (rain_mm_h >= extreme_thr)
    if mask_extreme.sum() == 0:
        return final_operational_mask, mask_extreme, n_operational_storms, 0

    mask_extreme = closing(mask_extreme, disk(2))
    mask_extreme = remove_small_objects(mask_extreme, min_size=10)
    labeled_ext, n_extreme_cores = label(mask_extreme)

    return final_operational_mask, mask_extreme, n_operational_storms, n_extreme_cores
