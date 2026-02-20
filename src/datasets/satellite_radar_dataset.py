"""
Satellite–Radar Temporal Dataset for Convective Nowcasting.

This module implements a PyTorch Dataset for learning radar nowcasting
from multi-channel satellite history and radar history.

For each sample, the dataset:
    1. Computes temporal satellite and radar history timestamps.
    2. Validates availability of all required files.
    3. Loads satellite channels and radar history.
    4. Applies normalization (no clipping on satellite — full dynamic range preserved).
    5. Applies spatial resizing/interpolation.
    6. Applies logarithmic radar transformation.
    7. Handles NaN/Inf values robustly.
    8. Returns tensors suitable for deep learning.

Key Features
------------
- Temporal history stacking.
- Multi-channel satellite support.
- Radar history as auxiliary input.
- Cached validation of file existence.
- Debug validation mode.
- Robust NaN handling.
- Optional normalization modes.
- Spatial resampling support.

Outputs per sample:
    - Input tensor: [C_in, H, W]
    - Target tensor: [1, H, W]
    - Optional mask tensor: [1, H, W]

Typical Usage
-------------
>>> dataset = SatelliteRadarDataset(
>>>     config,
>>>     "metadata/train.csv",
>>>     validate_files=True
>>> )

>>> loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""


import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

from utils.spatial_utils import apply_spatial
from utils.nan_utils import mask_and_fill_nans
from utils.transform_utils import clip, zscore, minmax
from utils.path_utils import satellite_file_paths, radar_file_path
from utils.time_utils import get_satellite_history_and_radar_target
from utils.io_utils import load_satellite_channel, load_radar_frame


# Load CHANNEL_STATS from JSON (computed by compute_channel_stats.py)
# JSON keys are strings, so we convert back to int for channel lookup.
_STATS_PATH = os.path.join(os.path.dirname(__file__), "../../metadata/channel_stats.json")

def _load_channel_stats(path: str) -> dict:
    """Load channel mean/std from JSON produced by compute_channel_stats.py."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"channel_stats.json not found at: {abs_path}\n"
            f"Run compute_channel_stats.py first to generate it."
        )
    with open(abs_path, "r") as f:
        raw = json.load(f)
    # JSON keys are strings — convert to int
    return {int(k): v for k, v in raw.items()}

CHANNEL_STATS = _load_channel_stats(_STATS_PATH)

RADAR_CONFIG = {
    'clip_max': 400,
}


class SoftLogTransform:
    """
    Logarithmic transformation for radar reflectivity/precipitation values.

    Applies a numerically stable log10 transform with epsilon offset and optional inverse transformation.

    Forward transform:
        y = log10(x + eps)

    Inverse transform:
        x = (10^y - eps) * scale

    Parameters
    ----------
    eps : float, optional
        Small constant added to input to avoid log(0).
        Default is 1e-3.
    inverse : bool, optional
        If True, apply inverse transform.
        Default is False.

    Notes
    -----
    The inverse transform clamps output to [0, 400] mm/hr.
    """
    def __init__(self, eps=1e-3, inverse=False):
        self.eps = eps
        self.inverse = inverse

    def __call__(self, x):
        return self.inv(x) if self.inverse else self.fwd(x)

    def fwd(self, x):
        if torch.is_tensor(x):
            return torch.log10(x + self.eps)
        else:
            return np.log10(x + self.eps)

    def inv(self, y):
        if torch.is_tensor(y):
            return torch.clamp((10.0**y - self.eps) * 12.0, min=0.0, max=400.0)
        else:
            return np.clip((10.0**y - self.eps) * 12.0, 0.0, 400.0)


def is_valid_sample_debug(
    satellite_root,
    radar_root,
    satellite_times,
    radar_time,
    channel_numbers,
):
    """
    Validate file availability for a sample with debug information.

    Checks existence of:
        - All satellite history channels.
        - Radar history files.
        - Radar target file.

    Returns detailed error messages for debugging.

    Parameters
    ----------
    satellite_root : str
        Root directory for satellite data.
    radar_root : str
        Root directory for radar data.
    satellite_times : list of datetime
        List of satellite history timestamps.
    radar_time : datetime
        Radar target timestamp.
    channel_numbers : list of int
        Satellite channel identifiers.

    Returns
    -------
    tuple
        (is_valid, error_message)

        is_valid : bool
            True if all required files exist.
        error_message : str or None
            Detailed error message if invalid.
    """
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
                return False, f"Missing satellite Ch{channel_number} at {timestamp}\n  Path: {numpy_path}"
    
    for timestamp in satellite_times:
        radar_hist_path = radar_file_path(radar_root, timestamp)
        if not os.path.exists(radar_hist_path):
            return False, f"Missing radar history at {timestamp}\n  Path: {radar_hist_path}"
    
    radar_path = radar_file_path(radar_root, radar_time)
    if not os.path.exists(radar_path):
        return False, f"Missing radar target at {radar_time}\n  Path: {radar_path}"
    
    return True, None


def is_valid_sample_fast(
    satellite_root,
    radar_root,
    satellite_times,
    radar_time,
    channel_numbers,
):
    """
    Fast file existence validation without debug output.

    Performs lightweight existence checks for satellite history,
    radar history, and radar target files.

    Intended for large-scale validation loops.

    Parameters
    ----------
    satellite_root : str
        Root directory for satellite data.
    radar_root : str
        Root directory for radar data.
    satellite_times : list of datetime
        List of satellite history timestamps.
    radar_time : datetime
        Radar target timestamp.
    channel_numbers : list of int
        Satellite channel identifiers.

    Returns
    -------
    bool
        True if all required files exist, False otherwise.
    """
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
                return False
    
    for timestamp in satellite_times:
        radar_hist_path = radar_file_path(radar_root, timestamp)
        if not os.path.exists(radar_hist_path):
            return False
    
    radar_path = radar_file_path(radar_root, radar_time)
    if not os.path.exists(radar_path):
        return False
    
    return True


class SatelliteRadarDataset(Dataset):
    """
    Temporal satellite–radar dataset for precipitation nowcasting.

    Each sample consists of:
        - Multi-channel satellite history.
        - Radar history sequence.
        - Radar target field.

    Input tensor structure:
        [sat_channels + radar_history_channels, H, W]

    Target tensor structure:
        [1, H, W]

    Optional mask tensor indicates valid target pixels.

    Parameters
    ----------
    config : object
        Configuration object loaded from YAML.
    metadata_csv : str
        Path to CSV file containing sample timestamps.
    mask_nans : bool, optional
        If True, mask and fill NaNs in target.
        Default is True.
    mean : float or None, optional
        Optional global mean (unused by default).
    std : float or None, optional
        Optional global standard deviation.
    validate_files : bool, optional
        If True, validate file existence before loading.
        Default is True.
    use_cache : bool, optional
        If True, cache validation results.
        Default is True.

    Attributes
    ----------
    df : pandas.DataFrame
        Filtered metadata table.
    radar_log_transform : SoftLogTransform
        Radar logarithmic transformation instance.

    Notes
    -----
    - Validation results are cached to disk.
    - Debug mode prints detailed file paths.
    - All NaNs/Inf values in input are replaced with zero.
    - Target NaNs are optionally masked.
    - Satellite data is NOT clipped — full dynamic range is preserved
      for convective storm detection (cold cloud tops matter).
    """

    def __init__(
        self,
        config,
        metadata_csv,
        *,
        mask_nans: bool = True,
        mean=None,
        std=None,
        validate_files: bool = True,
        use_cache: bool = True,
    ):
        self.config = config
        self.mask_nans = mask_nans
        self.mean = mean
        self.std = std
        
        self.radar_log_transform = SoftLogTransform(eps=1e-3, inverse=False)
        
        print(f"\n{'='*70}")
        print(f" SatelliteRadarDataset Initialized (TEMPORAL WITH RADAR HISTORY)")
        print(f"{'='*70}")
        print(f"Metadata: {metadata_csv}")
        print(f"Satellite Configuration:")
        for ch in self.config.satellite.channels:
            print(f"  Ch{ch}: mean={CHANNEL_STATS[ch]['mean']:.2f}, std={CHANNEL_STATS[ch]['std']:.2f} (no clipping)")
        print(f"Radar Configuration:")
        print(f"  Clip: [0, {RADAR_CONFIG['clip_max']:.2f}] mm/hr")
        print(f"  Transform: log10(x + 0.001)")
        print(f"Temporal Configuration:")
        print(f"  History: {self.config.temporal.history_minutes} min")
        print(f"  Cadence: {self.config.satellite.cadence_minutes} min")
        print(f"  Lead time: {self.config.temporal.radar_lead_minutes} min")
        
        df = pd.read_csv(metadata_csv)
        print(f"\nInitial samples: {len(df):,}")
        
        if validate_files:
            cache_file = f"{metadata_csv}.temporal_valid_cache.pkl"
            
            if use_cache and os.path.exists(cache_file):
                print(f"\n Loading cached validation from: {cache_file}")
                with open(cache_file, 'rb') as f:
                    valid_indices = pickle.load(f)
                print(f"   Cached valid samples: {len(valid_indices):,}")
            else:
                print(f"\n Validating file existence (satellite + radar history + target)...")
                
                if len(df) > 0:
                    print(f"\n{'='*70}")
                    print(f" DEBUG: Checking FIRST sample")
                    print(f"{'='*70}")
                    
                    row = df.iloc[0]
                    sample_time = datetime.strptime(
                        row["reference_time"],
                        self.config.metadata.datetime_format,
                    )
                    
                    satellite_times, radar_target_time = get_satellite_history_and_radar_target(
                        sample_time,
                        self.config.satellite.cadence_minutes,
                        self.config.temporal.history_minutes,
                        self.config.temporal.radar_lead_minutes,
                    )
                    
                    print(f"Reference time: {sample_time}")
                    print(f"Satellite times ({len(satellite_times)} timesteps):")
                    for i, t in enumerate(satellite_times[:3]):
                        print(f"  {i+1}. {t}")
                    print(f"  ... (showing first 3)")
                    print(f"Radar target: {radar_target_time}")
                    
                    print(f"\nChecking files...")
                    is_valid, error_msg = is_valid_sample_debug(
                        satellite_root=self.config.paths.satellite_root,
                        radar_root=self.config.paths.radar_root,
                        satellite_times=satellite_times,
                        radar_time=radar_target_time,
                        channel_numbers=self.config.satellite.channels,
                    )
                    
                    if is_valid:
                        print(f" First sample is VALID - all files exist!")
                    else:
                        print(f" First sample is INVALID:")
                        print(f"   {error_msg}")
                    
                    print(f"{'='*70}\n")
                
                valid_indices = []
                
                for idx in tqdm(range(len(df)), desc="Validating"):
                    row = df.iloc[idx]
                    sample_time = datetime.strptime(
                        row["reference_time"],
                        self.config.metadata.datetime_format,
                    )
                    
                    satellite_times, radar_target_time = get_satellite_history_and_radar_target(
                        sample_time,
                        self.config.satellite.cadence_minutes,
                        self.config.temporal.history_minutes,
                        self.config.temporal.radar_lead_minutes,
                    )
                    
                    if is_valid_sample_fast(
                        satellite_root=self.config.paths.satellite_root,
                        radar_root=self.config.paths.radar_root,
                        satellite_times=satellite_times,
                        radar_time=radar_target_time,
                        channel_numbers=self.config.satellite.channels,
                    ):
                        valid_indices.append(idx)
                
                if use_cache:
                    print(f" Caching validation results to: {cache_file}")
                    with open(cache_file, 'wb') as f:
                        pickle.dump(valid_indices, f)
            
            self.df = df.iloc[valid_indices].reset_index(drop=True)
            
            n_discarded = len(df) - len(self.df)
            print(f"\n Validation Results:")
            print(f"   Valid samples:    {len(self.df):,} ({100*len(self.df)/len(df):.1f}%)")
            print(f"   Discarded:        {n_discarded:,} ({100*n_discarded/len(df):.1f}%)")
        else:
            self.df = df
        
        n_timesteps = self.config.temporal.history_minutes // self.config.satellite.cadence_minutes
        n_sat_channels = n_timesteps * len(self.config.satellite.channels)
        n_radar_channels = n_timesteps * 1
        print(f"\n  → {n_timesteps} timesteps")
        print(f"  → Satellite: {n_sat_channels} channels")
        print(f"  → Radar history: {n_radar_channels} channels")
        print(f"  → Total input: {n_sat_channels + n_radar_channels} channels")
        print(f"{'='*70}\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
     row = self.df.iloc[index]

     sample_time = datetime.strptime(
        row["reference_time"],
        self.config.metadata.datetime_format,
    )

     satellite_times, radar_target_time = get_satellite_history_and_radar_target(
        sample_time,
        self.config.satellite.cadence_minutes,
        self.config.temporal.history_minutes,
        self.config.temporal.radar_lead_minutes,
    )

     satellite_planes = []
     for channel_number in self.config.satellite.channels:
        for sat_time in satellite_times:
            numpy_path, _ = satellite_file_paths(
                self.config.paths.satellite_root,
                sat_time,
                channel_number,
            )

            sat_array = load_satellite_channel(
                numpy_path,
                None,
                self.config.satellite.nodata_value,
                self.config.satellite.fill_value,
            )

            # No clipping — full dynamic range preserved for convective detection
            if self.config.transform.satellite.normalization == "zscore":
                channel_mean = CHANNEL_STATS[channel_number]['mean']
                channel_std = CHANNEL_STATS[channel_number]['std']
                sat_array = zscore(sat_array, channel_mean, channel_std)

            sat_array = apply_spatial(
                sat_array,
                self.config.spatial.mode,
                self.config.spatial.height,
                self.config.spatial.width,
                self.config.spatial.interpolation,
            )

            satellite_planes.append(sat_array)

     radar_history_planes = []
     for radar_hist_time in satellite_times:
        radar_hist_path = radar_file_path(
            self.config.paths.radar_root,
            radar_hist_time
        )
        
        radar_hist = load_radar_frame(radar_hist_path)
        radar_hist = clip(radar_hist, 0.0, RADAR_CONFIG['clip_max'])
        radar_hist = self.radar_log_transform(radar_hist)
        radar_hist = apply_spatial(
            radar_hist,
            self.config.spatial.mode,
            self.config.spatial.height,
            self.config.spatial.width,
            self.config.spatial.interpolation,
        )
        radar_history_planes.append(radar_hist)

     satellite_tensor = torch.from_numpy(np.stack(satellite_planes)).float()
     radar_history_tensor = torch.from_numpy(np.stack(radar_history_planes)).float()
     input_tensor = torch.cat([satellite_tensor, radar_history_tensor], dim=0)
    
     input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)

     radar_target_path = radar_file_path(
        self.config.paths.radar_root,
        radar_target_time
    )
    
     radar_target = load_radar_frame(radar_target_path)
     radar_target = clip(radar_target, 0.0, RADAR_CONFIG['clip_max'])
     radar_target = self.radar_log_transform(radar_target)
     radar_target = apply_spatial(
        radar_target,
        self.config.spatial.mode,
        self.config.spatial.height,
        self.config.spatial.width,
        self.config.spatial.interpolation,
    )

     if self.mask_nans:
        radar_target, radar_mask = mask_and_fill_nans(radar_target)
        target_tensor = torch.from_numpy(radar_target[None]).float()
        mask_tensor = torch.from_numpy(radar_mask[None]).bool()
        return input_tensor, target_tensor, mask_tensor
     else:
        target_tensor = torch.from_numpy(radar_target[None]).float()
        return input_tensor, target_tensor