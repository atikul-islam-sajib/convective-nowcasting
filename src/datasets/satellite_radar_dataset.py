import torch
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.spatial_utils import apply_spatial
from utils.path_utils import satellite_file_paths, radar_file_path
from utils.time_utils import get_satellite_history_and_radar_target
from utils.io_utils import load_satellite_channel, load_radar_frame
from utils.transform_utils import clip, zscore, minmax
from utils.nan_utils import mask_and_fill_nans


CHANNEL_STATS = {
    7: {'mean': 31.43, 'std': 16.41, 'clip_min': 9.00, 'clip_max': 63.63},
    9: {'mean': 56.38, 'std': 25.80, 'clip_min': 19.89, 'clip_max': 103.75},
}

RADAR_CONFIG = {
    'clip_max': 128,
}


class SoftLogTransform:
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
            return torch.clamp(torch.pow(10.0, y.float()) - self.eps, min=0.0, max=128.0)
        else:
            return np.clip(10.0**y - self.eps, 0.0, 128.0)


def is_valid_sample_debug(
    satellite_root,
    radar_root,
    satellite_times,
    radar_time,
    channel_numbers,
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
            print(f"  Ch{ch}: clip=[{CHANNEL_STATS[ch]['clip_min']:.2f}, {CHANNEL_STATS[ch]['clip_max']:.2f}], "
                  f"mean={CHANNEL_STATS[ch]['mean']:.2f}, std={CHANNEL_STATS[ch]['std']:.2f}")
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

            sat_array = clip(
                sat_array,
                CHANNEL_STATS[channel_number]['clip_min'],
                CHANNEL_STATS[channel_number]['clip_max'],
            )

            if self.config.transform.satellite.normalization == "zscore":
                channel_mean = CHANNEL_STATS[channel_number]['mean']
                channel_std = CHANNEL_STATS[channel_number]['std']
                sat_array = zscore(sat_array, channel_mean, channel_std)
            elif self.config.transform.satellite.normalization == "minmax":
                sat_array = minmax(
                    sat_array,
                    CHANNEL_STATS[channel_number]['clip_min'],
                    CHANNEL_STATS[channel_number]['clip_max'],
                )

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