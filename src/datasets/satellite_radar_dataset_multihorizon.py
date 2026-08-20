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
from utils.time_utils_multihorizon import (
    get_satellite_history_and_radar_targets,
    compute_forecast_horizons,
)
from utils.io_utils import load_satellite_channel, load_radar_frame


_STATS_PATH = os.path.join(os.path.dirname(__file__), "../../metadata/channel_stats.json")


def _load_channel_stats(path: str) -> dict:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"channel_stats.json not found at: {abs_path}\n"
            f"Run compute_channel_stats.py first to generate it."
        )
    with open(abs_path, "r") as f:
        raw = json.load(f)

    return {int(k): v for k, v in raw.items()}


CHANNEL_STATS = _load_channel_stats(_STATS_PATH)

RADAR_CONFIG = {
    'clip_max': 128.0,
}


class SoftLogTransform:
    CLIP_MAX  = 128.0
    EPS       = 1.0
    LOG_NORM  = np.log10(CLIP_MAX + EPS)
    ZEROVALUE = 0.02

    def __init__(self, inverse=False, zerovalue=0.01):
        self.inverse   = inverse
        self.zerovalue = zerovalue if zerovalue is not None else self.ZEROVALUE

    def __call__(self, x):
        return self.inv(x) if self.inverse else self.fwd(x)

    def fwd(self, x):
        if torch.is_tensor(x):
            zero_val = x.new_tensor(self.zerovalue)
            x = torch.where(x < zero_val, zero_val, x)
            return torch.log10(x + self.EPS) / self.LOG_NORM
        else:
            x = x.copy()
            x[x < self.zerovalue] = self.zerovalue
            return np.log10(x + self.EPS) / self.LOG_NORM

    def inv(self, y):
        if torch.is_tensor(y):
            mmh = torch.pow(10.0, y.float() * self.LOG_NORM) - self.EPS
            mmh = torch.clamp(mmh, min=0.0, max=self.CLIP_MAX).to(y.dtype)
            zero_val = mmh.new_tensor(self.zerovalue)
            return torch.where(mmh < zero_val, torch.zeros_like(mmh), mmh)
        else:
            mmh = 10.0 ** (y * self.LOG_NORM) - self.EPS
            mmh = np.clip(mmh, 0.0, self.CLIP_MAX)
            mmh[mmh < self.zerovalue] = 0.0
            return mmh


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

        self.radar_log_transform = SoftLogTransform(inverse=False)

        self.horizons = compute_forecast_horizons(config.temporal.radar_lead_minutes)
        self.n_horizons = len(self.horizons)

        print(f"\n{'='*70}")
        print(f" SatelliteRadarDataset Initialized (MULTI-HORIZON)")
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

        print(f"  Forecast horizons: {self.horizons} min")
        print(f"  n_horizons: {self.n_horizons}")

        df = pd.read_csv(metadata_csv)
        print(f"\nInitial samples: {len(df):,}")

        if validate_files:
            cache_file = f"{metadata_csv}.multihorizon_valid_cache.pkl"

            if use_cache and os.path.exists(cache_file):
                print(f"\n Loading cached validation from: {cache_file}")
                with open(cache_file, 'rb') as f:
                    valid_indices = pickle.load(f)
                print(f"   Cached valid samples: {len(valid_indices):,}")
            else:
                print(f"\n Validating file existence (satellite + radar history + ALL horizon targets)...")

                if len(df) > 0:
                    print(f"\n{'='*70}")
                    print(f" DEBUG: Checking FIRST sample")
                    print(f"{'='*70}")

                    row = df.iloc[0]
                    sample_time = datetime.strptime(
                        row["reference_time"],
                        self.config.metadata.datetime_format,
                    )

                    satellite_times, radar_target_times = get_satellite_history_and_radar_targets(
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

                    print(f"Radar targets ({len(radar_target_times)} horizons):")
                    for h, t in zip(self.horizons, radar_target_times):
                        print(f"  t+{h:3d} min → {t}")

                    print(f"{'='*70}\n")

                valid_indices = []

                for idx in tqdm(range(len(df)), desc="Validating"):
                    row = df.iloc[idx]
                    sample_time = datetime.strptime(
                        row["reference_time"],
                        self.config.metadata.datetime_format,
                    )

                    satellite_times, radar_target_times = get_satellite_history_and_radar_targets(
                        sample_time,
                        self.config.satellite.cadence_minutes,
                        self.config.temporal.history_minutes,
                        self.config.temporal.radar_lead_minutes,
                    )

                    from utils.validation_utils_multihorizon import is_valid_sample as is_valid_mh
                    if is_valid_mh(
                        satellite_root=self.config.paths.satellite_root,
                        radar_root=self.config.paths.radar_root,
                        satellite_times=satellite_times,
                        radar_times=radar_target_times,
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

        print(f"  → Target: [{self.n_horizons}, H, W] (one frame per horizon)")
        print(f"{'='*70}\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        sample_time = datetime.strptime(
            row["reference_time"],
            self.config.metadata.datetime_format,
        )

        satellite_times, radar_target_times = get_satellite_history_and_radar_targets(
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

                if self.config.transform.satellite.normalization == "zscore":
                    channel_mean = CHANNEL_STATS[channel_number]['mean']
                    channel_std  = CHANNEL_STATS[channel_number]['std']
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

        satellite_tensor    = torch.from_numpy(np.stack(satellite_planes)).float()
        radar_history_tensor = torch.from_numpy(np.stack(radar_history_planes)).float()
        input_tensor        = torch.cat([satellite_tensor, radar_history_tensor], dim=0)
        input_tensor        = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        target_planes = []
        mask_planes   = []

        for h, radar_target_time in zip(self.horizons, radar_target_times):

            radar_target_path = row[f'radar_path_{h}']

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
                target_planes.append(radar_target)
                mask_planes.append(radar_mask)
            else:
                target_planes.append(radar_target)

        target_tensor = torch.from_numpy(np.stack(target_planes)).float()

        if self.mask_nans:
            mask_tensor = torch.from_numpy(np.stack(mask_planes)).bool()
            return input_tensor, target_tensor, mask_tensor
        else:
            return input_tensor, target_tensor