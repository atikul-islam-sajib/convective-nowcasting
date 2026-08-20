import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import Dataset

from utils.nan_utils import mask_and_fill_nans
from utils.transform_utils import clip, zscore
from utils.io_utils import load_satellite_channel, load_radar_frame


_STATS_PATH = os.path.join(
    os.path.dirname(__file__), "../../metadata/channel_stats.json"
)


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

RADAR_CONFIG = {'clip_max': 128.0}

PATCH_SIZE = 256


class SoftLogTransform:
    CLIP_MAX  = 128.0
    EPS       = 1.0
    LOG_NORM  = np.log10(CLIP_MAX + EPS)
    ZEROVALUE = 0.0

    def __init__(self, inverse=False, zerovalue=0.0):
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


def _radar_path(radar_root, dt):
    return os.path.join(
        radar_root,
        dt.strftime('%y%m'),
        dt.strftime('%d'),
        dt.strftime('%y%m%d_%H%M') + '.npy',
    )


def _satellite_path(sat_root, dt, channel):
    return os.path.join(
        sat_root,
        dt.strftime('%Y'),
        dt.strftime('%m'),
        dt.strftime('%d'),
        dt.strftime('%H%M%S') + f'_{channel}.npy',
    )


def _get_history_times(current_time, num_frames, stride_minutes):
    start = current_time - (num_frames - 1) * timedelta(minutes=stride_minutes)
    return [start + timedelta(minutes=stride_minutes) * i for i in range(num_frames)]


def _get_target_times(current_time, horizons_minutes):
    return [current_time + timedelta(minutes=h) for h in horizons_minutes]


class SatelliteRadarPatchDataset(Dataset):

    def __init__(
        self,
        config,
        metadata_csv: str,
        *,
        mask_nans: bool = True,
        validate_files: bool = True,
        use_cache: bool = True,
        patch_size: int = PATCH_SIZE,
        radar_only: bool = False,
    ):
        self.config      = config
        self.mask_nans   = mask_nans
        self.patch_size  = patch_size
        self.radar_only  = radar_only

        self.radar_log_transform = SoftLogTransform(inverse=False)

        from utils.time_utils_multihorizon import compute_forecast_horizons
        self.horizons = compute_forecast_horizons(config.temporal.radar_lead_minutes)
        self.n_horizons = len(self.horizons)

        self.num_in_frames = (
            config.temporal.history_minutes // config.satellite.cadence_minutes
        )

        raw_channels = config.satellite.channels
        self.channels = [
            f'CH{c}' if not str(c).startswith('CH') else str(c)
            for c in raw_channels
        ]
        self.channel_ints = [
            int(str(c).replace('CH', '')) for c in self.channels
        ]

        print(f"\n{'='*70}")
        print(f" SatelliteRadarPatchDataset (PATCH-BASED MULTI-HORIZON)")
        print(f"{'='*70}")
        print(f" Metadata CSV:     {metadata_csv}")
        print(f" Radar root:       {config.paths.radar_root}")
        print(f" Satellite root:   {config.paths.satellite_root}")
        print(f" Patch size:       {patch_size}x{patch_size} (native 1km/pixel)")
        print(f" Input frames:     {self.num_in_frames} × {config.satellite.cadence_minutes}min")
        print(f" Horizons:         {self.horizons} min")
        print(f" n_horizons:       {self.n_horizons}")
        print(f" Radar clip max:   {RADAR_CONFIG['clip_max']} mm/h")
        print(f" Transform:        log10(x + 0.001)")
        if self.radar_only:
            print(f" Channels:         RADAR ONLY (satellite disabled)")
        else:
            print(f" Channels:         {self.channels}")
            for ch_str, ch_int in zip(self.channels, self.channel_ints):
                if ch_int in CHANNEL_STATS:
                    m = CHANNEL_STATS[ch_int]['mean']
                    s = CHANNEL_STATS[ch_int]['std']
                    print(f"  {ch_str}: mean={m:.2f}, std={s:.2f}")
        print(f"{'='*70}")

        df = pd.read_csv(metadata_csv)
        print(f" Initial samples: {len(df):,}")

        required_cols = ['datetime', 'patch_row', 'patch_col']
        for h in self.horizons:
            required_cols += [f'radar_time_{h}', f'radar_path_{h}']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Metadata CSV missing columns: {missing}\n"
                f"Available: {list(df.columns)}"
            )

        if validate_files:
            cache_file = f"{metadata_csv}.patch_valid_cache.pkl"

            if use_cache and os.path.exists(cache_file):
                print(f" Loading cached validation: {cache_file}")
                with open(cache_file, 'rb') as f:
                    valid_indices = pickle.load(f)
                print(f" Cached valid samples: {len(valid_indices):,}")
            else:
                print(f" Validating file existence...")
                valid_indices = []
                for idx in tqdm(range(len(df)), desc="Validating"):
                    row = df.iloc[idx]
                    if self._validate_row(row):
                        valid_indices.append(idx)
                if use_cache:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(valid_indices, f)
                    print(f" Cached to: {cache_file}")

            self.df = df.iloc[valid_indices].reset_index(drop=True)
            n_disc = len(df) - len(self.df)
            print(f" Valid:     {len(self.df):,} ({100*len(self.df)/max(len(df),1):.1f}%)")
            print(f" Discarded: {n_disc:,}")
        else:
            self.df = df

        n_radar_ch = self.num_in_frames
        if self.radar_only:
            n_in_ch  = n_radar_ch
            sat_line = f"   Satellite:   DISABLED (radar_only=True)"
        else:
            n_sat_ch = self.num_in_frames * len(self.channels)
            n_in_ch  = n_sat_ch + n_radar_ch
            sat_line = f"   Satellite:   {n_sat_ch} channels ({len(self.channels)} ch × {self.num_in_frames} frames)"

        print(f"\n Input tensor:  [{n_in_ch}, {patch_size}, {patch_size}]")
        print(sat_line)
        print(f"   Radar hist:  {n_radar_ch} channels (1 ch × {self.num_in_frames} frames)")
        print(f" Target tensor: [{self.n_horizons}, {patch_size}, {patch_size}]")
        print(f" Total samples: {len(self.df):,}")
        print(f"{'='*70}\n")

    def _validate_row(self, row):
        dt = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
        history_times = _get_history_times(
            dt, self.num_in_frames, self.config.satellite.cadence_minutes
        )

        for t in history_times:
            if not os.path.exists(_radar_path(self.config.paths.radar_root, t)):
                return False

        if not self.radar_only:
            for ch in self.channels:
                for t in history_times:
                    if not os.path.exists(_satellite_path(self.config.paths.satellite_root, t, ch)):
                        return False

        for h in self.horizons:
            if not os.path.exists(row[f'radar_path_{h}']):
                return False

        return True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        dt = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
        pi = int(row['patch_row'])
        pj = int(row['patch_col'])
        ps = self.patch_size

        history_times = _get_history_times(
            dt, self.num_in_frames, self.config.satellite.cadence_minutes
        )

        if pi + ps > 1100 or pj + ps > 900:
            raise ValueError(
                f"Patch out of bounds: pi={pi}, pj={pj}, patch_size={ps} "
                f"exceeds grid [1100, 900]"
            )

        satellite_planes = []

        if not self.radar_only:
            for ch_str, ch_int in zip(self.channels, self.channel_ints):
                for t in history_times:
                    sat_path = _satellite_path(
                        self.config.paths.satellite_root, t, ch_str
                    )
                    try:
                        sat_array = load_satellite_channel(
                            sat_path,
                            None,
                            self.config.satellite.nodata_value,
                            self.config.satellite.fill_value,
                        )
                    except Exception:
                        sat_array = np.full((1100, 900), np.nan, dtype=np.float32)

                    sat_patch = sat_array[pi:pi + ps, pj:pj + ps].astype(np.float32)

                    if self.config.transform.satellite.normalization == "zscore":
                        if ch_int in CHANNEL_STATS:
                            mean = CHANNEL_STATS[ch_int]['mean']
                            std  = CHANNEL_STATS[ch_int]['std']
                            sat_patch = zscore(sat_patch, mean, std)

                    satellite_planes.append(sat_patch)

        radar_history_planes = []

        for t in history_times:
            rpath = _radar_path(self.config.paths.radar_root, t)
            try:
                radar_hist = load_radar_frame(rpath)
            except Exception:
                radar_hist = np.full((1100, 900), np.nan, dtype=np.float32)

            radar_hist = np.nan_to_num(radar_hist, nan=0.0, posinf=0.0, neginf=0.0)
            radar_hist = clip(radar_hist, 0.0, RADAR_CONFIG['clip_max'])
            radar_hist = self.radar_log_transform(radar_hist)
            radar_patch = radar_hist[pi:pi + ps, pj:pj + ps].astype(np.float32)
            radar_history_planes.append(radar_patch)

        radar_history_tensor = torch.from_numpy(
            np.stack(radar_history_planes, axis=0)
        ).float()

        if self.radar_only:
            input_tensor = radar_history_tensor
        else:
            satellite_tensor = torch.from_numpy(
                np.stack(satellite_planes, axis=0)
            ).float()
            input_tensor = torch.cat(
                [satellite_tensor, radar_history_tensor], dim=0
            )

        input_tensor = torch.nan_to_num(
            input_tensor, nan=0.0, posinf=0.0, neginf=0.0
        )

        target_planes = []
        mask_planes   = []

        for h in self.horizons:
            rpath = row[f'radar_path_{h}']
            try:
                radar_target = load_radar_frame(rpath)
            except Exception:
                radar_target = np.full((1100, 900), np.nan, dtype=np.float32)

            radar_target = np.nan_to_num(radar_target, nan=0.0, posinf=0.0, neginf=0.0)
            radar_target = clip(radar_target, 0.0, RADAR_CONFIG['clip_max'])
            radar_target = self.radar_log_transform(radar_target)
            target_patch = radar_target[pi:pi + ps, pj:pj + ps].astype(np.float32)

            if self.mask_nans:
                target_patch, target_mask = mask_and_fill_nans(target_patch)
                target_planes.append(target_patch)
                mask_planes.append(target_mask)
            else:
                target_planes.append(target_patch)

        target_tensor = torch.from_numpy(
            np.stack(target_planes, axis=0)
        ).float()

        if self.mask_nans:
            mask_tensor = torch.from_numpy(
                np.stack(mask_planes, axis=0)
            ).bool()
            return input_tensor, target_tensor, mask_tensor
        else:
            return input_tensor, target_tensor