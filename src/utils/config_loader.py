import yaml
from typing import List
from datetime import datetime
from dataclasses import dataclass



@dataclass
class Paths:
    satellite_root: str
    radar_root: str


@dataclass
class Satellite:
    channels: List[int]
    cadence_minutes: int
    nodata_value: float
    fill_value: float


@dataclass
class Temporal:
    history_minutes: int
    radar_lead_minutes: int


@dataclass
class Spatial:
    mode: str
    height: int
    width: int
    interpolation: str


@dataclass
class SatelliteTransform:
    normalization: str
    clip_min: float
    clip_max: float


@dataclass
class RadarTransform:
    log_transform: bool
    clip_max: float


@dataclass
class Transform:
    satellite: SatelliteTransform
    radar: RadarTransform


@dataclass
class Patch:
    enabled: bool
    height: int
    width: int
    stride: int


@dataclass
class Metadata:
    enabled: bool
    percentiles: List[int]
    datetime_format: str
    csv_path: str


@dataclass
class Dataset:
    years: List[int]


@dataclass
class SplitRange:
    start: datetime
    end: datetime


@dataclass
class Splits:
    train: SplitRange
    val: SplitRange
    test: SplitRange


@dataclass
class DataLoaderCfg:
    batch_size: int
    num_workers: int
    shuffle_train: bool

@dataclass
class NanHandling:
    mode: str
    max_nan_ratio: float


@dataclass
class DataPolicy:
    nan_handling: NanHandling
    mask_nans: bool = True



@dataclass
class Config:
    paths: Paths
    satellite: Satellite
    temporal: Temporal
    spatial: Spatial
    transform: Transform
    patch: Patch
    metadata: Metadata
    dataset: Dataset
    splits: Splits
    dataloader: DataLoaderCfg
    data_policy: DataPolicy


def parse_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def load_config(path: str) -> Config:
    with open(path, "r") as file:
        config_dict = yaml.safe_load(file)

    splits = Splits(
        train=SplitRange(
            start=parse_datetime(config_dict["splits"]["train"]["start"]),
            end=parse_datetime(config_dict["splits"]["train"]["end"]),
        ),
        val=SplitRange(
            start=parse_datetime(config_dict["splits"]["val"]["start"]),
            end=parse_datetime(config_dict["splits"]["val"]["end"]),
        ),
        test=SplitRange(
            start=parse_datetime(config_dict["splits"]["test"]["start"]),
            end=parse_datetime(config_dict["splits"]["test"]["end"]),
        ),
    )

    return Config(
        paths=Paths(**config_dict["paths"]),
        satellite=Satellite(**config_dict["satellite"]),
        temporal=Temporal(**config_dict["temporal"]),
        spatial=Spatial(**config_dict["spatial"]),
        transform=Transform(
            satellite=SatelliteTransform(**config_dict["transform"]["satellite"]),
            radar=RadarTransform(**config_dict["transform"]["radar"]),
        ),
        patch=Patch(**config_dict["patch"]),
        metadata=Metadata(**config_dict["metadata"]),
        dataset=Dataset(**config_dict["dataset"]),
        splits=splits,
        dataloader=DataLoaderCfg(**config_dict["dataloader"]),
        data_policy=DataPolicy(
            nan_handling=NanHandling(**config_dict["data_policy"]["nan_handling"]),
            mask_nans=config_dict["data_policy"].get("mask_nans", True)  # ← Added this line
        ),
    )
