try:
    import argparse
    import pandas as pd
    
    from torch.utils.data import DataLoader
    from utils.data_policy import DataPolicy
    from utils.config_loader import load_config
    from datasets.satellite_radar_dataset import SatelliteRadarDataset
except ImportError as e:
    print(f"ImportError: Failed to import required dependencies.")
    print(f"Details: {e}")
    print("\nPlease ensure the following are installed:")
    print("  - pandas: pip install pandas")
    print("  - torch: pip install torch")
    print("\nAlso verify that the following modules exist in your project:")
    print("  - utils/data_policy.py")
    print("  - utils/config_loader.py")
    print("  - datasets/satellite_radar_dataset.py")
    print("\nIf issues persist, check your Python path and project structure.")
    exit(1)


def make_dataloaders(config_path):
    config = load_config(config_path)
    
    df = pd.read_csv("metadata/all_samples.csv")
    
    mask_nans = config.data_policy.mask_nans
    print(f"mask_nans from config: {mask_nans}")
    
    policy = DataPolicy(
        config.data_policy.nan_handling,
        mask_nans=mask_nans,
    )
    df = policy.apply(df)
    
    loaders = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        csv_path = f"metadata/{split}.csv"
        split_df.to_csv(csv_path, index=False)
        
        dataset = SatelliteRadarDataset(
            config=config,
            metadata_csv=csv_path,
            mask_nans=policy.mask_nans,
        )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=(split == "train"),
            num_workers=config.dataloader.num_workers,
        )
    
    print("\nDataloaders created with:")
    print(f"  nan_handling.mode = {policy.mode}")
    print(f"  mask_nans         = {policy.mask_nans}")
    
    return loaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dataloaders for satellite-radar weather prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yml",
        help="Path to the configuration YAML file"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Creating DataLoaders")
    print("="*60)
    print(f"Configuration file: {args.config}\n")
    
    loaders = make_dataloaders(args.config)
    
    print("\n" + "="*60)
    print("DataLoader Summary")
    print("="*60)
    print(f"Train: {len(loaders['train'])} batches, "
          f"{len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['val'])} batches, "
          f"{len(loaders['val'].dataset)} samples")
    print(f"Test:  {len(loaders['test'])} batches, "
          f"{len(loaders['test'].dataset)} samples")
    print("="*60)


if __name__ == "__main__":
    main()