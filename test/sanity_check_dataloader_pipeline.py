"""
Sanity Check: Verify Dataset Pipeline After Changes
====================================================
This script will:
1. Load samples through your actual dataset class
2. Check if the 12 mm/h clipping is removed
3. Verify transform is working correctly
4. Show actual min/max values being fed to the model
"""

import torch
import numpy as np
from tqdm import tqdm

from utils.config_loader import load_config
from utils.data_policy import DataPolicy
from datasets.satellite_radar_dataset import SatelliteRadarDataset, SoftLogTransform


def test_dataset_pipeline(config_yml, metadata_csv, n_samples=20):
    """
    Test the full dataset pipeline to verify changes worked
    """
    
    print("\n" + "="*80)
    print("DATASET PIPELINE SANITY CHECK")
    print("="*80)
    
    # Load config
    config = load_config(config_yml)
    policy = DataPolicy(
        config.data_policy.nan_handling, 
        mask_nans=config.data_policy.mask_nans
    )
    
    # Create dataset
    print(f"\nLoading dataset from: {metadata_csv}")
    dataset = SatelliteRadarDataset(
        config=config,
        metadata_csv=metadata_csv,
        mask_nans=policy.mask_nans,
        validate_files=False,
        use_cache=False
    )
    
    print(f"Dataset size: {len(dataset):,}")
    
    # Create inverse transform
    inv_transform = SoftLogTransform(eps=1e-3, inverse=True)
    
    # TEST SAMPLES
    print(f"\n" + "="*80)
    print(f"Testing {n_samples} random samples...")
    print("="*80)
    
    # Select random samples
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    all_target_mins = []
    all_target_maxs = []
    all_target_means = []
    
    samples_above_12 = 0
    samples_above_75 = 0
    
    print(f"\n{'Sample':<8} {'Target Min':<12} {'Target Max':<12} {'Target Mean':<12} {'Has >12?':<10} {'Has >75?'}")
    print("-" * 80)
    
    for i, idx in enumerate(tqdm(indices, desc="Processing samples")):
        try:
            # Load sample
            input_tensor, target_tensor, mask_tensor = dataset[idx]
            
            # Convert target back to mm/h
            target_log = target_tensor[0].numpy()
            target_mm_h = inv_transform(target_log)
            
            # Apply mask
            mask = mask_tensor[0].numpy()
            target_masked = np.where(mask, target_mm_h, np.nan)
            
            # Get statistics
            valid_values = target_masked[~np.isnan(target_masked)]
            
            if len(valid_values) > 0:
                min_val = valid_values.min()
                max_val = valid_values.max()
                mean_val = valid_values.mean()
                
                all_target_mins.append(min_val)
                all_target_maxs.append(max_val)
                all_target_means.append(mean_val)
                
                has_above_12 = "YES ✓" if max_val > 12.0 else "NO"
                has_above_75 = "YES ✓" if max_val > 75.0 else "NO"
                
                if max_val > 12.0:
                    samples_above_12 += 1
                if max_val > 75.0:
                    samples_above_75 += 1
                
                print(f"{i+1:<8} {min_val:<12.4f} {max_val:<12.2f} {mean_val:<12.2f} {has_above_12:<10} {has_above_75}")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    

    # SUMMARY STATISTICS    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if len(all_target_maxs) > 0:
        all_target_mins = np.array(all_target_mins)
        all_target_maxs = np.array(all_target_maxs)
        all_target_means = np.array(all_target_means)
        
        print(f"\n Target Statistics (after inverse transform):")
        print(f"   Global min:           {all_target_mins.min():.4f} mm/h")
        print(f"   Global max:           {all_target_maxs.max():.2f} mm/h")
        print(f"   Mean of means:        {all_target_means.mean():.2f} mm/h")
        print(f"   Median of maxes:      {np.median(all_target_maxs):.2f} mm/h")
        
        print(f"\n Percentiles of Max Values:")
        print(f"   P50:                  {np.percentile(all_target_maxs, 50):.2f} mm/h")
        print(f"   P75:                  {np.percentile(all_target_maxs, 75):.2f} mm/h")
        print(f"   P90:                  {np.percentile(all_target_maxs, 90):.2f} mm/h")
        print(f"   P95:                  {np.percentile(all_target_maxs, 95):.2f} mm/h")
        print(f"   P99:                  {np.percentile(all_target_maxs, 99):.2f} mm/h")
        
        print(f"\n Clipping Analysis:")
        print(f"   Samples with max > 12 mm/h:   {samples_above_12}/{len(indices)} ({100*samples_above_12/len(indices):.1f}%)")
        print(f"   Samples with max > 75 mm/h:   {samples_above_75}/{len(indices)} ({100*samples_above_75/len(indices):.1f}%)")
        
        # ============================================================
        # VERDICT
        # ============================================================
        
        print(f"\n" + "="*80)
        print("VERDICT")
        print("="*80)
        
        max_val_overall = all_target_maxs.max()
        
        if max_val_overall <= 12.1:
            print(f"\n PROBLEM DETECTED!")
            print(f"   All data is still clipped at ~12 mm/h")
            print(f"   Maximum value seen: {max_val_overall:.2f} mm/h")
            print(f"\n   Possible issues:")
            print(f"   1. RADAR_CONFIG['clip_max'] still set to 12.0")
            print(f"   2. SoftLogTransform.inv() still has max=12.0")
            print(f"   3. Changes not saved or wrong file edited")
            print(f"\n   ACTION: Double-check the 3 changes in satellite_radar_dataset.py")
        
        elif max_val_overall > 50.0:
            print(f"\n SUCCESS!")
            print(f"   Data pipeline is working correctly!")
            print(f"   Maximum value seen: {max_val_overall:.2f} mm/h")
            print(f"   {100*samples_above_12/len(indices):.1f}% of samples have rain > 12 mm/h")
            print(f"   {100*samples_above_75/len(indices):.1f}% of samples have storms > 75 mm/h")
            print(f"\n  12 mm/h clipping has been successfully removed")
            print(f"    Full intensity range is preserved")
            print(f"    Ready for training!")
        
        else:
            print(f"\n  UNCERTAIN")
            print(f"   Maximum value seen: {max_val_overall:.2f} mm/h")
            print(f"   This is above 12 mm/h but might be low by chance")
            print(f"   Recommendation: Test more samples (increase --n_samples)")
    
    else:
        print("\n No valid samples processed!")
    
    print("\n" + "="*80)
    
    # TEST INVERSE TRANSFORM DIRECTLY
    print("\n" + "="*80)
    print("TESTING INVERSE TRANSFORM DIRECTLY")
    print("="*80)
    
    # Test the transform with known values
    test_values_log = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # log10 space
    test_values_mm_h = inv_transform(test_values_log)
    
    print(f"\nLog10 values → mm/h:")
    for log_val, mm_h_val in zip(test_values_log, test_values_mm_h):
        expected = 10**log_val - 0.001
        print(f"   log10={log_val:.1f} → {mm_h_val:.2f} mm/h (expected: {expected:.2f})")
    
    # Check if capped at 12 or 400
    high_log_value = 2.5  # Should give ~316 mm/h
    high_mm_h = inv_transform(high_log_value)
    expected_high = 10**2.5 - 0.001  # ~316 mm/h
    
    print(f"\nHigh intensity test:")
    print(f"   log10={high_log_value} → {high_mm_h:.2f} mm/h (expected: {expected_high:.2f})")
    
    if high_mm_h < 13.0:
        print(f"    Still clipped at 12 mm/h in transform!")
    elif high_mm_h > 300.0:
        print(f"    Transform allows high values (cap at 400 mm/h)")
    
    print("\n" + "="*80)
    print("Sanity check complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sanity check for dataset pipeline after changes")
    parser.add_argument(
        "--config_yml",
        type=str,
        default="config/config.yml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="metadata/train_sampled.csv",
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of samples to test"
    )
    
    args = parser.parse_args()
    
    test_dataset_pipeline(args.config_yml, args.metadata_csv, args.n_samples)