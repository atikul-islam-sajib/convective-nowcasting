"""
custom_samplers.py

Custom PyTorch samplers for handling imbalanced precipitation data.
Includes two strategies for oversampling rare rainy samples during training.
"""

import numpy as np
from torch.utils.data import Sampler


# ============================================================
# Helper Function for Printing Bin Statistics
# ============================================================

def print_bincount(unique_bins, counts, bin_edges, p_bins=None, draws=None):
    """
    Print a formatted table showing bin distribution and sampling probabilities.
    
    Parameters
    ----------
    unique_bins : np.ndarray
        Array of bin indices that contain samples
    counts : np.ndarray
        Number of samples in each bin
    bin_edges : np.ndarray
        Bin edge values
    p_bins : np.ndarray, optional
        Sampling probability for each bin
    draws : int, optional
        Total number of draws (for expected count calculation)
    """
    bin_edges_str = [f"<={be:.2f}" for be in bin_edges[1:]]
    all_bins = np.arange(len(bin_edges) - 1)

    # Actual counts per bin (expand to all bins)
    all_counts = np.zeros_like(all_bins, dtype=np.int64)
    all_counts[unique_bins - 1] = counts

    # Optional: probabilities and expected counts
    all_p = None
    prob_strs = None
    exp_counts = None
    if p_bins is not None:
        all_p = np.zeros(len(all_bins), dtype=float)
        all_p[unique_bins - 1] = p_bins
        prob_strs = [f"{p*100:.2f}%" for p in all_p]

        T = int(draws) if draws is not None else int(counts.sum())
        exp_counts = np.round(all_p * T).astype(int)

    # Column widths per bin (ensure alignment across all printed rows)
    col_widths = []
    for i in range(len(all_bins)):
        items = [
            bin_edges_str[i],
            str(all_bins[i]),
            str(all_counts[i]),
        ]
        if prob_strs is not None:
            items.append(prob_strs[i])
        if exp_counts is not None:
            items.append(str(exp_counts[i]))
        col_widths.append(max(len(x) for x in items) + 1)

    # Print table
    print("-" * 100 + "\nBin counts:\n")
    print('Edge:        ' + ''.join(f"{s:<{w}}" for s, w in zip(bin_edges_str, col_widths)))
    print('Bin idx:     ' + ''.join(f"{str(b):<{w}}" for b, w in zip(all_bins, col_widths)))
    print('Counts:      ' + ''.join(f"{str(c):<{w}}" for c, w in zip(all_counts, col_widths)))
    if prob_strs is not None:
        print('Prob:        ' + ''.join(f"{s:<{w}}" for s, w in zip(prob_strs, col_widths)))
    if exp_counts is not None:
        print('E_p(counts): ' + ''.join(f"{str(e):<{w}}" for e, w in zip(exp_counts, col_widths)))

    print(f"\nOverall number of samples: {int(counts.sum()):,}\n".replace(",", " ") + "-" * 100)


# ============================================================
# Sampler 1: QuantileThresholdSampler (Simple Two-Bucket)
# ============================================================

class QuantileThresholdSampler(Sampler):
    """
    Two-bucket sampler via dataset quantile threshold.
    
    Separates samples into "top" (severe rain) and "rest" (dry/light rain) buckets
    based on a quantile threshold, then samples each bucket with different probabilities.

    Parameters
    ----------
    dataset : SatelliteRadarDataset
        Your dataset with a .df (Pandas DataFrame) containing "p99(x)" column
    batch_size : int
        Batch size for sampling
    random : bool
        Whether to shuffle indices each epoch (default: True for train)
    seed : int or None
        Random seed. None = different each epoch (typical for train)
    p99_top_ratio : float
        Fraction of samples to treat as "top" severe rain (default: 0.3 = top 30%)
    prob : float
        Sampling probability for the top bucket (default: 0.9 = sample top 90% of the time)
        The rest bucket is sampled with probability (1 - prob)

    Notes
    -----
    - The threshold is the empirical (1 - p99_top_ratio) quantile computed on p99(x).
    - Expected keep rate ≈ p99_top_ratio·prob + (1-p99_top_ratio)·(1-prob).
    - For train: set seed=None for different sampling each epoch
    - For val/test: set seed=42 and random=False for deterministic sampling
    
    Examples
    --------
    # Train: oversample top 30% rainy samples
    sampler = QuantileThresholdSampler(
        dataset, batch_size=32, 
        p99_top_ratio=0.3, prob=0.9, seed=None
    )
    
    # Val: deterministic subset
    sampler = QuantileThresholdSampler(
        dataset, batch_size=32,
        p99_top_ratio=0.3, prob=0.5, seed=42, random=False
    )
    """

    def __init__(self, 
                 dataset, 
                 batch_size, 
                 random=True, 
                 seed=None,
                 p99_top_ratio=0.3, 
                 prob=0.9):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.random = random
        self.seed = seed

        # Extract severity scores from Pandas DataFrame
        if not hasattr(dataset, 'df'):
            raise ValueError("Dataset must have a 'df' attribute (Pandas DataFrame)")
        
        if "p99(x)" not in dataset.df.columns:
            raise ValueError("Dataset metadata must contain 'p99(x)' column")
        
        s = dataset.df["p99(x)"].to_numpy()

        # Quantile threshold for "top" bucket
        p99_top_ratio = float(np.clip(p99_top_ratio, 0.0, 1.0))
        q = 1.0 - p99_top_ratio
        thr = np.quantile(s, q) if 0.0 < q < 1.0 else (np.min(s) if q <= 0 else np.max(s))

        # Separate into top and rest buckets
        top_mask = (s >= thr)
        rng = np.random.RandomState(seed) if seed is not None else np.random
        rand = rng.rand(len(s))

        # Sample top bucket with prob, rest with (1 - prob)
        keep_mask = np.where(top_mask, rand < prob, rand < (1.0 - prob))

        kept_idx = np.nonzero(keep_mask)[0]
        rng.shuffle(kept_idx)  # Shuffle once globally
        self.indices = kept_idx

        # Print summary statistics
        n_top = int(top_mask.sum())
        n_rest = len(s) - n_top
        n_top_kept = int((top_mask & keep_mask).sum())
        n_rest_kept = int((~top_mask & keep_mask).sum())
        
        print("-" * 80)
        print(f"QuantileThresholdSampler Summary:")
        print(f"  Quantile cutoff:     q={q:.3f} → threshold={thr:.6f}")
        print(f"  Original samples:    top={n_top:,} | rest={n_rest:,} | total={len(s):,}")
        print(f"  Sampling probs:      top={prob:.2f} | rest={1.0-prob:.2f}")
        print(f"  Kept samples:        top={n_top_kept:,} | rest={n_rest_kept:,} | total={len(self.indices):,}")
        print(f"  Keep rate:           {len(self.indices)/len(s)*100:.2f}%")
        print(f"  Top ratio (kept):    {n_top_kept/len(self.indices)*100:.2f}% (vs {p99_top_ratio*100:.1f}% original)")
        print("-" * 80)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        idx = self.indices.copy()
        if self.random:
            rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
            rng.shuffle(idx)
        
        for i in range(0, len(idx), self.batch_size):
            yield idx[i:i+self.batch_size]


# ============================================================
# Sampler 2: QuantileBinnedSampler (Multi-Bin Fine-Grained)
# ============================================================

class QuantileBinnedSampler(Sampler):
    """
    Quantile-binned sampler with explicit dry/near-dry bins (Pandas-compatible version).
    
    Design:
      - Bin 0: s <= dry_value (completely dry)
      - Bin 1: dry_value < s <= near_dry_edge (near-dry, computed from positives)
      - Bins 2+: equal-mass quantile bins over positives > near_dry_edge
      - Per-bin sampling prob ∝ count**(1 - smooth_oversampling)
          smooth_oversampling=0 → natural by-count; =1 → uniform by bin.
    
    Parameters
    ----------
    dataset : SatelliteRadarDataset
        Your dataset with a .df (Pandas DataFrame) containing "p99(x)" column
    batch_size : int
        Batch size for sampling
    num_bins : int
        Total number of bins including dry and near-dry (default: 20)
    seed : int or None
        Random seed. None = different each epoch (for training)
    dry_value : float
        Threshold for "completely dry" bin (default: 0.0)
    near_dry_quantile : float
        Quantile on positive samples to define near-dry edge (default: 0.20)
    smooth_oversampling : float
        Balance between natural (0.0) and uniform (1.0) sampling (default: 1.0)
        - 0.0 = sample proportional to bin counts (natural distribution)
        - 1.0 = sample uniformly across bins (maximum oversampling)
        - 0.5-0.8 = good middle ground for most cases
    
    Examples
    --------
    # Aggressive balanced sampling (uniform across bins)
    sampler = QuantileBinnedSampler(
        dataset, batch_size=32, num_bins=20,
        smooth_oversampling=1.0, seed=None
    )
    
    # Moderate balancing
    sampler = QuantileBinnedSampler(
        dataset, batch_size=32, num_bins=20,
        smooth_oversampling=0.7, seed=None
    )
    
    # Natural distribution (no oversampling)
    sampler = QuantileBinnedSampler(
        dataset, batch_size=32, num_bins=20,
        smooth_oversampling=0.0, seed=None
    )
    """
    
    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_bins: int = 20,
                 seed: int | None = None,
                 dry_value: float = 0.0,
                 near_dry_quantile: float = 0.20,
                 smooth_oversampling: float = 1.0):
        
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = seed
        self.smooth_oversampling = float(smooth_oversampling)

        # Extract severity scores from Pandas DataFrame
        if not hasattr(dataset, 'df'):
            raise ValueError("Dataset must have a 'df' attribute (Pandas DataFrame)")
        
        if "p99(x)" not in dataset.df.columns:
            raise ValueError("Dataset metadata must contain 'p99(x)' column")
        
        s = dataset.df["p99(x)"].to_numpy()
        
        finite = np.isfinite(s)
        assert finite.any(), "No finite values in p99(x)."
        s_f = s[finite]

        # Positive-only for edge construction
        s_pos = s_f[s_f > dry_value]
        if s_pos.size == 0:
            self._build_all_dry_bins(s)
            return

        # Near-dry edge from positives
        nd_q = float(np.clip(near_dry_quantile, 0.0, 1.0))
        near_dry_edge = np.quantile(s_pos, nd_q) if nd_q > 0 else s_pos.min()

        # Remaining positive bins (ensure at least one)
        remaining_bins = max(int(num_bins) - 2, 1)

        # Quantile edges on positives above near_dry_edge
        s_pos_high = s_pos[s_pos > near_dry_edge]
        if s_pos_high.size == 0:
            # Everything positive collapsed into near-dry → single positive bin
            pos_edges = np.array([near_dry_edge, s_pos.max()])
            remaining_bins = 1
        else:
            qs = np.linspace(0.0, 1.0, remaining_bins + 1)
            pos_edges = np.quantile(s_pos_high, qs)

        # --- Assemble global bin edges (finite top; no +inf) ---
        edges = np.concatenate(([-np.inf, dry_value, near_dry_edge], pos_edges[1:]))

        # Ensure strictly increasing edges
        edges = np.maximum.accumulate(edges + 0.0)
        for i in range(1, edges.size):
            if edges[i] <= edges[i-1]:
                edges[i] = np.nextafter(edges[i-1], np.inf)

        self.bin_edges = edges

        # --- Assign bins with capping to the top edge ---
        s_capped = np.minimum(s, self.bin_edges[-1])
        dm_bin_idx = np.searchsorted(self.bin_edges, s_capped, side="right") - 1
        dm_bin_idx = np.clip(dm_bin_idx, 0, len(self.bin_edges) - 2)

        # --- Bookkeeping ---
        self.unique_bins, self.counts = np.unique(dm_bin_idx, return_counts=True)
        self.idxs_bins = np.column_stack((np.arange(len(s), dtype=np.uint32), dm_bin_idx))
        self.bin2indices = {b: np.flatnonzero(self.idxs_bins[:, 1] == b) for b in self.unique_bins}

        # Write bin index back to Pandas DataFrame for inspection
        self.dataset.df["bin_idx"] = dm_bin_idx.astype(np.int32)

        # --- Per-bin sampling probabilities ---
        counts_f = self.counts.astype(float)
        p = counts_f ** (1.0 - self.smooth_oversampling)
        self.p_bins = p / p.sum()

        # Print summary
        print_bincount(self.unique_bins, self.counts, self.bin_edges, p_bins=self.p_bins)

    def _build_all_dry_bins(self, s):
        """Degenerate case: everything is dry/near-dry"""
        self.bin_edges = np.array([-np.inf, 0.0, 1e-12])  # two bins around zero
        dm_bin_idx = np.searchsorted(self.bin_edges, s, side="right") - 1
        dm_bin_idx = np.clip(dm_bin_idx, 0, len(self.bin_edges) - 2)
        self.unique_bins, self.counts = np.unique(dm_bin_idx, return_counts=True)
        self.idxs_bins = np.column_stack((np.arange(len(s), dtype=np.uint32), dm_bin_idx))
        self.bin2indices = {b: np.flatnonzero(self.idxs_bins[:, 1] == b) for b in self.unique_bins}
        self.dataset.df["bin_idx"] = dm_bin_idx.astype(np.int32)
        counts_f = self.counts.astype(float)
        p = counts_f ** 0.0  # uniform across bins
        self.p_bins = p / p.sum()

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = (np.random if self.seed is None else np.random.RandomState(self.seed))
        n = len(self.dataset)
        assert n > 0, "Empty dataset."
        produced = 0
        
        while produced < n - self.batch_size:
            # Sample bins according to probability distribution
            sampled_bins = rng.choice(self.unique_bins, size=self.batch_size, p=self.p_bins)
            row_indices = np.empty(self.batch_size, dtype=np.uint32)
            
            # For each unique bin in this batch, randomly sample from its pool
            for b in np.unique(sampled_bins):
                mask = (sampled_bins == b)
                pool = self.bin2indices[b]
                row_indices[mask] = pool[rng.randint(0, len(pool), size=mask.sum())]
            
            produced += self.batch_size
            yield self.idxs_bins[row_indices, 0]
