class DataPolicy:
    def __init__(self, nan_handling_cfg, mask_nans=True):
        """
        Metadata-level NaN policy + tensor-level NaN switch.
        
        Parameters
        ----------
        nan_handling_cfg : config node
            Contains mode + max_nan_ratio for metadata filtering.
        mask_nans : bool
            Whether tensors should mask NaNs and replace them with zero.
        """
        self.mode = nan_handling_cfg.mode
        self.max_nan_ratio = nan_handling_cfg.max_nan_ratio
        self.mask_nans = mask_nans
    
    def apply(self, df):
        """Apply metadata filtering based on NaN handling mode."""
        if self.mode == "keep_all":
            return df
        if self.mode == "discard_all_nan":
            return df[df["all_nan"] == False]
        if self.mode == "threshold":
            return df[df["nan_ratio"] <= self.max_nan_ratio]
        raise ValueError(f"Unknown nan_handling mode: {self.mode}")