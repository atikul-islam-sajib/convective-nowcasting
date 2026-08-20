class DataPolicy:
    def __init__(self, nan_handling_cfg, mask_nans=True):
        self.mode = nan_handling_cfg.mode
        self.max_nan_ratio = nan_handling_cfg.max_nan_ratio
        self.mask_nans = mask_nans

    def apply(self, df):
        if self.mode == "keep_all":
            return df
        if self.mode == "discard_all_nan":
            return df[df["all_nan"] == False]
        if self.mode == "threshold":
            return df[df["nan_ratio"] <= self.max_nan_ratio]
        raise ValueError(f"Unknown nan_handling mode: {self.mode}")