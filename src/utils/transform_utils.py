import torch
import numpy as np


class LogTransform:

    def __init__(self, threshold=0.1, zerovalue=0.02, inverse=False):
        self.threshold = threshold
        self.zerovalue = zerovalue
        self.inverse = inverse

    def __call__(self, x):
        return self.inv_transform(x) if self.inverse else self.transform(x)

    def transform(self, x):
        
        if torch.is_tensor(x): 
            thresh = x.new_tensor(self.threshold)
            zero  = x.new_tensor(self.zerovalue)
            x = torch.where(x >= thresh, x, zero) 
            return x.log10()
        else:
            x = x.copy() 
            x[x < self.threshold] = self.zerovalue
            return np.log10(x)
            
    def inv_transform(self, log_x):
        
        if torch.is_tensor(log_x):
            x = torch.pow(10, log_x)
            thresh = x.new_tensor(self.threshold)
            x = torch.where(x < thresh, torch.zeros_like(x), x)
            return x
        else:
            x = 10 ** log_x
            x[x < self.threshold] = 0.0
            return x
        
def clip(array, vmin=None, vmax=None):
    """Clip array values to [vmin, vmax]"""
    if vmin is not None:
        array = np.maximum(array, vmin)
    if vmax is not None:
        array = np.minimum(array, vmax)
    return array


def zscore(array, mean=None, std=None):
    """Z-score normalization"""
    if mean is None:
        mean = np.mean(array)
    if std is None:
        std = np.std(array)
    return (array - mean) / (std + 1e-8)


def minmax(array, vmin, vmax):
    """Min-max normalization to [0, 1]"""
    return (array - vmin) / (vmax - vmin + 1e-8)


def log_transform(array):
    """DEPRECATED: Use LogTransform class instead"""
    return np.log1p(array)
