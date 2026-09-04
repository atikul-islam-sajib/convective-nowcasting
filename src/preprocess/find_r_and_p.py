import pandas as pd
import numpy as np

df = pd.read_csv('metadata/train_patch_256x256_s64_summer.csv')
s = df['p99(x_seq)'].to_numpy()
s = s[np.isfinite(s)]

r_empirical = (s > 0).mean()
p_value = 1 - r_empirical

print(f"Fraction of patches with any rain : {r_empirical:.4f}")
print(f"P-value : {p_value:.4f}")