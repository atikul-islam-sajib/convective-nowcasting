# Evaluation Metrics

Five metrics are used to evaluate all models (and the pySTEPS baseline)
on the held-out test set: MAE, MSE, CSI, ETS, and PSNR. Lower is better for
MAE/MSE; higher is better for CSI/ETS/PSNR.

## Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{N} \qquad (4.6)
$$

## Mean Squared Error (MSE)

$$
\text{MSE} = \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{N} \qquad (4.7)
$$

$y_i$, $\hat{y}_i$ are the observed/predicted precipitation intensity at
pixel $i$; $N$ is the total number of pixels. MSE weights larger errors more
heavily than MAE.

## Critical Success Index (CSI)

$$
\text{CSI} = \frac{H}{H + M + FA} \qquad (4.8)
$$

where $H$, $M$, $FA$ are hits, misses, and false alarms at a given
precipitation threshold. CSI ignores correct negatives, making it suitable
for datasets dominated by non-precipitating pixels.

## Equitable Threat Score (ETS)

$$
\text{ETS} = \frac{H - H_c}{H + M + FA - H_c} \qquad (4.9)
$$

$$
H_c = \frac{(H+M)(H+FA)}{N} \qquad (4.10)
$$

where $H_c$ is the number of hits expected by chance and
$N = H+M+FA+CN$ is the total sample count ($CN$ = correct negatives). ETS
corrects CSI for chance agreement, and was used as the **model-selection
criterion** on the validation set.

## Peak Signal-to-Noise Ratio (PSNR)

$$
\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
\qquad (4.11)
$$

where MAX is the maximum precipitation value in the data.

!!! note
    CSI and ETS are averaged across the 5 and 15 mm/h thresholds throughout
    the results tables.
