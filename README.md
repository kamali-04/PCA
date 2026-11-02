# PCA for Image Analysis

A NumPy implementation of Principal Component Analysis (PCA), applied to three image tasks:

| Demo | What it does | Data |
|------|--------------|------|
| **Face recognition** | Eigenfaces + K-Nearest-Neighbours | Olivetti faces (400 images, 40 people) |
| **Image compression** | Low-rank image reconstruction with PCA | Any image (synthetic one included) |
| **Anomaly detection** | Flags images with high PCA reconstruction error | Synthetic squares vs. circles / triangles / noise |

The PCA itself lives in `PCAFromScratch` (`pca.py`) and uses **only NumPy linear algebra**. scikit-learn is used for the dataset, the KNN classifier and metrics, and as a reference in the unit tests. OpenCV handles image I/O and drawing.

## How PCA is implemented

1. Mean-centre the data.
2. Eigen-decompose the covariance matrix (`np.linalg.eigh`) and sort axes by variance.
3. Keep the top-*k* axes as `components_`; project with `transform`, reconstruct with `inverse_transform`.

**Eigenfaces trick:** a 64×64 face has 4096 pixels but there are only ~300 training images. When `n_features > n_samples`, the eigenvectors are computed from the small `n_samples × n_samples` Gram matrix and mapped back to pixel space (Turk & Pentland, 1991). The principal axes are identical, and it avoids decomposing a 4096×4096 matrix.

`PCAFromScratch` supports `whiten=True`, `fit_transform`, and explained-variance ratios, and follows scikit-learn's layout. `test_pca.py` checks it against `sklearn.decomposition.PCA`: variances, axes (up to sign), projections, orthonormality and round-trip reconstruction, for both the tall and wide cases.

## Quick start

```bash
pip install -r requirements.txt

python pca.py                        # run all demos
python pca.py --demo anomaly         # faces | compression | anomaly
python pca.py --image photo.jpg --demo compression
python pca.py --show                 # also open plot windows

pytest -q                            # unit tests
```

Figures and a `results.json` are written to `./outputs/`. The first face run downloads the Olivetti dataset (~1.5 MB) via scikit-learn.

## Demos and results

### 1. Face recognition (Eigenfaces)
- Stratified 75/25 split, so every person appears in both train and test sets.
- Compares three feature sets with the same KNN classifier: **raw pixels** (baseline), **PCA-100**, and **whitened PCA-100**.
- Also plots the mean face, the top eigenfaces, and accuracy vs. number of components.

The script prints the comparison table on every run. Whitening scales every component to unit variance, which amplifies noisy low-variance directions and typically hurts Euclidean KNN. The first version of this project used whitened PCA (150 components, k=5) and reached **~69%** test accuracy. It is kept here as an ablation row.

### 2. Image compression
Each image row is treated as a sample; keeping *k* components stores `h·k + k·w + w` numbers instead of `h·w`. Results on the built-in 300×400 synthetic image:

| k | Compression ratio | PSNR (dB) | Variance kept |
|---|------------------:|----------:|--------------:|
| 5  | 30.8× | 29.7 | 90.5% |
| 10 | 16.2× | 31.8 | 94.2% |
| 20 | 8.3×  | 34.9 | 97.1% |
| 50 | 3.4×  | 42.9 | 99.5% |

The built-in image is synthetic (smooth gradients and sharp shapes). Pass `--image` to try a real photo.

### 3. Anomaly detection
"Normal" images are noisy squares of random size and position. Anomalies are circles, triangles and uniform noise. PCA (10 components) is fit on normal images only; an image is flagged if its reconstruction error exceeds a threshold.

The normal data is split into **train** (fit PCA), **validation** (set the threshold at the 95th percentile of its errors) and **test** (evaluate). The threshold must come from images PCA has *not* seen. Training errors are optimistically small, so a threshold taken from them flags a large share of unseen normal images.

Results (300 normal images split 180 / 60 / 60 into train / validation / test, plus 60 anomalies; seed 42):

| Metric | Value |
|--------|------:|
| ROC-AUC | 0.977 |
| Recall (anomalies caught) | 88.3% |
| False positive rate (normal flagged) | 1.7% |
| Precision | 98.1% |
| F1 | 0.930 |

Recall by anomaly type: noise 100%, triangles 95%, circles 70%. Circles are hardest because a filled blob is close to what a low-rank "square" model can already represent. See `outputs/anomaly_hardest.png`.

## Project structure

```
pca.py            # PCAFromScratch + the three demos + CLI
test_pca.py       # unit tests against scikit-learn
requirements.txt
outputs/          # generated figures and results.json (created on run)
```

## Limitations and ideas
- PCA is linear, so it captures global structure well but not complex pose/lighting variation. Eigenfaces work on aligned, centred faces such as Olivetti and degrade on unconstrained photos.
- The anomaly dataset is synthetic. Real defects (e.g. MVTec AD) would need patch-based features or a nonlinear model.
- Possible extensions: kernel PCA, an autoencoder baseline for anomaly detection, cross-validated choice of `n_components`, and per-channel colour compression.

## References
- Turk & Pentland, *Eigenfaces for Recognition*, Journal of Cognitive Neuroscience, 1991.
- Olivetti faces dataset (AT&T Laboratories Cambridge), via `sklearn.datasets.fetch_olivetti_faces`.
