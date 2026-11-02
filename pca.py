"""
Principal Component Analysis (PCA) for Image Analysis
=====================================================

Three demos built on one NumPy-only PCA implementation:

1. Face recognition (Eigenfaces) on the Olivetti faces dataset + KNN
2. Image compression (PCA on the rows of a grayscale image)
3. Anomaly detection in images using PCA reconstruction error

Usage
-----
    python pca.py                       # run all demos, save figures to ./outputs
    python pca.py --demo faces          # only one demo: faces | compression | anomaly
    python pca.py --image my_photo.jpg  # use your own image for the compression demo
    python pca.py --show                # also open matplotlib windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
OUTPUT_DIR = Path("outputs")


##############################################
# Part 1: PCA implementation from scratch
##############################################

class PCAFromScratch:
    """PCA using only NumPy linear algebra (no scikit-learn).

    Follows the scikit-learn layout so it can be swapped in easily:
    ``components_`` has shape (n_components, n_features), one principal axis per row.

    When there are more features than samples (e.g. 4096 pixels vs ~300 faces)
    the eigenvectors are computed from the small (n_samples x n_samples) Gram
    matrix and mapped back to pixel space - the classic Turk & Pentland
    "eigenfaces" trick. This is much faster than decomposing a 4096 x 4096
    covariance matrix and gives the same principal axes.
    """

    def __init__(self, n_components: int, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # 1. Mean-centre the data
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2. Eigen-decomposition of the covariance matrix (or its Gram-matrix equivalent)
        if n_features > n_samples:
            gram = (Xc @ Xc.T) / (n_samples - 1)
            eigvals, eigvecs = np.linalg.eigh(gram)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            # Drop numerically-zero directions before mapping back (avoids divide-by-zero)
            keep = eigvals > eigvals[0] * 1e-12
            eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]
            axes = Xc.T @ eigvecs                      # map back to pixel space
            axes /= np.linalg.norm(axes, axis=0, keepdims=True)
        else:
            cov = np.cov(Xc, rowvar=False)
            eigvals, axes = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, axes = eigvals[order], axes[:, order]
            keep = eigvals > eigvals[0] * 1e-12
            eigvals, axes = eigvals[keep], axes[:, keep]
        k = min(self.n_components, len(eigvals))

        # 3. Keep the top-k axes. Fix the sign so results are reproducible.
        components = axes[:, :k].T
        signs = np.sign(components[np.arange(k), np.argmax(np.abs(components), axis=1)])
        signs[signs == 0] = 1
        self.components_ = components * signs[:, None]

        # 4. Variance bookkeeping (total variance = trace of covariance matrix)
        total_var = np.sum(Xc ** 2) / (n_samples - 1)
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T
        if self.whiten:
            Z /= np.sqrt(self.explained_variance_)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_


##############################################
# Helpers
##############################################

def finish_figure(name: str, show: bool) -> None:
    """Save the current figure to OUTPUT_DIR and optionally display it."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def show_images(images: np.ndarray, image_size: int, title: str, name: str, show: bool) -> None:
    n = min(10, len(images))
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.2), squeeze=False)
    for ax, img in zip(axes[0], images[:n]):
        ax.imshow(img.reshape(image_size, image_size), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    finish_figure(name, show)


##############################################
# Part 2: Face recognition using PCA (Eigenfaces)
##############################################

def evaluate_knn(X_train, X_test, y_train, y_test, k: int) -> float:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


def plot_eigenfaces(pca: PCAFromScratch, img_shape, show: bool) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(14, 5), subplot_kw={"xticks": [], "yticks": []})
    axes[0, 0].imshow(pca.mean_.reshape(img_shape), cmap="gray")
    axes[0, 0].set_title("Mean face")
    for i, ax in enumerate(axes.flat[1:]):
        ax.imshow(pca.components_[i].reshape(img_shape), cmap="gray")
        ax.set_title(f"Eigenface {i + 1}\n{100 * pca.explained_variance_ratio_[i]:.1f}% var")
    plt.tight_layout()
    finish_figure("eigenfaces.png", show)


def plot_component_sweep(sweep: dict, show: bool) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(list(sweep.keys()), list(sweep.values()), marker="o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Test accuracy")
    plt.title("Face recognition accuracy vs. number of components")
    plt.grid(alpha=0.3)
    finish_figure("accuracy_vs_components.png", show)


def plot_predictions(X_test, y_test, y_pred, img_shape, show: bool, n: int = 20) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(12, 11), subplot_kw={"xticks": [], "yticks": []})
    for i, ax in enumerate(axes.flat[:n]):
        ax.imshow(X_test[i].reshape(img_shape), cmap="gray")
        ok = y_pred[i] == y_test[i]
        color = "green" if ok else "red"
        ax.set_title(f"Person {y_pred[i]}" if ok else f"Pred {y_pred[i]} / True {y_test[i]}",
                     color=color, fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    plt.tight_layout()
    finish_figure("face_predictions.png", show)


def face_recognition_demo(show: bool = False, n_components: int = 100, k: int = 1) -> dict:
    """Eigenfaces + KNN, compared against a raw-pixel baseline and a whitened-PCA variant."""
    faces = fetch_olivetti_faces(shuffle=True, random_state=SEED)
    X, y = faces.data, faces.target
    img_shape = faces.images[0].shape

    # Stratified split: every person appears in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    print(f"  {len(X_train)} train / {len(X_test)} test images, "
          f"{len(np.unique(y))} people, {X.shape[1]} pixels each")

    results = {}

    # Baseline: KNN directly on the 4096 raw pixels
    results["raw_pixels"] = evaluate_knn(X_train, X_test, y_train, y_test, k)

    # Eigenfaces (PCA without whitening) - the main model
    pca = PCAFromScratch(n_components=n_components).fit(X_train)
    Z_train, Z_test = pca.transform(X_train), pca.transform(X_test)
    results[f"pca_{n_components}"] = evaluate_knn(Z_train, Z_test, y_train, y_test, k)
    print(f"  {n_components} components keep "
          f"{100 * pca.explained_variance_ratio_.sum():.1f}% of the variance")

    # Whitened PCA: equalises variance of every component, which boosts noisy
    # low-variance directions and usually hurts Euclidean KNN. Kept as an ablation.
    pca_w = PCAFromScratch(n_components=n_components, whiten=True).fit(X_train)
    results[f"pca_{n_components}_whitened"] = evaluate_knn(
        pca_w.transform(X_train), pca_w.transform(X_test), y_train, y_test, k)

    # Sweep the number of components
    sweep = {}
    for nc in (5, 10, 25, 50, 100, 150):
        p = PCAFromScratch(n_components=nc).fit(X_train)
        sweep[nc] = evaluate_knn(p.transform(X_train), p.transform(X_test), y_train, y_test, k)

    print(f"\n  KNN (k={k}) test accuracy")
    for name, acc in results.items():
        print(f"    {name:<22s} {acc:.4f}")
    print("  Accuracy by number of components:",
          ", ".join(f"{nc}: {a:.3f}" for nc, a in sweep.items()))

    plot_eigenfaces(pca, img_shape, show)
    plot_component_sweep(sweep, show)
    y_pred = KNeighborsClassifier(n_neighbors=k).fit(Z_train, y_train).predict(Z_test)
    plot_predictions(X_test, y_test, y_pred, img_shape, show)

    return {"k": k, "n_components": n_components, "accuracy": results, "sweep": sweep}


##############################################
# Part 3: Image compression using PCA
##############################################

def make_sample_image(h: int = 300, w: int = 400) -> np.ndarray:
    """Synthetic BGR image (gradient + texture + shapes) used when no image is supplied."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[:h, :w]
    base = 90 + 60 * xx / w + 40 * np.sin(yy / 18.0) + rng.normal(0, 6, (h, w))
    img = np.clip(np.dstack([base, base * 0.8, base * 0.6]), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
    cv2.circle(img, (300, 150), 100, (0, 255, 0), -1)
    cv2.line(img, (0, 0), (w, h), (0, 0, 255), 5)
    return cv2.GaussianBlur(img, (3, 3), 0)


def load_image(path: str | None) -> np.ndarray:
    if path:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img
    default = Path("sample_image.jpg")
    if default.exists():
        img = cv2.imread(str(default))
        if img is not None:
            return img
    return make_sample_image()


def compress_image_pca(image: np.ndarray, n_components_list, show: bool = False) -> list:
    """Treat each image row as a sample, keep k principal components, and reconstruct."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float64)
    h, w = gray.shape

    fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(4 * (len(n_components_list) + 1), 4))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    stats = []
    for ax, k in zip(axes[1:], n_components_list):
        pca = PCAFromScratch(n_components=k).fit(gray)
        scores = pca.transform(gray)
        recon = np.clip(pca.inverse_transform(scores), 0, 255)

        # Storage cost: scores (h*k) + components (k*w) + mean (w)
        ratio = (h * w) / (scores.size + pca.components_.size + w)
        mse = np.mean((gray - recon) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
        stats.append({"k": k, "ratio": ratio, "mse": mse, "psnr_db": psnr,
                      "variance_kept": float(pca.explained_variance_ratio_.sum())})

        ax.imshow(recon, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"k={k}\nratio {ratio:.1f}x | PSNR {psnr:.1f} dB")
        ax.axis("off")

    plt.tight_layout()
    finish_figure("compression.png", show)

    print(f"  {'k':>4} {'ratio':>8} {'MSE':>9} {'PSNR(dB)':>9} {'var kept':>9}")
    for s in stats:
        print(f"  {s['k']:>4} {s['ratio']:>7.2f}x {s['mse']:>9.2f} {s['psnr_db']:>9.2f} {s['variance_kept']:>8.1%}")
    return stats


##############################################
# Part 4: Anomaly detection in images using PCA
##############################################

def generate_normal_and_anomalous_images(n_normal=300, n_anomalous=60, image_size=20, seed=SEED):
    """Synthetic dataset: 'normal' = noisy squares; anomalies = circles, triangles, noise."""
    rng = np.random.default_rng(seed)

    def add_noise(img):
        return np.clip(img + rng.normal(0, 0.1, img.shape), 0, 1).ravel()

    normal = []
    for _ in range(n_normal):
        img = np.zeros((image_size, image_size))
        size = rng.integers(5, 10)
        x = rng.integers(0, image_size - size)
        y = rng.integers(0, image_size - size)
        img[y:y + size, x:x + size] = 1
        normal.append(add_noise(img))

    anomalous = []
    yy, xx = np.ogrid[:image_size, :image_size]
    for i in range(n_anomalous):
        img = np.zeros((image_size, image_size))
        if i % 3 == 0:    # circle
            c = image_size // 2
            img[np.sqrt((xx - c) ** 2 + (yy - c) ** 2) <= rng.integers(5, 9)] = 1
        elif i % 3 == 1:  # triangle (cv2 needs int32 points)
            pts = np.array([[image_size // 2, 5], [5, image_size - 5],
                            [image_size - 5, image_size - 5]], dtype=np.int32)
            cv2.fillPoly(img, [pts], 1.0)
        else:             # uniform random noise
            img = rng.random((image_size, image_size))
        anomalous.append(add_noise(img))

    return np.array(normal), np.array(anomalous), image_size


def reconstruction_error(X: np.ndarray, pca: PCAFromScratch) -> np.ndarray:
    """Mean squared reconstruction error per image."""
    return np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2, axis=1)


def anomaly_detection_demo(show: bool = False, n_components: int = 10, percentile: float = 95) -> dict:
    X_normal, X_anom, size = generate_normal_and_anomalous_images()

    show_images(X_normal, size, "Normal images (squares)", "anomaly_normal_samples.png", show)
    show_images(X_anom, size, "Anomalous images", "anomaly_anomalous_samples.png", show)

    # Three disjoint normal splits: fit PCA / choose threshold / evaluate.
    # The threshold must come from data PCA has NOT seen, otherwise it is too low
    # (training errors are optimistically small) and false positives balloon.
    X_train, X_rest = train_test_split(X_normal, test_size=0.4, random_state=SEED)
    X_val, X_test = train_test_split(X_rest, test_size=0.5, random_state=SEED)

    pca = PCAFromScratch(n_components=n_components).fit(X_train)
    err_train = reconstruction_error(X_train, pca)
    err_val = reconstruction_error(X_val, pca)
    err_test = reconstruction_error(X_test, pca)
    err_anom = reconstruction_error(X_anom, pca)

    threshold = np.percentile(err_val, percentile)

    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, max(err_anom.max(), err_test.max()), 30)
    plt.hist(err_train, bins=bins, alpha=0.5, label="Normal (train)")
    plt.hist(err_test, bins=bins, alpha=0.5, label="Normal (test)")
    plt.hist(err_anom, bins=bins, alpha=0.5, label="Anomalous")
    plt.axvline(threshold, color="r", linestyle="--", label=f"Threshold = {threshold:.4f}")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Count")
    plt.title("PCA reconstruction error: normal vs. anomalous images")
    plt.legend()
    finish_figure("anomaly_error_histogram.png", show)

    # Metrics: anomalies are the positive class
    y_true = np.r_[np.zeros(len(err_test)), np.ones(len(err_anom))]
    scores = np.r_[err_test, err_anom]
    y_pred = scores > threshold
    tp, fp = np.sum(y_pred & (y_true == 1)), np.sum(y_pred & (y_true == 0))
    fn = np.sum(~y_pred & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    fpr = fp / np.sum(y_true == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc_score(y_true, scores)

    # Per-anomaly-type recall (the generator cycles circle / triangle / noise)
    per_type = {name: float(np.mean(err_anom[i::3] > threshold))
                for i, name in enumerate(("circle", "triangle", "noise"))}

    print(f"  Threshold ({percentile:.0f}th percentile of validation error): {threshold:.4f}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"  Recall (true positive rate):  {recall:.1%}")
    print(f"  False positive rate:          {fpr:.1%}")
    print(f"  Precision: {precision:.1%}   F1: {f1:.3f}")
    print("  Recall by anomaly type:", ", ".join(f"{k} {v:.0%}" for k, v in per_type.items()))

    show_images(X_test[np.argsort(err_test)[-10:]], size,
                "Normal images with highest error (false-alarm candidates)",
                "anomaly_hard_normal.png", show)
    show_images(X_anom[np.argsort(err_anom)[-10:]], size,
                "Anomalies with highest error (easiest to detect)",
                "anomaly_easiest.png", show)
    show_images(X_anom[np.argsort(err_anom)[:10]], size,
                "Anomalies with lowest error (hardest to detect)",
                "anomaly_hardest.png", show)

    return {"threshold": float(threshold), "roc_auc": float(auc), "recall": float(recall),
            "false_positive_rate": float(fpr), "precision": float(precision),
            "f1": float(f1), "recall_by_type": per_type}


##############################################
# Entry point
##############################################

def main() -> None:
    parser = argparse.ArgumentParser(description="PCA for image analysis demos")
    parser.add_argument("--demo", choices=["all", "faces", "compression", "anomaly"], default="all")
    parser.add_argument("--image", help="image file for the compression demo")
    parser.add_argument("--show", action="store_true", help="display figures as well as saving them")
    args = parser.parse_args()

    np.random.seed(SEED)
    results = {}

    if args.demo in ("all", "faces"):
        print("\n[1/3] Face recognition (Eigenfaces)")
        results["faces"] = face_recognition_demo(args.show)

    if args.demo in ("all", "compression"):
        print("\n[2/3] Image compression")
        results["compression"] = compress_image_pca(load_image(args.image), [5, 10, 20, 50], args.show)

    if args.demo in ("all", "anomaly"):
        print("\n[3/3] Anomaly detection")
        results["anomaly"] = anomaly_detection_demo(args.show)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    if results_path.exists():  # keep results from earlier single-demo runs
        results = {**json.loads(results_path.read_text()), **results}
    results_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nFigures and results.json saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
