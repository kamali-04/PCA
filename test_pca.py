"""Sanity checks: PCAFromScratch must agree with scikit-learn's PCA.  Run with: pytest -q"""
import numpy as np
import pytest
from sklearn.decomposition import PCA

from pca import PCAFromScratch


@pytest.fixture(params=["tall", "wide"])  # wide = more features than samples (Gram-matrix path)
def data(request):
    rng = np.random.default_rng(0)
    shape = (200, 15) if request.param == "tall" else (40, 120)
    return rng.normal(size=shape) @ rng.normal(size=(shape[1], shape[1]))


@pytest.mark.parametrize("whiten", [False, True])
def test_matches_sklearn(data, whiten):
    k = 8
    ours = PCAFromScratch(k, whiten=whiten).fit(data)
    ref = PCA(k, whiten=whiten, svd_solver="full").fit(data)

    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-6)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-6)
    # Principal axes can differ by sign only
    dots = np.abs(np.sum(ours.components_ * ref.components_, axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-6)
    np.testing.assert_allclose(np.abs(ours.transform(data)), np.abs(ref.transform(data)), atol=1e-6)


def test_reconstruction_roundtrip(data):
    full = PCAFromScratch(min(data.shape[0] - 1, data.shape[1])).fit(data)  # full rank
    np.testing.assert_allclose(full.inverse_transform(full.transform(data)), data, atol=1e-6)


def test_components_orthonormal(data):
    pca = PCAFromScratch(8).fit(data)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(8), atol=1e-8)
