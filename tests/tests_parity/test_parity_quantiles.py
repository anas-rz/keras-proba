"""Numerical parity tests comparing the Keras 3 quantiles against TensorFlow Probability's implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "reference/probability"
if not REFERENCE_PATH.exists():
    pytest.skip(f"Reference implementation not found at {REFERENCE_PATH}", allow_module_level=True)
sys.path.insert(0, str(REFERENCE_PATH))
tfp_quantiles = pytest.importorskip("tensorflow_probability.python.stats.quantiles")

from keras import ops  # noqa: E402

from keras_proba.stats.quantiles import percentile, quantiles  # noqa: E402

INTERPOLATIONS = [None, "nearest", "linear", "lower", "higher", "midpoint"]


def assert_same(actual, expected):
    actual, expected = ops.convert_to_numpy(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("q", [0.0, 30.0, 50.0, 87.5, 100.0, [5.0, 25.0, 50.0, 99.0]])
def test_percentile_parity_flat(interpolation, q):
    x = np.random.default_rng(0).normal(size=(37,)).astype("float32")
    assert_same(
        percentile(x, q, interpolation=interpolation),
        tfp_quantiles.percentile(x, q, interpolation=interpolation),
    )


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("axis", [None, 0, 1, -1, [0, 2], [1, 2]])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("q", [40.0, [10.0, 60.0, 90.0]])
def test_percentile_parity_axis(interpolation, axis, keepdims, q):
    x = np.random.default_rng(1).normal(size=(6, 5, 7)).astype("float32")
    assert_same(
        percentile(x, q, axis=axis, interpolation=interpolation, keepdims=keepdims),
        tfp_quantiles.percentile(x, q, axis=axis, interpolation=interpolation, keepdims=keepdims),
    )


def test_percentile_parity_nan():
    x = np.random.default_rng(2).normal(size=(4, 9)).astype("float32")
    x[1, 3] = np.nan
    assert_same(percentile(x, [25.0, 75.0], axis=1), tfp_quantiles.percentile(x, [25.0, 75.0], axis=1))


@pytest.mark.parametrize("interpolation", ["nearest", "lower", "higher"])
def test_percentile_parity_integers(interpolation):
    x = np.random.default_rng(3).integers(-50, 50, size=(5, 8)).astype("int32")
    assert_same(
        percentile(x, [20.0, 80.0], axis=1, interpolation=interpolation),
        tfp_quantiles.percentile(x, [20.0, 80.0], axis=1, interpolation=interpolation),
    )


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("num_quantiles", [1, 3, 4, 10])
@pytest.mark.parametrize("axis", [None, 0, [0, 1]])
def test_quantiles_parity(interpolation, num_quantiles, axis):
    x = np.random.default_rng(num_quantiles).uniform(size=(11, 4, 3)).astype("float32")
    assert_same(
        quantiles(x, num_quantiles, axis=axis, interpolation=interpolation),
        tfp_quantiles.quantiles(x, num_quantiles, axis=axis, interpolation=interpolation),
    )


def test_quantiles_docstring_parity():
    x = np.arange(11.0, dtype="float32")
    for interpolation in ["nearest", "linear", "lower"]:
        assert_same(
            quantiles(x, 4, interpolation=interpolation),
            tfp_quantiles.quantiles(x, 4, interpolation=interpolation),
        )


def test_matches_numpy():
    """Without NaNs, the interpolation modes agree with `numpy.percentile`, except numpy's tie rule for "nearest"."""
    x = np.random.default_rng(4).normal(size=(6, 21)).astype("float32")
    q = [3.0, 35.0, 50.0, 72.0]
    for interpolation in ["linear", "lower", "higher", "midpoint"]:
        np.testing.assert_allclose(
            ops.convert_to_numpy(percentile(x, q, axis=1, interpolation=interpolation)),
            np.percentile(x, q, axis=1, method=interpolation),
            rtol=1e-5,
        )
