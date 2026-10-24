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

# (q, axis, keepdims) covering scalar and vector q, every axis form and keepdims.
PERCENTILE_CASES = [
    (30.0, None, False),
    ([5.0, 50.0, 99.0], None, True),
    (40.0, 1, False),
    ([10.0, 60.0, 90.0], -1, True),
    ([0.0, 100.0], [0, 2], False),
    (87.5, [1, 2], True),
]


def assert_same(actual, expected):
    actual, expected = ops.convert_to_numpy(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("interpolation", ["nearest", "linear", "lower", "higher", "midpoint"])
def test_percentile_and_quantiles_parity(interpolation):
    x = np.random.default_rng(1).normal(size=(6, 5, 7)).astype("float32")
    for q, axis, keepdims in PERCENTILE_CASES:
        assert_same(
            percentile(x, q, axis=axis, interpolation=interpolation, keepdims=keepdims),
            tfp_quantiles.percentile(x, q, axis=axis, interpolation=interpolation, keepdims=keepdims),
        )
    for num_quantiles, axis in [(1, None), (4, 0), (10, [0, 1])]:
        assert_same(
            quantiles(x, num_quantiles, axis=axis, interpolation=interpolation),
            tfp_quantiles.quantiles(x, num_quantiles, axis=axis, interpolation=interpolation),
        )


def test_percentile_parity_nan_and_integers():
    x = np.random.default_rng(2).normal(size=(4, 9)).astype("float32")
    x[1, 3] = np.nan
    assert_same(percentile(x, [25.0, 75.0], axis=1), tfp_quantiles.percentile(x, [25.0, 75.0], axis=1))

    x = np.random.default_rng(3).integers(-50, 50, size=(5, 8)).astype("int32")
    for interpolation in ["nearest", "lower", "higher"]:
        assert_same(
            percentile(x, [20.0, 80.0], axis=1, interpolation=interpolation),
            tfp_quantiles.percentile(x, [20.0, 80.0], axis=1, interpolation=interpolation),
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
