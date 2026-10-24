"""Smoke tests for percentiles and quantiles, from TensorFlow Probability's docstring examples. Run once per backend."""

import numpy as np
import pytest

from keras import ops

from keras_proba.stats.quantiles import percentile, quantiles


def to_numpy(x):
    return ops.convert_to_numpy(x)


def test_percentile_docstring_examples():
    x = np.array([1.0, 2.0, 3.0, 4.0], "float32")
    np.testing.assert_allclose(to_numpy(percentile(x, q=30.0)), 2.0)
    np.testing.assert_allclose(to_numpy(percentile(x, q=30.0, interpolation="linear")), 1.9, rtol=1e-6)
    np.testing.assert_allclose(to_numpy(percentile(x, q=[30.0, 70.0], interpolation="lower")), [1.0, 3.0])
    x = np.array([[1.0, 2.0], [3.0, 4.0]], "float32")
    np.testing.assert_allclose(to_numpy(percentile(x, q=100.0)), 4.0)
    np.testing.assert_allclose(to_numpy(percentile(x, q=100.0, axis=[0])), [3.0, 4.0])


def test_quantiles_docstring_examples():
    x = np.arange(11.0, dtype="float32")
    np.testing.assert_allclose(to_numpy(quantiles(x, num_quantiles=4, interpolation="nearest")), [0, 2, 5, 8, 10])
    np.testing.assert_allclose(to_numpy(quantiles(x, num_quantiles=4, interpolation="linear")), [0, 2.5, 5, 7.5, 10])
    np.testing.assert_allclose(to_numpy(quantiles(x, num_quantiles=4, interpolation="lower")), [0, 2, 5, 7, 10])


def test_shapes():
    x = np.random.randn(4, 5, 6).astype("float32")
    assert ops.shape(percentile(x, 50.0, axis=1)) == (4, 6)
    assert ops.shape(percentile(x, [10.0, 90.0], axis=[0, 2])) == (2, 5)
    assert ops.shape(percentile(x, [10.0, 90.0], axis=1, keepdims=True)) == (2, 4, 1, 6)
    assert ops.shape(percentile(x, 50.0, keepdims=True)) == (1, 1, 1)
    assert ops.shape(quantiles(x, 3, axis=0)) == (4, 5, 6)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 50.0, interpolation="cubic")
    with pytest.raises(TypeError):
        percentile(np.array([1, 2, 3], "int32"), 50.0, interpolation="linear")
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], [[50.0]])
