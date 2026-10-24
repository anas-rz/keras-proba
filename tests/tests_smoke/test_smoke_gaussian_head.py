"""Smoke and correctness tests for GaussianHead and GaussianNLL. Run once per backend via `KERAS_BACKEND`."""

import math

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.gaussian_head import GaussianHead, GaussianNLL, gaussian_nll, split_gaussian_params


@pytest.mark.parametrize("units", [1, 4])
def test_output_shape_and_positive_scale(units):
    head = GaussianHead(units, min_scale=0.05)
    x = np.random.randn(8, 6).astype("float32") * 10
    out = head(x)
    assert ops.shape(out) == (8, 2 * units)
    assert head.compute_output_shape((None, 6)) == (None, 2 * units)
    mean, scale = (ops.convert_to_numpy(t) for t in split_gaussian_params(out))
    assert mean.shape == scale.shape == (8, units)
    assert np.all(scale >= 0.05)


def test_higher_rank_inputs():
    head = GaussianHead(3)
    out = head(np.random.randn(2, 5, 7).astype("float32"))
    assert ops.shape(out) == (2, 5, 6)


def test_functional_model():
    inputs = keras.Input((4,))
    model = keras.Model(inputs, GaussianHead(2)(keras.layers.Dense(8)(inputs)))
    assert model.output_shape == (None, 4)


def test_formula():
    head = GaussianHead(2, min_scale=0.1)
    x = np.random.randn(3, 4).astype("float32")
    out = ops.convert_to_numpy(head(x))
    w_m, b_m = (ops.convert_to_numpy(w) for w in head.mean_dense.weights)
    w_s, b_s = (ops.convert_to_numpy(w) for w in head.scale_dense.weights)
    np.testing.assert_allclose(out[:, :2], x @ w_m + b_m, atol=1e-5)
    np.testing.assert_allclose(out[:, 2:], np.log1p(np.exp(x @ w_s + b_s)) + 0.1, atol=1e-5)


def test_initial_scale_from_bias():
    head = GaussianHead(1, min_scale=0.0, kernel_initializer="zeros", scale_bias_initializer=keras.initializers.Constant(2.0))
    _, scale = split_gaussian_params(head(np.random.randn(4, 3).astype("float32")))
    np.testing.assert_allclose(ops.convert_to_numpy(scale), np.log1p(np.exp(2.0)), rtol=1e-6)


def test_weights_and_regularization():
    head = GaussianHead(2, kernel_regularizer=keras.regularizers.L2(0.1), bias_regularizer="l1")
    head.build((None, 3))
    # Drop the (uniquified) layer name from each path.
    assert sorted(w.path.split("/", 1)[1] for w in head.trainable_weights) == [
        "mean/bias",
        "mean/kernel",
        "scale/bias",
        "scale/kernel",
    ]
    head(np.zeros((1, 3), "float32"))
    assert len(head.losses) == 4


def test_nll_values():
    # Standard normal at its mean: 0.5 * log(2 pi).
    y_pred = np.array([[0.0, 1.0]], "float32")
    np.testing.assert_allclose(ops.convert_to_numpy(gaussian_nll(np.zeros((1, 1)), y_pred)), [0.5 * math.log(2 * math.pi)], rtol=1e-6)
    # One scale away from the mean, with scale 2.
    y_pred = np.array([[1.0, 2.0]], "float32")
    expected = 0.5 * math.log(2 * math.pi) + math.log(2.0) + 0.5
    np.testing.assert_allclose(ops.convert_to_numpy(gaussian_nll(np.array([[3.0]]), y_pred)), [expected], rtol=1e-6)


def test_nll_accepts_rank_one_targets():
    y_pred = np.array([[0.0, 1.0], [1.0, 2.0]], "float32")
    np.testing.assert_allclose(
        ops.convert_to_numpy(gaussian_nll(np.array([0.0, 3.0]), y_pred)),
        ops.convert_to_numpy(gaussian_nll(np.array([[0.0], [3.0]]), y_pred)),
    )


def test_nll_is_minimized_at_true_parameters():
    """The expected NLL over samples from N(2, 0.5^2) is lowest at mean 2, scale 0.5."""
    y = np.random.default_rng(0).normal(2.0, 0.5, size=(20000, 1)).astype("float32")
    loss = GaussianNLL()

    def nll(mean, scale):
        y_pred = np.tile(np.array([[mean, scale]], "float32"), (len(y), 1))
        return float(ops.convert_to_numpy(loss(y, y_pred)))

    best = nll(2.0, 0.5)
    for mean, scale in [(1.8, 0.5), (2.2, 0.5), (2.0, 0.4), (2.0, 0.6)]:
        assert nll(mean, scale) > best


@pytest.mark.parametrize("kwargs", [{"units": 0}, {"units": 2, "min_scale": -1.0}])
def test_invalid_args(kwargs):
    with pytest.raises(ValueError):
        GaussianHead(**kwargs)
