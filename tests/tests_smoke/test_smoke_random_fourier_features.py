"""Smoke and correctness tests for RandomFourierFeatures. Run once per backend via `KERAS_BACKEND`."""

import math

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.random_fourier_features import RandomFourierFeatures


@pytest.mark.parametrize("kernel_initializer", ["gaussian", "laplacian"])
def test_output_shape_and_range(kernel_initializer):
    layer = RandomFourierFeatures(32, kernel_initializer=kernel_initializer)
    x = np.random.randn(4, 10).astype("float32")
    out = ops.convert_to_numpy(layer(x))
    assert out.shape == (4, 32)
    assert np.all(np.abs(out) <= 1.0 + 1e-6)
    assert layer.compute_output_shape((None, 10)) == (None, 32)


def test_higher_rank_inputs():
    layer = RandomFourierFeatures(16)
    x = np.random.randn(2, 3, 5, 8).astype("float32")
    out = ops.convert_to_numpy(layer(x))
    assert out.shape == (2, 3, 5, 16)
    # Each position is mapped independently with the same random features.
    flat = ops.convert_to_numpy(layer(x.reshape(-1, 8)))
    np.testing.assert_allclose(out.reshape(-1, 16), flat, atol=1e-6)


def test_symbolic_functional_model():
    inputs = keras.Input(shape=(6,))
    outputs = RandomFourierFeatures(24)(inputs)
    model = keras.Model(inputs, outputs)
    assert model.output_shape == (None, 24)
    assert model.predict(np.zeros((3, 6), "float32"), verbose=0).shape == (3, 24)


def test_formula():
    layer = RandomFourierFeatures(20, scale=2.5)
    x = np.random.randn(5, 3).astype("float32")
    out = ops.convert_to_numpy(layer(x))
    w = ops.convert_to_numpy(layer.unscaled_kernel)
    b = ops.convert_to_numpy(layer.bias)
    np.testing.assert_allclose(out, np.cos(x @ w / 2.5 + b), atol=1e-5)


def test_weights_and_defaults():
    layer = RandomFourierFeatures(50)
    layer.build((None, 18))
    assert layer.unscaled_kernel.shape == (18, 50)
    assert layer.bias.shape == (50,)
    assert layer.kernel_scale.shape == (1,)
    assert math.isclose(layer.scale, 3.0)
    assert layer.trainable_weights == []
    bias = ops.convert_to_numpy(layer.bias)
    assert np.all((bias >= 0) & (bias <= 2 * math.pi))

    layer = RandomFourierFeatures(50, kernel_initializer="laplacian")
    layer.build((None, 18))
    assert layer.scale == 1.0


def test_custom_initializer():
    layer = RandomFourierFeatures(8, kernel_initializer=keras.initializers.Ones(), scale=1.0)
    x = np.random.randn(3, 4).astype("float32")
    out = ops.convert_to_numpy(layer(x))
    b = ops.convert_to_numpy(layer.bias)
    np.testing.assert_allclose(out, np.cos(x.sum(-1, keepdims=True) + b), atol=1e-5)

    layer = RandomFourierFeatures(8, kernel_initializer={"class_name": "Zeros", "config": {}})
    layer.build((None, 4))
    assert np.all(ops.convert_to_numpy(layer.unscaled_kernel) == 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_dim": 0},
        {"output_dim": -3},
        {"output_dim": 4, "kernel_initializer": "polynomial"},
        {"output_dim": 4, "scale": 0.0},
    ],
)
def test_invalid_args(kwargs):
    with pytest.raises(ValueError):
        RandomFourierFeatures(**kwargs)


def test_invalid_input_shapes():
    with pytest.raises(ValueError):
        RandomFourierFeatures(4).build((None,))
    with pytest.raises(ValueError):
        RandomFourierFeatures(4).build((None, None))


@pytest.mark.parametrize(
    "kernel_initializer, scale, kernel_fn",
    [
        ("gaussian", 1.5, lambda d, s: np.exp(-np.sum(d**2, -1) / (2 * s**2))),
        ("laplacian", 2.0, lambda d, s: np.exp(-np.sum(np.abs(d), -1) / s)),
    ],
)
def test_kernel_approximation(kernel_initializer, scale, kernel_fn):
    keras.utils.set_random_seed(0)
    output_dim = 20000
    layer = RandomFourierFeatures(output_dim, kernel_initializer=kernel_initializer, scale=scale)
    x = np.random.randn(10, 3).astype("float32") * 0.8
    phi = ops.convert_to_numpy(layer(x)).astype("float64")
    approx = (2.0 / output_dim) * phi @ phi.T
    exact = kernel_fn(x[:, None, :] - x[None, :, :], scale)
    assert np.abs(approx - exact).max() < 0.05
