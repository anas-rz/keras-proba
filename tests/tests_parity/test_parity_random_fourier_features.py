"""Numerical parity tests comparing Keras 3 RandomFourierFeatures against `tf_keras.layers.experimental.RandomFourierFeatures`."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
tf_keras = pytest.importorskip("tf_keras")

import keras
from keras import ops

from keras_proba.layers.random_fourier_features import RandomFourierFeatures

TFRandomFourierFeatures = tf_keras.layers.experimental.RandomFourierFeatures


def build_pair(input_dim, output_dim, **kwargs):
    """Builds both layers and copies the reference weights into the Keras 3 layer."""
    tf_layer = TFRandomFourierFeatures(output_dim, **kwargs)
    tf_layer.build((None, input_dim))
    k_layer = RandomFourierFeatures(output_dim, **kwargs)
    k_layer.build((None, input_dim))
    k_layer.unscaled_kernel.assign(tf_layer.unscaled_kernel.numpy())
    k_layer.bias.assign(tf_layer.bias.numpy())
    k_layer.kernel_scale.assign(tf_layer.kernel_scale.numpy())
    return tf_layer, k_layer


@pytest.mark.parametrize("kernel_initializer", ["gaussian", "laplacian"])
@pytest.mark.parametrize("scale", [None, 0.5, 3.0])
def test_forward_parity(kernel_initializer, scale):
    np.random.seed(0)
    tf_layer, k_layer = build_pair(16, 64, kernel_initializer=kernel_initializer, scale=scale)
    x = np.random.randn(8, 16).astype("float32")

    tf_out = tf_layer(x).numpy()
    k_out = ops.convert_to_numpy(k_layer(x))
    diff = np.abs(tf_out - k_out).max()
    # Heavy-tailed Cauchy weights give large pre-activations, where float32 cos
    # differs across matmul implementations by more than for Gaussian weights.
    tol = 1e-4 if kernel_initializer == "laplacian" else 1e-5
    assert diff < tol, f"Forward parity diff {diff}"


@pytest.mark.parametrize("kernel_initializer", ["gaussian", "laplacian", "GAUSSIAN"])
@pytest.mark.parametrize("input_dim", [1, 7, 32])
def test_default_scale_parity(kernel_initializer, input_dim):
    tf_layer = TFRandomFourierFeatures(10, kernel_initializer=kernel_initializer)
    tf_layer.build((None, input_dim))
    k_layer = RandomFourierFeatures(10, kernel_initializer=kernel_initializer)
    k_layer.build((None, input_dim))

    assert np.isclose(k_layer.scale, tf_layer.scale)
    np.testing.assert_allclose(
        ops.convert_to_numpy(k_layer.kernel_scale), tf_layer.kernel_scale.numpy(), rtol=1e-6
    )


def test_weight_structure_parity():
    tf_layer, k_layer = build_pair(5, 12)
    tf_weights = {w.name.split("/")[-1].split(":")[0]: tuple(w.shape) for w in tf_layer.weights}
    k_weights = {w.name: tuple(w.shape) for w in k_layer.weights}
    assert k_weights == tf_weights
    # Neither layer exposes trainable weights unless `trainable=True`.
    assert len(k_layer.trainable_weights) == len(tf_layer.trainable_weights) == 0

    tf_layer, k_layer = build_pair(5, 12, trainable=True)
    assert [w.name for w in k_layer.trainable_weights] == ["kernel_scale"]
    assert len(tf_layer.trainable_weights) == 1


def test_scale_gradient_parity():
    np.random.seed(1)
    tf_layer, k_layer = build_pair(6, 32, scale=2.0, trainable=True)
    x = np.random.randn(4, 6).astype("float32")

    with tf.GradientTape() as tape:
        tf_loss = tf.reduce_sum(tf.square(tf_layer(x)))
    tf_grad = tape.gradient(tf_loss, tf_layer.kernel_scale).numpy()

    # Gradient of the same loss computed analytically from the Keras 3 forward pass:
    # d/ds sum(cos(z)^2) with z = x W / s + b  ->  sum(-2 cos(z) sin(z) * (-(x W) / s^2)).
    w = ops.convert_to_numpy(k_layer.unscaled_kernel)
    b = ops.convert_to_numpy(k_layer.bias)
    s = ops.convert_to_numpy(k_layer.kernel_scale)
    xw = x @ w
    z = xw / s + b
    np.testing.assert_allclose(ops.convert_to_numpy(k_layer(x)), np.cos(z), atol=1e-5)
    k_grad = np.sum(-2 * np.cos(z) * np.sin(z) * (-xw / s**2))
    np.testing.assert_allclose(k_grad, tf_grad[0], rtol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_dim": 0},
        {"output_dim": 4, "kernel_initializer": "polynomial"},
        {"output_dim": 4, "scale": 0.0},
        {"output_dim": 4, "scale": -1.0},
    ],
)
def test_invalid_args_parity(kwargs):
    with pytest.raises(ValueError):
        TFRandomFourierFeatures(**kwargs)
    with pytest.raises(ValueError):
        RandomFourierFeatures(**kwargs)


def test_config_parity():
    tf_layer = TFRandomFourierFeatures(20, kernel_initializer="laplacian", scale=1.5, trainable=True)
    k_layer = RandomFourierFeatures(20, kernel_initializer="laplacian", scale=1.5, trainable=True)
    tf_config, k_config = tf_layer.get_config(), k_layer.get_config()
    for key in ["output_dim", "kernel_initializer", "scale", "trainable"]:
        assert k_config[key] == tf_config[key], key
