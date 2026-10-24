"""Numerical parity tests comparing the Keras 3 spectral normalization wrappers against TensorFlow Model Garden's implementation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
tf_keras = pytest.importorskip("tf_keras")

import keras
from keras import ops

from keras_proba.layers.spectral_normalization import (
    SpectralNormalization,
    SpectralNormalizationConv2D,
)

REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "reference/models/official/nlp/modeling/layers/spectral_normalization.py"
)
if not REFERENCE_PATH.exists():
    pytest.skip(f"Reference implementation not found at {REFERENCE_PATH}", allow_module_level=True)

# Load the single module by path to avoid importing the whole Model Garden package.
_spec = importlib.util.spec_from_file_location("tfm_spectral_normalization", REFERENCE_PATH)
tfm_sn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfm_sn)


class _TFWithLegacyInitializers:
    """`tf` whose `initializers` are tf_keras's, so the reference does not run on the Keras 3 backend.

    `tf.initializers` resolves to Keras 3 when it is installed, which breaks the
    reference under `KERAS_BACKEND=jax`.
    """

    initializers = tf_keras.initializers

    def __getattr__(self, name):
        return getattr(tf, name)


tfm_sn.tf = _TFWithLegacyInitializers()


def copy_weights(tf_wrapper, k_wrapper):
    """Copies the reference wrapper's kernel, bias and power iteration vectors."""
    k_wrapper.layer.set_weights(tf_wrapper.layer.get_weights())
    k_wrapper.u.assign(tf_wrapper.u.numpy())
    k_wrapper.v.assign(tf_wrapper.v.numpy())


def assert_state_close(tf_wrapper, k_wrapper, atol=1e-5):
    np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper.u), tf_wrapper.u.numpy(), atol=atol)
    np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper.v), tf_wrapper.v.numpy(), atol=atol)
    np.testing.assert_allclose(
        ops.convert_to_numpy(k_wrapper.layer.kernel), tf_wrapper.layer.kernel.numpy(), atol=atol
    )


def build_dense_pair(input_dim=6, units=5, **kwargs):
    tf_wrapper = tfm_sn.SpectralNormalization(tf_keras.layers.Dense(units), **kwargs)
    tf_wrapper.build((None, input_dim))
    k_wrapper = SpectralNormalization(keras.layers.Dense(units), **kwargs)
    k_wrapper.build((None, input_dim))
    copy_weights(tf_wrapper, k_wrapper)
    return tf_wrapper, k_wrapper


def build_conv_pair(input_shape, conv_kwargs, **kwargs):
    tf_wrapper = tfm_sn.SpectralNormalizationConv2D(tf_keras.layers.Conv2D(**conv_kwargs), **kwargs)
    tf_wrapper.build(input_shape)
    k_wrapper = SpectralNormalizationConv2D(keras.layers.Conv2D(**conv_kwargs), **kwargs)
    k_wrapper.build(input_shape)
    copy_weights(tf_wrapper, k_wrapper)
    return tf_wrapper, k_wrapper


@pytest.mark.parametrize("iteration", [1, 3])
@pytest.mark.parametrize("norm_multiplier", [0.95, 0.5, 100.0])
def test_dense_forward_parity(iteration, norm_multiplier):
    rng = np.random.default_rng(0)
    tf_wrapper, k_wrapper = build_dense_pair(iteration=iteration, norm_multiplier=norm_multiplier)
    # Scale the kernel up so that normalization is active.
    kernel = rng.normal(size=(6, 5)).astype("float32") * 3
    tf_wrapper.layer.kernel.assign(kernel)
    k_wrapper.layer.kernel.assign(kernel)

    for _ in range(3):
        x = rng.normal(size=(8, 6)).astype("float32")
        np.testing.assert_allclose(
            ops.convert_to_numpy(k_wrapper(x, training=True)), tf_wrapper(x, training=True).numpy(), atol=1e-5
        )
        assert_state_close(tf_wrapper, k_wrapper)

    x = rng.normal(size=(8, 6)).astype("float32")
    np.testing.assert_allclose(
        ops.convert_to_numpy(k_wrapper(x, training=False)), tf_wrapper(x, training=False).numpy(), atol=1e-5
    )


@pytest.mark.parametrize("iteration", [1, 3])
def test_dense_reparameterization_forward_parity(iteration):
    """Without optimizer steps, reparameterization gives the reference's outputs and u/v."""
    rng = np.random.default_rng(0)
    tf_wrapper, _ = build_dense_pair(iteration=iteration)
    k_wrapper = SpectralNormalization(keras.layers.Dense(5), iteration=iteration, mode="reparameterization")
    k_wrapper.build((None, 6))
    kernel = rng.normal(size=(6, 5)).astype("float32") * 3
    tf_wrapper.layer.kernel.assign(kernel)
    copy_weights(tf_wrapper, k_wrapper)

    for _ in range(3):
        x = rng.normal(size=(8, 6)).astype("float32")
        np.testing.assert_allclose(
            ops.convert_to_numpy(k_wrapper(x, training=True)), tf_wrapper(x, training=True).numpy(), atol=1e-5
        )
        np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper.u), tf_wrapper.u.numpy(), atol=1e-5)
        np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper.v), tf_wrapper.v.numpy(), atol=1e-5)
    # The reference projected its kernel; ours stays as it was.
    np.testing.assert_array_equal(ops.convert_to_numpy(k_wrapper.layer.kernel), kernel)


def test_conv_reparameterization_forward_parity():
    rng = np.random.default_rng(0)
    conv_kwargs = {"filters": 4, "kernel_size": 3, "padding": "same"}
    tf_wrapper, _ = build_conv_pair((None, 8, 8, 3), conv_kwargs)
    k_wrapper = SpectralNormalizationConv2D(keras.layers.Conv2D(**conv_kwargs), mode="reparameterization")
    k_wrapper.build((None, 8, 8, 3))
    copy_weights(tf_wrapper, k_wrapper)

    for _ in range(3):
        x = rng.normal(size=(2, 8, 8, 3)).astype("float32")
        np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper(x)), tf_wrapper(x).numpy(), atol=1e-4)


@pytest.mark.parametrize("training", [True, False])
def test_dense_default_training_parity(training):
    rng = np.random.default_rng(1)
    tf_wrapper, k_wrapper = build_dense_pair(training=training)
    x = rng.normal(size=(4, 6)).astype("float32")
    np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper(x)), tf_wrapper(x).numpy(), atol=1e-5)
    assert_state_close(tf_wrapper, k_wrapper)


def test_dense_build_normalizes_kernel_parity():
    """Both implementations normalize the kernel once at build time."""
    keras.utils.set_random_seed(0)
    tf_layer = tf_keras.layers.Dense(5, kernel_initializer="ones")
    k_layer = keras.layers.Dense(5, kernel_initializer="ones")
    tf_wrapper = tfm_sn.SpectralNormalization(tf_layer)
    k_wrapper = SpectralNormalization(k_layer)
    tf_wrapper.build((None, 6))
    k_wrapper.build((None, 6))
    # For an all-ones kernel, any u/v start converges in one step to sigma = sqrt(30).
    np.testing.assert_allclose(
        ops.convert_to_numpy(k_wrapper.layer.kernel), tf_wrapper.layer.kernel.numpy(), atol=1e-5
    )


@pytest.mark.parametrize(
    "input_shape, conv_kwargs",
    [
        ((None, 8, 8, 3), {"filters": 4, "kernel_size": 3, "padding": "same"}),
        ((None, 8, 8, 3), {"filters": 4, "kernel_size": 3, "padding": "valid"}),
        ((None, 8, 8, 3), {"filters": 4, "kernel_size": 3, "strides": 2, "padding": "same"}),
        ((None, 12, 8, 2), {"filters": 5, "kernel_size": (5, 3), "strides": (2, 1), "padding": "same"}),
    ],
)
@pytest.mark.parametrize("iteration", [1, 3])
def test_conv_forward_parity(input_shape, conv_kwargs, iteration):
    rng = np.random.default_rng(0)
    tf_wrapper, k_wrapper = build_conv_pair(input_shape, conv_kwargs, iteration=iteration)

    for _ in range(3):
        x = rng.normal(size=(2,) + input_shape[1:]).astype("float32")
        np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper(x)), tf_wrapper(x).numpy(), atol=1e-4)
        assert_state_close(tf_wrapper, k_wrapper, atol=1e-4)


def test_conv_no_power_iteration_parity():
    rng = np.random.default_rng(0)
    tf_wrapper, k_wrapper = build_conv_pair(
        (None, 8, 8, 3), {"filters": 4, "kernel_size": 3, "padding": "same"}, training=False
    )
    x = rng.normal(size=(2, 8, 8, 3)).astype("float32")
    np.testing.assert_allclose(ops.convert_to_numpy(k_wrapper(x)), tf_wrapper(x).numpy(), atol=1e-4)
    assert_state_close(tf_wrapper, k_wrapper, atol=1e-4)


def test_conv_layer_name_parity():
    tf_wrapper = tfm_sn.SpectralNormalizationConv2D(tf_keras.layers.Conv2D(2, 3, name="conv"))
    k_wrapper = SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3, name="conv"))
    assert k_wrapper.layer.name == tf_wrapper.layer.name == "conv_spec_norm"


def test_inhere_layer_name_parity():
    tf_wrapper = tfm_sn.SpectralNormalization(tf_keras.layers.Dense(2, name="dense_x"), inhere_layer_name=True)
    k_wrapper = SpectralNormalization(keras.layers.Dense(2, name="dense_x"), inhere_layer_name=True)
    assert k_wrapper.name == tf_wrapper.name == "dense_x"


def test_invalid_layer_parity():
    with pytest.raises(ValueError):
        tfm_sn.SpectralNormalization("dense")
    with pytest.raises(ValueError):
        SpectralNormalization("dense")
    with pytest.raises(ValueError):
        tfm_sn.SpectralNormalizationConv2D(tf_keras.layers.Dense(2))
    with pytest.raises(ValueError):
        SpectralNormalizationConv2D(keras.layers.Dense(2))


@pytest.mark.skipif(
    keras.backend.backend() == "jax",
    reason="JAX discards assignments to trainable variables in `fit`, so the kernel projection is not persisted.",
)
def test_dense_training_parity():
    """Full-batch SGD steps produce the same kernel as the reference."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 6)).astype("float32")
    y = rng.normal(size=(32, 5)).astype("float32") * 5

    tf_wrapper, k_wrapper = build_dense_pair(iteration=2, norm_multiplier=0.9)
    tf_model = tf_keras.Sequential([tf_keras.Input((6,)), tf_wrapper])
    tf_model.compile(optimizer=tf_keras.optimizers.SGD(0.1), loss="mse")
    k_model = keras.Sequential([keras.Input((6,)), k_wrapper])
    k_model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")

    tf_history = tf_model.fit(x, y, batch_size=32, epochs=5, shuffle=False, verbose=0)
    k_history = k_model.fit(x, y, batch_size=32, epochs=5, shuffle=False, verbose=0)

    np.testing.assert_allclose(k_history.history["loss"], tf_history.history["loss"], rtol=1e-4)
    assert_state_close(tf_wrapper, k_wrapper, atol=1e-4)
    np.testing.assert_allclose(
        ops.convert_to_numpy(k_wrapper.layer.bias), tf_wrapper.layer.bias.numpy(), atol=1e-4
    )
