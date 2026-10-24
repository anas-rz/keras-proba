"""Smoke and correctness tests for the spectral normalization wrappers. Run once per backend via `KERAS_BACKEND`.

The spectral norm magnitude and Lipschitz tests are ported from TensorFlow Model Garden's
`spectral_normalization_test.py`.

## References:

[1] Hanie Sedghi, Vineet Gupta, Philip M. Long.
    The Singular Values of Convolutional Layers.
    In _International Conference on Learning Representations_, 2019.
"""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.spectral_normalization import (
    SpectralNormalization,
    SpectralNormalizationConv2D,
)

NUM_ITERATIONS = 1000
NORM_MULTIPLIER = 0.95


def compute_spectral_norm(weight):
    if weight.ndim > 2:
        # Computes Conv2D via FFT transform as in [1].
        weight = np.fft.fft2(weight, weight.shape[1:3], axes=[0, 1])
    return np.max(np.linalg.svd(weight, compute_uv=False))


@pytest.mark.parametrize(
    "input_shape, make_layer, norm_wrapper",
    [
        ((None, 10), lambda: keras.layers.Dense(10), SpectralNormalization),
        (
            (None, 32, 32, 3),
            lambda: keras.layers.Conv2D(filters=64, kernel_size=3, padding="valid"),
            SpectralNormalizationConv2D,
        ),
    ],
)
def test_spec_norm_magnitude(input_shape, make_layer, norm_wrapper):
    """Tests if the weights' spectral norm converges to norm_multiplier."""
    keras.utils.set_random_seed(0)
    layer = make_layer()
    layer.build(input_shape)
    sn_layer = norm_wrapper(layer, iteration=NUM_ITERATIONS, norm_multiplier=NORM_MULTIPLIER)

    # Perform normalization.
    sn_layer.build(input_shape)
    sn_layer.update_weights()
    normalized_kernel = ops.convert_to_numpy(sn_layer.layer.kernel)

    spectral_norm_computed = compute_spectral_norm(normalized_kernel)
    np.testing.assert_allclose(spectral_norm_computed, NORM_MULTIPLIER, atol=1e-1)

    # The normalized layer is K-Lipschitz: ||f(x1) - f(x2)|| <= K * ||x1 - x2||.
    rng = np.random.default_rng(0)
    new_input_shape = (16,) + input_shape[1:]
    new_input = rng.uniform(size=new_input_shape).astype("float32")
    delta_vec = rng.uniform(size=new_input_shape).astype("float32")
    output1 = ops.convert_to_numpy(sn_layer(new_input))
    output2 = ops.convert_to_numpy(sn_layer(new_input + delta_vec))

    delta_input = np.linalg.norm(delta_vec.reshape(-1))
    delta_output = np.linalg.norm((output2 - output1).reshape(-1))
    assert delta_output <= NORM_MULTIPLIER * delta_input


def test_dense_kernel_is_bounded_after_calls():
    keras.utils.set_random_seed(0)
    layer = keras.layers.Dense(8, kernel_initializer=keras.initializers.RandomNormal(stddev=5.0))
    sn = SpectralNormalization(layer, iteration=1, norm_multiplier=0.5)
    x = np.random.randn(4, 12).astype("float32")
    for _ in range(50):
        sn(x, training=True)
    sigma = compute_spectral_norm(ops.convert_to_numpy(sn.layer.kernel))
    np.testing.assert_allclose(sigma, 0.5, rtol=1e-3)


def test_small_kernel_is_unchanged():
    layer = keras.layers.Dense(4, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))
    layer.build((None, 4))
    kernel = ops.convert_to_numpy(layer.kernel).copy()
    sn = SpectralNormalization(layer, norm_multiplier=10.0)
    sn(np.random.randn(2, 4).astype("float32"), training=True)
    np.testing.assert_allclose(ops.convert_to_numpy(sn.layer.kernel), kernel, rtol=1e-6)


def test_inference_does_not_update_power_iteration():
    sn = SpectralNormalization(keras.layers.Dense(4))
    sn.build((None, 3))
    u, v = ops.convert_to_numpy(sn.u).copy(), ops.convert_to_numpy(sn.v).copy()
    sn(np.random.randn(2, 3).astype("float32"), training=False)
    np.testing.assert_array_equal(ops.convert_to_numpy(sn.u), u)
    np.testing.assert_array_equal(ops.convert_to_numpy(sn.v), v)

    sn(np.random.randn(2, 3).astype("float32"), training=True)
    assert not np.allclose(ops.convert_to_numpy(sn.u), u)


def test_weights_and_shapes():
    sn = SpectralNormalization(keras.layers.Dense(7))
    out = sn(np.zeros((2, 3, 5), "float32"))
    assert ops.shape(out) == (2, 3, 7)
    assert sn.u.shape == (1, 7)
    assert sn.v.shape == (1, 5)
    assert sorted(w.name for w in sn.trainable_weights) == ["bias", "kernel"]
    assert sorted(w.name for w in sn.non_trainable_weights) == ["u", "v"]

    conv = SpectralNormalizationConv2D(keras.layers.Conv2D(6, 3, strides=2, padding="same"))
    out = conv(np.zeros((2, 8, 8, 3), "float32"))
    assert ops.shape(out) == (2, 4, 4, 6)
    assert conv.v.shape == (1, 8, 8, 3)
    assert conv.u.shape == (1, 4, 4, 6)


def test_functional_model():
    inputs = keras.Input(shape=(16, 16, 3))
    h = SpectralNormalizationConv2D(keras.layers.Conv2D(4, 3, padding="same"))(inputs)
    h = keras.layers.GlobalAveragePooling2D()(h)
    outputs = SpectralNormalization(keras.layers.Dense(2))(h)
    model = keras.Model(inputs, outputs)
    assert model.output_shape == (None, 2)
    assert model.predict(np.zeros((3, 16, 16, 3), "float32"), verbose=0).shape == (3, 2)


def test_legacy_mode_uses_batch_size():
    conv = SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3, padding="same"), legacy_mode=True)
    conv.build((4, 6, 6, 1))
    assert conv.v.shape == (4, 6, 6, 1)
    assert conv.u.shape == (4, 6, 6, 2)


def test_invalid_layers():
    with pytest.raises(ValueError):
        SpectralNormalization("dense")
    with pytest.raises(ValueError):
        SpectralNormalizationConv2D(keras.layers.Dense(2))
    with pytest.raises(ValueError):
        SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3, data_format="channels_first"))
    with pytest.raises(ValueError):
        SpectralNormalization(keras.layers.ReLU()).build((None, 3))
    with pytest.raises(ValueError):
        SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3)).build((None, None, None, 3))


def test_names():
    assert SpectralNormalization(keras.layers.Dense(2, name="d"), inhere_layer_name=True).name == "d"
    assert SpectralNormalization(keras.layers.Dense(2, name="d"), name="sn").name == "sn"
    assert SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3, name="c")).layer.name == "c_spec_norm"
