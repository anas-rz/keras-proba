"""Smoke and correctness tests for the spectral normalization wrappers. Run once per backend via `KERAS_BACKEND`.

The spectral norm magnitude and Lipschitz tests are ported from TensorFlow Model Garden's
`spectral_normalization_test.py`.

## References:

[1] Hanie Sedghi, Vineet Gupta, Philip M. Long.
    The Singular Values of Convolutional Layers.
    In _International Conference on Learning Representations_, 2019.
"""

import warnings

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.spectral_normalization import (
    SpectralNormalization,
    SpectralNormalizationConv2D,
)

# The reference uses 1000 power iterations; 100 converge to the same tolerance.
NUM_ITERATIONS = 100
NORM_MULTIPLIER = 0.95


def compute_spectral_norm(weight):
    if weight.ndim > 2:
        # Computes Conv2D via FFT transform as in [1].
        weight = np.fft.fft2(weight, weight.shape[1:3], axes=[0, 1])
    return np.max(np.linalg.svd(weight, compute_uv=False))


@pytest.mark.parametrize(
    "input_shape, make_layer, norm_wrapper, mode",
    [
        ((None, 10), lambda: keras.layers.Dense(10), SpectralNormalization, "projection"),
        (
            (None, 32, 32, 3),
            lambda: keras.layers.Conv2D(filters=64, kernel_size=3, padding="valid"),
            SpectralNormalizationConv2D,
            "reparameterization",
        ),
    ],
)
def test_spec_norm_magnitude(input_shape, make_layer, norm_wrapper, mode):
    """Tests if the weights' spectral norm converges to norm_multiplier."""
    keras.utils.set_random_seed(0)
    layer = make_layer()
    layer.build(input_shape)
    sn_layer = norm_wrapper(layer, iteration=NUM_ITERATIONS, norm_multiplier=NORM_MULTIPLIER, mode=mode)

    # Perform normalization.
    sn_layer.build(input_shape)
    normalized_kernel = ops.convert_to_numpy(sn_layer.update_weights())
    if mode == "projection":
        np.testing.assert_allclose(ops.convert_to_numpy(sn_layer.layer.kernel), normalized_kernel)

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


@pytest.mark.parametrize("wrapper_cls", [SpectralNormalization, SpectralNormalizationConv2D])
def test_reparameterization_leaves_kernel_unchanged(wrapper_cls):
    keras.utils.set_random_seed(0)
    init = keras.initializers.RandomNormal(stddev=5.0)
    if wrapper_cls is SpectralNormalization:
        layer, x = keras.layers.Dense(8, kernel_initializer=init), np.random.randn(4, 6).astype("float32")
    else:
        layer = keras.layers.Conv2D(4, 3, padding="same", kernel_initializer=init)
        x = np.random.randn(2, 8, 8, 3).astype("float32")
    sn = wrapper_cls(layer, iteration=3, norm_multiplier=0.5, mode="reparameterization")
    sn.build(x.shape)
    kernel = ops.convert_to_numpy(sn.layer.kernel).copy()
    for _ in range(3):
        out = ops.convert_to_numpy(sn(x, training=True))
    np.testing.assert_array_equal(ops.convert_to_numpy(sn.layer.kernel), kernel)

    # The output uses the normalized kernel, i.e. the raw kernel scaled by c / sigma < 1.
    # Recompute it from the stored u and v, without another power iteration.
    if wrapper_cls is SpectralNormalization:
        w_norm = sn.update_weights(training=False)
    else:
        sn.do_power_iteration = False
        w_norm = sn.update_weights()
    w_norm = ops.convert_to_numpy(w_norm)
    ratio = (w_norm / kernel).ravel()
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)
    assert ratio[0] < 1
    expected = layer.stateless_call([w_norm, layer.bias.value], [], x)[0]
    np.testing.assert_allclose(out, ops.convert_to_numpy(expected), atol=1e-5)


def test_reparameterization_inference_is_bounded():
    keras.utils.set_random_seed(0)
    layer = keras.layers.Dense(8, kernel_initializer=keras.initializers.RandomNormal(stddev=5.0))
    sn = SpectralNormalization(layer, iteration=50, norm_multiplier=0.5, mode="reparameterization")
    sn.build((None, 12))
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=(64, 12)).astype("float32")
    x2 = x1 + rng.normal(size=(64, 12)).astype("float32")
    out1 = ops.convert_to_numpy(sn(x1, training=False))
    out2 = ops.convert_to_numpy(sn(x2, training=False))
    ratios = np.linalg.norm(out2 - out1, axis=-1) / np.linalg.norm(x2 - x1, axis=-1)
    assert ratios.max() <= 0.5 * 1.001


@pytest.mark.parametrize(
    "make_wrapper",
    [
        lambda mode: SpectralNormalization(keras.layers.Dense(2), mode=mode),
        lambda mode: SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3), mode=mode),
    ],
)
def test_projection_mode_warns_on_jax(make_wrapper):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_wrapper("projection")
    messages = [str(w.message) for w in caught if 'mode="reparameterization"' in str(w.message)]
    assert len(messages) == (1 if keras.backend.backend() == "jax" else 0)
    # The warning points at the caller, not at this module.
    assert all(w.filename == __file__ for w in caught if 'mode="reparameterization"' in str(w.message))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_wrapper("reparameterization")
    assert not [w for w in caught if 'mode="reparameterization"' in str(w.message)]


def test_invalid_mode():
    with pytest.raises(ValueError):
        SpectralNormalization(keras.layers.Dense(2), mode="power")
    with pytest.raises(ValueError):
        SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3), mode="power")


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


