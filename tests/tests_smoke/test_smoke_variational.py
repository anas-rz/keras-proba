"""Smoke and correctness tests for the variational layers. Run once per backend via `KERAS_BACKEND`.

KL penalty, Sequential and gradient tests are ported from TensorFlow Probability's
`dense_variational_test.py` and `conv_variational_test.py`.
"""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.variational import (
    Conv2DFlipout,
    Conv2DReparameterization,
    DenseFlipout,
    DenseLocalReparameterization,
    DenseReparameterization,
    normal_kl_divergence,
)

DENSE_CLASSES = [DenseReparameterization, DenseLocalReparameterization, DenseFlipout]
CONV_CLASSES = [Conv2DReparameterization, Conv2DFlipout]


def to_numpy(x):
    return ops.convert_to_numpy(x)


@pytest.mark.parametrize("layer_cls", DENSE_CLASSES)
def test_dense_shapes_and_kl_penalty(layer_cls):
    layer = layer_cls(4)
    out = layer(np.random.randn(3, 5).astype("float32"))
    assert ops.shape(out) == (3, 4)
    assert layer.compute_output_shape((None, 5)) == (None, 4)
    assert sorted(w.name for w in layer.trainable_weights) == [
        "bias_posterior_loc",
        "kernel_posterior_loc",
        "kernel_posterior_untransformed_scale",
    ]
    assert len(layer.losses) == 1
    np.testing.assert_allclose(to_numpy(layer.losses[0]), to_numpy(layer.kl_divergence()), rtol=1e-6)


@pytest.mark.parametrize("layer_cls", CONV_CLASSES)
def test_conv_shapes_and_kl_penalty(layer_cls):
    layer = layer_cls(5, 3, strides=2, padding="same")
    out = layer(np.random.randn(2, 9, 8, 3).astype("float32"))
    assert ops.shape(out) == (2, 5, 4, 5) == layer.compute_output_shape((2, 9, 8, 3))
    assert layer.kernel_posterior_loc.shape == (3, 3, 3, 5)
    assert len(layer.losses) == 1


@pytest.mark.parametrize("layer_cls", DENSE_CLASSES + CONV_CLASSES)
def test_kl_penalty_without_bias(layer_cls):
    layer = layer_cls(2, 3) if layer_cls in CONV_CLASSES else layer_cls(2)
    layer.use_bias = False
    x = np.random.randn(2, 5, 5, 1).astype("float32") if layer_cls in CONV_CLASSES else np.random.randn(2, 3).astype("float32")
    layer(x)
    assert layer.bias_posterior_loc is None
    assert len(layer.trainable_weights) == 2
    assert len(layer.losses) == 1


def test_kl_divergence_values():
    # KL(N(0, 1) || N(0, 1)) = 0 and KL(N(1, 1) || N(0, 1)) = 0.5 per element.
    np.testing.assert_allclose(to_numpy(normal_kl_divergence(ops.zeros((3,)), ops.ones((3,)), 1.0)), 0.0, atol=1e-7)
    np.testing.assert_allclose(to_numpy(normal_kl_divergence(ops.ones((4,)), ops.ones((4,)), 1.0)), 2.0, rtol=1e-6)
    # Scale part: log(2) - log(0.5) + 0.25 / 8 - 0.5.
    expected = np.log(2.0) - np.log(0.5) + 0.25 / 8.0 - 0.5
    np.testing.assert_allclose(to_numpy(normal_kl_divergence(ops.zeros((1,)), 0.5 * ops.ones((1,)), 2.0)), expected, rtol=1e-6)


def test_kl_weight_scales_loss():
    layer = DenseFlipout(3, kl_weight=0.25)
    layer(np.random.randn(2, 4).astype("float32"))
    np.testing.assert_allclose(to_numpy(layer.losses[0]), 0.25 * to_numpy(layer.kl_divergence()), rtol=1e-6)


def test_posterior_scale_initialization():
    layer = DenseReparameterization(64)
    layer.build((None, 64))
    scale = to_numpy(layer.kernel_posterior_scale)
    # softplus(-3) ~ 0.0486: initialized close to a point mass.
    np.testing.assert_allclose(scale.mean(), np.log1p(np.exp(-3.0)), rtol=0.05)


@pytest.mark.parametrize("layer_cls", DENSE_CLASSES + CONV_CLASSES)
def test_forward_pass_is_stochastic(layer_cls):
    layer = layer_cls(3, 3) if layer_cls in CONV_CLASSES else layer_cls(3)
    x = np.random.randn(2, 6, 6, 2).astype("float32") if layer_cls in CONV_CLASSES else np.random.randn(2, 4).astype("float32")
    outputs = [to_numpy(layer(x)) for _ in range(2)]
    assert not np.allclose(outputs[0], outputs[1])


def test_seed_makes_samples_reproducible():
    x = np.random.randn(3, 4).astype("float32")
    layers = [DenseFlipout(5, seed=7, kernel_loc_initializer="zeros", bias_initializer="zeros") for _ in range(2)]
    for layer in layers:
        layer.build(x.shape)
        layer.kernel_posterior_untransformed_scale.assign(np.zeros((4, 5), "float32"))
    np.testing.assert_allclose(to_numpy(layers[0](x)), to_numpy(layers[1](x)))

    other = DenseFlipout(5, seed=8, kernel_loc_initializer="zeros", bias_initializer="zeros")
    other.build(x.shape)
    other.kernel_posterior_untransformed_scale.assign(np.zeros((4, 5), "float32"))
    assert not np.allclose(to_numpy(layers[0](x)), to_numpy(other(x)))


@pytest.mark.parametrize("layer_cls", DENSE_CLASSES)
def test_output_distribution(layer_cls):
    """Under q(W), outputs have mean `x @ loc + b` and variance `x^2 @ scale^2` for every estimator."""
    keras.utils.set_random_seed(0)
    x = np.random.default_rng(0).normal(size=(2, 3)).astype("float32")
    layer = layer_cls(2, kernel_untransformed_scale_initializer=keras.initializers.Constant(0.0))
    layer.build(x.shape)
    loc = to_numpy(layer.kernel_posterior_loc)
    scale = to_numpy(layer.kernel_posterior_scale)
    bias = to_numpy(layer.bias_posterior_loc)

    # 1500 samples: the variance estimate has a relative standard error of ~4%.
    samples = np.stack([to_numpy(layer(x)) for _ in range(1500)])
    np.testing.assert_allclose(samples.mean(0), x @ loc + bias, atol=0.1)
    np.testing.assert_allclose(samples.var(0), np.square(x) @ np.square(scale), rtol=0.15)


def test_flipout_decorrelates_examples():
    """Flipout perturbations of identical examples are uncorrelated, unlike a single shared kernel sample."""
    keras.utils.set_random_seed(0)
    x = np.ones((2, 8), "float32")
    correlations = {}
    for layer_cls in [DenseFlipout, DenseReparameterization]:
        layer = layer_cls(1, kernel_loc_initializer="zeros", bias_initializer="zeros",
                          kernel_untransformed_scale_initializer=keras.initializers.Constant(0.0))
        samples = np.stack([to_numpy(layer(x))[:, 0] for _ in range(800)])
        correlations[layer_cls] = np.corrcoef(samples.T)[0, 1]
    # The correlation estimate has a standard error of ~0.035.
    assert abs(correlations[DenseFlipout]) < 0.15
    assert correlations[DenseReparameterization] > 0.99


def test_local_reparameterization_zero_inputs_have_finite_gradients():
    """An all-zero example (e.g. after a ReLU) must not make the scale gradient NaN."""
    inputs = keras.Input((3,))
    model = keras.Model(inputs, DenseLocalReparameterization(2)(inputs))
    model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], "float32")
    model.fit(x, np.zeros((2, 2), "float32"), batch_size=2, epochs=1, verbose=0)
    for weight in model.trainable_weights:
        assert np.all(np.isfinite(to_numpy(weight))), weight.name


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow", reason="TensorFlow's CPU convolution does not support channels_first."
)
def test_conv_channels_first_matches_channels_last():
    x = np.random.default_rng(0).normal(size=(2, 7, 6, 3)).astype("float32")
    noise = np.random.default_rng(1).normal(size=(3, 3, 3, 4)).astype("float32")
    signs = [np.sign(np.random.default_rng(i).normal(size=s)).astype("float32") for i, s in [(2, (2, 3)), (3, (2, 4))]]
    outputs = {}
    for data_format in ["channels_last", "channels_first"]:
        layer = Conv2DFlipout(4, 3, padding="same", data_format=data_format, seed=1)
        layer.build(x.shape if data_format == "channels_last" else (2, 3, 7, 6))
        if data_format == "channels_last":
            weights = [to_numpy(w) for w in layer.trainable_weights]
        else:
            for w, value in zip(layer.trainable_weights, weights):
                w.assign(value)
        queue = list(signs)
        layer._standard_normal = lambda shape, dtype: ops.cast(noise, dtype)
        layer._rademacher = lambda shape, dtype: ops.reshape(ops.cast(queue.pop(0), dtype), shape)
        inputs = x if data_format == "channels_last" else np.transpose(x, (0, 3, 1, 2))
        out = to_numpy(layer(inputs))
        outputs[data_format] = out if data_format == "channels_last" else np.transpose(out, (0, 2, 3, 1))
    np.testing.assert_allclose(outputs["channels_first"], outputs["channels_last"], rtol=1e-4, atol=1e-5)


def test_dense_layers_in_sequential():
    x = np.random.uniform(-1.0, 1.0, size=(10, 3)).astype("float32")
    y = np.random.uniform(-1.0, 1.0, size=(10, 4)).astype("float32")
    model = keras.Sequential(
        [
            keras.Input((3,)),
            DenseReparameterization(6, activation="relu"),
            DenseFlipout(6, activation="relu"),
            DenseLocalReparameterization(4),
        ]
    )
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, batch_size=2, epochs=3, verbose=0)
    assert ops.shape(model(np.zeros((2, 3), "float32"))) == (2, 4)


def test_conv_layers_in_sequential():
    model = keras.Sequential(
        [
            keras.Input((8, 8, 1)),
            Conv2DReparameterization(2, 3, activation="relu"),
            Conv2DFlipout(2, 3),
            keras.layers.Flatten(),
            DenseFlipout(1),
        ]
    )
    model.compile(loss="mse", optimizer="adam")
    model.fit(np.random.randn(8, 8, 8, 1), np.random.randn(8, 1), batch_size=4, epochs=2, verbose=0)
    assert model.output_shape == (None, 1)


@pytest.mark.parametrize(
    "make_layer, kwargs",
    [
        (DenseFlipout, {"units": 0}),
        (DenseFlipout, {"units": 2, "kernel_prior_scale": 0.0}),
        (Conv2DFlipout, {"filters": 0, "kernel_size": 3}),
        (Conv2DFlipout, {"filters": 2, "kernel_size": 3, "padding": "causal"}),
        (Conv2DFlipout, {"filters": 2, "kernel_size": (3, 3, 3)}),
        (Conv2DFlipout, {"filters": 2, "kernel_size": 3, "data_format": "nhwc"}),
    ],
)
def test_invalid_args(make_layer, kwargs):
    with pytest.raises(ValueError):
        make_layer(**kwargs)
