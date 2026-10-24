"""Training tests for the variational layers. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.variational import (
    Conv1DFlipout,
    Conv1DReparameterization,
    Conv2DFlipout,
    Conv3DFlipout,
    DenseFlipout,
    DenseLocalReparameterization,
    DenseReparameterization,
)


def to_numpy(x):
    return ops.convert_to_numpy(x)


def test_sgd_step_matches_analytic_gradient():
    """One full-batch SGD step with fixed noise matches the gradient of MSE + KL derived by hand."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3)).astype("float32")
    y = rng.normal(size=(16, 2)).astype("float32")
    noise = rng.normal(size=(3, 2)).astype("float32")
    lr, kl_weight, prior_scale = 0.1, 0.05, 1.5

    layer = DenseReparameterization(2, kl_weight=kl_weight, kernel_prior_scale=prior_scale)
    model = keras.Sequential([keras.Input((3,)), layer])
    model.compile(optimizer=keras.optimizers.SGD(lr), loss="mse")
    layer._standard_normal = lambda shape, dtype: ops.cast(noise, dtype)

    loc = to_numpy(layer.kernel_posterior_loc).astype("float64")
    u = to_numpy(layer.kernel_posterior_untransformed_scale).astype("float64")
    bias = to_numpy(layer.bias_posterior_loc).astype("float64")
    model.fit(x, y, batch_size=16, epochs=1, shuffle=False, verbose=0)

    eps = np.finfo("float32").eps
    scale = eps + np.log1p(np.exp(u))
    kernel = loc + scale * noise
    # d(MSE)/d(kernel) and d(MSE)/d(bias) for the sampled kernel.
    residual = 2.0 / y.size * (x @ kernel + bias - y)
    grad_kernel = x.T @ residual
    grad_bias = residual.sum(0)
    # KL(N(loc, scale^2) || N(0, prior^2)) gradients.
    grad_loc = grad_kernel + kl_weight * loc / prior_scale**2
    grad_scale = grad_kernel * noise + kl_weight * (-1.0 / scale + scale / prior_scale**2)
    grad_u = grad_scale / (1.0 + np.exp(-u))  # d softplus(u) / du = sigmoid(u).

    np.testing.assert_allclose(to_numpy(layer.kernel_posterior_loc), loc - lr * grad_loc, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(to_numpy(layer.kernel_posterior_untransformed_scale), u - lr * grad_u, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(to_numpy(layer.bias_posterior_loc), bias - lr * grad_bias, rtol=1e-4, atol=1e-6)


def make_regression_data(n=512, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 1)).astype("float32")
    y = (np.sin(2 * x) + rng.normal(size=(n, 1)) * 0.1).astype("float32")
    return x, y


@pytest.mark.parametrize("layer_cls", [DenseReparameterization, DenseLocalReparameterization, DenseFlipout])
def test_bayesian_regression(layer_cls):
    keras.utils.set_random_seed(0)
    x, y = make_regression_data()
    kl_weight = 1.0 / len(x)
    model = keras.Sequential(
        [
            keras.Input((1,)),
            layer_cls(64, activation="relu", kl_weight=kl_weight),
            layer_cls(64, activation="relu", kl_weight=kl_weight),
            layer_cls(1, kl_weight=kl_weight),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    model.fit(x, y, epochs=150, batch_size=64, verbose=0)

    # Monte Carlo predictive mean and spread.
    x_in = np.linspace(-1.5, 1.5, 31, dtype="float32")[:, None]
    x_out = np.linspace(4.0, 6.0, 31, dtype="float32")[:, None]
    samples_in = np.stack([model.predict(x_in, verbose=0) for _ in range(30)])
    samples_out = np.stack([model.predict(x_out, verbose=0) for _ in range(30)])
    assert np.abs(samples_in.mean(0) - np.sin(2 * x_in)).mean() < 0.2
    # More epistemic uncertainty away from the data.
    assert samples_out.std(0).mean() > 2 * samples_in.std(0).mean()


def test_kl_weight_pulls_posterior_towards_prior():
    """Without data fit pressure, a larger KL weight moves the posterior scale towards the prior's."""
    keras.utils.set_random_seed(0)
    x, y = make_regression_data(n=128)
    scales = {}
    for kl_weight in [1e-4, 1.0]:
        layer = DenseFlipout(8, kl_weight=kl_weight)
        model = keras.Sequential([keras.Input((1,)), layer, keras.layers.Dense(1)])
        model.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
        model.fit(x, y, epochs=30, batch_size=32, verbose=0)
        scales[kl_weight] = to_numpy(layer.kernel_posterior_scale).mean()
    assert scales[1.0] > 2 * scales[1e-4]
    assert scales[1.0] < 1.2


@pytest.mark.parametrize("conv_cls", [Conv1DFlipout, Conv2DFlipout, Conv3DFlipout, Conv1DReparameterization])
def test_conv_classifier(conv_cls):
    """The label is whether the first half of the input (along the first spatial axis) is brighter."""
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    spatial = (8,) * conv_cls.rank
    x = rng.normal(size=(256, *spatial, 1)).astype("float32")
    other_axes = tuple(range(1, conv_cls.rank + 2))
    y = (x[:, :4].mean(other_axes) > x[:, 4:].mean(other_axes)).astype("int32")
    kl_weight = 1.0 / len(x)
    model = keras.Sequential(
        [
            keras.Input(spatial + (1,)),
            conv_cls(8, 3, padding="same", activation="relu", kl_weight=kl_weight),
            # The label depends on where the signal is, so keep the spatial layout.
            keras.layers.Flatten(),
            DenseFlipout(2, kl_weight=kl_weight),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(x, y, epochs=30, batch_size=32, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]
    assert history.history["accuracy"][-1] > 0.8


def test_repeated_predict_calls_sample_differently():
    """Monte Carlo prediction needs fresh noise on every call, on every backend."""
    model = keras.Sequential([keras.Input((3,)), DenseFlipout(4)])
    x = np.random.randn(5, 3).astype("float32")
    assert not np.allclose(model.predict(x, verbose=0), model.predict(x, verbose=0))
