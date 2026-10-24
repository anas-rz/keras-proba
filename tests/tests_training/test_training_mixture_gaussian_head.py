"""Training tests for MixtureGaussianHead and MixtureGaussianNLL. Run once per backend via `KERAS_BACKEND`."""

import numpy as np

import keras

from keras_proba.layers.gaussian_head import GaussianHead, GaussianNLL
from keras_proba.layers.mixture_gaussian_head import (
    MixtureGaussianHead,
    MixtureGaussianNLL,
    mixture_gaussian_nll,
    split_mixture_params,
)


def make_bimodal_data(n=4000, seed=0):
    """y = +-x^2 with equal probability, plus noise: two modes per input."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 1)).astype("float32")
    sign = np.where(rng.uniform(size=(n, 1)) < 0.5, -1.0, 1.0)
    y = (sign * x**2 + rng.normal(size=(n, 1)) * 0.1).astype("float32")
    return x, y


def build(head, loss):
    # ReLU features: with tanh at this learning rate the network saturates and
    # the mixture can get stuck covering only one mode.
    model = keras.Sequential(
        [keras.Input((1,)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(64, activation="relu"), head]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.005), loss=loss)
    return model


def test_mixture_captures_bimodal_targets():
    keras.utils.set_random_seed(0)
    x, y = make_bimodal_data()
    mixture = build(MixtureGaussianHead(3), MixtureGaussianNLL())
    gaussian = build(GaussianHead(1), GaussianNLL())
    mixture.fit(x, y, epochs=40, batch_size=128, verbose=0)
    gaussian.fit(x, y, epochs=40, batch_size=128, verbose=0)

    x_val, y_val = make_bimodal_data(n=2000, seed=1)
    params = mixture.predict(x_val, verbose=0)
    mixture_nll = float(np.mean(keras.ops.convert_to_numpy(mixture_gaussian_nll(y_val, params))))
    gaussian_nll = float(keras.ops.convert_to_numpy(GaussianNLL()(y_val, gaussian.predict(x_val, verbose=0))))
    # A single Gaussian cannot represent two modes (NLL ~1.15); the mixture reaches ~-0.2.
    assert mixture_nll < 0.1, mixture_nll
    assert mixture_nll < gaussian_nll - 0.8, (mixture_nll, gaussian_nll)

    # Away from x = 0, each mode at +x^2 and -x^2 has a component with substantial weight near it.
    x_test = np.array([[-1.5], [1.0], [1.8]], "float32")
    logits, means, _ = (keras.ops.convert_to_numpy(t) for t in split_mixture_params(mixture.predict(x_test, verbose=0)))
    weights = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    for i, xi in enumerate(x_test[:, 0]):
        weighted_means = means[i, weights[i] > 0.2, 0]
        for mode in (-(xi**2), xi**2):
            assert np.min(np.abs(weighted_means - mode)) < 0.3, (xi, mode, means[i, :, 0], weights[i])


