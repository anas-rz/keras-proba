"""Training tests for GaussianHead and GaussianNLL. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras

from keras_proba.layers.gaussian_head import GaussianHead, GaussianNLL, split_gaussian_params


def make_heteroscedastic_data(n=2048, seed=0):
    """y = sin(x) + noise whose standard deviation grows linearly with |x|."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 1)).astype("float32")
    std = 0.05 + 0.3 * np.abs(x)
    y = (np.sin(x) + rng.normal(size=(n, 1)) * std).astype("float32")
    return x, y, std


def build_model():
    model = keras.Sequential(
        [
            keras.Input((1,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            GaussianHead(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.005), loss=GaussianNLL())
    return model


@pytest.fixture(scope="module")
def trained_model():
    keras.utils.set_random_seed(0)
    x, y, _ = make_heteroscedastic_data()
    model = build_model()
    model.fit(x, y, epochs=60, batch_size=128, verbose=0)
    return model


def test_learns_mean_and_heteroscedastic_scale(trained_model):
    model = trained_model

    x_test = np.linspace(-2.5, 2.5, 51, dtype="float32")[:, None]
    mean, scale = (keras.ops.convert_to_numpy(t) for t in split_gaussian_params(model.predict(x_test, verbose=0)))
    true_std = 0.05 + 0.3 * np.abs(x_test)

    # Typical runs reach ~0.05 mean error and ~0.03 scale error; the bounds leave room across seeds and backends.
    assert np.abs(mean - np.sin(x_test)).mean() < 0.15
    # The predicted scale tracks the true noise level.
    assert np.corrcoef(scale[:, 0], true_std[:, 0])[0, 1] > 0.9
    assert np.abs(scale - true_std).mean() < 0.12


def test_predicted_intervals_are_calibrated(trained_model):
    model = trained_model

    x_val, y_val, _ = make_heteroscedastic_data(n=4096, seed=1)
    mean, scale = (keras.ops.convert_to_numpy(t) for t in split_gaussian_params(model.predict(x_val, verbose=0)))
    coverage = np.mean(np.abs(y_val - mean) <= 1.96 * scale)
    assert 0.9 < coverage < 0.99, coverage
