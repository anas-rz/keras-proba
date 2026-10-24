"""Training tests for the spectral normalization wrappers. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.spectral_normalization import (
    SpectralNormalization,
    SpectralNormalizationConv2D,
)


def spectral_norm(kernel):
    return np.linalg.svd(kernel.reshape(-1, kernel.shape[-1]), compute_uv=False)[0]


def make_regression_data(n=128, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype("float32")
    y = (np.sin(x[:, :1]) + 0.5 * x[:, 1:2]).astype("float32")
    return x, y


def test_fit_reduces_loss_and_updates_power_iteration():
    keras.utils.set_random_seed(0)
    x, y = make_regression_data()
    sn = SpectralNormalization(keras.layers.Dense(32, activation="relu"), norm_multiplier=0.95)
    model = keras.Sequential([keras.Input((4,)), sn, keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    u_before = ops.convert_to_numpy(sn.u).copy()

    history = model.fit(x, y, epochs=20, batch_size=32, verbose=0)
    assert history.history["loss"][-1] < 0.5 * history.history["loss"][0]
    # u and v are non-trainable state, persisted on every backend.
    assert not np.allclose(ops.convert_to_numpy(sn.u), u_before)


def test_trained_layer_is_lipschitz_at_inference():
    """Whatever the backend, inference uses a kernel with bounded spectral norm."""
    keras.utils.set_random_seed(0)
    x, y = make_regression_data()
    sn = SpectralNormalization(keras.layers.Dense(16), iteration=5, norm_multiplier=0.5)
    model = keras.Sequential([keras.Input((4,)), sn, keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)

    # Measure the Lipschitz constant of the wrapper itself at inference.
    infer = keras.Model(model.inputs, sn.output)
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=(256, 4)).astype("float32")
    x2 = x1 + rng.normal(size=(256, 4)).astype("float32") * 0.1
    out1, out2 = infer.predict(x1, verbose=0), infer.predict(x2, verbose=0)
    ratios = np.linalg.norm(out2 - out1, axis=-1) / np.linalg.norm(x2 - x1, axis=-1)
    assert ratios.max() <= 0.5 * 1.02


@pytest.mark.skipif(
    keras.backend.backend() == "jax",
    reason="JAX discards assignments to trainable variables in `fit`, so the projected kernel is not persisted.",
)
def test_fit_projects_stored_kernel():
    keras.utils.set_random_seed(0)
    x, y = make_regression_data()
    sn = SpectralNormalization(keras.layers.Dense(16), iteration=5, norm_multiplier=0.5)
    model = keras.Sequential([keras.Input((4,)), sn, keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # Only the last optimizer step can push the stored kernel past the bound.
    np.testing.assert_allclose(spectral_norm(ops.convert_to_numpy(sn.layer.kernel)), 0.5, rtol=0.02)


def test_conv_fit():
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 8, 8, 2)).astype("float32")
    y = x.mean(axis=(1, 2, 3))[:, None].astype("float32")
    conv = SpectralNormalizationConv2D(keras.layers.Conv2D(8, 3, padding="same", activation="relu"))
    model = keras.Sequential(
        [keras.Input((8, 8, 2)), conv, keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(1)]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    kernel_before = ops.convert_to_numpy(conv.layer.kernel).copy()
    history = model.fit(x, y, epochs=10, batch_size=16, verbose=0)
    assert np.isfinite(history.history["loss"][-1])
    assert history.history["loss"][-1] < history.history["loss"][0]
    assert not np.allclose(ops.convert_to_numpy(conv.layer.kernel), kernel_before)
