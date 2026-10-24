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


@pytest.mark.parametrize("mode", ["projection", "reparameterization"])
def test_trained_layer_is_lipschitz_at_inference(mode):
    """Whatever the backend, inference uses a kernel with bounded spectral norm."""
    keras.utils.set_random_seed(0)
    x, y = make_regression_data()
    sn = SpectralNormalization(keras.layers.Dense(16), iteration=5, norm_multiplier=0.5, mode=mode)
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


def test_reparameterization_gradient_flows_through_sigma():
    """One full-batch SGD step matches the analytic gradient of W * c / sigma(W)."""
    keras.utils.set_random_seed(1)
    rng = np.random.default_rng(0)
    c, lr = 0.5, 0.1
    layer = keras.layers.Dense(3, kernel_initializer=keras.initializers.RandomNormal(stddev=2.0))
    sn = SpectralNormalization(layer, iteration=3, norm_multiplier=c, mode="reparameterization")
    model = keras.Sequential([keras.Input((5,)), sn])
    model.compile(optimizer=keras.optimizers.SGD(lr), loss="mse")
    x = rng.normal(size=(16, 5)).astype("float32")
    y = rng.normal(size=(16, 3)).astype("float32")

    w0 = ops.convert_to_numpy(sn.layer.kernel).astype("float64")
    b0 = ops.convert_to_numpy(sn.layer.bias).astype("float64")
    model.fit(x, y, batch_size=16, epochs=1, shuffle=False, verbose=0)

    # u and v after the step are the ones used in its forward pass.
    u = ops.convert_to_numpy(sn.u).astype("float64")
    v = ops.convert_to_numpy(sn.v).astype("float64")
    sigma = (v @ w0 @ u.T).item()
    assert c / sigma < 1
    # G = dL/dW_sn; dL/dW = (c / sigma) G - (c / sigma^2) <G, W> v^T u.
    grad_w_sn = 2.0 / y.size * x.T @ (x @ (w0 * c / sigma) + b0 - y)
    grad_w = (c / sigma) * grad_w_sn - (c / sigma**2) * np.sum(grad_w_sn * w0) * (v.T @ u)
    np.testing.assert_allclose(ops.convert_to_numpy(sn.layer.kernel), w0 - lr * grad_w, atol=1e-5)


def test_reparameterization_fit_does_not_assign_kernel():
    keras.utils.set_random_seed(0)
    x, y = make_regression_data()
    layer = keras.layers.Dense(16, kernel_initializer=keras.initializers.RandomNormal(stddev=3.0))
    sn = SpectralNormalization(layer, iteration=5, norm_multiplier=0.5, mode="reparameterization")
    model = keras.Sequential([keras.Input((4,)), sn, keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    history = model.fit(x, y, epochs=20, batch_size=32, verbose=0)
    assert history.history["loss"][-1] < 0.5 * history.history["loss"][0]
    # The raw kernel stays unnormalized; only the forward pass is normalized.
    assert spectral_norm(ops.convert_to_numpy(sn.layer.kernel)) > 1.0
    np.testing.assert_allclose(spectral_norm(ops.convert_to_numpy(sn.update_weights())), 0.5, rtol=0.02)


def test_conv_fit():
    mode = "reparameterization"
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 8, 8, 2)).astype("float32")
    y = x.mean(axis=(1, 2, 3))[:, None].astype("float32")
    conv = SpectralNormalizationConv2D(keras.layers.Conv2D(8, 3, padding="same", activation="relu"), mode=mode)
    model = keras.Sequential(
        [keras.Input((8, 8, 2)), conv, keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(1)]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    kernel_before = ops.convert_to_numpy(conv.layer.kernel).copy()
    history = model.fit(x, y, epochs=10, batch_size=16, verbose=0)
    assert np.isfinite(history.history["loss"][-1])
    assert history.history["loss"][-1] < history.history["loss"][0]
    assert not np.allclose(ops.convert_to_numpy(conv.layer.kernel), kernel_before)
