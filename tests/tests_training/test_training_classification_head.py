"""Training tests for the SNGP head and ResetCovarianceCallback. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.callbacks.reset_covariance import ResetCovarianceCallback
from keras_proba.layers.classification_head import GaussianProcessClassificationHead
from keras_proba.layers.gaussian_process import RandomFeatureGaussianProcess
from keras_proba.layers.spectral_normalization import SpectralNormalization


def make_two_moons(n=512, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, size=n)
    y = rng.integers(0, 2, size=n)
    x = np.stack([np.cos(t), np.sin(t)], -1)
    x = np.where(y[:, None] == 1, np.array([1.0, 0.5]) - x, x)
    x = x + rng.normal(size=x.shape) * 0.1
    return x.astype("float32"), y.astype("int32")


def build_sngp(**head_kwargs):
    """A distance-preserving SNGP model, following the SNGP tutorial's 2-D setup.

    The head's BERT-oriented defaults (tanh pooler, input layer normalization and
    small random feature weights) are replaced so that distance from the
    training data reaches the Gaussian process.
    """
    inputs = keras.Input((2,))
    # A frozen random projection and residual spectral-normalized blocks preserve distances.
    hidden = keras.layers.Dense(64, trainable=False)(inputs)
    for _ in range(2):
        block = SpectralNormalization(
            keras.layers.Dense(64, activation="relu"), norm_multiplier=0.9, mode="reparameterization"
        )
        hidden = hidden + block(hidden)
    head = GaussianProcessClassificationHead(
        inner_dim=0,
        num_classes=2,
        num_inducing=1024,
        normalize_input=False,
        scale_random_features=True,
        custom_random_features_initializer=keras.initializers.RandomNormal(stddev=1.0),
        **head_kwargs,
    )
    model = keras.Model(inputs, head(hidden))
    model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model, head


def test_sngp_fit_and_ood_uncertainty():
    keras.utils.set_random_seed(0)
    x, y = make_two_moons()
    model, head = build_sngp(temperature=np.pi / 8, gp_cov_momentum=-1)
    history = model.fit(x, y, epochs=30, batch_size=64, verbose=0, callbacks=[ResetCovarianceCallback()])
    assert history.history["accuracy"][-1] > 0.95

    probs_in = ops.convert_to_numpy(ops.softmax(model.predict(x[:128], verbose=0)))
    x_ood = np.random.default_rng(1).normal(size=(128, 2)).astype("float32") * 0.5 + np.array([6.0, 6.0], "float32")
    probs_ood = ops.convert_to_numpy(ops.softmax(model.predict(x_ood, verbose=0)))
    # Confident in distribution (~0.99), close to uniform far away from the data (~0.55-0.73).
    assert probs_in.max(-1).mean() > 0.95
    assert probs_ood.max(-1).mean() < 0.8


@pytest.mark.parametrize("nested", [False, True])
def test_reset_covariance_callback(nested):
    """With features fixed and momentum=-1, the precision only sums the last epoch when reset."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 6)).astype("float32")
    y = rng.integers(0, 3, 64)
    num_inducing, epochs = 32, 3

    def run(callbacks):
        keras.utils.set_random_seed(0)
        inputs = keras.Input((6,))
        if nested:
            gp = GaussianProcessClassificationHead(inner_dim=8, num_classes=3, num_inducing=num_inducing, gp_cov_momentum=-1)
            outputs = gp(inputs)
            cov_layer = lambda: gp.out_proj._gp_cov_layer
        else:
            gp = RandomFeatureGaussianProcess(3, num_inducing=num_inducing, gp_cov_momentum=-1)
            outputs = gp(inputs)[0]
            cov_layer = lambda: gp._gp_cov_layer
        model = keras.Model(inputs, outputs)
        # A zero learning rate keeps the features identical across epochs.
        model.compile(optimizer=keras.optimizers.SGD(0.0), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        model.fit(x, y, epochs=epochs, batch_size=16, verbose=0, callbacks=callbacks)
        return np.trace(ops.convert_to_numpy(cov_layer().precision_matrix)) - num_inducing

    accumulated = run([])
    reset = run([ResetCovarianceCallback()])
    np.testing.assert_allclose(accumulated, epochs * reset, rtol=1e-3)


def test_callback_ignores_models_without_gp():
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(2)])
    model.compile(optimizer="sgd", loss="mse")
    model.fit(np.zeros((4, 3)), np.zeros((4, 2)), epochs=2, verbose=0, callbacks=[ResetCovarianceCallback()])
