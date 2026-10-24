"""Serialization and reload tests for MixtureGaussianHead and MixtureGaussianNLL. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras

from keras_proba.layers.mixture_gaussian_head import MixtureGaussianHead, MixtureGaussianNLL


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_components": 3},
        {
            "num_components": 5,
            "units": 2,
            "min_scale": 0.01,
            "kernel_initializer": "he_normal",
            "scale_bias_initializer": keras.initializers.Constant(-1.0),
            "kernel_regularizer": keras.regularizers.L2(0.01),
            "bias_regularizer": "l1",
        },
    ],
)
def test_config_round_trip(kwargs):
    layer = MixtureGaussianHead(**kwargs)
    config = layer.get_config()
    assert MixtureGaussianHead.from_config(config).get_config() == config


def test_loss_round_trip():
    loss = MixtureGaussianNLL(units=2, reduction="sum")
    restored = keras.losses.deserialize(keras.losses.serialize(loss))
    assert isinstance(restored, MixtureGaussianNLL)
    assert restored.get_config() == loss.get_config()


def test_model_save_and_load(tmp_path):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(8, activation="relu"), MixtureGaussianHead(3, units=2)])
    model.compile(optimizer="adam", loss=MixtureGaussianNLL())
    x = np.random.randn(32, 3).astype("float32")
    y = np.random.randn(32, 2).astype("float32")
    model.fit(x, y, epochs=1, verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    assert isinstance(loaded.layers[-1], MixtureGaussianHead)
    assert isinstance(loaded.loss, MixtureGaussianNLL)
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6)
    loaded.fit(x, y, epochs=1, verbose=0)
