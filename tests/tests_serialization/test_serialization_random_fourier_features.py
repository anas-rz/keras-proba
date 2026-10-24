"""Serialization and reload tests for RandomFourierFeatures. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.random_fourier_features import RandomFourierFeatures


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_dim": 16, "kernel_initializer": "laplacian", "scale": 2.0, "trainable": True},
        {"output_dim": 16, "kernel_initializer": keras.initializers.RandomNormal(stddev=0.5, seed=3)},
    ],
)
def test_config_round_trip(kwargs):
    layer = RandomFourierFeatures(**kwargs)
    config = layer.get_config()
    restored = RandomFourierFeatures.from_config(config)
    assert restored.get_config() == config


def test_config_records_default_scale_after_build():
    layer = RandomFourierFeatures(8)
    assert layer.get_config()["scale"] is None
    layer.build((None, 8))
    assert layer.get_config()["scale"] == 2.0


def test_model_save_and_load(tmp_path):
    kernel_initializer = "laplacian"
    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [
            keras.Input(shape=(5,)),
            RandomFourierFeatures(64, kernel_initializer=kernel_initializer, trainable=True),
            keras.layers.Dense(3),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    x = np.random.randn(16, 5).astype("float32")
    model.fit(x, np.random.randn(16, 3).astype("float32"), epochs=1, verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    assert isinstance(loaded.layers[0], RandomFourierFeatures)
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6)
    for a, b in zip(model.layers[0].weights, loaded.layers[0].weights):
        np.testing.assert_array_equal(ops.convert_to_numpy(a), ops.convert_to_numpy(b))


