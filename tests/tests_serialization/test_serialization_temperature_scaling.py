"""Serialization and reload tests for temperature scaling. Run once per backend via `KERAS_BACKEND`."""

import numpy as np

import keras
from keras import ops

from keras_proba.postprocessing.temperature_scaling import TemperatureScaling


def test_config_round_trip():
    layer = TemperatureScaling(1.7, name="ts")
    config = layer.get_config()
    restored = TemperatureScaling.from_config(config)
    assert restored.get_config() == config
    np.testing.assert_allclose(float(ops.convert_to_numpy(restored.temperature)), 1.7, rtol=1e-6)


def test_model_save_and_load_keeps_fitted_temperature(tmp_path):
    keras.utils.set_random_seed(0)
    base = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3)])
    scaling = TemperatureScaling()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4)).astype("float32")
    temperature = scaling.fit(base.predict(x, verbose=0) * 5, rng.integers(0, 3, size=64))
    model = keras.Sequential([base, scaling])

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    loaded_scaling = loaded.layers[-1]
    assert isinstance(loaded_scaling, TemperatureScaling)
    np.testing.assert_allclose(float(ops.convert_to_numpy(loaded_scaling.temperature)), temperature, rtol=1e-6)
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-6, atol=1e-6)
