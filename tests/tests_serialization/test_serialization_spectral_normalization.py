"""Serialization and reload tests for the spectral normalization wrappers. Run once per backend via `KERAS_BACKEND`."""

import copy

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.spectral_normalization import (
    SpectralNormalization,
    SpectralNormalizationConv2D,
)


@pytest.mark.parametrize(
    "make_wrapper",
    [
        lambda: SpectralNormalization(
            keras.layers.Dense(4, name="dense_sn"),
            iteration=3,
            norm_multiplier=0.5,
            training=False,
            aggregation="sum",
            inhere_layer_name=True,
        ),
        lambda: SpectralNormalizationConv2D(
            keras.layers.Conv2D(4, 3), iteration=2, norm_multiplier=0.8, training=False, legacy_mode=True
        ),
    ],
)
def test_config_round_trip(make_wrapper):
    wrapper = make_wrapper()
    config = wrapper.get_config()
    # `Wrapper.from_config` pops the wrapped layer from the dict it is given.
    restored = type(wrapper).from_config(copy.deepcopy(config))
    assert restored.get_config() == config


def test_conv_name_suffix_is_not_repeated():
    wrapper = SpectralNormalizationConv2D(keras.layers.Conv2D(2, 3, name="conv"))
    restored = keras.layers.deserialize(keras.layers.serialize(wrapper))
    assert restored.layer.name == "conv_spec_norm"


def test_model_save_and_load(tmp_path):
    # Projection mode is saved inside the SNGP head's save/load test.
    mode = "reparameterization"
    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(8, 8, 3))
    conv = keras.layers.Conv2D(4, 3, padding="same", activation="relu")
    h = SpectralNormalizationConv2D(conv, mode=mode)(inputs)
    h = keras.layers.Flatten()(h)
    outputs = SpectralNormalization(keras.layers.Dense(2), iteration=2, mode=mode)(h)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    x = np.random.randn(16, 8, 8, 3).astype("float32")
    model.fit(x, np.random.randn(16, 2).astype("float32"), epochs=1, batch_size=8, verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    assert [w.name for w in loaded.weights] == [w.name for w in model.weights]
    for original, restored in zip(model.weights, loaded.weights):
        np.testing.assert_array_equal(ops.convert_to_numpy(restored), ops.convert_to_numpy(original))
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-5)
