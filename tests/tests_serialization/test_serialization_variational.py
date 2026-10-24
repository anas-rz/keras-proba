"""Serialization and reload tests for the variational layers. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.variational import (
    Conv1DReparameterization,
    Conv2DFlipout,
    Conv3DFlipout,
    DenseFlipout,
    DenseReparameterization,
)


@pytest.mark.parametrize(
    "layer",
    [
        DenseFlipout(
            3,
            kernel_loc_initializer="he_normal",
            kernel_untransformed_scale_initializer=keras.initializers.Constant(-5.0),
            kernel_prior_scale=0.5,
            kl_weight=1e-3,
            seed=3,
            activity_regularizer="l2",
        ),
        Conv1DReparameterization(4, 5, dilation_rate=2, use_bias=False),
        Conv2DFlipout(4, (3, 2), strides=2, padding="same", dilation_rate=1, data_format="channels_first", seed=1),
        Conv3DFlipout(2, (3, 2, 1), padding="same", kl_weight=0.1),
    ],
)
def test_config_round_trip(layer):
    config = layer.get_config()
    restored = type(layer).from_config(config)
    assert restored.get_config() == config
    assert keras.layers.deserialize(keras.layers.serialize(layer)).get_config() == config


def test_model_save_and_load(tmp_path):
    conv_cls = Conv2DFlipout
    keras.utils.set_random_seed(0)
    spatial = (6,) * conv_cls.rank
    model = keras.Sequential(
        [
            keras.Input(spatial + (1,)),
            conv_cls(2, 3, activation="relu", kl_weight=0.01),
            keras.layers.Flatten(),
            DenseReparameterization(3, kl_weight=0.01),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    x = np.random.randn(16, *spatial, 1).astype("float32")
    model.fit(x, np.random.randn(16, 3).astype("float32"), epochs=1, verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    for original, restored in zip(model.trainable_weights, loaded.trainable_weights):
        assert original.name == restored.name
        np.testing.assert_array_equal(ops.convert_to_numpy(restored), ops.convert_to_numpy(original))
    # Predictions are stochastic; without noise both predict with the posterior mean.
    for m in [model, loaded]:
        for layer in m.layers:
            if hasattr(layer, "_standard_normal"):
                layer._standard_normal = lambda shape, dtype: ops.zeros(shape, dtype)
                layer._rademacher = lambda shape, dtype: ops.ones(shape, dtype)
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5, atol=1e-6)
