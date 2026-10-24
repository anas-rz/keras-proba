"""Serialization and reload tests for the Gaussian process layer. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.gaussian_process import RandomFeatureGaussianProcess


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "units": 2,
            "num_inducing": 64,
            "gp_kernel_scale": 2.0,
            "normalize_input": True,
            "gp_cov_momentum": -1,
            "gp_cov_likelihood": "poisson",
            "return_random_features": True,
        },
        {
            "units": 2,
            "custom_random_features_initializer": keras.initializers.RandomNormal(stddev=0.5, seed=1),
            "custom_random_features_activation": "relu",
        },
        {"units": 2, "kernel_initializer": "zeros", "kernel_constraint": keras.constraints.NonNeg()},
    ],
)
def test_config_round_trip(kwargs):
    layer = RandomFeatureGaussianProcess(**kwargs)
    config = layer.get_config()
    restored = RandomFeatureGaussianProcess.from_config(config)
    assert restored.get_config() == config


def test_gp_output_kwargs_round_trip():
    layer = RandomFeatureGaussianProcess(units=2, num_inducing=8, kernel_initializer="zeros")
    restored = RandomFeatureGaussianProcess.from_config(layer.get_config())
    restored.build((None, 3))
    np.testing.assert_array_equal(ops.convert_to_numpy(restored._gp_output_layer.kernel), 0)


def test_model_save_and_load(tmp_path):
    kwargs = {"use_custom_random_features": False, "normalize_input": True, "return_random_features": True}
    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(4,))
    outputs = RandomFeatureGaussianProcess(units=2, num_inducing=32, **kwargs)(inputs)
    model = keras.Model(inputs, outputs)
    # Train through the logits only; the saved model shares the trained GP layer.
    train_model = keras.Model(inputs, outputs[0])
    train_model.compile(optimizer="adam", loss="mse")

    x = np.random.randn(16, 4).astype("float32")
    y = np.random.randn(16, 2).astype("float32")
    train_model.fit(x, y, epochs=1, batch_size=16, verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    expected = model.predict(x, batch_size=16, verbose=0)
    actual = loaded.predict(x, batch_size=16, verbose=0)
    for e, a in zip(expected, actual):
        np.testing.assert_allclose(a, e, atol=1e-5)

    # The learned precision matrix survives the round trip.
    gp, loaded_gp = model.layers[1], loaded.layers[1]
    np.testing.assert_array_equal(
        ops.convert_to_numpy(loaded_gp._gp_cov_layer.precision_matrix),
        ops.convert_to_numpy(gp._gp_cov_layer.precision_matrix),
    )
