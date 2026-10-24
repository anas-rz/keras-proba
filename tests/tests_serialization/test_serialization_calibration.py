"""Serialization and reload tests for the calibration metrics. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras

from keras_proba.metrics.calibration import BrierScore, ExpectedCalibrationError


@pytest.mark.parametrize(
    "metric",
    [ExpectedCalibrationError(num_bins=7, from_logits=True, name="my_ece"), BrierScore(from_logits=True, name="my_brier")],
)
def test_config_round_trip(metric):
    restored = keras.metrics.deserialize(keras.metrics.serialize(metric))
    assert type(restored) is type(metric)
    assert restored.get_config() == metric.get_config()


def test_model_save_and_load_with_metrics(tmp_path):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(4)])
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[ExpectedCalibrationError(num_bins=5, from_logits=True), BrierScore(from_logits=True)],
    )
    x = np.random.randn(64, 3).astype("float32")
    y = np.random.randint(0, 4, size=64)
    model.fit(x, y, epochs=1, verbose=0)
    expected = model.evaluate(x, y, verbose=0, return_dict=True)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    actual = loaded.evaluate(x, y, verbose=0, return_dict=True)
    for key in ["ece", "brier_score"]:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=1e-6)
