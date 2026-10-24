"""Tests of the calibration metrics inside `fit` and `evaluate`. Run once per backend via `KERAS_BACKEND`."""

import numpy as np

import keras
from keras import ops

from keras_proba.metrics.calibration import BrierScore, ExpectedCalibrationError, brier_score, expected_calibration_error


def make_data(n=1024, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype("float32")
    logits = x @ rng.normal(size=(4, 3))
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(probs, -1)).sum(-1).clip(0, 2)
    return x, y.astype("int32")


def build_model():
    model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(16, activation="relu"), keras.layers.Dense(3)])
    model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[ExpectedCalibrationError(num_bins=10, from_logits=True), BrierScore(from_logits=True)],
    )
    return model


def test_metrics_in_fit_and_evaluate():
    keras.utils.set_random_seed(0)
    x, y = make_data()
    model = build_model()
    history = model.fit(x, y, epochs=5, batch_size=64, verbose=0)
    assert {"ece", "brier_score"} <= set(history.history)
    assert all(np.isfinite(history.history["ece"]))

    # `evaluate` accumulates over batches, so it equals the functions on all predictions.
    results = model.evaluate(x, y, batch_size=100, verbose=0, return_dict=True)
    logits = model.predict(x, batch_size=100, verbose=0)
    np.testing.assert_allclose(
        results["ece"], ops.convert_to_numpy(expected_calibration_error(10, logits=logits, labels_true=y)), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        results["brier_score"], ops.convert_to_numpy(ops.mean(brier_score(y, logits))), rtol=1e-4, atol=1e-5
    )


