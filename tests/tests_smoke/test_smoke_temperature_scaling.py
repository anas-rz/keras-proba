"""Smoke and correctness tests for temperature scaling. Run once per backend via `KERAS_BACKEND`."""


import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.metrics.calibration import expected_calibration_error
from keras_proba.postprocessing.temperature_scaling import TemperatureScaling, fit_temperature


def to_numpy(x):
    return ops.convert_to_numpy(x)


def calibrated_data(rng, n=20000, num_classes=4):
    """Logits with labels sampled from their softmax, i.e. perfectly calibrated."""
    logits = rng.normal(size=(n, num_classes)) * 2
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    labels = (rng.uniform(size=(n, 1)) > np.cumsum(probs, -1)).sum(-1).clip(0, num_classes - 1)
    return logits.astype("float32"), labels


def test_layer_divides_by_temperature():
    layer = TemperatureScaling(2.0)
    x = np.random.randn(3, 5).astype("float32")
    np.testing.assert_allclose(to_numpy(layer(x)), x / 2.0, rtol=1e-6)
    np.testing.assert_allclose(float(to_numpy(layer.temperature)), 2.0, rtol=1e-6)
    assert [w.name for w in layer.trainable_weights] == ["log_temperature"]
    assert layer.compute_output_shape((None, 5)) == (None, 5)


def test_fit_recovers_known_temperature():
    sharpness = 3.0
    """Scaling calibrated logits by `s` is undone by a temperature close to `s`."""
    logits, labels = calibrated_data(np.random.default_rng(0))
    np.testing.assert_allclose(fit_temperature(logits * sharpness, labels), sharpness, rtol=0.05)


def test_fit_binary_sigmoid_logits():
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000) * 2
    labels = (rng.uniform(size=20000) < 1 / (1 + np.exp(-z))).astype("int32")
    for logits in [3 * z, (3 * z)[:, None]]:
        np.testing.assert_allclose(fit_temperature(logits, labels), 3.0, rtol=0.05)


def test_one_hot_labels_and_sample_weight():
    rng = np.random.default_rng(2)
    logits, labels = calibrated_data(rng, n=500)
    logits = logits * 2
    t = fit_temperature(logits, labels)
    np.testing.assert_allclose(fit_temperature(logits, np.eye(4)[labels]), t, rtol=1e-9)
    weights = rng.integers(0, 4, size=500)
    np.testing.assert_allclose(
        fit_temperature(logits, labels, sample_weight=weights.astype("float32")),
        fit_temperature(np.repeat(logits, weights, axis=0), np.repeat(labels, weights)),
        rtol=1e-6,
    )


def test_fit_accepts_backend_tensors():
    logits, labels = calibrated_data(np.random.default_rng(3), n=500)
    t = fit_temperature(ops.convert_to_tensor(logits * 2), ops.convert_to_tensor(labels))
    np.testing.assert_allclose(t, fit_temperature(logits * 2, labels), rtol=1e-9)


def test_separable_data_returns_lower_bound():
    logits = np.array([[5.0, -5.0], [-5.0, 5.0]], "float32")
    assert fit_temperature(logits, np.array([0, 1]), temperature_range=(0.1, 10.0)) == 0.1


def test_uninformative_logits_return_upper_bound():
    # Confident and always wrong: the NLL keeps decreasing as T grows.
    logits = np.array([[5.0, -5.0], [-5.0, 5.0]], "float32")
    assert fit_temperature(logits, np.array([1, 0]), temperature_range=(0.1, 10.0)) == 10.0


def test_scaling_preserves_predictions_and_improves_calibration():
    logits, labels = calibrated_data(np.random.default_rng(4))
    overconfident = logits * 3
    layer = TemperatureScaling()
    t = layer.fit(overconfident, labels)
    np.testing.assert_allclose(float(to_numpy(layer.temperature)), t, rtol=1e-6)
    scaled = to_numpy(layer(overconfident))
    np.testing.assert_array_equal(scaled.argmax(-1), overconfident.argmax(-1))
    ece_before = float(to_numpy(expected_calibration_error(15, logits=overconfident, labels_true=labels)))
    ece_after = float(to_numpy(expected_calibration_error(15, logits=scaled, labels_true=labels)))
    assert ece_after < 0.02 < ece_before


def test_invalid_inputs():
    for temperature_range in [(0.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError):
            fit_temperature(np.zeros((2, 2)), np.array([0, 1]), temperature_range=temperature_range)
    with pytest.raises(ValueError):
        TemperatureScaling(-1.0)
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((3, 2)), np.array([0, 1]))
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((2, 2, 2)), np.array([0, 1]))


