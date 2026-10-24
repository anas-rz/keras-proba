"""Parity tests for temperature scaling.

Applying a temperature is compared with TensorFlow Model Garden's object
detection `calibration_builder` temperature scaling calibrator. Fitting it is not
in the reference, so the fitted temperature is compared with a generic bounded
scalar minimization of the same NLL with SciPy.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from keras import ops

from keras_proba.postprocessing.temperature_scaling import (
    TemperatureScaling,
    fit_temperature,
    temperature_scaled_nll,
)

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "reference/models/research"


@pytest.fixture(scope="module")
def calibration_builder():
    pytest.importorskip("tensorflow")
    if not REFERENCE_PATH.exists():
        pytest.skip(f"Reference implementation not found at {REFERENCE_PATH}")
    sys.path.insert(0, str(REFERENCE_PATH))
    return pytest.importorskip("object_detection.builders.calibration_builder")


def temperature_config(scaler):
    """A stand-in for `calibration_pb2.CalibrationConfig` with a temperature scaling calibrator."""
    return SimpleNamespace(
        WhichOneof=lambda name: "temperature_scaling_calibration",
        temperature_scaling_calibration=SimpleNamespace(scaler=scaler),
    )


@pytest.mark.parametrize("temperature", [0.3, 1.0, 2.5])
def test_apply_temperature_parity(calibration_builder, temperature):
    import tensorflow as tf

    logits = np.random.default_rng(0).normal(size=(2, 7, 4)).astype("float32")
    expected = calibration_builder.build(temperature_config(temperature))(tf.constant(logits)).numpy()
    actual = ops.convert_to_numpy(TemperatureScaling(temperature)(logits))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_reference_rejects_non_positive_temperature(calibration_builder):
    import tensorflow as tf

    with pytest.raises(ValueError):
        calibration_builder.build(temperature_config(0.0))(tf.constant([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        TemperatureScaling(0.0)


def scipy_temperature(logits, labels, sample_weight=None, bounds=(0.05, 20.0)):
    optimize = pytest.importorskip("scipy.optimize")
    result = optimize.minimize_scalar(
        lambda log_t: temperature_scaled_nll(logits, labels, np.exp(log_t), sample_weight),
        bounds=np.log(bounds),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(np.exp(result.x))


def overconfident_logits(rng, n, num_classes, sharpness):
    true_logits = rng.normal(size=(n, num_classes)) * 1.5
    probs = np.exp(true_logits) / np.exp(true_logits).sum(-1, keepdims=True)
    labels = (rng.uniform(size=(n, 1)) > np.cumsum(probs, -1)).sum(-1).clip(0, num_classes - 1)
    return (true_logits * sharpness).astype("float32"), labels


@pytest.mark.parametrize("num_classes, sharpness", [(2, 3.0), (5, 0.4), (10, 2.0)])
def test_fit_matches_scipy(num_classes, sharpness):
    rng = np.random.default_rng(num_classes)
    logits, labels = overconfident_logits(rng, 2000, num_classes, sharpness)
    np.testing.assert_allclose(fit_temperature(logits, labels), scipy_temperature(logits, labels), rtol=1e-5)


def test_fit_matches_scipy_binary_and_weighted():
    rng = np.random.default_rng(0)
    z = rng.normal(size=1000) * 4
    labels = (rng.uniform(size=1000) < 1 / (1 + np.exp(-z / 4))).astype("int32")
    weights = rng.uniform(size=1000)
    np.testing.assert_allclose(fit_temperature(z, labels), scipy_temperature(z, labels), rtol=1e-5)
    np.testing.assert_allclose(
        fit_temperature(z[:, None], labels, sample_weight=weights),
        scipy_temperature(z[:, None], labels, sample_weight=weights),
        rtol=1e-5,
    )
