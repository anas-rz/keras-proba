"""Training tests for temperature scaling. Run once per backend via `KERAS_BACKEND`."""

import numpy as np

import keras
from keras import ops

from keras_proba.metrics.calibration import ExpectedCalibrationError
from keras_proba.postprocessing.temperature_scaling import TemperatureScaling, fit_temperature


def make_data(n, seed):
    """Noisy 3-class data, so that a long-trained network becomes overconfident."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 10)).astype("float32")
    logits = x[:, :3] * 1.5
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(probs, -1)).sum(-1).clip(0, 2).astype("int32")
    return x, y


def train_overconfident_classifier():
    keras.utils.set_random_seed(0)
    x, y = make_data(300, seed=0)
    model = keras.Sequential(
        [keras.Input((10,)), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(3)]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.003), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x, y, epochs=150, batch_size=32, verbose=0)
    return model


def evaluate_calibration(model, x, y):
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    logits = model.predict(x, verbose=0)
    ece = ExpectedCalibrationError(num_bins=10, from_logits=True)
    ece.update_state(y, logits)
    return float(ops.convert_to_numpy(loss(y, logits))), float(ops.convert_to_numpy(ece.result()))


def test_temperature_scaling_improves_held_out_calibration():
    model = train_overconfident_classifier()
    x_val, y_val = make_data(2000, seed=1)
    x_test, y_test = make_data(4000, seed=2)

    scaling = TemperatureScaling()
    temperature = scaling.fit(model.predict(x_val, verbose=0), y_val)
    assert temperature > 1.5  # The overfit network is overconfident.
    calibrated = keras.Sequential([model, scaling])

    nll_before, ece_before = evaluate_calibration(model, x_test, y_test)
    nll_after, ece_after = evaluate_calibration(calibrated, x_test, y_test)
    assert nll_after < nll_before
    assert ece_after < 0.5 * ece_before
    # Accuracy is unchanged.
    np.testing.assert_array_equal(
        calibrated.predict(x_test, verbose=0).argmax(-1), model.predict(x_test, verbose=0).argmax(-1)
    )


def test_gradient_training_agrees_with_closed_form_fit():
    """Training only the temperature with `model.fit` reaches the closed-form optimum."""
    model = train_overconfident_classifier()
    model.trainable = False
    x_val, y_val = make_data(2000, seed=1)
    expected = fit_temperature(model.predict(x_val, verbose=0), y_val)

    scaling = TemperatureScaling()
    calibrated = keras.Sequential([model, scaling])
    calibrated.compile(
        optimizer=keras.optimizers.Adam(0.05), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    # The optimum is far from T = 1 (log T ~ 2.2), so give Adam enough full-batch steps.
    calibrated.fit(x_val, y_val, epochs=300, batch_size=2000, verbose=0)
    np.testing.assert_allclose(float(ops.convert_to_numpy(scaling.temperature)), expected, rtol=0.02)
