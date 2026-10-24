"""Probabilistic temperature forecasting with a Gaussian head.

Dataset: Jena climate (Max Planck Institute for Biogeochemistry), hourly weather 2009-2016.
Task: forecast the temperature 24 hours ahead from the previous 48 hours of weather.
Technique (training): `keras_proba.layers.GaussianHead` trained with `GaussianNLL`, so the model
predicts a mean and a standard deviation for every forecast. The standard deviation is the
aleatoric uncertainty: how unpredictable the weather is from this input.

The script compares it with a point forecaster trained with MSE, and checks whether the predicted
intervals contain the stated fraction of the test-year temperatures.

Run:
    KERAS_BACKEND=jax python examples/forecasting/training/probabilistic_forecasting.py
"""

import glob
import os

import numpy as np

import keras
from keras_proba import layers, losses, ops

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 2 if QUICK else 15
LOOKBACK, HORIZON = 48, 24  # Hours.
keras.utils.set_random_seed(0)


# ---------------------------------------------------------------------------------------------
# Data: hourly windows of the last 48 hours -> temperature 24 hours later.
# ---------------------------------------------------------------------------------------------
def load_jena():
    """Returns hourly features (n, 8) and the temperature (n,), in chronological order."""
    path = keras.utils.get_file(
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        extract=True,
    )
    csv = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)[0]
    # Columns: p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, H2OC, rho, wv, max. wv, wd.
    raw = np.genfromtxt(csv, delimiter=",", skip_header=1, usecols=range(1, 15), dtype="float32")
    raw = raw[5::6]  # One sample every 10 minutes: keep one per hour.
    pressure, temperature, dew_point, humidity, wind = raw[:, 0], raw[:, 1], raw[:, 3], raw[:, 4], raw[:, 11]
    wind = np.maximum(wind, 0.0)  # -9999 marks missing wind measurements.
    hours = np.arange(len(raw), dtype="float32")
    day, year = 2 * np.pi * hours / 24, 2 * np.pi * hours / (24 * 365.25)
    features = np.stack(
        [pressure, temperature, dew_point, humidity, wind, np.sin(day), np.cos(day), np.sin(year)], axis=-1
    )
    return features.astype("float32"), temperature


def make_windows(features, target, start, stop, step=1):
    """Flattened windows `features[t - LOOKBACK:t]`, targets `target[t + HORIZON - 1]` and their hour index."""
    ends = np.arange(max(start, LOOKBACK), stop - HORIZON + 1, step)
    x = np.stack([features[t - LOOKBACK : t].reshape(-1) for t in ends])
    return x, target[ends + HORIZON - 1][:, None], ends + HORIZON - 1


features, temperature = load_jena()
n = len(features)
train_end, val_end = int(0.7 * n), int(0.85 * n)  # 2009-2014 / 2014-2015 / 2015-2016.
mean, std = features[:train_end].mean(0), features[:train_end].std(0)
features = (features - mean) / std

step = 6 if QUICK else 1
x_train, y_train, _ = make_windows(features, temperature, 0, train_end, step)
x_val, y_val, _ = make_windows(features, temperature, train_end, val_end, step)
x_test, y_test, test_hours = make_windows(features, temperature, val_end, n, step)
print(f"train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")


# ---------------------------------------------------------------------------------------------
# Models: the same network with a point head (MSE) and with a Gaussian head (NLL).
# ---------------------------------------------------------------------------------------------
def build(head):
    return keras.Sequential(
        [
            keras.Input((x_train.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            head,
        ]
    )


callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]

point_model = build(keras.layers.Dense(1))
point_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
point_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=256, callbacks=callbacks, verbose=2)

gaussian_model = build(layers.GaussianHead(1))
gaussian_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses.GaussianNLL())
gaussian_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=256, callbacks=callbacks, verbose=2)


# ---------------------------------------------------------------------------------------------
# Evaluation on the held-out last year.
# ---------------------------------------------------------------------------------------------
point = point_model.predict(x_test, verbose=0)
mean_pred, scale = (keras.ops.convert_to_numpy(t) for t in ops.split_gaussian_params(gaussian_model.predict(x_test, verbose=0)))

print(f"\nMAE (degC): point model {np.abs(point - y_test).mean():.2f}, Gaussian model {np.abs(mean_pred - y_test).mean():.2f}")
nll = keras.ops.convert_to_numpy(losses.GaussianNLL()(y_test, np.concatenate([mean_pred, scale], -1)))
print(f"Gaussian model test NLL: {float(nll):.3f}")

# A point forecast only has one spread: the residual standard deviation on the validation set.
point_std = float(np.std(point_model.predict(x_val, verbose=0) - y_val))
print("\nCentral interval coverage on the test year (target -> constant width | Gaussian head):")
for level, z in [(0.5, 0.674), (0.8, 1.282), (0.95, 1.960)]:
    constant = np.mean(np.abs(y_test - point) <= z * point_std)
    adaptive = np.mean(np.abs(y_test - mean_pred) <= z * scale)
    print(f"  {level:.0%} -> {constant:.1%} (width {2 * z * point_std:.2f}) | {adaptive:.1%} (mean width {2 * z * scale.mean():.2f})")

# Coverage within each season: an exogenous split, unlike splitting by the predicted std itself
# (forecasts with an overestimated std would land in the "uncertain" group and over-cover).
# The Gaussian head is sharper and calibrated overall, but it only models the noise seen in
# training (aleatoric uncertainty): a test-year season unlike the training years can be
# under-covered. `bayesian_forecasting.py` adds epistemic uncertainty, and
# `../postprocessing/conformal_forecast_intervals.py` adds a coverage guarantee.
# The data starts on 2009-01-01, one sample per hour.
month = ((test_hours / 24) % 365.25 / 30.44).astype(int) % 12
seasons = {"winter": [11, 0, 1], "spring": [2, 3, 4], "summer": [5, 6, 7], "autumn": [8, 9, 10]}
print("\n80% interval coverage and mean width by season (constant width | Gaussian head):")
for season, months in seasons.items():
    mask = np.isin(month, months)
    constant = np.mean(np.abs(y_test[mask] - point[mask]) <= 1.282 * point_std)
    adaptive = np.mean(np.abs(y_test[mask] - mean_pred[mask]) <= 1.282 * scale[mask])
    print(f"  {season:7s} {constant:.1%} (width {2.564 * point_std:.2f}) | {adaptive:.1%} (width {2.564 * scale[mask].mean():.2f})")
