"""Conformal prediction intervals for temperature forecasts.

Dataset: Jena climate (Max Planck Institute for Biogeochemistry), hourly weather 2009-2016.
Task: forecast the temperature 24 hours ahead from the previous 48 hours of weather.
Technique (post-processing): split conformal prediction with
`keras_proba.postprocessing.SplitConformalRegressor`. Held-out residuals turn any forecaster into
one whose intervals contain the truth with a chosen probability, here 90%:

- around a point forecast, with a constant width (absolute residual scores);
- around a Gaussian forecast, with a width proportional to its predicted standard deviation
  (normalized residual scores), so the intervals adapt to each input.

Conformal guarantees assume calibration and test data are exchangeable. Weather is not
stationary, so the calibration data is the most recent period before the test year, and the
coverage is checked on the test year.

Run:
    KERAS_BACKEND=jax python examples/forecasting/postprocessing/conformal_forecast_intervals.py
"""

import glob
import os

import numpy as np

import keras
from keras_proba import layers, losses, ops, postprocessing

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 2 if QUICK else 15
LOOKBACK, HORIZON = 48, 24  # Hours.
ALPHA = 0.1  # Target miscoverage: 90% intervals.
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
# Train on 2009-2014, calibrate on the following year, test on the last year.
train_end, cal_end = int(0.7 * n), int(0.85 * n)
mean, std = features[:train_end].mean(0), features[:train_end].std(0)
features = (features - mean) / std

step = 6 if QUICK else 1
x_train, y_train, _ = make_windows(features, temperature, 0, train_end, step)
x_cal, y_cal, _ = make_windows(features, temperature, train_end, cal_end, step)
x_test, y_test, test_hours = make_windows(features, temperature, cal_end, n, step)
print(f"train {x_train.shape}, calibration {x_cal.shape}, test {x_test.shape}")


# ---------------------------------------------------------------------------------------------
# A probabilistic forecaster: any trained model works, conformal prediction only needs its
# predictions on held-out data.
# ---------------------------------------------------------------------------------------------
model = keras.Sequential(
    [
        keras.Input((x_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        layers.GaussianHead(1),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses.GaussianNLL())
# The last 20% of the training years are used for early stopping, keeping the calibration year
# untouched.
model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=2,
)

mean_cal, scale_cal = ops.split_gaussian_params(model.predict(x_cal, verbose=0))
mean_test, scale_test = ops.split_gaussian_params(model.predict(x_test, verbose=0))


# ---------------------------------------------------------------------------------------------
# Three ways to get 90% intervals.
# ---------------------------------------------------------------------------------------------
def season_of(hours):
    """Season index (0 winter, 1 spring, 2 summer, 3 autumn); the data starts on 2009-01-01, hourly."""
    month = ((hours / 24) % 365.25 / 30.44).astype(int) % 12
    return ((month + 1) % 12) // 3


z = 1.645  # Two-sided 90% quantile of the standard normal.
intervals = {
    "Gaussian head (no calibration)": (mean_test - z * scale_test, mean_test + z * scale_test),
    "conformal, constant width": postprocessing.SplitConformalRegressor(ALPHA).fit(y_cal, mean_cal).predict_interval(mean_test),
    "conformal, scaled by the std": postprocessing.SplitConformalRegressor(ALPHA)
    .fit(y_cal, mean_cal, scale=scale_cal)
    .predict_interval(mean_test, scale=scale_test),
}

seasons = ["winter", "spring", "summer", "autumn"]
test_season = season_of(test_hours)

print(f"\n{1 - ALPHA:.0%} intervals on the test year: coverage, mean width (degC), and coverage by season")
for name, (lower, upper) in intervals.items():
    lower, upper = keras.ops.convert_to_numpy(lower), keras.ops.convert_to_numpy(upper)
    inside = (y_test >= lower) & (y_test <= upper)
    by_season = ", ".join(f"{name} {inside[test_season == i].mean():.0%}" for i, name in enumerate(seasons))
    print(f"  {name:32s} {inside.mean():.1%}  width {(upper - lower).mean():5.2f}  ({by_season})")

# Conformal calibration restores the overall coverage that the Gaussian head alone can miss on a
# new year. The guarantee is marginal: it holds over all forecasts together, not within each
# season, and it relies on the calibration period resembling the test period. Calibrating each
# season separately ("Mondrian" conformal prediction) targets per-season coverage, but only helps
# if each season of the calibration year resembles the same season of the test year.
