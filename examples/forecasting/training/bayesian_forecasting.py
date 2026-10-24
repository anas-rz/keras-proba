"""Bayesian temperature forecasting: aleatoric and epistemic uncertainty.

Dataset: Jena climate (Max Planck Institute for Biogeochemistry), hourly weather 2009-2016.
Task: forecast the temperature 24 hours ahead from the previous 48 hours of weather.
Technique (training): a Bayesian `Conv1DFlipout` encoder (uncertainty over the weights) followed
by a `GaussianHead` (uncertainty over the noise). Averaging Monte Carlo predictions gives the
total predictive variance, split by the law of total variance into

    aleatoric = mean of the predicted variances   (noise the model cannot explain)
    epistemic = variance of the predicted means   (what the model does not know)

Run:
    KERAS_BACKEND=jax python examples/forecasting/training/bayesian_forecasting.py
"""

import glob
import os

import numpy as np

import keras
from keras_proba import layers, losses, ops

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 2 if QUICK else 15
NUM_SAMPLES = 5 if QUICK else 30  # Monte Carlo predictions.
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
    """Windows `features[t - LOOKBACK:t]` of shape (LOOKBACK, features), targets and their hour index."""
    ends = np.arange(max(start, LOOKBACK), stop - HORIZON + 1, step)
    x = np.stack([features[t - LOOKBACK : t] for t in ends])
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
# Model: Bayesian convolutional encoder + Gaussian head.
# ---------------------------------------------------------------------------------------------
# The KL term is added on every batch, so weight it by 1 / num_train_examples to count it once
# per epoch.
kl_weight = 1.0 / len(x_train)
model = keras.Sequential(
    [
        keras.Input(x_train.shape[1:]),
        layers.Conv1DFlipout(32, 5, padding="same", activation="relu", kl_weight=kl_weight),
        layers.Conv1DFlipout(32, 5, strides=2, padding="same", activation="relu", kl_weight=kl_weight),
        keras.layers.Flatten(),
        layers.DenseFlipout(64, activation="relu", kl_weight=kl_weight),
        layers.GaussianHead(1),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses.GaussianNLL())
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=2,
)


# ---------------------------------------------------------------------------------------------
# Monte Carlo prediction: every call samples new weights.
# ---------------------------------------------------------------------------------------------
def predict_distribution(x):
    samples = [ops.split_gaussian_params(model.predict(x, batch_size=1024, verbose=0)) for _ in range(NUM_SAMPLES)]
    means = np.stack([keras.ops.convert_to_numpy(m) for m, _ in samples])
    scales = np.stack([keras.ops.convert_to_numpy(s) for _, s in samples])
    aleatoric = np.mean(scales**2, axis=0)
    epistemic = np.var(means, axis=0)
    return means.mean(0), aleatoric, epistemic


mean_pred, aleatoric, epistemic = predict_distribution(x_test)
total_std = np.sqrt(aleatoric + epistemic)
print(f"\nMAE (degC): {np.abs(mean_pred - y_test).mean():.2f}")
print(f"Mean std: aleatoric {np.sqrt(aleatoric).mean():.2f}, epistemic {np.sqrt(epistemic).mean():.2f}, total {total_std.mean():.2f}")

print("\nCentral interval coverage on the test year (aleatoric only | aleatoric + epistemic):")
for level, z in [(0.5, 0.674), (0.8, 1.282), (0.95, 1.960)]:
    aleatoric_only = np.mean(np.abs(y_test - mean_pred) <= z * np.sqrt(aleatoric))
    total = np.mean(np.abs(y_test - mean_pred) <= z * total_std)
    print(f"  {level:.0%} -> {aleatoric_only:.1%} | {total:.1%}")

# Epistemic uncertainty is where the model is unsure of its own forecast: compare by season,
# and on inputs unlike the training data.
month = ((test_hours / 24) % 365.25 / 30.44).astype(int) % 12
seasons = {"winter": [11, 0, 1], "spring": [2, 3, 4], "summer": [5, 6, 7], "autumn": [8, 9, 10]}
print("\nBy season: mean aleatoric std, mean epistemic std, 80% coverage of the total interval:")
for season, months in seasons.items():
    mask = np.isin(month, months)
    covered = np.mean(np.abs(y_test[mask] - mean_pred[mask]) <= 1.282 * total_std[mask])
    print(f"  {season:7s} {np.sqrt(aleatoric[mask]).mean():.2f}  {np.sqrt(epistemic[mask]).mean():.3f}  {covered:.1%}")

# Shift the inputs far outside the training range (e.g. a sensor fault): the model's epistemic
# uncertainty grows, while a deterministic model would forecast with unchanged confidence.
x_shifted = x_test[:2000] + 3.0  # +3 training standard deviations on every feature.
_, _, epistemic_shifted = predict_distribution(x_shifted)
print(
    f"\nMean epistemic std: test inputs {np.sqrt(epistemic[:2000]).mean():.3f}, "
    f"inputs shifted by 3 standard deviations {np.sqrt(epistemic_shifted).mean():.3f}"
)
