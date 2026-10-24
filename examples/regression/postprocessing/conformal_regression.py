"""Conformal prediction intervals for house prices.

Dataset: California housing (1990 US census), median house value per block group.
Task: predict the house value from 8 features (income, house age, rooms, location, ...).
Technique (post-processing): split conformal prediction with
`keras_proba.postprocessing.SplitConformalRegressor`, which turns held-out residuals into
intervals that contain the true value with probability at least 90%:

- around a point prediction, with a constant width (absolute residuals);
- around a Gaussian prediction, with a width proportional to its predicted standard deviation
  (normalized residuals), so cheap and predictable areas get narrow intervals.

The guarantee only assumes that calibration and test examples are exchangeable, which holds for
this random split.

Run:
    KERAS_BACKEND=jax python examples/regression/postprocessing/conformal_regression.py
"""

import os

import numpy as np

import keras
from keras_proba import layers, losses, ops, postprocessing

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 3 if QUICK else 100
ALPHA = 0.1  # Target miscoverage: 90% intervals.
keras.utils.set_random_seed(0)

# ---------------------------------------------------------------------------------------------
# Data, with values in units of $100,000, split into train / calibration / test.
# ---------------------------------------------------------------------------------------------
(x, y), (x_test, y_test) = keras.datasets.california_housing.load_data(version="large", test_split=0.2)
y, y_test = (y / 1e5).astype("float32")[:, None], (y_test / 1e5).astype("float32")[:, None]
num_cal = len(x) // 4
x_train, y_train, x_cal, y_cal = x[num_cal:], y[num_cal:], x[:num_cal], y[:num_cal]
mean, std = x_train.mean(0), x_train.std(0)
x_train, x_cal, x_test = (((v - mean) / std).astype("float32") for v in (x_train, x_cal, x_test))
income = x_test[:, 0]  # Standardized median income of the block group.
print(f"train {x_train.shape}, calibration {x_cal.shape}, test {x_test.shape}")


def build(head):
    return keras.Sequential(
        [keras.Input((8,)), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(128, activation="relu"), head]
    )


def fit(model, loss):
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=128,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )
    return model


# Any trained model works: conformal prediction only needs predictions on held-out data.
point_model = fit(build(keras.layers.Dense(1)), "mse")
gaussian_model = fit(build(layers.GaussianHead(1)), losses.GaussianNLL())

mean_cal, scale_cal = ops.split_gaussian_params(gaussian_model.predict(x_cal, verbose=0))
mean_test, scale_test = ops.split_gaussian_params(gaussian_model.predict(x_test, verbose=0))

intervals = {
    "point model, constant width": postprocessing.SplitConformalRegressor(ALPHA)
    .fit(y_cal, point_model.predict(x_cal, verbose=0))
    .predict_interval(point_model.predict(x_test, verbose=0)),
    "Gaussian model, scaled by its std": postprocessing.SplitConformalRegressor(ALPHA)
    .fit(y_cal, mean_cal, scale=scale_cal)
    .predict_interval(mean_test, scale=scale_test),
}

# ---------------------------------------------------------------------------------------------
# Coverage overall and within income groups. Both are valid overall; the scaled intervals are
# narrow where prices are predictable and wide where they are not, so their coverage is more even
# across groups.
# ---------------------------------------------------------------------------------------------
edges = np.quantile(income, [0, 0.25, 0.5, 0.75, 1.0])
groups = np.clip(np.searchsorted(edges, income, side="right") - 1, 0, 3)
print(f"\n{1 - ALPHA:.0%} conformal intervals on the test set ($100k):")
print(f"  {'':34s} {'coverage':>9s} {'width':>6s}   coverage / width by income quartile (low -> high)")
for name, (lower, upper) in intervals.items():
    lower, upper = keras.ops.convert_to_numpy(lower), keras.ops.convert_to_numpy(upper)
    inside = ((y_test >= lower) & (y_test <= upper))[:, 0]
    width = (upper - lower)[:, 0]
    by_group = "  ".join(f"{inside[groups == g].mean():.0%}/{width[groups == g].mean():.2f}" for g in range(4))
    print(f"  {name:34s} {inside.mean():9.1%} {width.mean():6.2f}   {by_group}")

# Without conformal calibration, the Gaussian model's own 90% intervals are not guaranteed.
raw_inside = np.abs(y_test - keras.ops.convert_to_numpy(mean_test)) <= 1.645 * keras.ops.convert_to_numpy(scale_test)
print(f"\nThe Gaussian model's own 90% intervals, without calibration, cover {raw_inside.mean():.1%}.")
