"""House price prediction with a mixture density network.

Dataset: California housing (1990 US census), median house value per block group.
Task: predict the house value from 8 features (income, house age, rooms, location, ...).
Technique (training): `keras_proba.layers.MixtureGaussianHead` trained with `MixtureGaussianNLL`
predicts a full distribution per input as a mixture of Gaussians, compared with a single
`GaussianHead`.

House values in the census were capped at $500,001: about 5% of the block groups sit exactly at
the cap, a spike that a single Gaussian cannot represent but a mixture can.

Run:
    KERAS_BACKEND=jax python examples/regression/training/mixture_density_network.py
"""

import os
from math import erf, sqrt

import numpy as np

import keras
from keras_proba import layers, losses, ops

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 3 if QUICK else 100
NUM_COMPONENTS = 5
CAP = 5.00001  # $500,001, in units of $100,000.
keras.utils.set_random_seed(0)

# ---------------------------------------------------------------------------------------------
# Data, with values in units of $100,000.
# ---------------------------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.california_housing.load_data(version="large", test_split=0.2)
y_train, y_test = (y_train / 1e5).astype("float32")[:, None], (y_test / 1e5).astype("float32")[:, None]
mean, std = x_train.mean(0), x_train.std(0)
x_train, x_test = ((x_train - mean) / std).astype("float32"), ((x_test - mean) / std).astype("float32")
print(f"train {x_train.shape}, test {x_test.shape}, at the cap: {np.mean(y_test >= CAP - 1e-3):.1%} of the test set")


def build(head):
    return keras.Sequential(
        [
            keras.Input((8,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            head,
        ]
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


gaussian = fit(build(layers.GaussianHead(1)), losses.GaussianNLL())
mixture = fit(build(layers.MixtureGaussianHead(NUM_COMPONENTS)), losses.MixtureGaussianNLL())

gaussian_params = gaussian.predict(x_test, verbose=0)
mixture_params = mixture.predict(x_test, verbose=0)

# ---------------------------------------------------------------------------------------------
# Likelihood: how probable the true values are under each predicted distribution.
# ---------------------------------------------------------------------------------------------
gaussian_nll = keras.ops.convert_to_numpy(losses.gaussian_nll(y_test, gaussian_params))
mixture_nll = keras.ops.convert_to_numpy(losses.mixture_gaussian_nll(y_test, mixture_params))
capped = y_test[:, 0] >= CAP - 1e-3
print("\nTest negative log-likelihood (lower is better):")
print(f"  {'':8s} {'all':>7s} {'below cap':>10s} {'at cap':>8s}")
for name, nll in [("Gaussian", gaussian_nll), ("mixture", mixture_nll)]:
    print(f"  {name:8s} {nll.mean():7.3f} {nll[~capped].mean():10.3f} {nll[capped].mean():8.3f}")

# ---------------------------------------------------------------------------------------------
# The mixture puts a narrow component on the cap for expensive areas.
# ---------------------------------------------------------------------------------------------
logits, means, scales = (keras.ops.convert_to_numpy(t) for t in ops.split_mixture_params(mixture_params))
weights = np.exp(logits - logits.max(-1, keepdims=True))
weights /= weights.sum(-1, keepdims=True)
near_cap = np.abs(means[..., 0] - CAP) < 0.1  # Components centered within $10,000 of the cap.
mass_at_cap = (weights * near_cap).sum(-1)
print("\nMixture weight of the components centered at the cap:")
print(f"  houses at the cap: {mass_at_cap[capped].mean():.2f}, other houses: {mass_at_cap[~capped].mean():.2f}")

# ---------------------------------------------------------------------------------------------
# Predictive mean and spread, and 90% intervals from each predicted distribution.
# ---------------------------------------------------------------------------------------------
gaussian_mean, gaussian_std = (keras.ops.convert_to_numpy(t) for t in ops.split_gaussian_params(gaussian_params))
mixture_mean, mixture_std = (keras.ops.convert_to_numpy(t) for t in ops.mixture_mean_and_stddev(mixture_params))
print(f"\nMAE ($100k): Gaussian {np.abs(gaussian_mean - y_test).mean():.3f}, mixture {np.abs(mixture_mean - y_test).mean():.3f}")


def mixture_quantile(q):
    """Quantile of each predicted mixture, by bisection on its CDF."""
    normal_cdf = np.vectorize(lambda v: 0.5 * (1 + erf(v / sqrt(2))))
    lo, hi = np.full(len(y_test), -5.0), np.full(len(y_test), 15.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        cdf = (weights * normal_cdf((mid[:, None] - means[..., 0]) / scales[..., 0])).sum(-1)
        lo, hi = np.where(cdf < q, mid, lo), np.where(cdf < q, hi, mid)
    return ((lo + hi) / 2)[:, None]


lower, upper = mixture_quantile(0.05), mixture_quantile(0.95)
gaussian_inside = np.abs(y_test - gaussian_mean) <= 1.645 * gaussian_std
mixture_inside = (y_test >= lower) & (y_test <= upper)
print("\n90% interval coverage and mean width ($100k), from each predicted distribution:")
print(f"  Gaussian {gaussian_inside.mean():.1%}, width {(2 * 1.645 * gaussian_std).mean():.2f}")
print(f"  mixture  {mixture_inside.mean():.1%}, width {(upper - lower).mean():.2f}")
