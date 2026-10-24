"""Coverage tests for split conformal prediction on trained models. Run once per backend via `KERAS_BACKEND`.

For continuous scores, the coverage of split conformal prediction averaged over
random calibration/test splits is exactly `k / (n + 1)` with
`k = ceil((n + 1) (1 - alpha))`, which these tests check.
"""

import math

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.gaussian_head import GaussianHead, GaussianNLL, split_gaussian_params
from keras_proba.postprocessing.conformal import (
    SplitConformalClassifier,
    SplitConformalRegressor,
    average_set_size,
    interval_coverage,
    set_coverage,
)


def to_numpy(x):
    return ops.convert_to_numpy(x)


def heteroscedastic_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 1)).astype("float32")
    std = 0.1 + 0.4 * np.abs(x)
    return x, (np.sin(x) + rng.normal(size=(n, 1)) * std).astype("float32"), std


@pytest.fixture(scope="module")
def regression_predictions():
    keras.utils.set_random_seed(0)
    x, y, _ = heteroscedastic_data(2000, seed=0)
    model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(64, activation="relu"), GaussianHead(1)])
    model.compile(optimizer=keras.optimizers.Adam(0.005), loss=GaussianNLL())
    model.fit(x, y, epochs=40, batch_size=128, verbose=0)
    x_pool, y_pool, std_pool = heteroscedastic_data(4000, seed=1)
    mean, scale = (to_numpy(t) for t in split_gaussian_params(model.predict(x_pool, verbose=0)))
    return x_pool, y_pool, std_pool, mean, scale


@pytest.mark.parametrize("normalized", [False, True])
def test_regression_expected_coverage(regression_predictions, normalized):
    _, y, _, mean, scale = regression_predictions
    alpha, n_cal = 0.1, 99
    expected = math.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1)
    rng = np.random.default_rng(0)
    coverages = []
    for _ in range(100):
        idx = rng.permutation(len(y))
        cal, test = idx[:n_cal], idx[n_cal:]
        conformal = SplitConformalRegressor(alpha).fit(y[cal], mean[cal], scale=scale[cal] if normalized else None)
        lower, upper = conformal.predict_interval(mean[test], scale=scale[test] if normalized else None)
        coverages.append(float(to_numpy(interval_coverage(y[test], lower, upper))[0]))
    # Mean coverage over 100 splits has a standard error of about 0.003.
    assert abs(np.mean(coverages) - expected) < 0.01, (np.mean(coverages), expected)


def test_normalized_intervals_adapt_to_noise(regression_predictions):
    x, y, std, mean, scale = regression_predictions
    cal, test = slice(0, 1000), slice(1000, None)
    intervals = {}
    for normalized in [False, True]:
        conformal = SplitConformalRegressor(0.1).fit(y[cal], mean[cal], scale=scale[cal] if normalized else None)
        intervals[normalized] = [to_numpy(t) for t in conformal.predict_interval(mean[test], scale=scale[test] if normalized else None)]

    width = intervals[True][1] - intervals[True][0]
    assert np.corrcoef(width[:, 0], std[test][:, 0])[0, 1] > 0.9
    # Constant intervals over-cover low-noise inputs and under-cover high-noise ones; normalized ones do not.
    noisy = np.abs(x[test][:, 0]) > 2
    for normalized, (lower, upper) in intervals.items():
        inside = (y[test] >= lower) & (y[test] <= upper)
        if normalized:
            assert inside[noisy].mean() > 0.85
        else:
            assert inside[noisy].mean() < 0.85 < inside[~noisy].mean()


@pytest.fixture(scope="module")
def classification_predictions():
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(5, 2)) * 2

    def sample(n, seed):
        r = np.random.default_rng(seed)
        y = r.integers(0, 5, size=n)
        return (centers[y] + r.normal(size=(n, 2)) * 1.2).astype("float32"), y.astype("int32")

    x, y = sample(2000, 1)
    model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(5, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="sparse_categorical_crossentropy")
    model.fit(x, y, epochs=20, batch_size=64, verbose=0)
    x_pool, y_pool = sample(4000, 2)
    return y_pool, model.predict(x_pool, verbose=0)


@pytest.mark.parametrize("score, randomized", [("lac", True), ("aps", True), ("aps", False)])
def test_classification_coverage(classification_predictions, score, randomized):
    y, probs = classification_predictions
    alpha, n_cal = 0.1, 199
    expected = math.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1)
    rng = np.random.default_rng(0)
    coverages, sizes = [], []
    for _ in range(100):
        idx = rng.permutation(len(y))
        cal, test = idx[:n_cal], idx[n_cal:]
        conformal = SplitConformalClassifier(alpha, score=score, randomized=randomized, seed=0)
        sets = conformal.fit(y[cal], probs[cal]).predict_set(probs[test])
        coverages.append(float(to_numpy(set_coverage(y[test], sets))))
        sizes.append(float(to_numpy(average_set_size(sets))))
    if randomized:
        # LAC and randomized APS have exact expected coverage (~0.90 here).
        assert abs(np.mean(coverages) - expected) < 0.01, (np.mean(coverages), expected)
        # Informative sets: much smaller than all 5 classes (~2.0 for LAC, ~2.2 for APS).
        assert np.mean(sizes) < 2.6
    else:
        # Non-randomized APS is valid but conservative on this hard task (~0.996 coverage).
        assert np.mean(coverages) > expected
        assert np.mean(sizes) > 3.0


def test_aps_sets_grow_with_uncertainty(classification_predictions):
    y, probs = classification_predictions
    sets = to_numpy(SplitConformalClassifier(0.1, score="aps").fit(y[:1000], probs[:1000]).predict_set(probs[1000:]))
    entropy = -np.sum(probs[1000:] * np.log(probs[1000:] + 1e-12), axis=-1)
    uncertain = entropy > np.median(entropy)
    assert sets[uncertain].sum(-1).mean() > sets[~uncertain].sum(-1).mean() + 0.5
