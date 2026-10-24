"""Smoke and correctness tests for split conformal prediction. Run once per backend via `KERAS_BACKEND`."""

import math

import numpy as np
import pytest

from keras import ops

from keras_proba.postprocessing.conformal import (
    SplitConformalClassifier,
    SplitConformalRegressor,
    average_set_size,
    conformal_quantile,
    interval_coverage,
    set_coverage,
)


def to_numpy(x):
    return ops.convert_to_numpy(x)


def softmax(logits):
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_conformal_quantile_order_statistic():
    scores = np.arange(1.0, 101.0, dtype="float32")  # n = 100.
    # k = ceil(101 * 0.9) = 91.
    assert to_numpy(conformal_quantile(scores, 0.1)) == 91.0
    # Too few scores for 99.5% coverage: everything is included.
    assert np.isinf(to_numpy(conformal_quantile(scores[:50], 0.005)))


def test_conformal_quantile_per_column():
    scores = np.stack([np.arange(10.0), 10 * np.arange(10.0)], -1).astype("float32")
    k = math.ceil(11 * 0.8)
    np.testing.assert_allclose(to_numpy(conformal_quantile(scores, 0.2)), [k - 1, 10 * (k - 1)])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        SplitConformalRegressor(alpha)
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), alpha)


def test_regressor_constant_and_scaled_intervals():
    rng = np.random.default_rng(0)
    y_pred = rng.normal(size=(500,)).astype("float32")
    y = (y_pred + rng.normal(size=500)).astype("float32")
    conformal = SplitConformalRegressor(0.2).fit(y, y_pred)
    lower, upper = conformal.predict_interval(y_pred[:5])
    width = to_numpy(upper - lower)
    np.testing.assert_allclose(width, width[0], rtol=1e-6)
    np.testing.assert_allclose(width[0] / 2, np.quantile(np.abs(y - y_pred), 0.8), rtol=0.1)

    scale = rng.uniform(0.5, 2.0, size=500).astype("float32")
    conformal = SplitConformalRegressor(0.2).fit(y, y_pred, scale=scale)
    lower, upper = conformal.predict_interval(y_pred[:5], scale=scale[:5])
    np.testing.assert_allclose(to_numpy(upper - lower), 2 * to_numpy(conformal.quantile_) * scale[:5], rtol=1e-6)


def test_regressor_requires_consistent_scale():
    conformal = SplitConformalRegressor().fit(np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError):
        conformal.predict_interval(np.zeros(3), scale=np.ones(3))
    with pytest.raises(ValueError):
        SplitConformalRegressor().predict_interval(np.zeros(3))


def test_lac_sets():
    probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]], "float32")
    conformal = SplitConformalClassifier(0.1)
    conformal.quantile_ = ops.convert_to_tensor(np.float32(0.7))
    np.testing.assert_array_equal(to_numpy(conformal.predict_set(probs)), [[True, False, False], [True, True, False]])


def test_non_randomized_aps_sets_are_never_empty_and_nested():
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(200, 5)) * 3).astype("float32")
    labels = rng.integers(0, 5, size=200)
    sizes = []
    for alpha in [0.5, 0.2, 0.05]:
        sets = to_numpy(SplitConformalClassifier(alpha, score="aps", randomized=False).fit(labels, probs).predict_set(probs))
        assert sets.any(-1).all()
        sizes.append(sets)
    # Lower alpha gives supersets.
    assert np.all(sizes[0] <= sizes[1]) and np.all(sizes[1] <= sizes[2])


def test_aps_scores():
    probs = np.array([[0.5, 0.3, 0.2]], "float32")
    conformal = SplitConformalClassifier(score="aps", randomized=False)
    np.testing.assert_allclose(to_numpy(conformal.scores(np.array([0]), probs)), [0.5], rtol=1e-6)
    np.testing.assert_allclose(to_numpy(conformal.scores(np.array([2]), probs)), [1.0], rtol=1e-6)
    np.testing.assert_allclose(to_numpy(conformal.scores(np.array([1]), probs)), [0.8], rtol=1e-6)


def test_randomized_aps_scores_and_seed():
    probs = np.tile(np.array([[0.5, 0.3, 0.2]], "float32"), (1000, 1))
    scores = to_numpy(SplitConformalClassifier(score="aps", seed=1).scores(np.full(1000, 1), probs))
    # Mass before the label (0.5) plus u * 0.3, with u ~ Uniform(0, 1).
    assert scores.min() >= 0.5 and scores.max() <= 0.8
    np.testing.assert_allclose(scores.mean(), 0.65, atol=0.02)
    again = to_numpy(SplitConformalClassifier(score="aps", seed=1).scores(np.full(1000, 1), probs))
    np.testing.assert_array_equal(scores, again)


def test_label_and_prediction_formats():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(100, 4)).astype("float32")
    labels = rng.integers(0, 4, size=100)
    reference = to_numpy(SplitConformalClassifier(0.1).fit(labels, softmax(logits)).quantile_)
    for y_true in [labels[:, None], np.eye(4)[labels]]:
        np.testing.assert_allclose(to_numpy(SplitConformalClassifier(0.1).fit(y_true, softmax(logits)).quantile_), reference, rtol=1e-6)
    np.testing.assert_allclose(
        to_numpy(SplitConformalClassifier(0.1, from_logits=True).fit(labels, logits).quantile_), reference, rtol=1e-5
    )


def test_binary_sigmoid_outputs():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=(300, 1)).astype("float32")
    labels = (rng.uniform(size=(300, 1)) < p).astype("int32")
    sets = SplitConformalClassifier(0.2).fit(labels, p).predict_set(p)
    assert ops.shape(sets) == (300, 2)
    two_class = np.concatenate([1 - p, p], -1)
    np.testing.assert_array_equal(to_numpy(sets), to_numpy(SplitConformalClassifier(0.2).fit(labels, two_class).predict_set(two_class)))


def test_coverage_helpers():
    sets = np.array([[True, False], [True, True], [False, True]])
    np.testing.assert_allclose(to_numpy(set_coverage(np.array([0, 0, 0]), sets)), 2 / 3, rtol=1e-6)
    np.testing.assert_allclose(to_numpy(average_set_size(sets)), 4 / 3, rtol=1e-6)
    np.testing.assert_allclose(
        to_numpy(interval_coverage(np.array([0.0, 2.0, 5.0]), np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 6.0]))),
        2 / 3,
        rtol=1e-6,
    )


def test_invalid_score():
    with pytest.raises(ValueError):
        SplitConformalClassifier(score="raps")
