"""Parity tests for split conformal prediction.

Neither TensorFlow Model Garden nor TensorFlow Probability implements conformal
prediction, so the Keras 3 implementation is compared with direct NumPy
implementations of the procedures in Angelopoulos & Bates, "A Gentle
Introduction to Conformal Prediction" (https://arxiv.org/abs/2107.07511), and
Romano et al., "Classification with Valid and Adaptive Coverage" (APS).
"""

import math

import numpy as np
import pytest

from keras import ops

from keras_proba.postprocessing.conformal import (
    SplitConformalClassifier,
    SplitConformalRegressor,
    conformal_quantile,
)


def numpy_conformal_quantile(scores, alpha):
    """The k-th smallest score, k = ceil((n + 1)(1 - alpha))."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else np.sort(scores, axis=0)[k - 1]


def softmax(logits):
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


@pytest.mark.parametrize("n, alpha", [(1, 0.3), (19, 0.1), (100, 0.05), (1001, 0.01)])
def test_conformal_quantile_parity(n, alpha):
    scores = np.random.default_rng(n).exponential(size=n).astype("float32")
    np.testing.assert_allclose(ops.convert_to_numpy(conformal_quantile(scores, alpha)), numpy_conformal_quantile(scores, alpha))


def test_higher_quantile_selects_the_next_order_statistic():
    """Documents why the threshold is not `np.quantile(scores, k / n, method="higher")`."""
    scores = np.arange(1.0, 20.0)  # n = 19.
    alpha = 0.1
    k = math.ceil(20 * 0.9)  # 18.
    assert ops.convert_to_numpy(conformal_quantile(scores, alpha)) == 18.0
    assert np.quantile(scores, k / 19, method="higher") == 19.0


@pytest.mark.parametrize("normalized", [False, True])
def test_regression_parity(normalized):
    units = 3
    rng = np.random.default_rng(units)
    y_cal, y_pred_cal = rng.normal(size=(200, units)), rng.normal(size=(200, units))
    scale_cal = rng.uniform(0.5, 2.0, size=(200, units)) if normalized else None
    y_pred_test = rng.normal(size=(50, units))
    scale_test = rng.uniform(0.5, 2.0, size=(50, units)) if normalized else None

    scores = np.abs(y_cal - y_pred_cal) / (scale_cal if normalized else 1.0)
    q_hat = numpy_conformal_quantile(scores, 0.1)
    width = q_hat * (scale_test if normalized else 1.0)

    conformal = SplitConformalRegressor(alpha=0.1).fit(y_cal.astype("float32"), y_pred_cal.astype("float32"), scale=None if scale_cal is None else scale_cal.astype("float32"))
    lower, upper = conformal.predict_interval(y_pred_test.astype("float32"), scale=None if scale_test is None else scale_test.astype("float32"))
    np.testing.assert_allclose(ops.convert_to_numpy(lower), y_pred_test - width, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ops.convert_to_numpy(upper), y_pred_test + width, rtol=1e-5, atol=1e-5)


def numpy_lac_sets(probs_cal, y_cal, probs_test, alpha):
    q_hat = numpy_conformal_quantile(1 - probs_cal[np.arange(len(y_cal)), y_cal], alpha)
    return probs_test >= 1 - q_hat


def numpy_aps_sets(probs_cal, y_cal, probs_test, alpha, u_cal=None, u_test=None):
    """APS of Romano et al., written per example with explicit loops.

    Randomized when `u_cal` and `u_test` are given; otherwise the conservative
    non-randomized sets, which include the class crossing the threshold.
    """

    def score(p, label, u):
        order = list(np.argsort(-p, kind="stable"))
        rank = order.index(label)
        return p[order][:rank].sum() + u * p[label]

    randomized = u_cal is not None
    u_cal = u_cal if randomized else np.ones(len(y_cal))
    q_hat = numpy_conformal_quantile(np.array([score(p, y, u) for p, y, u in zip(probs_cal, y_cal, u_cal)]), alpha)
    sets = np.zeros(probs_test.shape, bool)
    for i, p in enumerate(probs_test):
        mass_before = 0.0
        for k in np.argsort(-p, kind="stable"):
            sets[i, k] = mass_before + u_test[i] * p[k] <= q_hat if randomized else mass_before < q_hat
            mass_before += p[k]
    return sets


def float64(x):
    return x.astype("float32").astype("float64")


def test_lac_parity():
    alpha = 0.1
    rng = np.random.default_rng(0)
    probs_cal, y_cal = softmax(rng.normal(size=(300, 6)) * 2), rng.integers(0, 6, size=300)
    probs_test = softmax(rng.normal(size=(80, 6)) * 2)
    conformal = SplitConformalClassifier(alpha=alpha, score="lac").fit(y_cal, probs_cal.astype("float32"))
    expected = numpy_lac_sets(float64(probs_cal), y_cal, float64(probs_test), alpha)
    np.testing.assert_array_equal(ops.convert_to_numpy(conformal.predict_set(probs_test.astype("float32"))), expected)


@pytest.mark.parametrize("randomized", [False, True])
def test_aps_parity(randomized):
    alpha = 0.1
    rng = np.random.default_rng(0)
    probs_cal, y_cal = softmax(rng.normal(size=(300, 6)) * 2), rng.integers(0, 6, size=300)
    probs_test = softmax(rng.normal(size=(80, 6)) * 2)
    u_cal, u_test = rng.uniform(size=300), rng.uniform(size=80)

    conformal = SplitConformalClassifier(alpha=alpha, score="aps", randomized=randomized)
    if randomized:
        # Feed the same uniform draws to both implementations.
        draws = [u_cal, u_test]
        conformal._uniform = lambda batch_size, dtype: ops.cast(draws.pop(0)[:, None], dtype)
    conformal.fit(y_cal, probs_cal.astype("float32"))
    actual = ops.convert_to_numpy(conformal.predict_set(probs_test.astype("float32")))
    expected = numpy_aps_sets(
        float64(probs_cal), y_cal, float64(probs_test), alpha, *((float64(u_cal), float64(u_test)) if randomized else ())
    )
    np.testing.assert_array_equal(actual, expected)
