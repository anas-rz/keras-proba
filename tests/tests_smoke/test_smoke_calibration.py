"""Smoke and correctness tests for the calibration metrics. Run once per backend via `KERAS_BACKEND`.

Statistical and docstring tests are ported from TensorFlow Probability's `calibration_test.py`.
"""

import numpy as np
import pytest

from keras import ops

from keras_proba.metrics.calibration import (
    BrierScore,
    ExpectedCalibrationError,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    expected_calibration_error_quantiles,
)


def to_numpy(x):
    return ops.convert_to_numpy(x)


def softmax(logits):
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sample_labels(rng, logits):
    """Samples one label per row from `Categorical(logits)`."""
    cdf = np.cumsum(softmax(logits), axis=-1)
    return (rng.uniform(size=(len(logits), 1)) > cdf).sum(-1).clip(0, logits.shape[-1] - 1).astype("int32")


@pytest.mark.parametrize("temperature", [0.01, 1.0, 5.0])
@pytest.mark.parametrize("num_classes", [2, 4])
def test_brier_decomposition(temperature, num_classes):
    """The decomposition recovers the Brier score up to estimation error."""
    rng = np.random.default_rng(1)
    logits = (rng.normal(size=(8192, num_classes)) / temperature).astype("float32")
    labels = rng.integers(0, num_classes, size=8192).astype("int32")

    uncertainty, resolution, reliability = (to_numpy(t) for t in brier_decomposition(labels, logits))
    brier = uncertainty - resolution + reliability
    brier_direct = to_numpy(brier_score(labels, logits)).mean()

    assert resolution >= 0.0
    assert reliability >= 0.0
    assert abs(brier - brier_direct) < 0.1


def test_brier_decomposition_is_exact_for_binned_forecasts():
    """With one forecast per argmax class, Uncertainty - Resolution + Reliability is the Brier score."""
    rng = np.random.default_rng(0)
    for num_classes in [2, 3, 5]:
        forecasts = rng.dirichlet(np.ones(num_classes), size=num_classes)
        forecasts[np.arange(num_classes), np.arange(num_classes)] += 2.0
        forecasts /= forecasts.sum(-1, keepdims=True)
        pred = rng.integers(0, num_classes, size=4000)
        logits = np.log(forecasts[pred]).astype("float32")
        labels = rng.choice(num_classes, p=rng.dirichlet(np.ones(num_classes) * 0.5), size=4000).astype("int32")

        uncertainty, resolution, reliability = (to_numpy(t) for t in brier_decomposition(labels, logits))
        brier_direct = to_numpy(brier_score(labels, logits)).mean()
        np.testing.assert_allclose(uncertainty - resolution + reliability, brier_direct, atol=1e-5)


def test_brier_decomposition_batching():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3, 5, 1024, 4)).astype("float32")
    labels = rng.integers(0, 4, size=(3, 5, 1024)).astype("int32")
    batched = np.stack([to_numpy(t) for t in brier_decomposition(labels, logits)])
    for i in range(3):
        for j in range(5):
            single = np.stack([to_numpy(t) for t in brier_decomposition(labels[i, j], logits[i, j])])
            np.testing.assert_allclose(batched[:, i, j], single, rtol=1e-5, atol=1e-6)


def test_reliability_grows_with_perturbation():
    """Perturbing the true model makes its predictions less reliable."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(2000, 5)).astype("float32")
    weights = rng.normal(size=(5, 4)).astype("float32")
    labels = sample_labels(rng, data @ weights)

    reliabilities = []
    for perturbation in np.linspace(0.0, 3.0, 6):
        values = []
        for _ in range(5):
            perturbed = weights + perturbation * rng.normal(size=weights.shape).astype("float32")
            values.append(to_numpy(brier_decomposition(labels, data @ perturbed)[2]))
        reliabilities.append(np.mean(values))
    for previous, current in zip(reliabilities[:-1], reliabilities[1:]):
        assert previous <= current + 0.05


def test_brier_score_values():
    # A certain correct prediction scores -1, a uniform one over 2 classes -0.5.
    logits = np.log(np.array([[1.0, 1e-30], [0.5, 0.5]], "float32"))
    np.testing.assert_allclose(to_numpy(brier_score(np.array([0, 1]), logits)), [-1.0, -0.5], atol=1e-6)


def test_expected_calibration_error():
    probs = np.array([[0.3, 0.7], [0.9, 0.1], [0.4, 0.6]], "float32")
    ece = to_numpy(expected_calibration_error(3, logits=np.log(probs), labels_true=np.array([1, 0, 0])))
    np.testing.assert_allclose(ece, 1.0 / 3.0, rtol=1e-6)


@pytest.mark.parametrize("num_classes", [2, 5, 50])
def test_expected_calibration_error_wellcalibrated(num_classes):
    rng = np.random.default_rng(1)
    logits = (2.0 * rng.normal(size=(50000, num_classes))).astype("float32")
    labels = sample_labels(rng, logits)
    assert to_numpy(expected_calibration_error(10, logits=logits, labels_true=labels)) < 0.01


@pytest.mark.parametrize("num_classes", [2, 5, 50])
def test_expected_calibration_error_uncalibrated(num_classes):
    rng = np.random.default_rng(1)
    logits = (2.0 * rng.normal(size=(50000, num_classes))).astype("float32")
    labels = sample_labels(rng, logits)
    logits_other = (2.0 * rng.normal(size=(50000, num_classes))).astype("float32")
    assert to_numpy(expected_calibration_error(10, logits=logits_other, labels_true=labels)) > 0.2


def test_ece_quantiles_docstring_simple():
    label = np.array([0, 0, 1, 0, 1, 1], bool)
    log_pred = np.log(np.array([0.1, 0.05, 0.5, 0.2, 0.99, 0.99], "float32"))
    ece, acc, conf, cnt, edges, buckets = (
        to_numpy(t) for t in expected_calibration_error_quantiles(label, log_pred, num_buckets=3)
    )
    np.testing.assert_allclose(ece, 0.145, atol=1e-6)
    np.testing.assert_allclose(acc, [0.0, 0.0, 1.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(conf, [0.075, 0.2, 0.826667], rtol=1e-6)
    np.testing.assert_allclose(cnt, [2.0, 1.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(edges, np.log([0.05, 0.2, 0.5, 0.99]), atol=1e-6)
    np.testing.assert_array_equal(buckets, [0, 0, 2, 1, 2, 2])


def test_ece_quantiles_categorical_shapes():
    y = np.array([[0, 1], [2, 1], [1, 1], [2, 2]])
    logits = np.array(
        [
            [[0.0, -1, -2], [-1.0, 0, -1]],
            [[2.0, -1, 0], [-1.0, 0, -1]],
            [[-1.0, 0, -1], [0.0, -2, -1]],
            [[-1.0, 0, -2], [0, -1, -2]],
        ],
        "float32",
    )
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    yhat = log_probs.argmax(-1)
    pred_log_prob = np.take_along_axis(log_probs, yhat[..., None], -1)[..., 0]
    ece, acc, conf, cnt, edges, buckets = expected_calibration_error_quantiles(
        y == yhat, pred_log_prob, num_buckets=10
    )
    assert ece.shape == (2,)
    assert acc.shape == conf.shape == cnt.shape == (10, 2)
    assert edges.shape == (11, 2)
    assert buckets.shape == (4, 2)


def test_ece_quantiles_multidim_and_axis():
    label = np.array([[0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1]], bool)
    log_pred = np.log(np.array([[0.1, 0.05, 0.5, 0.2, 0.99, 0.99]] * 2, "float32"))
    ece, acc, conf, cnt, edges, buckets = (
        to_numpy(t) for t in expected_calibration_error_quantiles(label, log_pred, num_buckets=3, axis=1)
    )
    np.testing.assert_allclose(ece, [0.145, 0.145], atol=1e-6)
    np.testing.assert_allclose(acc, np.transpose([[0.0, 0.0, 1.0]] * 2), atol=1e-6)
    np.testing.assert_allclose(conf, np.transpose([[0.075, 0.2, 0.826667]] * 2), rtol=1e-6)
    np.testing.assert_allclose(cnt, np.transpose([[2.0, 1.0, 3.0]] * 2), rtol=1e-6)
    np.testing.assert_allclose(edges, np.log(np.transpose([[0.05, 0.2, 0.5, 0.99]] * 2)), atol=1e-6)
    np.testing.assert_array_equal(buckets, [[0, 0, 2, 1, 2, 2]] * 2)


def test_ece_quantiles_edge_point_goes_to_upper_bucket():
    """A value equal to an interior edge belongs to the bucket starting at that edge."""
    log_pred = np.log(np.array([0.2, 0.4, 0.6, 0.8, 1.0], "float32"))
    *_, edges, buckets = expected_calibration_error_quantiles(np.ones(5, bool), log_pred, num_buckets=2)
    np.testing.assert_allclose(np.exp(to_numpy(edges)), [0.2, 0.6, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(to_numpy(buckets), [0, 0, 1, 1, 1])


@pytest.mark.parametrize("from_logits", [False, True])
def test_ece_metric_matches_function(from_logits):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 4)).astype("float32")
    labels = rng.integers(0, 4, size=500)
    metric = ExpectedCalibrationError(num_bins=10, from_logits=from_logits)
    y_pred = logits if from_logits else softmax(logits)
    for start in range(0, 500, 64):
        metric.update_state(labels[start : start + 64], y_pred[start : start + 64])
    expected = to_numpy(expected_calibration_error(10, logits=logits, labels_true=labels))
    np.testing.assert_allclose(to_numpy(metric.result()), expected, rtol=1e-5, atol=1e-6)


def test_metric_label_formats():
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(64, 3)).astype("float32"))
    labels = rng.integers(0, 3, size=64)
    for metric_cls in [ExpectedCalibrationError, BrierScore]:
        results = []
        for y_true in [labels, labels[:, None], np.eye(3)[labels]]:
            metric = metric_cls()
            metric.update_state(y_true, probs)
            results.append(to_numpy(metric.result()))
        np.testing.assert_allclose(results, results[0], rtol=1e-6)


@pytest.mark.parametrize("from_logits", [False, True])
def test_binary_sigmoid_outputs(from_logits):
    """A single output unit is treated as `[1 - p, p]`."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(200, 1)).astype("float32")
    labels = (rng.uniform(size=(200, 1)) < p).astype("int32")
    y_pred = np.log(p / (1 - p)) if from_logits else p
    two_class_logits = np.log(np.concatenate([1 - p, p], -1))
    for metric, expected in [
        (ExpectedCalibrationError(num_bins=5, from_logits=from_logits), expected_calibration_error(5, two_class_logits, labels[:, 0])),
        (BrierScore(from_logits=from_logits), ops.mean(brier_score(labels[:, 0], two_class_logits))),
    ]:
        metric.update_state(labels, y_pred)
        np.testing.assert_allclose(to_numpy(metric.result()), to_numpy(expected), rtol=1e-4, atol=1e-6)


def test_sample_weight_matches_repetition():
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(20, 3)).astype("float32"))
    labels = rng.integers(0, 3, size=20)
    weights = rng.integers(0, 4, size=20)
    for metric_cls in [ExpectedCalibrationError, BrierScore]:
        weighted, repeated = metric_cls(), metric_cls()
        weighted.update_state(labels, probs, sample_weight=weights.astype("float32"))
        repeated.update_state(np.repeat(labels, weights), np.repeat(probs, weights, axis=0))
        np.testing.assert_allclose(to_numpy(weighted.result()), to_numpy(repeated.result()), rtol=1e-5, atol=1e-6)


def test_reset_state():
    metric = ExpectedCalibrationError()
    assert to_numpy(metric.result()) == 0.0
    metric.update_state(np.array([0, 1]), np.array([[0.9, 0.1], [0.8, 0.2]], "float32"))
    assert to_numpy(metric.result()) > 0.0
    metric.reset_state()
    assert to_numpy(metric.result()) == 0.0


def test_invalid_num_bins():
    with pytest.raises(ValueError):
        ExpectedCalibrationError(num_bins=0)
