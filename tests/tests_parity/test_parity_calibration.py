"""Numerical parity tests comparing the Keras 3 calibration metrics against TensorFlow Probability's implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "reference/probability"
if not REFERENCE_PATH.exists():
    pytest.skip(f"Reference implementation not found at {REFERENCE_PATH}", allow_module_level=True)
sys.path.insert(0, str(REFERENCE_PATH))
tfp_calibration = pytest.importorskip("tensorflow_probability.python.stats.calibration")

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


def random_logits(rng, shape, temperature=1.0):
    return (rng.normal(size=shape) / temperature).astype("float32")


@pytest.mark.parametrize("shape", [(64, 3), (2, 5, 32, 4)])
def test_brier_score_parity(shape):
    rng = np.random.default_rng(0)
    logits = random_logits(rng, shape)
    labels = rng.integers(0, shape[-1], size=shape[:-1]).astype("int32")
    if len(shape) == 2:
        expected = tfp_calibration.brier_score(labels, logits).numpy()
    else:
        # The reference applies softmax on axis 1, so compare on flattened examples.
        flat = tfp_calibration.brier_score(labels.reshape(-1), logits.reshape(-1, shape[-1])).numpy()
        expected = flat.reshape(shape[:-1])
    np.testing.assert_allclose(to_numpy(brier_score(labels, logits)), expected, rtol=1e-5, atol=1e-6)


def numpy_resolution(labels, logits):
    """Textbook Brier resolution, `sum_k w_k ||dist_mean[k] - pbar||^2`, over argmax bins."""
    num_classes = logits.shape[-1]
    pred = logits.argmax(-1)
    confusion = np.zeros((num_classes, num_classes))
    np.add.at(confusion, (pred, labels), 1.0)
    weights = confusion.sum(-1) / confusion.sum()
    pbar = confusion.sum(0) / confusion.sum()
    dist_mean = confusion / np.maximum(confusion.sum(-1, keepdims=True), 1.0)
    return np.sum(weights * np.sum((dist_mean - pbar) ** 2, axis=-1))


def assert_decomposition_close(labels, logits, actual):
    """Uncertainty and reliability match TFP; resolution matches the textbook formula.

    TFP's resolution indexes the average label distribution by the predicted
    class (see `brier_decomposition`), so it is checked against numpy instead.
    """
    uncertainty, _, reliability = tfp_calibration.brier_decomposition(labels, logits)
    np.testing.assert_allclose(to_numpy(actual[0]), uncertainty.numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(to_numpy(actual[2]), reliability.numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(to_numpy(actual[1]), numpy_resolution(labels, logits), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("temperature, num_classes", [(0.01, 2), (1.0, 4)])
def test_brier_decomposition_parity(temperature, num_classes):
    rng = np.random.default_rng(0)
    logits = random_logits(rng, (512, num_classes), temperature)
    labels = rng.integers(0, num_classes, size=512).astype("int32")
    assert_decomposition_close(labels, logits, brier_decomposition(labels, logits))


def test_reference_resolution_differs_with_imbalanced_labels():
    """Documents the TFP deviation: with imbalanced labels its resolution is not the textbook one."""
    rng = np.random.default_rng(0)
    logits = random_logits(rng, (2000, 3))
    labels = rng.choice(3, p=[0.7, 0.2, 0.1], size=2000).astype("int32")
    _, reference_resolution, _ = tfp_calibration.brier_decomposition(labels, logits)
    assert not np.isclose(reference_resolution.numpy(), numpy_resolution(labels, logits), rtol=1e-2)
    np.testing.assert_allclose(
        to_numpy(brier_decomposition(labels, logits)[1]), numpy_resolution(labels, logits), rtol=1e-4
    )


@pytest.mark.parametrize("num_bins, num_classes, use_predicted", [(1, 2, False), (15, 5, False), (10, 5, True)])
def test_expected_calibration_error_parity(num_bins, num_classes, use_predicted):
    rng = np.random.default_rng(num_bins)
    logits = random_logits(rng, (1000, num_classes), temperature=0.5)
    labels = rng.integers(0, num_classes, size=1000).astype("int32")
    predicted = rng.integers(0, num_classes, size=1000).astype("int32") if use_predicted else None
    expected = tfp_calibration.expected_calibration_error(
        num_bins, logits=logits, labels_true=labels, labels_predicted=predicted
    ).numpy()
    actual = to_numpy(expected_calibration_error(num_bins, logits=logits, labels_true=labels, labels_predicted=predicted))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_expected_calibration_error_bin_edges_parity():
    """Probabilities exactly on bin edges, including 0 and 1, fall in the same bins."""
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.3, 0.7], [0.0, 1.0], [0.9, 0.1], [0.2, 0.8]], "float32")
    logits = np.log(np.clip(probs, 1e-30, 1.0))
    labels = np.array([0, 1, 1, 0, 0, 1], "int32")
    for num_bins in [2, 5, 10]:
        expected = tfp_calibration.expected_calibration_error(num_bins, logits=logits, labels_true=labels).numpy()
        actual = to_numpy(expected_calibration_error(num_bins, logits=logits, labels_true=labels))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def assert_quantile_outputs_close(expected, actual):
    names = ["ece", "accuracy", "confidence", "count", "edges", "bucket"]
    for name, e, a in zip(names, expected, actual):
        np.testing.assert_allclose(to_numpy(a), np.asarray(e), rtol=1e-5, atol=1e-6, err_msg=name)


def assert_prob_space_quantile_outputs_close(expected, actual, pred_log_prob, axis=0):
    """Parity for probability-space buckets, up to points lying exactly on an edge.

    Bucket edges are data points. TFP finds buckets by comparing `log p` with
    `log(exp(log p_edge))`, and TF's vectorized exp/log can be off by one ulp
    depending on an element's position, so a point lying on an edge may fall in
    the bucket below. Ours compares in probability space, where an edge equals
    its data point exactly, and puts it in the bucket starting at that edge.
    Points within float tolerance of an edge can move either way.
    """
    # Edges are log-probabilities near 0, where one ulp in probability is a large relative error.
    np.testing.assert_allclose(to_numpy(actual[4]), np.asarray(expected[4]), rtol=0.0, atol=1e-6, err_msg="edges")
    probs = np.moveaxis(np.exp(pred_log_prob), axis, 0)
    edges = np.exp(np.asarray(expected[4]))
    on_edge = np.any(np.isclose(probs[None], edges[:, None], rtol=1e-6, atol=0.0), axis=0)
    expected_bucket = np.moveaxis(np.asarray(expected[5]), axis, 0)
    actual_bucket = np.moveaxis(to_numpy(actual[5]), axis, 0)
    np.testing.assert_array_equal(actual_bucket[~on_edge], expected_bucket[~on_edge])
    assert np.all(np.isin(actual_bucket[on_edge] - expected_bucket[on_edge], [-1, 0, 1]))
    # Each moved point changes the ECE by at most 2 / n.
    moved = np.sum(actual_bucket != expected_bucket, axis=0)
    np.testing.assert_array_less(
        np.abs(to_numpy(actual[0]) - np.asarray(expected[0])), 2.0 * moved / probs.shape[0] + 1e-6
    )


@pytest.mark.parametrize("num_buckets, log_space_buckets", [(10, False), (20, True)])
def test_ece_quantiles_parity(num_buckets, log_space_buckets):
    rng = np.random.default_rng(num_buckets)
    probs = rng.uniform(0.2, 1.0, size=500).astype("float32")
    hit = rng.uniform(size=500) < probs
    log_pred = np.log(probs)
    expected = tfp_calibration.expected_calibration_error_quantiles(
        hit, log_pred, num_buckets=num_buckets, log_space_buckets=log_space_buckets
    )
    actual = expected_calibration_error_quantiles(
        hit, log_pred, num_buckets=num_buckets, log_space_buckets=log_space_buckets
    )
    if log_space_buckets:
        # No round trip: exact parity.
        assert_quantile_outputs_close(expected, actual)
    else:
        assert_prob_space_quantile_outputs_close(expected, actual, log_pred)


def test_ece_quantiles_ties_parity():
    """Repeated probabilities produce duplicated edges and empty buckets in both."""
    log_pred = np.log(np.array([0.5, 0.5, 0.5, 0.9, 0.9, 0.2, 0.5, 0.9], "float32"))
    hit = np.array([1, 0, 1, 1, 1, 0, 0, 1], bool)
    expected = tfp_calibration.expected_calibration_error_quantiles(hit, log_pred, num_buckets=5)
    actual = expected_calibration_error_quantiles(hit, log_pred, num_buckets=5)
    assert_quantile_outputs_close(expected, actual)


def test_ece_quantiles_axis_parity():
    axis = 1
    rng = np.random.default_rng(0)
    # Every axis has enough samples for 4 buckets.
    probs = rng.uniform(0.3, 1.0, size=(40, 36, 30)).astype("float32")
    hit = rng.uniform(size=probs.shape) < probs
    log_pred = np.log(probs)
    expected = tfp_calibration.expected_calibration_error_quantiles(hit, log_pred, num_buckets=4, axis=axis)
    actual = expected_calibration_error_quantiles(hit, log_pred, num_buckets=4, axis=axis)
    assert_prob_space_quantile_outputs_close(expected, actual, log_pred, axis=axis)
    # Exact parity with log-space buckets.
    assert_quantile_outputs_close(
        tfp_calibration.expected_calibration_error_quantiles(
            hit, log_pred, num_buckets=4, axis=axis, log_space_buckets=True
        ),
        expected_calibration_error_quantiles(hit, log_pred, num_buckets=4, axis=axis, log_space_buckets=True),
    )


def test_streaming_ece_matches_reference():
    """Accumulating over batches gives the reference ECE on all the data."""
    rng = np.random.default_rng(0)
    logits = random_logits(rng, (999, 4), temperature=0.7)
    labels = rng.integers(0, 4, size=999).astype("int32")
    metric = ExpectedCalibrationError(num_bins=12, from_logits=True)
    for start in range(0, 999, 128):
        metric.update_state(labels[start : start + 128], logits[start : start + 128])
    expected = tfp_calibration.expected_calibration_error(12, logits=logits, labels_true=labels).numpy()
    np.testing.assert_allclose(to_numpy(metric.result()), expected, rtol=1e-5, atol=1e-6)


def test_streaming_brier_matches_reference():
    rng = np.random.default_rng(0)
    logits = random_logits(rng, (300, 3))
    labels = rng.integers(0, 3, size=300).astype("int32")
    metric = BrierScore(from_logits=True)
    for start in range(0, 300, 64):
        metric.update_state(labels[start : start + 64], logits[start : start + 64])
    expected = tfp_calibration.brier_score(labels, logits).numpy().mean()
    np.testing.assert_allclose(to_numpy(metric.result()), expected, rtol=1e-5)
