"""Multi-backend calibration metrics in Keras 3.

Ported from TensorFlow Probability's `tensorflow_probability/python/stats/calibration.py`.

Functions (computed over a full set of predictions):
    brier_score: Per-example Brier score.
    brier_decomposition: Uncertainty, resolution and reliability of the Brier score.
    expected_calibration_error: ECE over equal-width probability bins [1].
    expected_calibration_error_quantiles: ECE over equal-count (quantile) buckets.

Streaming metrics (accumulated over batches, e.g. in `model.compile`):
    ExpectedCalibrationError: Equal-width bin ECE.
    BrierScore: Mean Brier score.

[1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. On Calibration of
     Modern Neural Networks. In _International Conference on Machine
     Learning_, 2017. https://arxiv.org/abs/1706.04599
"""

import numpy as np

import keras
from keras import ops


def _finfo(dtype):
    return np.finfo(keras.backend.standardize_dtype(dtype))


def brier_score(labels, logits):
    r"""Computes the Brier score for a probabilistic prediction.

    For a probability vector `p` and a realized outcome `k`, the Brier score is
    `sum_i p[i]^2 - 2 p[k]`. Smaller values are better. It is a strictly proper
    scoring rule, and can be negative. This form differs from
    `sum_i (p[i] - onehot(k)[i])^2` by a constant of 1.

    Args:
        labels: Integer tensor of shape `(N1, ..., Nk)` with class labels.
        logits: Tensor of shape `(N1, ..., Nk, num_classes)`.

    Returns:
        Tensor of shape `(N1, ..., Nk)` with elementwise Brier scores; reduce
        with a mean over examples.
    """
    logits = ops.convert_to_tensor(logits)
    return _brier_score_from_probs(labels, ops.softmax(logits, axis=-1))


def _brier_score_from_probs(labels, probabilities):
    num_classes = probabilities.shape[-1]
    plabel = ops.sum(probabilities * ops.one_hot(labels, num_classes, dtype=probabilities.dtype), axis=-1)
    return ops.sum(ops.square(probabilities), axis=-1) - 2.0 * plabel


def brier_decomposition(labels, logits):
    r"""Decomposes the Brier score into uncertainty, resolution, and reliability.

    Any proper scoring rule admits a unique decomposition as
    `Score = Uncertainty - Resolution + Reliability`, where:

    * `Uncertainty` is a generalized entropy of the average predictive
      distribution; it can be positive or negative.
    * `Resolution` is a generalized variance of individual predictive
      distributions; it is always non-negative, and larger is better.
    * `Reliability` measures calibration of predictions against the true
      frequency of events; it is always non-negative, and lower is better.

    The space of predictive distributions is discretized by their argmax. The
    estimation error of each component is O(k/n) for n instances and k labels.

    Note: TFP's implementation computes the resolution as
    `sum_k w_k sum_j (pbar[k] - dist_mean[k, j])^2`, indexing the average label
    distribution by the predicted class. Here it is
    `sum_k w_k sum_j (pbar[j] - dist_mean[k, j])^2`, for which the decomposition
    reproduces the Brier score exactly when all predictions with the same
    argmax are identical. Uncertainty and reliability match TFP.

    Args:
        labels: Integer tensor of shape `(..., n)` with class labels.
        logits: Tensor of shape `(..., n, num_classes)`.

    Returns:
        Tuple of `uncertainty`, `resolution` and `reliability`, each of shape
        `(...)`.
    """
    logits = ops.convert_to_tensor(logits)
    dtype = logits.dtype
    num_classes = logits.shape[-1]

    pred_class = ops.argmax(logits, axis=-1)
    pred_one_hot = ops.one_hot(pred_class, num_classes, dtype=dtype)
    label_one_hot = ops.one_hot(labels, num_classes, dtype=dtype)
    # confusion_matrix[..., k, j] counts predictions of class k with label j.
    confusion_matrix = ops.einsum("...nk,...nj->...kj", pred_one_hot, label_one_hot)

    dist_weights = ops.sum(confusion_matrix, axis=-1)
    dist_weights = dist_weights / ops.sum(dist_weights, axis=-1, keepdims=True)
    pbar = ops.sum(confusion_matrix, axis=-2)
    pbar = pbar / ops.sum(pbar, axis=-1, keepdims=True)

    eps = _finfo(dtype).eps
    # dist_mean[k, :] is the empirical label distribution when predicting k.
    # Some outcomes may not realize, corresponding to dist_weights[k] = 0.
    dist_mean = confusion_matrix / (eps + ops.sum(confusion_matrix, axis=-1, keepdims=True))

    # Uncertainty: quadratic entropy of the average label distribution.
    uncertainty = -ops.sum(ops.square(pbar), axis=-1)

    # Resolution: expected quadratic divergence of predictive to mean.
    resolution = ops.square(ops.expand_dims(pbar, -2) - dist_mean)
    resolution = ops.sum(dist_weights * ops.sum(resolution, axis=-1), axis=-1)

    # Reliability: expected quadratic divergence of predictive to true.
    prob_true = ops.einsum("...nk,...kj->...nj", pred_one_hot, dist_mean)
    prob_pred = ops.softmax(logits, axis=-1)
    reliability = ops.mean(ops.sum(ops.square(prob_pred - prob_true), axis=-1), axis=-1)

    return uncertainty, resolution, reliability


def _calibration_bin_statistics(num_bins, probabilities, labels_true, labels_predicted=None, sample_weight=None):
    """Per-bin weighted counts, correct counts and confidence sums of the decided labels."""
    if labels_predicted is None:
        # The argmax decision is optimal under 0/1 loss.
        labels_predicted = ops.argmax(probabilities, axis=-1)
    labels_true = ops.cast(labels_true, "int32")
    labels_predicted = ops.cast(labels_predicted, "int32")
    dtype = probabilities.dtype

    correct = ops.cast(ops.equal(labels_predicted, labels_true), dtype)
    # Predicted probability of the decision, p(pred_y | x).
    prob_y = ops.take_along_axis(probabilities, ops.expand_dims(labels_predicted, -1), axis=-1)[..., 0]

    # Equal-width bins over [0, 1], as `tf.histogram_fixed_width_bins`.
    bins = ops.clip(ops.cast(ops.floor(prob_y * num_bins), "int32"), 0, num_bins - 1)
    bin_one_hot = ops.one_hot(bins, num_bins, dtype=dtype)
    if sample_weight is not None:
        bin_one_hot = bin_one_hot * ops.expand_dims(ops.cast(sample_weight, dtype), -1)

    counts = ops.sum(bin_one_hot, axis=0)
    correct_counts = ops.sum(bin_one_hot * ops.expand_dims(correct, -1), axis=0)
    confidence_sums = ops.sum(bin_one_hot * ops.expand_dims(prob_y, -1), axis=0)
    return counts, correct_counts, confidence_sums


def _ece_from_bin_statistics(counts, correct_counts, confidence_sums):
    tiny = _finfo(counts.dtype).tiny
    pbins = counts / ops.sum(counts)
    pcorrect = correct_counts / (counts + tiny)
    pmean_observed = confidence_sums / (counts + tiny)
    return ops.sum(pbins * ops.abs(pcorrect - pmean_observed))


def expected_calibration_error(num_bins, logits=None, labels_true=None, labels_predicted=None):
    """Computes the Expected Calibration Error (ECE).

    Implements equation (3) of [1], using the probability of the decided label
    being correct. ECE estimates are unreliable when some of the `num_bins`
    equal-width bins contain few samples; see also
    `expected_calibration_error_quantiles`.

    Args:
        num_bins: Number of equal-width probability bins, e.g. 10.
        logits: Tensor of shape `(n, num_classes)`.
        labels_true: Integer tensor of shape `(n,)` with class labels.
        labels_predicted: Integer tensor of shape `(n,)` with the decisions of
            the predictive system. If `None`, the argmax of `logits` is used.

    Returns:
        Scalar ECE.
    """
    probabilities = ops.softmax(ops.cast(logits, "float32"), axis=-1)
    stats = _calibration_bin_statistics(num_bins, probabilities, labels_true, labels_predicted)
    return _ece_from_bin_statistics(*stats)


def _quantile_edges(x, num_quantiles):
    """`num_quantiles + 1` edges of `x` along axis 0, with "nearest" interpolation.

    Equivalent to `tfp.stats.quantiles(x, num_quantiles, axis=0)`.
    """
    n = x.shape[0]
    if n is None:
        raise ValueError("The size of the bucketed axis must be known.")
    frac = np.linspace(0.0, 1.0, num_quantiles + 1)
    indices = np.clip(np.round((n - 1) * frac).astype("int32"), 0, n - 1)
    return ops.take(ops.sort(x, axis=0), indices, axis=0)


def _find_buckets(x, edges):
    """Bucket of each `x` given sorted `edges` along axis 0, as TFP's `_find_bins`.

    The bucket is the index of the last edge not larger than `x`, with the
    maximum assigned to the last bucket.
    """
    num_buckets = edges.shape[0] - 1
    edges_not_larger = ops.sum(ops.cast(ops.expand_dims(x, 0) >= ops.expand_dims(edges, 1), "int32"), axis=0)
    return ops.minimum(edges_not_larger, num_buckets) - 1


def expected_calibration_error_quantiles(hit, pred_log_prob, num_buckets=20, axis=0, log_space_buckets=False):
    """Expected calibration error over equal-count buckets of predicted probability.

    A model is calibrated if buckets of predicted probabilities have the same
    within-bucket average accuracy. The ECE is the average absolute difference
    between predicted probability and bucket accuracy:

    ```python
    bucket_weight = bucket_count / sum(bucket_count)
    bucket_error = abs(bucket_accuracy - bucket_confidence)
    ece = sum(bucket_weight * bucket_error)
    ```

    where buckets are `num_buckets`-quantiles of `exp(pred_log_prob)`.

    Example:

    ```python
    hit = [False, False, True, False, True, True]
    log_pred = np.log([0.1, 0.05, 0.5, 0.2, 0.99, 0.99])
    ece, acc, conf, cnt, edges, bucket = expected_calibration_error_quantiles(hit, log_pred, num_buckets=3)
    # ece  ==> 0.145
    # acc  ==> [0., 0., 1.]
    # conf ==> [0.075, 0.2, 0.826665]
    # cnt  ==> [2., 1., 3.]
    ```

    Args:
        hit: Boolean tensor, `True` where the model prediction was correct.
            Must broadcast with `pred_log_prob`.
        pred_log_prob: The model's predicted log probability for each `hit`.
        num_buckets: Number of quantile buckets.
        axis: Axis over which to compute buckets and aggregate statistics.
        log_space_buckets: Whether to compute bucket edges from
            `pred_log_prob` instead of `exp(pred_log_prob)`.

    Returns:
        Tuple of:
        - `ece`: Expected calibration error, with the shape of
          `pred_log_prob` reduced over `axis`.
        - `bucket_accuracy`: Within-bucket average hits, shape
          `(num_buckets, ...)`.
        - `bucket_confidence`: Within-bucket average probability, shape
          `(num_buckets, ...)`.
        - `bucket_count`: Number of observations per bucket, shape
          `(num_buckets, ...)`.
        - `bucket_pred_log_prob`: Bucket edges in log space, shape
          `(num_buckets + 1, ...)`.
        - `bucket`: Bucket of each observation, with the shape of
          `pred_log_prob`.
    """
    pred_log_prob = ops.convert_to_tensor(pred_log_prob)
    if not keras.backend.is_float_dtype(pred_log_prob.dtype):
        pred_log_prob = ops.cast(pred_log_prob, "float32")
    dtype = pred_log_prob.dtype
    hit = ops.broadcast_to(ops.cast(hit, dtype), ops.shape(pred_log_prob))

    # Move the bucketed axis first.
    pred_log_prob = ops.moveaxis(pred_log_prob, axis, 0)
    hit = ops.moveaxis(hit, axis, 0)

    # Compute quantiles in probability space, unless asked otherwise. Buckets
    # are found in the same space as the edges: each edge is one of the values,
    # so comparing there is exact, whereas a log/exp round trip can move values
    # lying on an edge into the next bucket depending on the backend.
    if log_space_buckets:
        bucket_pred_log_prob = _quantile_edges(pred_log_prob, num_buckets)
        bucket = _find_buckets(pred_log_prob, bucket_pred_log_prob)
    else:
        pred_prob = ops.exp(pred_log_prob)
        bucket_pred_prob = _quantile_edges(pred_prob, num_buckets)
        bucket = _find_buckets(pred_prob, bucket_pred_prob)
        bucket_pred_log_prob = ops.log(bucket_pred_prob)

    # keep[i, n, ...] is whether observation n falls in bucket i.
    keep = ops.expand_dims(bucket, 0) == ops.reshape(
        ops.arange(num_buckets, dtype=bucket.dtype), (num_buckets,) + (1,) * len(bucket.shape)
    )
    bucket_total_hit = ops.sum(ops.where(keep, hit, ops.zeros_like(hit)), axis=1)
    bucket_count = ops.sum(ops.cast(keep, dtype), axis=1)
    bucket_log_total_pred_prob = ops.logsumexp(
        ops.where(keep, pred_log_prob, ops.full_like(pred_log_prob, -np.inf)), axis=1
    )

    n = ops.maximum(bucket_count, 1.0)
    bucket_accuracy = bucket_total_hit / n
    bucket_confidence = ops.exp(bucket_log_total_pred_prob - ops.log(n))
    bucket_error = ops.abs(bucket_accuracy - bucket_confidence)
    ece = ops.sum(bucket_count * bucket_error, axis=0) / ops.cast(pred_log_prob.shape[0], dtype)
    return (
        ece,
        bucket_accuracy,
        bucket_confidence,
        bucket_count,
        bucket_pred_log_prob,
        ops.moveaxis(bucket, 0, axis),
    )


def _to_probabilities(y_pred, from_logits):
    """Converts predictions to class probabilities, treating one unit as a sigmoid output."""
    y_pred = ops.convert_to_tensor(y_pred)
    if not keras.backend.is_float_dtype(y_pred.dtype):
        y_pred = ops.cast(y_pred, "float32")
    if y_pred.shape[-1] == 1:
        p = ops.sigmoid(y_pred) if from_logits else y_pred
        return ops.concatenate([1.0 - p, p], axis=-1)
    return ops.softmax(y_pred, axis=-1) if from_logits else y_pred


def _to_sparse_labels(y_true, y_pred):
    """Converts one-hot or `(batch, 1)` labels to integer labels of shape `(batch,)`."""
    y_true = ops.convert_to_tensor(y_true)
    if len(y_true.shape) == len(y_pred.shape):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, -1)
        else:
            y_true = ops.argmax(y_true, axis=-1)
    return ops.cast(y_true, "int32")


@keras.saving.register_keras_serializable(package="keras_proba")
class ExpectedCalibrationError(keras.metrics.Metric):
    """Streaming Expected Calibration Error over equal-width bins [1].

    Accumulates per-bin counts, correct counts and confidence sums over
    batches, so the result equals `expected_calibration_error` on all the data
    seen since the last reset.

    Example:

    ```python
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=[ExpectedCalibrationError(num_bins=15)])
    ```

    Args:
        num_bins: Number of equal-width probability bins.
        from_logits: Whether `y_pred` contains logits instead of probabilities.
            A single output unit is treated as a binary sigmoid output.
        name: Name of the metric.
        dtype: Dtype of the metric state.

    Call arguments:
        y_true: Integer labels of shape `(batch,)` or `(batch, 1)`, or one-hot
            labels of shape `(batch, num_classes)`.
        y_pred: Probabilities or logits of shape `(batch, num_classes)`.
        sample_weight: Optional weights of shape `(batch,)`.
    """

    def __init__(self, num_bins=15, from_logits=False, name="ece", dtype=None):
        super().__init__(name=name, dtype=dtype)
        if num_bins <= 0:
            raise ValueError(f"`num_bins` should be a positive integer. Received: {num_bins}")
        self.num_bins = num_bins
        self.from_logits = from_logits
        self.counts = self.add_variable(shape=(num_bins,), initializer="zeros", name="counts")
        self.correct_counts = self.add_variable(shape=(num_bins,), initializer="zeros", name="correct_counts")
        self.confidence_sums = self.add_variable(shape=(num_bins,), initializer="zeros", name="confidence_sums")

    def update_state(self, y_true, y_pred, sample_weight=None):
        probabilities = ops.cast(_to_probabilities(y_pred, self.from_logits), self.dtype)
        labels = _to_sparse_labels(y_true, probabilities)
        if sample_weight is not None:
            sample_weight = ops.reshape(sample_weight, (-1,))
        counts, correct_counts, confidence_sums = _calibration_bin_statistics(
            self.num_bins, probabilities, labels, sample_weight=sample_weight
        )
        self.counts.assign_add(counts)
        self.correct_counts.assign_add(correct_counts)
        self.confidence_sums.assign_add(confidence_sums)

    def result(self):
        total = ops.sum(self.counts)
        ece = _ece_from_bin_statistics(self.counts, self.correct_counts, self.confidence_sums)
        return ops.where(total > 0, ece, ops.zeros_like(ece))

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype, "num_bins": self.num_bins, "from_logits": self.from_logits}


@keras.saving.register_keras_serializable(package="keras_proba")
class BrierScore(keras.metrics.Mean):
    """Streaming mean Brier score, `sum_i p[i]^2 - 2 p[label]` (see `brier_score`).

    Args:
        from_logits: Whether `y_pred` contains logits instead of probabilities.
            A single output unit is treated as a binary sigmoid output.
        name: Name of the metric.
        dtype: Dtype of the metric state.

    Call arguments:
        y_true: Integer labels of shape `(batch,)` or `(batch, 1)`, or one-hot
            labels of shape `(batch, num_classes)`.
        y_pred: Probabilities or logits of shape `(batch, num_classes)`.
        sample_weight: Optional weights of shape `(batch,)`.
    """

    def __init__(self, from_logits=False, name="brier_score", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        probabilities = ops.cast(_to_probabilities(y_pred, self.from_logits), self.dtype)
        labels = _to_sparse_labels(y_true, probabilities)
        scores = _brier_score_from_probs(labels, probabilities)
        if sample_weight is not None:
            sample_weight = ops.reshape(sample_weight, (-1,))
        return super().update_state(scores, sample_weight=sample_weight)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype, "from_logits": self.from_logits}
