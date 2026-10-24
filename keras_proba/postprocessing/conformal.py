"""Multi-backend split conformal prediction in Keras 3.

Split conformal prediction [1, 2] turns any trained model into one with
prediction intervals (regression) or prediction sets (classification) that
contain the true label with probability at least `1 - alpha`, assuming only
that calibration and test examples are exchangeable:

1. Compute a nonconformity score `s_i` for each of `n` held-out calibration
   examples, e.g. the absolute residual `|y_i - y_pred_i|`.
2. Take `q_hat`, the `k`-th smallest score with `k = ceil((n + 1) (1 - alpha))`.
3. For a test input, return every label whose score is at most `q_hat`.

Then `P(y_test in C(x_test)) >= 1 - alpha`, and for continuous scores the
expected coverage is exactly `k / (n + 1) < 1 - alpha + 1 / (n + 1)`.

Neither TensorFlow Model Garden nor TensorFlow Probability implements conformal
prediction; this follows [2]. The threshold is the exact order statistic,
rather than `numpy.quantile(scores, k / n, method="higher")`, which selects
the next larger score and over-covers.

[1]: Vladimir Vovk, Alexander Gammerman, Glenn Shafer. Algorithmic Learning in
     a Random World. Springer, 2005.
[2]: Anastasios N. Angelopoulos, Stephen Bates. A Gentle Introduction to
     Conformal Prediction and Distribution-Free Uncertainty Quantification.
     https://arxiv.org/abs/2107.07511
[3]: Yaniv Romano, Matteo Sesia, Emmanuel J. Candes. Classification with Valid
     and Adaptive Coverage. In _Neural Information Processing Systems_, 2020.
     https://arxiv.org/abs/2006.02544
"""

import math

import numpy as np

import keras
from keras import ops


def _validate_alpha(alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must be in (0, 1). Received: {alpha}")


def conformal_quantile(scores, alpha):
    """The split conformal threshold: the `ceil((n + 1) (1 - alpha))`-th smallest score.

    Args:
        scores: Calibration scores of shape `(n, ...)`; thresholds are computed
            independently for each trailing position.
        alpha: Miscoverage level in `(0, 1)`.

    Returns:
        Tensor of shape `scores.shape[1:]`. If `n` is too small for the
        requested coverage (`k > n`), the threshold is `inf`, i.e. every label
        is included.
    """
    _validate_alpha(alpha)
    scores = ops.convert_to_tensor(scores)
    if not keras.backend.is_float_dtype(scores.dtype):
        scores = ops.cast(scores, "float32")
    n = scores.shape[0]
    if n is None or n == 0:
        raise ValueError(f"At least one calibration score is needed, with a static count. Received shape: {scores.shape}")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return ops.full(tuple(scores.shape[1:]), np.inf, dtype=scores.dtype)
    return ops.take(ops.sort(scores, axis=0), k - 1, axis=0)


def _as_float(x):
    x = ops.convert_to_tensor(x)
    return x if keras.backend.is_float_dtype(x.dtype) else ops.cast(x, "float32")


class SplitConformalRegressor:
    """Split conformal prediction intervals for regression.

    Without `scale`, the score is the absolute residual `|y - y_pred|` and
    intervals have a constant width `2 q_hat`. With a per-example `scale`
    (e.g. the standard deviation of a `GaussianHead`, `MixtureGaussianHead` or
    Monte Carlo samples), the score is `|y - y_pred| / scale` and intervals
    `y_pred +- q_hat * scale` adapt to each input [2]. Each output dimension
    gets its own threshold, so coverage holds for each dimension separately.

    Example:

    ```python
    mean, scale = split_gaussian_params(model.predict(x_cal))
    conformal = SplitConformalRegressor(alpha=0.1).fit(y_cal, mean, scale=scale)
    mean, scale = split_gaussian_params(model.predict(x_test))
    lower, upper = conformal.predict_interval(mean, scale=scale)
    ```

    Args:
        alpha: Miscoverage level in `(0, 1)`; intervals cover with probability
            at least `1 - alpha`.
    """

    def __init__(self, alpha=0.1):
        _validate_alpha(alpha)
        self.alpha = alpha
        self.normalized = None
        self.quantile_ = None

    def _scores(self, y_true, y_pred, scale):
        y_pred = _as_float(y_pred)
        y_true = ops.reshape(ops.cast(y_true, y_pred.dtype), ops.shape(y_pred))
        residual = ops.abs(y_true - y_pred)
        return residual if scale is None else residual / ops.cast(scale, y_pred.dtype)

    def fit(self, y_true, y_pred, scale=None):
        """Computes the threshold from held-out calibration data.

        Args:
            y_true: Calibration targets, of the same shape as `y_pred`.
            y_pred: Point predictions of shape `(n,)` or `(n, units)`.
            scale: Optional positive per-example scales, of the same shape.

        Returns:
            `self`.
        """
        self.normalized = scale is not None
        self.quantile_ = conformal_quantile(self._scores(y_true, y_pred, scale), self.alpha)
        return self

    def predict_interval(self, y_pred, scale=None):
        """Returns the prediction intervals `(lower, upper)` for `y_pred`.

        Args:
            y_pred: Point predictions of shape `(n,)` or `(n, units)`.
            scale: Per-example scales, required if and only if `fit` was given
                scales.

        Returns:
            Tuple of `lower` and `upper` bounds, each of the shape of `y_pred`.
        """
        if self.quantile_ is None:
            raise ValueError("Call `fit` on calibration data first.")
        if (scale is not None) != self.normalized:
            raise ValueError(
                "`scale` must be given to `predict_interval` if and only if it was given to `fit`."
            )
        y_pred = _as_float(y_pred)
        half_width = ops.cast(self.quantile_, y_pred.dtype)
        if scale is not None:
            half_width = half_width * ops.cast(scale, y_pred.dtype)
        return y_pred - half_width, y_pred + half_width

    def get_config(self):
        quantile = None if self.quantile_ is None else ops.convert_to_numpy(self.quantile_).tolist()
        return {"alpha": self.alpha, "normalized": self.normalized, "quantile": quantile}

    @classmethod
    def from_config(cls, config):
        regressor = cls(alpha=config["alpha"])
        regressor.normalized = config["normalized"]
        if config["quantile"] is not None:
            regressor.quantile_ = ops.convert_to_tensor(np.asarray(config["quantile"], "float32"))
        return regressor


_CLASSIFICATION_SCORES = ("lac", "aps")


def _to_probabilities(y_pred, from_logits):
    y_pred = _as_float(y_pred)
    if y_pred.shape[-1] == 1:
        # A single sigmoid output is the two-class distribution [1 - p, p].
        p = ops.sigmoid(y_pred) if from_logits else y_pred
        return ops.concatenate([1.0 - p, p], axis=-1)
    return ops.softmax(y_pred, axis=-1) if from_logits else y_pred


def _to_sparse_labels(y_true, probs):
    y_true = ops.convert_to_tensor(y_true)
    if len(y_true.shape) == len(probs.shape):
        y_true = ops.squeeze(y_true, -1) if y_true.shape[-1] == 1 else ops.argmax(y_true, axis=-1)
    return ops.cast(y_true, "int32")


def _sorted_mass(probs):
    """Per class, the probability mass of the classes ranked before it, in descending probability."""
    order = ops.argsort(-probs, axis=-1)
    sorted_probs = ops.take_along_axis(probs, order, axis=-1)
    mass_before_sorted = ops.cumsum(sorted_probs, axis=-1) - sorted_probs
    # Scatter back to the original class order.
    ranks = ops.argsort(order, axis=-1)
    return ops.take_along_axis(mass_before_sorted, ranks, axis=-1)


class SplitConformalClassifier:
    """Split conformal prediction sets for classification.

    Scores:

    - `"lac"` (least ambiguous set-valued classifier): `1 - p_label`. Sets are
      `{k : p_k >= 1 - q_hat}`, with exact coverage and the smallest average
      size [2]; they can be empty for ambiguous inputs.
    - `"aps"` (adaptive prediction sets) [3]: the probability mass of the
      classes ranked before the label, plus `u * p_label`. Sets grow with the
      model's uncertainty about each input. With `randomized=True` (default),
      `u ~ Uniform(0, 1)` and sets `{k : mass before k + u * p_k <= q_hat}`
      have exact coverage. With `randomized=False`, `u = 1` and the sets
      `{k : mass before k < q_hat}` are never empty but conservative: on hard
      tasks they can cover far more than `1 - alpha`.

    Example:

    ```python
    conformal = SplitConformalClassifier(alpha=0.1, score="aps").fit(y_cal, model.predict(x_cal))
    sets = conformal.predict_set(model.predict(x_test))  # Boolean, (n, num_classes).
    ```

    Args:
        alpha: Miscoverage level in `(0, 1)`; sets contain the label with
            probability at least `1 - alpha`.
        score: `"lac"` or `"aps"`.
        from_logits: Whether predictions are logits instead of probabilities.
            A single output unit is treated as a binary sigmoid output.
        randomized: Whether to randomize APS scores and sets for exact
            coverage. Ignored for `"lac"`.
        seed: Seed for the APS randomization.
    """

    def __init__(self, alpha=0.1, score="lac", from_logits=False, randomized=True, seed=None):
        _validate_alpha(alpha)
        if score not in _CLASSIFICATION_SCORES:
            raise ValueError(f"`score` must be one of {_CLASSIFICATION_SCORES}. Received: {score}")
        self.alpha = alpha
        self.score = score
        self.from_logits = from_logits
        self.randomized = randomized
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)
        self.quantile_ = None

    def _uniform(self, batch_size, dtype):
        """One `Uniform(0, 1)` draw per example, of shape `(batch_size, 1)`."""
        if not self.randomized:
            return ops.ones((batch_size, 1), dtype)
        return keras.random.uniform((batch_size, 1), dtype=dtype, seed=self.seed_generator)

    def scores(self, y_true, y_pred):
        """Nonconformity scores of the labels `y_true`, of shape `(n,)`."""
        probs = _to_probabilities(y_pred, self.from_logits)
        labels = ops.expand_dims(_to_sparse_labels(y_true, probs), -1)
        label_prob = ops.take_along_axis(probs, labels, axis=-1)[:, 0]
        if self.score == "lac":
            return 1.0 - label_prob
        u = self._uniform(ops.shape(probs)[0], probs.dtype)[:, 0]
        return ops.take_along_axis(_sorted_mass(probs), labels, axis=-1)[:, 0] + u * label_prob

    def fit(self, y_true, y_pred):
        """Computes the threshold from held-out calibration labels and predictions.

        Args:
            y_true: Integer labels of shape `(n,)` or `(n, 1)`, or one-hot labels.
            y_pred: Probabilities or logits of shape `(n, num_classes)`.

        Returns:
            `self`.
        """
        self.quantile_ = conformal_quantile(self.scores(y_true, y_pred), self.alpha)
        return self

    def predict_set(self, y_pred):
        """Returns boolean prediction sets of shape `(n, num_classes)`."""
        if self.quantile_ is None:
            raise ValueError("Call `fit` on calibration data first.")
        probs = _to_probabilities(y_pred, self.from_logits)
        q_hat = ops.cast(self.quantile_, probs.dtype)
        if self.score == "lac":
            return probs >= 1.0 - q_hat
        if not self.randomized:
            return _sorted_mass(probs) < q_hat
        return _sorted_mass(probs) + self._uniform(ops.shape(probs)[0], probs.dtype) * probs <= q_hat

    def get_config(self):
        quantile = None if self.quantile_ is None else float(ops.convert_to_numpy(self.quantile_))
        return {
            "alpha": self.alpha,
            "score": self.score,
            "from_logits": self.from_logits,
            "randomized": self.randomized,
            "seed": self.seed,
            "quantile": quantile,
        }

    @classmethod
    def from_config(cls, config):
        classifier = cls(
            alpha=config["alpha"],
            score=config["score"],
            from_logits=config["from_logits"],
            randomized=config["randomized"],
            seed=config["seed"],
        )
        if config["quantile"] is not None:
            classifier.quantile_ = ops.convert_to_tensor(np.float32(config["quantile"]))
        return classifier


def interval_coverage(y_true, lower, upper):
    """Fraction of targets inside `[lower, upper]`, per output dimension for 2-D inputs."""
    lower = _as_float(lower)
    y_true = ops.reshape(ops.cast(y_true, lower.dtype), ops.shape(lower))
    inside = ops.logical_and(y_true >= lower, y_true <= ops.cast(upper, lower.dtype))
    return ops.mean(ops.cast(inside, lower.dtype), axis=0)


def set_coverage(y_true, prediction_sets):
    """Fraction of labels contained in their boolean prediction sets."""
    prediction_sets = ops.cast(prediction_sets, "float32")
    labels = ops.expand_dims(_to_sparse_labels(y_true, prediction_sets), -1)
    return ops.mean(ops.take_along_axis(prediction_sets, labels, axis=-1))


def average_set_size(prediction_sets):
    """Average number of classes in the boolean prediction sets."""
    return ops.mean(ops.sum(ops.cast(prediction_sets, "float32"), axis=-1))
