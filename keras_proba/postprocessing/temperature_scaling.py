"""Multi-backend temperature scaling for classifier calibration in Keras 3.

Temperature scaling [1] divides a trained classifier's logits by a single
scalar `T > 0`, fitted to minimize the negative log-likelihood on held-out
data. It does not change the predicted class, only the confidence: `T > 1`
softens overconfident predictions and `T < 1` sharpens underconfident ones.

Applying a temperature matches the temperature scaling calibrator of
TensorFlow Model Garden's object detection `calibration_builder.py`, which
divides logits by a configured `scaler`. Fitting it is not part of the
reference: `fit_temperature` minimizes the NLL exactly, using that it is convex
in `1 / T`.

Example:

```python
logits = model.predict(x_val)
scaling = TemperatureScaling()
scaling.fit(logits, y_val)
calibrated_model = keras.Sequential([model, scaling])
```

[1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. On Calibration of
     Modern Neural Networks. In _International Conference on Machine
     Learning_, 2017. https://arxiv.org/abs/1706.04599
"""

import math

import numpy as np

import keras
from keras import ops


def _prepare(logits, labels, sample_weight):
    """Returns float64 logits of shape `(n, num_classes)`, integer labels and weights."""
    logits = np.asarray(ops.convert_to_numpy(logits), dtype="float64")
    labels = np.asarray(ops.convert_to_numpy(labels))
    if logits.ndim == 1:
        logits = logits[:, None]
    if logits.ndim != 2:
        raise ValueError(f"`logits` must have shape (n, num_classes) or (n,). Received: {logits.shape}")
    if labels.ndim == 2 and labels.shape[-1] == logits.shape[-1] and logits.shape[-1] > 1:
        labels = labels.argmax(-1)  # One-hot labels.
    labels = labels.reshape(-1).astype("int64")
    if logits.shape[-1] == 1:
        # A single sigmoid logit z is the two-class softmax of [0, z].
        logits = np.concatenate([np.zeros_like(logits), logits], axis=-1)
    if len(labels) != len(logits):
        raise ValueError(f"Got {len(logits)} logits and {len(labels)} labels.")
    if sample_weight is None:
        sample_weight = np.ones(len(labels))
    sample_weight = np.asarray(ops.convert_to_numpy(sample_weight), dtype="float64").reshape(-1)
    return logits, labels, sample_weight / sample_weight.sum()


def _nll_slope(beta, logits, labels, weights):
    """Derivative of the mean NLL of `softmax(beta * logits)` with respect to `beta`.

    The NLL is convex in `beta`, so this is non-decreasing:
    `d NLL / d beta = mean(E_p[z] - z_label)`, with `p = softmax(beta * z)`.
    """
    scaled = beta * logits
    scaled = scaled - scaled.max(-1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(-1, keepdims=True)
    expected_logit = (probs * logits).sum(-1)
    label_logit = np.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    return float(np.sum(weights * (expected_logit - label_logit)))


def temperature_scaled_nll(logits, labels, temperature, sample_weight=None):
    """Mean negative log-likelihood of `softmax(logits / temperature)`.

    Args:
        logits: Array of shape `(n, num_classes)`, or `(n,)` / `(n, 1)` for
            binary sigmoid logits.
        labels: Integer labels of shape `(n,)` or one-hot labels.
        temperature: Positive temperature.
        sample_weight: Optional weights of shape `(n,)`.

    Returns:
        The (weighted) mean NLL as a Python float.
    """
    logits, labels, weights = _prepare(logits, labels, sample_weight)
    scaled = logits / temperature
    log_norm = np.log(np.exp(scaled - scaled.max(-1, keepdims=True)).sum(-1)) + scaled.max(-1)
    label_logit = np.take_along_axis(scaled, labels[:, None], axis=-1)[:, 0]
    return float(np.sum(weights * (log_norm - label_logit)))


def fit_temperature(
    logits,
    labels,
    sample_weight=None,
    temperature_range=(0.05, 20.0),
    tolerance=1e-7,
    max_iterations=200,
):
    """Finds the temperature that minimizes the NLL of `softmax(logits / T)`.

    The NLL is convex in `beta = 1 / T`, so the minimizer is the root of its
    non-decreasing derivative, found by bisection in `log(beta)`. This is exact,
    deterministic, needs no learning rate, and is computed in float64 NumPy, so
    it is the same on every backend.

    Args:
        logits: Held-out logits of shape `(n, num_classes)`, or `(n,)` /
            `(n, 1)` for binary sigmoid logits.
        labels: Integer labels of shape `(n,)` or one-hot labels.
        sample_weight: Optional weights of shape `(n,)`.
        temperature_range: `(min, max)` temperatures searched. If the optimum
            lies outside, the nearest bound is returned; e.g. on perfectly
            separable data the NLL keeps decreasing as `T -> 0`.
        tolerance: Relative tolerance on the temperature.
        max_iterations: Maximum number of bisection steps.

    Returns:
        The fitted temperature as a Python float.
    """
    t_min, t_max = temperature_range
    if not 0 < t_min < t_max:
        raise ValueError(f"`temperature_range` must satisfy 0 < min < max. Received: {temperature_range}")
    logits, labels, weights = _prepare(logits, labels, sample_weight)

    log_lo, log_hi = -math.log(t_max), -math.log(t_min)  # log(beta) bounds.
    if _nll_slope(math.exp(log_lo), logits, labels, weights) >= 0:
        return t_max
    if _nll_slope(math.exp(log_hi), logits, labels, weights) <= 0:
        return t_min
    for _ in range(max_iterations):
        log_mid = 0.5 * (log_lo + log_hi)
        if _nll_slope(math.exp(log_mid), logits, labels, weights) > 0:
            log_hi = log_mid
        else:
            log_lo = log_mid
        if log_hi - log_lo < tolerance:
            break
    return math.exp(-0.5 * (log_lo + log_hi))


@keras.saving.register_keras_serializable(package="keras_proba")
class TemperatureScaling(keras.layers.Layer):
    """Divides logits by a learned temperature.

    The temperature is stored as a trainable `log_temperature` weight, so it
    stays positive. Fit it in closed form on held-out logits with `fit`, or
    train it with `model.fit` on a model whose other layers are frozen.

    Args:
        temperature: Initial temperature, a positive float.
        **kwargs: Keyword arguments for `keras.layers.Layer`.

    Input shape:
        Logits of any shape.

    Output shape:
        Same as the input.
    """

    def __init__(self, temperature=1.0, **kwargs):
        if temperature <= 0:
            raise ValueError(f"`temperature` should be positive. Received: {temperature}")
        super().__init__(**kwargs)
        self.initial_temperature = temperature
        self.log_temperature = self.add_weight(
            name="log_temperature",
            shape=(),
            initializer=keras.initializers.Constant(math.log(temperature)),
        )

    @property
    def temperature(self):
        """The current temperature, `exp(log_temperature)`."""
        return ops.exp(self.log_temperature)

    def fit(self, logits, labels, **kwargs):
        """Fits the temperature to held-out `logits` and `labels`; see `fit_temperature`.

        Returns:
            The fitted temperature as a Python float.
        """
        temperature = fit_temperature(logits, labels, **kwargs)
        self.log_temperature.assign(math.log(temperature))
        return temperature

    def call(self, logits):
        return logits / ops.cast(self.temperature, logits.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "temperature": self.initial_temperature}
