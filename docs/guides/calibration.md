# Calibration

A classifier is calibrated when its confidence matches its accuracy: among predictions made with
80% confidence, 80% are correct. Modern networks are often overconfident [^guo2017].

## Measuring calibration

[`ExpectedCalibrationError`](../api/metrics.md#keras_proba.metrics.ExpectedCalibrationError) bins
predictions by confidence and averages the gap between confidence and accuracy per bin;
[`BrierScore`](../api/metrics.md#keras_proba.metrics.BrierScore) is a proper scoring rule for the
whole predicted distribution. Both accumulate over batches, so `evaluate` gives the value on the
whole dataset:

```python
import numpy as np
import keras
from keras_proba import metrics, postprocessing

rng = np.random.default_rng(0)
x = rng.normal(size=(2300, 10)).astype("float32")
y = (1.5 * x[:, :3] + rng.gumbel(size=(2300, 3))).argmax(-1)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = (x[:300], y[:300]), (x[300:1300], y[300:1300]), (x[1300:], y[1300:])

model = keras.Sequential([keras.Input((10,)), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(3)])
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[metrics.ExpectedCalibrationError(num_bins=15, from_logits=True), metrics.BrierScore(from_logits=True)],
)
model.fit(x_train, y_train, epochs=100, verbose=0)  # Overfits, so becomes overconfident.
print(model.evaluate(x_test, y_test, verbose=0, return_dict=True))
```

The metrics take probabilities or logits (`from_logits`), sparse or one-hot labels, and a single
sigmoid output for binary classification. The functions
[`expected_calibration_error`](../api/metrics.md#keras_proba.metrics.expected_calibration_error),
[`expected_calibration_error_quantiles`](../api/metrics.md#keras_proba.metrics.expected_calibration_error_quantiles)
(equal-count bins) and [`brier_decomposition`](../api/metrics.md#keras_proba.metrics.brier_decomposition)
(uncertainty, resolution and reliability) evaluate a full set of predictions at once.

## Temperature scaling

[`TemperatureScaling`](../api/postprocessing.md#keras_proba.postprocessing.TemperatureScaling)
divides the logits of a trained model by a single temperature $T$, fitted to minimize the
negative log-likelihood on held-out data. It changes the confidence, never the predicted class:

```python
scaling = postprocessing.TemperatureScaling()
temperature = scaling.fit(model.predict(x_val, verbose=0), y_val)  # > 1 for an overconfident model.
calibrated = keras.Sequential([model, scaling])
calibrated.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[metrics.ExpectedCalibrationError(num_bins=15, from_logits=True)],
)
print(temperature, calibrated.evaluate(x_test, y_test, verbose=0, return_dict=True))
```

`fit` finds the exact optimum: the NLL is convex in $1 / T$, so it is minimized by bisection in
float64, with the same result on every backend and no learning rate to tune. The layer can also be
trained with `model.fit` on a model whose other layers are frozen.

[^guo2017]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
