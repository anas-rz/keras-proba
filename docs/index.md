# Keras Probability

Uncertainty awareness for deep learning in **Keras 3**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Keras 3](https://img.shields.io/badge/keras-3.0+-red.svg)](https://keras.io/)
[![Multi-Backend](https://img.shields.io/badge/backends-PyTorch%20%7C%20JAX%20%7C%20TensorFlow-brightgreen)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`keras_proba` gives Keras models calibrated, distance-aware uncertainty estimates. Every
layer, loss and metric runs on the **TensorFlow**, **JAX** and **PyTorch** backends, plugs into
`model.compile` / `model.fit`, and saves with `model.save`.

The distance-aware and Bayesian layers are ports of
[TensorFlow Model Garden](https://github.com/tensorflow/models) and
[TensorFlow Probability](https://github.com/tensorflow/probability), tested for numerical
parity with them.

---

## What's inside

| Need | Use |
|---|---|
| A classifier that knows when inputs are far from the training data | SNGP: [`SpectralNormalization`](api/layers.md#keras_proba.layers.SpectralNormalization) + [`RandomFeatureGaussianProcess`](api/layers.md#keras_proba.layers.RandomFeatureGaussianProcess), or the all-in-one [`GaussianProcessClassificationHead`](api/layers.md#keras_proba.layers.GaussianProcessClassificationHead) |
| Uncertainty over the weights | Bayesian layers: `DenseFlipout`, `DenseReparameterization`, `Conv{1,2,3}DFlipout`, ... |
| Regression with a predicted noise level | [`GaussianHead`](api/layers.md#keras_proba.layers.GaussianHead) with [`GaussianNLL`](api/losses.md#keras_proba.losses.GaussianNLL) |
| Multi-modal or skewed regression targets | [`MixtureGaussianHead`](api/layers.md#keras_proba.layers.MixtureGaussianHead) with [`MixtureGaussianNLL`](api/losses.md#keras_proba.losses.MixtureGaussianNLL) |
| Measuring calibration | [`ExpectedCalibrationError`](api/metrics.md#keras_proba.metrics.ExpectedCalibrationError), [`BrierScore`](api/metrics.md#keras_proba.metrics.BrierScore) |
| Fixing an overconfident trained classifier | [`TemperatureScaling`](api/postprocessing.md#keras_proba.postprocessing.TemperatureScaling) |
| Intervals or sets with a coverage guarantee | [`SplitConformalRegressor`](api/postprocessing.md#keras_proba.postprocessing.SplitConformalRegressor), [`SplitConformalClassifier`](api/postprocessing.md#keras_proba.postprocessing.SplitConformalClassifier) |

---

## Installation

```bash
pip install keras-proba
```

Keras 3 needs one backend; install it with an extra, e.g. `pip install "keras-proba[jax]"` (`[torch]`, `[jax]` or `[tensorflow]`).

---

## A first example

A classifier whose predictions become uncertain away from the training data:

```python
import numpy as np
import keras
import keras_proba

x = np.random.normal(size=(512, 2)).astype("float32")
y = (x[:, 0] * x[:, 1] > 0).astype("int32")

inputs = keras.Input((2,))
hidden = keras.layers.Dense(64, trainable=False)(inputs)  # A frozen random projection.
hidden = keras_proba.layers.SpectralNormalization(keras.layers.Dense(64, activation="relu"))(hidden)
logits = keras_proba.layers.GaussianProcessClassificationHead(
    inner_dim=0,
    num_classes=2,
    temperature=np.pi / 8,
    normalize_input=False,
    scale_random_features=True,
    custom_random_features_initializer=keras.initializers.RandomNormal(stddev=1.0),
)(hidden)
model = keras.Model(inputs, logits)

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy", keras_proba.metrics.ExpectedCalibrationError(from_logits=True)],
)
model.fit(x, y, epochs=10, verbose=0, callbacks=[keras_proba.callbacks.ResetCovarianceCallback()])

probs = keras.ops.softmax(model.predict(np.array([[0.5, 0.5], [8.0, -8.0]], "float32"), verbose=0))
# Confident near the data, close to uniform far from it.
```

Continue with [Getting Started](getting_started.md), or jump to a [guide](guides/sngp.md).

---

## Verified against the references

Each component is tested on all three backends for:

- **Parity** with the original implementation (Model Garden, TensorFlow Probability, PyTorch, SciPy), feeding both the same weights and random draws.
- **Training**: the method does its job on real fits, e.g. out-of-distribution uncertainty, improved calibration, or the theoretical conformal coverage.
- **Serialization**: `model.save` / `keras.models.load_model` round trips.

See [Testing](testing.md) for details, including the bugs in the references that these tests surfaced.
