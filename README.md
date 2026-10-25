# Keras Probability

Uncertainty awareness for deep learning in **Keras 3**.

[![PyPI](https://img.shields.io/pypi/v/keras-proba.svg)](https://pypi.org/project/keras-proba/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Keras 3](https://img.shields.io/badge/keras-3.0+-red.svg)](https://keras.io/)
[![Multi-Backend](https://img.shields.io/badge/backends-PyTorch%20%7C%20JAX%20%7C%20TensorFlow-brightgreen)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Documentation](https://anas-rz.github.io/keras-proba/)** ·
**[Examples](https://anas-rz.github.io/keras-proba/examples/)** ·
**[API reference](https://anas-rz.github.io/keras-proba/api/layers/)**

`keras_proba` gives Keras models calibrated, distance-aware uncertainty estimates. Every layer,
loss and metric runs on the **TensorFlow**, **JAX** and **PyTorch** backends, plugs into
`model.compile` / `model.fit`, and saves with `model.save`.

The distance-aware and Bayesian layers are ports of
[TensorFlow Model Garden](https://github.com/tensorflow/models) and
[TensorFlow Probability](https://github.com/tensorflow/probability), tested for numerical parity
with them.

## Installation

```bash
pip install keras-proba
```

Keras 3 needs one backend; install it with an extra, e.g. `pip install "keras-proba[jax]"`
(`[torch]`, `[jax]` or `[tensorflow]`).

## What's inside

| Need | Use |
|---|---|
| A classifier that knows when inputs are far from the training data | SNGP: `SpectralNormalization` + `RandomFeatureGaussianProcess`, or the all-in-one `GaussianProcessClassificationHead`, with `ResetCovarianceCallback` |
| Uncertainty over the weights | Bayesian layers: `DenseFlipout`, `DenseReparameterization`, `DenseLocalReparameterization`, `Conv{1,2,3}DFlipout`, `Conv{1,2,3}DReparameterization` |
| Regression with a predicted noise level | `GaussianHead` with `GaussianNLL` |
| Multi-modal or skewed regression targets | `MixtureGaussianHead` with `MixtureGaussianNLL` |
| Measuring calibration | `ExpectedCalibrationError`, `BrierScore`, `brier_decomposition` |
| Fixing an overconfident trained classifier | `TemperatureScaling` |
| Intervals or sets with a coverage guarantee, for any trained model | `SplitConformalRegressor`, `SplitConformalClassifier` |

The package is organized as:

| Module | Contents |
|---|---|
| `keras_proba.layers` | Distance-aware (SNGP), Bayesian (variational) and probabilistic regression layers |
| `keras_proba.losses` | Negative log-likelihoods for the probabilistic regression heads |
| `keras_proba.metrics` | Calibration metrics |
| `keras_proba.callbacks` | Training callbacks, e.g. resetting the Gaussian process covariance |
| `keras_proba.postprocessing` | Temperature scaling and split conformal prediction |
| `keras_proba.ops` | Operations on layer outputs, e.g. `mean_field_logits`, `split_gaussian_params` |
| `keras_proba.stats` | Percentiles and quantiles |

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

Adding guaranteed intervals to any trained regressor, after training:

```python
conformal = keras_proba.postprocessing.SplitConformalRegressor(alpha=0.1)  # 90% coverage.
conformal.fit(y_cal, model.predict(x_cal))
lower, upper = conformal.predict_interval(model.predict(x_test))
```

## Examples

End-to-end [notebooks on real datasets](https://github.com/anas-rz/keras-proba/tree/main/examples),
each runnable on Google Colab and rendered in the
[documentation](https://anas-rz.github.io/keras-proba/examples/):

| Task | Training | Post-processing |
|---|---|---|
| **Forecasting** (Jena climate, ERA5) | Probabilistic forecasting with `GaussianHead`; Bayesian forecasting with `Conv1DFlipout` | Conformal forecast intervals; probabilistic forecasts from a pre-trained FourCastNet ([keras-climate](https://github.com/anas-rz/keras-climate)), with no training |
| **Regression** (California housing) | Mixture density network | Adaptive conformal intervals |
| **Classification** (Fashion-MNIST, MNIST) | Detecting unfamiliar inputs with SNGP; a Bayesian CNN | Temperature scaling; conformal prediction sets |

## Verified against the references

Each component is tested on all three backends for:

- **Parity** with the original implementation (Model Garden, TensorFlow Probability, PyTorch,
  SciPy), feeding both the same weights and random draws.
- **Training**: the method does its job on real fits, e.g. out-of-distribution uncertainty,
  improved calibration, or the theoretical conformal coverage.
- **Serialization**: `model.save` / `keras.models.load_model` round trips.

See [Testing](https://anas-rz.github.io/keras-proba/testing/) for details, including the bugs in
the references that these tests surfaced.

## Development

```bash
git clone https://github.com/anas-rz/keras-proba.git
cd keras-proba
pip install -e ".[dev]"
KERAS_BACKEND=jax pytest          # Or torch, tensorflow.
```

## License

[MIT](https://github.com/anas-rz/keras-proba/blob/main/LICENSE)
