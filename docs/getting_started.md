# Getting Started

## Choosing your backend

`keras_proba` runs on every Keras 3 backend. Set `KERAS_BACKEND` before importing Keras:

=== "PyTorch"
    ```python
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    ```

=== "JAX"
    ```python
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    ```

=== "TensorFlow"
    ```python
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import keras
    ```

!!! note "JAX and spectral normalization"
    In its default `mode="projection"`, `SpectralNormalization` behaves like the reference
    only on TensorFlow and PyTorch. Use `mode="reparameterization"` for identical behavior on
    every backend; see the [SNGP guide](guides/sngp.md#spectral-normalization).

## The package layout

The public API mirrors Keras's own layout:

| Namespace | Contents |
|---|---|
| `keras_proba.layers` | SNGP, Bayesian and probabilistic regression layers |
| `keras_proba.losses` | `GaussianNLL`, `MixtureGaussianNLL` |
| `keras_proba.metrics` | `ExpectedCalibrationError`, `BrierScore` and calibration functions |
| `keras_proba.callbacks` | `ResetCovarianceCallback` |
| `keras_proba.postprocessing` | Temperature scaling and split conformal prediction |
| `keras_proba.ops` | Helpers for layer outputs, e.g. `split_gaussian_params`, `mean_field_logits` |
| `keras_proba.stats` | `percentile`, `quantiles` |

```python
from keras_proba import callbacks, layers, losses, metrics, ops, postprocessing, stats
```

## From a model to calibrated uncertainty

A typical workflow on a classifier:

```python
import numpy as np
import keras
from keras_proba import metrics, postprocessing

rng = np.random.default_rng(0)
x = rng.normal(size=(3000, 10)).astype("float32")
y = (x[:, :3] @ rng.normal(size=(3, 4)) + rng.normal(size=(3000, 4))).argmax(-1)
(x_train, y_train), (x_cal, y_cal), (x_test, y_test) = [(x[s], y[s]) for s in (slice(0, 1500), slice(1500, 2200), slice(2200, None))]

# 1. Train, tracking calibration alongside accuracy.
model = keras.Sequential([keras.Input((10,)), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(4)])
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy", metrics.ExpectedCalibrationError(from_logits=True)],
)
model.fit(x_train, y_train, epochs=30, verbose=0)

# 2. Calibrate the confidence on held-out data.
scaling = postprocessing.TemperatureScaling()
scaling.fit(model.predict(x_cal, verbose=0), y_cal)
calibrated = keras.Sequential([model, scaling])

# 3. Turn predictions into sets that contain the label 90% of the time.
probs_cal = keras.ops.softmax(calibrated.predict(x_cal, verbose=0))
conformal = postprocessing.SplitConformalClassifier(alpha=0.1, score="aps").fit(y_cal, probs_cal)
sets = conformal.predict_set(keras.ops.softmax(calibrated.predict(x_test, verbose=0)))
print(float(postprocessing.set_coverage(y_test, sets)), float(postprocessing.average_set_size(sets)))
```

Calibration and conformal prediction work on any trained classifier. To also make the model
itself uncertainty-aware, use an [SNGP head](guides/sngp.md) or
[Bayesian layers](guides/bayesian.md).

## Saving and loading

Every layer, loss and metric is registered with Keras, so models save and load as usual:

```python
calibrated.save("calibrated.keras")
restored = keras.models.load_model("calibrated.keras")
```

Fitted conformal predictors are plain Python objects; save them with their JSON-compatible
`get_config()` and restore them with `from_config`.
