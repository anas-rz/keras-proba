"""Keras Probability: uncertainty awareness in deep learning with multi-backend Keras 3.

Subpackages:
    layers: Distance-aware (SNGP), Bayesian (variational) and probabilistic regression layers.
    losses: Negative log-likelihoods for the probabilistic regression heads.
    metrics: Calibration metrics (expected calibration error, Brier score).
    callbacks: Training callbacks, e.g. resetting the Gaussian process covariance.
    postprocessing: Temperature scaling and split conformal prediction.
    ops: Tensor operations on layer outputs, e.g. `mean_field_logits`.
    stats: Percentiles and quantiles.

Example:

```python
import keras_proba

head = keras_proba.layers.GaussianProcessClassificationHead(inner_dim=128, num_classes=10)
model.compile(metrics=[keras_proba.metrics.ExpectedCalibrationError(from_logits=True)])
```
"""

from keras_proba import callbacks, layers, losses, metrics, ops, postprocessing, stats
from keras_proba.version import __version__

__all__ = [
    "callbacks",
    "layers",
    "losses",
    "metrics",
    "ops",
    "postprocessing",
    "stats",
]
