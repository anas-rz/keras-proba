"""Post-hoc calibration and conformal prediction for trained models.

Temperature scaling:
    TemperatureScaling, fit_temperature, temperature_scaled_nll.

Split conformal prediction:
    SplitConformalRegressor, SplitConformalClassifier, conformal_quantile,
    interval_coverage, set_coverage, average_set_size.
"""

from keras_proba.postprocessing.conformal import (
    SplitConformalClassifier,
    SplitConformalRegressor,
    average_set_size,
    conformal_quantile,
    interval_coverage,
    set_coverage,
)
from keras_proba.postprocessing.temperature_scaling import (
    TemperatureScaling,
    fit_temperature,
    temperature_scaled_nll,
)

__all__ = [
    # Temperature scaling.
    "TemperatureScaling",
    "fit_temperature",
    "temperature_scaled_nll",
    # Split conformal prediction.
    "SplitConformalRegressor",
    "SplitConformalClassifier",
    "conformal_quantile",
    "interval_coverage",
    "set_coverage",
    "average_set_size",
]
