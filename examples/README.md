# Examples

End-to-end examples of `keras_proba` on real datasets. Each script is standalone: it downloads its
data with Keras, trains a model, and prints the uncertainty metrics that matter for the task.

```
examples/
├── forecasting/      Jena climate: temperature 24 hours ahead
│   ├── training/        uncertainty-aware models
│   └── postprocessing/  uncertainty added to a trained model
├── regression/       California housing: house values
│   ├── training/
│   └── postprocessing/
└── classification/   Fashion-MNIST, with MNIST digits as out-of-distribution inputs
    ├── training/
    └── postprocessing/
```

`training/` examples make the model itself uncertainty-aware; `postprocessing/` examples add
calibrated uncertainty to an already trained model, without retraining it.

## Running

```bash
pip install "keras-proba[jax]"   # Or [torch], [tensorflow].
KERAS_BACKEND=jax python examples/forecasting/training/probabilistic_forecasting.py
```

Every example runs on the TensorFlow, JAX and PyTorch backends, on CPU in a few minutes. Set
`KERAS_PROBA_QUICK=1` for a fast smoke run on a fraction of the data (the metrics are then not
meaningful).

## Examples

Results from one run on CPU with the PyTorch backend; they vary a little between runs and backends.

### Forecasting: [Jena climate](https://www.bgc-jena.mpg.de/wetter/), temperature 24 hours ahead from the last 48 hours

| Example | Technique | Result on the held-out last year |
|---|---|---|
| [`training/probabilistic_forecasting.py`](forecasting/training/probabilistic_forecasting.py) | `GaussianHead` + `GaussianNLL` | 80% intervals cover 79.3% of the test year and are 7% narrower than constant-width intervals |
| [`training/bayesian_forecasting.py`](forecasting/training/bayesian_forecasting.py) | `Conv1DFlipout` encoder + `GaussianHead`, Monte Carlo prediction | Uncertainty split into aleatoric and epistemic; epistemic uncertainty grows 4.4× on inputs shifted outside the training range |
| [`postprocessing/conformal_forecast_intervals.py`](forecasting/postprocessing/conformal_forecast_intervals.py) | `SplitConformalRegressor`, constant and scaled by the predicted std | A Gaussian head's 90% intervals cover 78.8% of the new year; conformal calibration restores 90.8% |

### Regression: [California housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), house values

| Example | Technique | Result on the test set |
|---|---|---|
| [`training/mixture_density_network.py`](regression/training/mixture_density_network.py) | `MixtureGaussianHead` + `MixtureGaussianNLL` vs `GaussianHead` | Test NLL 0.34 vs 0.55; at the $500k census cap, 0.17 vs 1.85, with a mixture component placed on the cap |
| [`postprocessing/conformal_regression.py`](regression/postprocessing/conformal_regression.py) | `SplitConformalRegressor`, constant vs scaled by a `GaussianHead` std | Both cover 90%; scaled intervals are 9% narrower, with coverage of 89–92% in every income quartile vs 86–95% |

### Classification: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), with [MNIST](https://yann.lecun.com/exdb/mnist/) digits as out-of-distribution inputs

| Example | Technique | Result on the test set |
|---|---|---|
| [`training/sngp_ood_detection.py`](classification/training/sngp_ood_detection.py) | SNGP: `SpectralNormalization` + `GaussianProcessClassificationHead` + `ResetCovarianceCallback` | Same accuracy (88%) as a standard network; detects digits with AUROC 0.92 vs 0.67 |
| [`training/bayesian_cnn.py`](classification/training/bayesian_cnn.py) | `Conv2DFlipout` + `DenseFlipout`, Monte Carlo prediction | Averaging 20 samples: accuracy 88.8% vs 87.4% for one; mutual information detects digits with AUROC 0.92 |
| [`postprocessing/temperature_scaling.py`](classification/postprocessing/temperature_scaling.py) | `TemperatureScaling`, `ExpectedCalibrationError`, `BrierScore` | ECE 0.036 → 0.009 and NLL 0.265 → 0.228, with unchanged accuracy |
| [`postprocessing/conformal_prediction_sets.py`](classification/postprocessing/conformal_prediction_sets.py) | `SplitConformalClassifier`, LAC and APS | APS sets cover 90.0% with 1.18 classes on average, larger for shirts and for wrong predictions |

## Reading the results

Each script prints its own diagnostics, and a few of them show the limits of a technique as well
as its strengths:

- A `GaussianHead` only models the noise seen in training: in the forecasting examples, a test-year
  season unlike the training years can be under-covered.
- Conformal guarantees hold on average over all inputs, and assume calibration and test data are
  exchangeable. Weather is not stationary, and the official Fashion-MNIST test images are slightly
  harder than the training images: both examples show what this does to the coverage.
