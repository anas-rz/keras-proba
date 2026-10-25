# Examples

End-to-end notebooks on real datasets. Each one explains a single uncertainty technique, and runs
on Google Colab in a few minutes: click a badge, then *Runtime → Run all*. They are also rendered
in the [documentation](https://anas-rz.github.io/keras-proba/examples/).

`training/` notebooks make the model itself uncertainty-aware; `postprocessing/` notebooks add
calibrated uncertainty to an already trained model, without retraining it.

## Forecasting: temperature 24 hours ahead ([Jena climate](https://www.bgc-jena.mpg.de/wetter/), [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5))

| Notebook | Technique | |
|---|---|---|
| [Probabilistic forecasting](forecasting/training/probabilistic_forecasting.ipynb) | Predict a mean and a standard deviation with `GaussianHead` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/forecasting/training/probabilistic_forecasting.ipynb) |
| [Bayesian forecasting](forecasting/training/bayesian_forecasting.ipynb) | Separate noise from model uncertainty with Bayesian `Conv1DFlipout` layers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/forecasting/training/bayesian_forecasting.ipynb) |
| [Conformal forecast intervals](forecasting/postprocessing/conformal_forecast_intervals.ipynb) | Add guaranteed intervals to a point forecaster with `SplitConformalRegressor` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/forecasting/postprocessing/conformal_forecast_intervals.ipynb) |
| [Probabilistic forecasts from a pre-trained FourCastNet](forecasting/postprocessing/pretrained_fourcastnet_conformal.ipynb) | 90% intervals at every grid point for a pre-trained global weather model ([keras-climate](https://github.com/anas-rz/keras-climate), [ERA5](https://weatherbench2.readthedocs.io/)), with no training; needs a GPU runtime | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/forecasting/postprocessing/pretrained_fourcastnet_conformal.ipynb) |

## Regression: house values ([California housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

| Notebook | Technique | |
|---|---|---|
| [Mixture density network](regression/training/mixture_density_network.ipynb) | Predict a mixture of Gaussians with `MixtureGaussianHead` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/regression/training/mixture_density_network.ipynb) |
| [Adaptive conformal intervals](regression/postprocessing/conformal_regression.ipynb) | Conformal intervals whose width follows the model's uncertainty | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/regression/postprocessing/conformal_regression.ipynb) |

## Classification: clothing images ([Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)), with digits as unfamiliar inputs

| Notebook | Technique | |
|---|---|---|
| [Detecting unfamiliar inputs with SNGP](classification/training/sngp_ood_detection.ipynb) | Distance-aware uncertainty with `SpectralNormalization` and a Gaussian process head | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/classification/training/sngp_ood_detection.ipynb) |
| [A Bayesian CNN](classification/training/bayesian_cnn.ipynb) | Average and compare repeated predictions of `Conv2DFlipout` layers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/classification/training/bayesian_cnn.ipynb) |
| [Temperature scaling](classification/postprocessing/temperature_scaling.ipynb) | Fix an overconfident classifier with `TemperatureScaling` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/classification/postprocessing/temperature_scaling.ipynb) |
| [Conformal prediction sets](classification/postprocessing/conformal_prediction_sets.ipynb) | Sets of classes that contain the true class 90% of the time | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-proba/blob/main/examples/classification/postprocessing/conformal_prediction_sets.ipynb) |

## Running locally

```bash
pip install "keras-proba[jax]" matplotlib jupyter
jupyter notebook examples/
```

Each notebook chooses its Keras backend in its first code cell: `"jax"`, `"tensorflow"` or
`"torch"`.
