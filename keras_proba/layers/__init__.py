"""Uncertainty-aware Keras 3 layers.

Distance-aware (SNGP) layers:
    RandomFourierFeatures, RandomFeatureGaussianProcess, LaplaceRandomFeatureCovariance,
    SpectralNormalization, SpectralNormalizationConv2D,
    ClassificationHead, GaussianProcessClassificationHead.

Bayesian (variational) layers:
    DenseReparameterization, DenseLocalReparameterization, DenseFlipout,
    Conv{1,2,3}DReparameterization, Conv{1,2,3}DFlipout.

Probabilistic regression heads:
    GaussianHead, MixtureGaussianHead.
"""

from keras_proba.layers.classification_head import ClassificationHead, GaussianProcessClassificationHead
from keras_proba.layers.gaussian_head import GaussianHead
from keras_proba.layers.gaussian_process import LaplaceRandomFeatureCovariance, RandomFeatureGaussianProcess
from keras_proba.layers.mixture_gaussian_head import MixtureGaussianHead
from keras_proba.layers.random_fourier_features import RandomFourierFeatures
from keras_proba.layers.spectral_normalization import SpectralNormalization, SpectralNormalizationConv2D
from keras_proba.layers.variational import (
    Conv1DFlipout,
    Conv1DReparameterization,
    Conv2DFlipout,
    Conv2DReparameterization,
    Conv3DFlipout,
    Conv3DReparameterization,
    DenseFlipout,
    DenseLocalReparameterization,
    DenseReparameterization,
)

__all__ = [
    # Distance-aware (SNGP).
    "RandomFourierFeatures",
    "RandomFeatureGaussianProcess",
    "LaplaceRandomFeatureCovariance",
    "SpectralNormalization",
    "SpectralNormalizationConv2D",
    "ClassificationHead",
    "GaussianProcessClassificationHead",
    # Bayesian (variational).
    "DenseReparameterization",
    "DenseLocalReparameterization",
    "DenseFlipout",
    "Conv1DReparameterization",
    "Conv1DFlipout",
    "Conv2DReparameterization",
    "Conv2DFlipout",
    "Conv3DReparameterization",
    "Conv3DFlipout",
    # Probabilistic regression heads.
    "GaussianHead",
    "MixtureGaussianHead",
]
