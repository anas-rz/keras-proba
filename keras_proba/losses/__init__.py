"""Losses for probabilistic regression heads.

`GaussianNLL` pairs with `keras_proba.layers.GaussianHead`, and
`MixtureGaussianNLL` with `keras_proba.layers.MixtureGaussianHead`.
"""

from keras_proba.layers.gaussian_head import GaussianNLL, gaussian_nll
from keras_proba.layers.mixture_gaussian_head import MixtureGaussianNLL, mixture_gaussian_nll

__all__ = [
    "GaussianNLL",
    "MixtureGaussianNLL",
    "gaussian_nll",
    "mixture_gaussian_nll",
]
