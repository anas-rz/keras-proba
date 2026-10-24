"""Tensor operations on the outputs and parameters of `keras_proba` layers.

    mean_field_logits: Uncertainty-adjusted logits from a Gaussian process output.
    split_gaussian_params: `(mean, scale)` of a `GaussianHead` output.
    split_mixture_params: `(logits, means, scales)` of a `MixtureGaussianHead` output.
    mixture_mean_and_stddev: Predictive mean and standard deviation of a mixture.
    normal_kl_divergence: KL divergence of a Normal posterior from a zero-mean Normal prior.
"""

from keras_proba.layers.gaussian_head import split_gaussian_params
from keras_proba.layers.gaussian_process import mean_field_logits
from keras_proba.layers.mixture_gaussian_head import mixture_mean_and_stddev, split_mixture_params
from keras_proba.layers.variational import normal_kl_divergence

__all__ = [
    "mean_field_logits",
    "split_gaussian_params",
    "split_mixture_params",
    "mixture_mean_and_stddev",
    "normal_kl_divergence",
]
