"""Smoke and correctness tests for MixtureGaussianHead and MixtureGaussianNLL. Run once per backend via `KERAS_BACKEND`."""

import math

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.gaussian_head import gaussian_nll
from keras_proba.layers.mixture_gaussian_head import (
    MixtureGaussianHead,
    MixtureGaussianNLL,
    mixture_gaussian_nll,
    mixture_mean_and_stddev,
    split_mixture_params,
)


def to_numpy(x):
    return ops.convert_to_numpy(x)


def test_output_shape_and_split():
    k, d = 5, 2
    head = MixtureGaussianHead(k, units=d, min_scale=0.05)
    out = head(np.random.randn(6, 4).astype("float32") * 10)
    assert ops.shape(out) == (6, k * (1 + 2 * d))
    assert head.compute_output_shape((None, 4)) == (None, k * (1 + 2 * d))
    logits, means, scales = (to_numpy(t) for t in split_mixture_params(out, d))
    assert logits.shape == (6, k) and means.shape == scales.shape == (6, k, d)
    assert np.all(scales >= 0.05)


def test_single_component_matches_gaussian_nll():
    """With one component and one unit, the mixture NLL is the Gaussian NLL."""
    rng = np.random.default_rng(0)
    mean = rng.normal(size=(10, 1)).astype("float32")
    scale = rng.uniform(0.5, 2.0, size=(10, 1)).astype("float32")
    y = rng.normal(size=(10, 1)).astype("float32")
    mixture = np.concatenate([np.zeros((10, 1), "float32"), mean, scale], -1)
    np.testing.assert_allclose(
        to_numpy(mixture_gaussian_nll(y, mixture)),
        to_numpy(gaussian_nll(y, np.concatenate([mean, scale], -1))),
        rtol=1e-5,
    )


def test_nll_values():
    # Two equally weighted standard normals at -1 and 1, evaluated at 0.
    params = np.array([[0.0, 0.0, -1.0, 1.0, 1.0, 1.0]], "float32")
    expected = 0.5 * math.log(2 * math.pi) + 0.5
    np.testing.assert_allclose(to_numpy(mixture_gaussian_nll(np.zeros((1,)), params)), [expected], rtol=1e-6)
    # Units are summed: two independent unit-variance dims at their means.
    params = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]], "float32")
    np.testing.assert_allclose(
        to_numpy(mixture_gaussian_nll(np.zeros((1, 2)), params, units=2)), [math.log(2 * math.pi)], rtol=1e-6
    )


def test_rank_one_targets():
    params = np.random.default_rng(0).normal(size=(4, 9)).astype("float32")
    params[:, 6:] = np.abs(params[:, 6:]) + 0.1
    np.testing.assert_allclose(
        to_numpy(mixture_gaussian_nll(np.arange(4.0), params)),
        to_numpy(mixture_gaussian_nll(np.arange(4.0)[:, None], params)),
    )


def test_numerically_stable_far_from_components():
    """The log-sum-exp formulation keeps the NLL finite where component densities underflow."""
    params = np.array([[0.0, 0.0, -1.0, 1.0, 0.01, 0.01]], "float32")
    nll = to_numpy(mixture_gaussian_nll(np.array([[50.0]]), params))
    assert np.all(np.isfinite(nll)) and nll[0] > 1e6


def test_mean_and_stddev_of_two_point_mixture():
    # 0.5 N(-2, 1) + 0.5 N(2, 1): mean 0, variance 1 + 4.
    params = np.array([[0.0, 0.0, -2.0, 2.0, 1.0, 1.0]], "float32")
    mean, stddev = (to_numpy(t) for t in mixture_mean_and_stddev(params))
    np.testing.assert_allclose(mean, [[0.0]], atol=1e-6)
    np.testing.assert_allclose(stddev, [[np.sqrt(5.0)]], rtol=1e-6)


def test_invalid_args():
    for kwargs in [{"num_components": 0}, {"num_components": 2, "units": 0}, {"num_components": 2, "min_scale": -1.0}]:
        with pytest.raises(ValueError):
            MixtureGaussianHead(**kwargs)


def test_mismatched_params_size():
    with pytest.raises(ValueError):
        split_mixture_params(np.zeros((2, 7), "float32"), units=2)


