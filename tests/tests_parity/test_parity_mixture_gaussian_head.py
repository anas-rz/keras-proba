"""Numerical parity tests comparing the mixture density head against TensorFlow Probability's `MixtureNormal`."""

import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "reference/probability"
if not REFERENCE_PATH.exists():
    pytest.skip(f"Reference implementation not found at {REFERENCE_PATH}", allow_module_level=True)
sys.path.insert(0, str(REFERENCE_PATH))
tfp_layers = pytest.importorskip("tensorflow_probability.python.layers.distribution_layer")
from tensorflow_probability.python.distributions import categorical as tfd_categorical  # noqa: E402
from tensorflow_probability.python.distributions import independent as tfd_independent  # noqa: E402
from tensorflow_probability.python.distributions import mixture_same_family as tfd_mixture  # noqa: E402
from tensorflow_probability.python.distributions import normal as tfd_normal  # noqa: E402

from keras import ops  # noqa: E402

from keras_proba.layers.mixture_gaussian_head import (  # noqa: E402
    MixtureGaussianHead,
    MixtureGaussianNLL,
    mixture_gaussian_nll,
    mixture_mean_and_stddev,
    split_mixture_params,
)


def random_params(rng, batch, k, d):
    logits = rng.normal(size=(batch, k)).astype("float32")
    means = rng.normal(size=(batch, k * d)).astype("float32") * 2
    scales = rng.uniform(0.2, 2.0, size=(batch, k * d)).astype("float32")
    return np.concatenate([logits, means, scales], axis=-1)


def tfd_mixture_from(params, d):
    logits, means, scales = (ops.convert_to_numpy(t) for t in split_mixture_params(params, d))
    components = tfd_independent.Independent(tfd_normal.Normal(loc=means, scale=scales), reinterpreted_batch_ndims=1)
    return tfd_mixture.MixtureSameFamily(tfd_categorical.Categorical(logits=logits), components)


def to_tfp_layout(params, k, d):
    """Our `[logits, means, scales]` as TFP's `[logits, (loc_d, raw_scale_d) per component]`."""
    logits = params[:, :k]
    means = params[:, k : k + k * d].reshape(-1, k, d)
    scales = params[:, k + k * d :].reshape(-1, k, d)
    raw_scales = np.log(np.expm1(scales.astype("float64"))).astype("float32")  # Inverse softplus.
    per_component = np.concatenate([means, raw_scales], axis=-1).reshape(-1, k * 2 * d)
    return np.concatenate([logits, per_component], axis=-1)


@pytest.mark.parametrize("k, d", [(1, 1), (3, 1), (5, 2), (4, 3)])
def test_log_prob_parity(k, d):
    rng = np.random.default_rng(k * 10 + d)
    params = random_params(rng, 32, k, d)
    y = rng.normal(size=(32, d)).astype("float32") * 2
    expected = -tfd_mixture_from(params, d).log_prob(y).numpy()
    np.testing.assert_allclose(ops.convert_to_numpy(mixture_gaussian_nll(y, params)), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ops.convert_to_numpy(MixtureGaussianNLL()(y, params)), expected.mean(), rtol=1e-5)


@pytest.mark.parametrize("k, d", [(3, 1), (5, 2)])
def test_tfp_mixture_normal_layer_parity(k, d):
    """TFP's `MixtureNormal` layer on our parameters, converted to its layout, is the same distribution."""
    rng = np.random.default_rng(0)
    params = random_params(rng, 16, k, d)
    y = rng.normal(size=(16, d)).astype("float32")
    tfp_params = to_tfp_layout(params, k, d)
    assert tfp_params.shape[-1] == tfp_layers.MixtureNormal.params_size(k, [d])
    dist = tfp_layers.MixtureNormal(k, [d])(tfp_params)
    np.testing.assert_allclose(
        ops.convert_to_numpy(mixture_gaussian_nll(y, params)), -dist.log_prob(y).numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("k, d", [(1, 1), (3, 1), (5, 2)])
def test_mean_and_stddev_parity(k, d):
    rng = np.random.default_rng(1)
    params = random_params(rng, 32, k, d)
    dist = tfd_mixture_from(params, d)
    mean, stddev = (ops.convert_to_numpy(t) for t in mixture_mean_and_stddev(params, d))
    np.testing.assert_allclose(mean, dist.mean().numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(stddev, dist.stddev().numpy(), rtol=1e-4, atol=1e-6)


def test_head_output_parity():
    """The head's output defines the same distribution as TFP's layer on the converted parameters."""
    rng = np.random.default_rng(0)
    head = MixtureGaussianHead(num_components=4, units=2, min_scale=0.0)
    x = rng.normal(size=(8, 5)).astype("float32")
    params = ops.convert_to_numpy(head(x))
    y = rng.normal(size=(8, 2)).astype("float32")
    dist = tfp_layers.MixtureNormal(4, [2])(to_tfp_layout(params, 4, 2))
    np.testing.assert_allclose(
        ops.convert_to_numpy(mixture_gaussian_nll(y, params)), -dist.log_prob(y).numpy(), rtol=1e-4, atol=1e-5
    )
