"""Numerical parity tests comparing the Keras 3 Gaussian process layer against TensorFlow Model Garden's implementation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("tf_keras")

from keras import ops

from keras_proba.layers.gaussian_process import (
    LaplaceRandomFeatureCovariance,
    RandomFeatureGaussianProcess,
    mean_field_logits,
)

REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "reference/models/official/nlp/modeling/layers/gaussian_process.py"
)
if not REFERENCE_PATH.exists():
    pytest.skip(f"Reference implementation not found at {REFERENCE_PATH}", allow_module_level=True)

# Load the single module by path to avoid importing the whole Model Garden package.
_spec = importlib.util.spec_from_file_location("tfm_gaussian_process", REFERENCE_PATH)
tfm_gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfm_gp)


def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") and not hasattr(x, "detach") else ops.convert_to_numpy(x)


def build_pair(input_dim, **kwargs):
    """Builds both GP layers and copies the reference weights into the Keras 3 layer."""
    tf_layer = tfm_gp.RandomFeatureGaussianProcess(**kwargs)
    tf_layer.build((None, input_dim))
    k_layer = RandomFeatureGaussianProcess(**kwargs)
    k_layer.build((None, input_dim))

    pairs = [
        (k_layer._random_feature, tf_layer._random_feature),
        (k_layer._gp_output_layer, tf_layer._gp_output_layer),
    ]
    if kwargs.get("normalize_input"):
        pairs.append((k_layer._input_norm_layer, tf_layer._input_norm_layer))
    for k_sub, tf_sub in pairs:
        tf_weights = tf_sub.get_weights()
        assert len(k_sub.weights) == len(tf_weights)
        k_sub.set_weights(tf_weights)
    k_layer._gp_output_bias.assign(tf_layer._gp_output_bias.numpy())
    return tf_layer, k_layer


def as_list(outputs):
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


def assert_outputs_close(tf_outputs, k_outputs, atol=1e-4, rtol=1e-4):
    tf_outputs, k_outputs = as_list(tf_outputs), as_list(k_outputs)
    assert len(tf_outputs) == len(k_outputs)
    for tf_out, k_out in zip(tf_outputs, k_outputs):
        np.testing.assert_allclose(to_numpy(k_out), tf_out.numpy(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"normalize_input": True},
        {"gp_kernel_scale": 4.0, "gp_output_bias": 0.5},
        {"scale_random_features": False},
        {"use_custom_random_features": False},
        {"use_custom_random_features": False, "gp_kernel_scale": 2.0},
        {"gp_kernel_type": "linear", "scale_random_features": False},
        {"return_gp_cov": False},
        {"return_random_features": True},
        {"return_gp_cov": False, "return_random_features": True},
    ],
)
def test_gp_forward_parity(kwargs):
    np.random.seed(0)
    kwargs = {"units": 3, "num_inducing": 64, **kwargs}
    tf_layer, k_layer = build_pair(8, **kwargs)
    x_tr = np.random.randn(16, 8).astype("float32")
    x_ts = np.random.randn(10, 8).astype("float32")

    # Training: logits, identity covariance, and features match.
    assert_outputs_close(tf_layer(x_tr, training=True), k_layer(x_tr, training=True))
    if kwargs.get("return_gp_cov", True):
        np.testing.assert_allclose(
            to_numpy(k_layer._gp_cov_layer.precision_matrix),
            tf_layer._gp_cov_layer.precision_matrix.numpy(),
            atol=1e-5,
        )
    # Inference: logits and predictive covariance match.
    assert_outputs_close(tf_layer(x_ts, training=False), k_layer(x_ts, training=False))


@pytest.mark.parametrize("momentum", [0.999, 0.5, -1.0])
@pytest.mark.parametrize("likelihood", ["gaussian", "binary_logistic", "poisson"])
def test_covariance_update_parity(momentum, likelihood):
    rng = np.random.default_rng(0)
    kwargs = {"momentum": momentum, "ridge_penalty": 0.5, "likelihood": likelihood}
    tf_layer = tfm_gp.LaplaceRandomFeatureCovariance(**kwargs)
    k_layer = LaplaceRandomFeatureCovariance(**kwargs)

    for _ in range(5):
        features = rng.normal(size=(12, 6)).astype("float32")
        logits = rng.normal(size=(12, 1)).astype("float32")
        tf_null = tf_layer(features, logits=logits, training=True)
        k_null = k_layer(features, logits=logits, training=True)
        np.testing.assert_allclose(to_numpy(k_null), tf_null.numpy())

    np.testing.assert_allclose(
        to_numpy(k_layer.precision_matrix), tf_layer.precision_matrix.numpy(), rtol=1e-5, atol=1e-5
    )

    features = rng.normal(size=(7, 6)).astype("float32")
    np.testing.assert_allclose(
        to_numpy(k_layer(features, training=False)),
        tf_layer(features, training=False).numpy(),
        rtol=1e-4,
        atol=1e-4,
    )


def test_reset_covariance_parity():
    tf_layer, k_layer = build_pair(4, units=1, num_inducing=16)
    x = np.random.randn(8, 4).astype("float32")
    tf_layer(x, training=True)
    k_layer(x, training=True)
    tf_layer.reset_covariance_matrix()
    k_layer.reset_covariance_matrix()
    np.testing.assert_allclose(
        to_numpy(k_layer._gp_cov_layer.precision_matrix),
        tf_layer._gp_cov_layer.precision_matrix.numpy(),
    )


def test_invalid_likelihood_parity():
    with pytest.raises(ValueError):
        tfm_gp.LaplaceRandomFeatureCovariance(likelihood="bernoulli")
    with pytest.raises(ValueError):
        LaplaceRandomFeatureCovariance(likelihood="bernoulli")

    features = np.ones((4, 3), "float32")
    for layer_cls in [tfm_gp.LaplaceRandomFeatureCovariance, LaplaceRandomFeatureCovariance]:
        layer = layer_cls(likelihood="binary_logistic")
        with pytest.raises(ValueError):
            layer(features, logits=None, training=True)
        layer = layer_cls(likelihood="poisson")
        with pytest.raises(ValueError):
            layer(features, logits=np.ones((4, 2), "float32"), training=True)


@pytest.mark.parametrize("mean_field_factor", [None, -1.0, 0.0, 1.0, np.pi / 8, 3.0])
@pytest.mark.parametrize("use_covariance", [False, True])
def test_mean_field_logits_parity(mean_field_factor, use_covariance):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(10, 4)).astype("float32")
    covmat = None
    if use_covariance:
        a = rng.normal(size=(10, 10)).astype("float32")
        covmat = a @ a.T

    expected = tfm_gp.mean_field_logits(logits, covmat, mean_field_factor=mean_field_factor)
    actual = mean_field_logits(logits, covmat, mean_field_factor=mean_field_factor)
    np.testing.assert_allclose(to_numpy(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_mean_field_logits_1d_parity():
    logits = np.linspace(-2, 2, 6).astype("float32")
    covmat = np.diag(np.linspace(0.1, 2.0, 6)).astype("float32")
    expected = tfm_gp.mean_field_logits(logits, covmat, mean_field_factor=2.0)
    actual = mean_field_logits(logits, covmat, mean_field_factor=2.0)
    np.testing.assert_allclose(to_numpy(actual), expected.numpy(), rtol=1e-5)
