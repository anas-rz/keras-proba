"""Smoke and correctness tests for the Gaussian process layer. Run once per backend via `KERAS_BACKEND`.

The statistical tests are ported from TensorFlow Model Garden's `gaussian_process_test.py`.
"""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.gaussian_process import (
    LaplaceRandomFeatureCovariance,
    RandomFeatureGaussianProcess,
    mean_field_logits,
)

NUM_DATA_DIM = 10
NUM_TRAIN_SAMPLE = 1024
NUM_TEST_SAMPLE = 256
PREC_TOLERANCE = {"atol": 1e-3, "rtol": 5e-2}
COV_TOLERANCE = {"atol": 5e-2, "rtol": 2.0}


def exact_gaussian_kernel(x1, x2):
    """Computes the exact Gaussian kernel exp(-|x1 - x2|^2 / 2)."""
    square = np.sum(x1**2, -1)[:, None] + np.sum(x2**2, -1)[None, :] - 2 * x1 @ x2.T
    return np.exp(-square / 2.0)


def generate_normal_data(rng, num_sample, num_dim, loc):
    return rng.normal(size=(num_sample, num_dim), loc=loc, scale=1.0 / np.sqrt(num_dim)).astype("float32")


def generate_rbf_data(x_data, orthogonal=True):
    """Generates high-dim data that is the eigen components of a RBF kernel."""
    k_rbf = exact_gaussian_kernel(x_data, x_data)
    x_orth, x_diag, _ = np.linalg.svd(k_rbf)
    if orthogonal:
        return x_orth
    return np.diag(np.sqrt(x_diag)).dot(x_orth.T)


def compute_posterior_kernel(x_tr, x_ts, ridge_penalty):
    """Computes the posterior covariance matrix of an exact Gaussian process."""
    k_tt_inv = np.linalg.inv(exact_gaussian_kernel(x_tr, x_tr) + ridge_penalty * np.eye(len(x_tr)))
    k_ts = exact_gaussian_kernel(x_tr, x_ts)
    k_ss = exact_gaussian_kernel(x_ts, x_ts)
    return k_ss - k_ts.T @ k_tt_inv @ k_ts


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x_tr = generate_normal_data(rng, NUM_TRAIN_SAMPLE, NUM_DATA_DIM, loc=0.0)
    x_ts = generate_normal_data(rng, NUM_TEST_SAMPLE, NUM_DATA_DIM, loc=1.0)
    return x_tr, x_ts


def test_layer_build(data):
    x_tr, _ = data
    layer = RandomFeatureGaussianProcess(units=1)
    layer.build(input_shape=x_tr.shape)
    assert layer.built


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, [(None, 4), (None, None)]),
        ({"return_gp_cov": False}, (None, 4)),
        ({"return_random_features": True}, [(None, 4), (None, None), (None, 32)]),
        ({"return_gp_cov": False, "return_random_features": True}, [(None, 4), (None, 32)]),
        (
            {"gp_kernel_type": "linear", "return_gp_cov": False, "return_random_features": True},
            [(None, 4), (None, 6)],
        ),
        ({"use_custom_random_features": False, "return_random_features": True}, [(None, 4), (None, None), (None, 32)]),
    ],
)
def test_output_shapes(kwargs, expected):
    layer = RandomFeatureGaussianProcess(units=4, num_inducing=32, **kwargs)
    inputs = keras.Input(shape=(6,))
    model = keras.Model(inputs, layer(inputs))
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = [tuple(s) for s in output_shape]
    assert output_shape == expected
    assert layer.compute_output_shape((None, 6)) == (tuple(expected) if isinstance(expected, list) else expected)

    x = np.random.randn(5, 6).astype("float32")
    for training in [True, False]:
        outputs = layer(x, training=training)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        expected_list = expected if isinstance(expected, list) else [expected]
        for out, shape in zip(outputs, expected_list):
            assert ops.shape(out) == tuple(5 if d is None else d for d in shape)


@pytest.mark.parametrize("orthogonal", [False, True])
def test_laplace_covariance_minibatch(data, orthogonal):
    """Tests if the layer correctly learns the population-level precision matrix."""
    _, x_ts = data
    batch_size, epochs = 50, 1000
    x_data = generate_rbf_data(x_ts, orthogonal).astype("float32")

    cov_estimator = LaplaceRandomFeatureCovariance(momentum=0.999, ridge_penalty=0)
    # Equivalent to `dataset.repeat(epochs).batch(batch_size)`: batches span epoch boundaries.
    num_rows = len(x_data) * epochs
    for start in range(0, num_rows, batch_size):
        indices = np.arange(start, min(start + batch_size, num_rows)) % len(x_data)
        cov_estimator(x_data[indices], training=True)

    prec_mat_expected = x_data.T.dot(x_data)
    prec_mat_computed = ops.convert_to_numpy(cov_estimator.precision_matrix) * NUM_TEST_SAMPLE
    np.testing.assert_allclose(prec_mat_computed, prec_mat_expected, **PREC_TOLERANCE)


def test_random_feature_prior_approximation(data):
    """Tests the random feature GP's ability to approximate the exact GP prior."""
    x_tr, _ = data
    keras.utils.set_random_seed(0)
    layer = RandomFeatureGaussianProcess(
        units=1,
        num_inducing=10240,
        normalize_input=False,
        gp_kernel_type="gaussian",
        return_random_features=True,
    )
    _, _, gp_feature = layer(x_tr, training=True)
    gp_feature = ops.convert_to_numpy(gp_feature)

    prior_kernel_computed = gp_feature.dot(gp_feature.T)
    prior_kernel_expected = exact_gaussian_kernel(x_tr, x_tr)
    np.testing.assert_allclose(prior_kernel_computed, prior_kernel_expected, **COV_TOLERANCE)


def test_random_feature_posterior_approximation(data):
    """Tests the random feature GP's ability to approximate the exact GP posterior."""
    x_tr, x_ts = data
    keras.utils.set_random_seed(0)
    # momentum = 0.5 makes the posterior precision matrix 0.5 * (I + K).
    gp_cov_momentum = 0.5
    gp_cov_ridge_penalty = 1.0
    layer = RandomFeatureGaussianProcess(
        units=1,
        num_inducing=1024,
        normalize_input=False,
        gp_kernel_type="gaussian",
        gp_cov_momentum=gp_cov_momentum,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
    )
    layer(x_tr, training=True)
    _, gp_cov_ts = layer(x_ts, training=False)

    # Scale up the covariance since the precision matrix is down-scaled by momentum.
    post_kernel_computed = ops.convert_to_numpy(gp_cov_ts) * gp_cov_momentum
    post_kernel_expected = compute_posterior_kernel(x_tr, x_ts, gp_cov_ridge_penalty)
    np.testing.assert_allclose(post_kernel_computed, post_kernel_expected, **COV_TOLERANCE)


def test_random_feature_linear_kernel(data):
    """Tests if the linear kernel leads to an identity mapping."""
    x_tr, _ = data
    layer = RandomFeatureGaussianProcess(
        units=1,
        normalize_input=False,
        gp_kernel_type="linear",
        scale_random_features=False,
        use_custom_random_features=True,
        return_random_features=True,
    )
    _, _, gp_feature = layer(x_tr, training=True)
    np.testing.assert_allclose(ops.convert_to_numpy(gp_feature), x_tr, **PREC_TOLERANCE)


def test_no_matrix_update_during_test(data):
    """Tests that the precision matrix is not updated during inference."""
    x_tr, x_ts = data
    layer = RandomFeatureGaussianProcess(units=1)

    _, gp_covmat_null = layer(x_tr, training=True)
    precision_before = ops.convert_to_numpy(layer._gp_cov_layer.precision_matrix).copy()
    layer(x_ts, training=False)
    layer(x_ts)
    precision_after = ops.convert_to_numpy(layer._gp_cov_layer.precision_matrix)

    np.testing.assert_allclose(ops.convert_to_numpy(gp_covmat_null), np.eye(NUM_TRAIN_SAMPLE), atol=1e-4)
    np.testing.assert_allclose(precision_before, precision_after, atol=1e-4)


def test_reset_covariance_matrix(data):
    x_tr, _ = data
    layer = RandomFeatureGaussianProcess(units=1, num_inducing=32, gp_cov_ridge_penalty=2.0)
    layer(x_tr, training=True)
    assert not np.allclose(ops.convert_to_numpy(layer._gp_cov_layer.precision_matrix), 2.0 * np.eye(32))
    layer.reset_covariance_matrix()
    np.testing.assert_allclose(ops.convert_to_numpy(layer._gp_cov_layer.precision_matrix), 2.0 * np.eye(32))


def test_predictive_variance_grows_away_from_data(data):
    """The GP should be more uncertain on out-of-distribution inputs."""
    x_tr, _ = data
    keras.utils.set_random_seed(0)
    layer = RandomFeatureGaussianProcess(units=1, num_inducing=512, gp_cov_momentum=-1)
    layer(x_tr, training=True)
    _, cov_in = layer(x_tr[:64], training=False)
    _, cov_out = layer(x_tr[:64] + 5.0, training=False)
    var_in = np.diag(ops.convert_to_numpy(cov_in))
    var_out = np.diag(ops.convert_to_numpy(cov_out))
    assert var_out.mean() > 5 * var_in.mean()


def test_custom_random_features():
    layer = RandomFeatureGaussianProcess(
        units=2,
        num_inducing=16,
        custom_random_features_initializer="glorot_uniform",
        custom_random_features_activation="relu",
        return_gp_cov=False,
        return_random_features=True,
        scale_random_features=False,
    )
    _, features = layer(np.random.randn(4, 3).astype("float32"))
    assert np.all(ops.convert_to_numpy(features) >= 0)


def test_trainable_weights():
    layer = RandomFeatureGaussianProcess(units=2, num_inducing=16)
    layer.build((None, 3))
    assert [w.name for w in layer.trainable_weights] == ["kernel"]
    assert layer.trainable_weights[0] is layer._gp_output_layer.kernel

    layer = RandomFeatureGaussianProcess(
        units=2,
        num_inducing=16,
        use_custom_random_features=False,
        gp_kernel_scale_trainable=True,
        gp_output_bias_trainable=True,
        normalize_input=True,
    )
    layer.build((None, 3))
    names = sorted(w.name for w in layer.trainable_weights)
    assert names == ["beta", "gamma", "gp_output_bias", "kernel", "kernel_scale"]


def test_l2_regularization_loss():
    layer = RandomFeatureGaussianProcess(units=2, num_inducing=16, l2_regularization=0.1)
    layer(np.random.randn(4, 3).astype("float32"))
    kernel = ops.convert_to_numpy(layer._gp_output_layer.kernel)
    assert len(layer.losses) == 1
    np.testing.assert_allclose(ops.convert_to_numpy(layer.losses[0]), 0.1 * np.sum(kernel**2), rtol=1e-5)


def test_gp_output_kwargs():
    layer = RandomFeatureGaussianProcess(units=2, num_inducing=8, kernel_initializer="zeros", gp_output_bias=1.5)
    logits, _ = layer(np.random.randn(4, 3).astype("float32"))
    np.testing.assert_allclose(ops.convert_to_numpy(logits), 1.5 * np.ones((4, 2)))


def test_invalid_likelihood():
    with pytest.raises(ValueError):
        LaplaceRandomFeatureCovariance(likelihood="bernoulli")
    layer = LaplaceRandomFeatureCovariance(likelihood="binary_logistic")
    with pytest.raises(ValueError):
        layer(np.ones((4, 3), "float32"), logits=None, training=True)


def test_mean_field_logits_likelihood():
    """Tests if the scaling is correct under different likelihoods."""
    batch_size, num_classes, variance, mean_field_factor = 10, 12, 1.5, 2.0
    logits = np.random.default_rng(0).normal(size=(batch_size, num_classes)).astype("float32")
    covmat = np.diag([variance] * batch_size).astype("float32")
    adjusted = mean_field_logits(logits, covmat, mean_field_factor=mean_field_factor)
    np.testing.assert_allclose(ops.convert_to_numpy(adjusted), logits / 2.0, atol=1e-4)


def test_mean_field_logits_temperature_scaling():
    """Tests using mean_field_logits as temperature scaling."""
    logits = np.random.default_rng(0).normal(size=(10, 12)).astype("float32")
    # No change when mean_field_factor < 0.
    unchanged = mean_field_logits(logits, covariance_matrix=None, mean_field_factor=-1)
    # Temperature = sqrt(1 + mean_field_factor) when no covariance is given.
    halved = mean_field_logits(logits, covariance_matrix=None, mean_field_factor=3.0)
    np.testing.assert_allclose(ops.convert_to_numpy(unchanged), logits, atol=1e-4)
    np.testing.assert_allclose(ops.convert_to_numpy(halved), logits / 2.0, atol=1e-4)
