"""Numerical parity tests comparing the Keras 3 variational layers against TensorFlow Probability's implementation.

The layers are stochastic, so the reference's random draws are recorded and fed
to the Keras 3 layer: the Normal noise is recovered from the sampled tensors the
reference keeps, and its Rademacher signs are recorded by wrapping
`tfp.random.rademacher`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "reference/probability"
if not REFERENCE_PATH.exists():
    pytest.skip(f"Reference implementation not found at {REFERENCE_PATH}", allow_module_level=True)
sys.path.insert(0, str(REFERENCE_PATH))
tfp_dense = pytest.importorskip("tensorflow_probability.python.layers.dense_variational")
tfp_conv = pytest.importorskip("tensorflow_probability.python.layers.conv_variational")
from tensorflow_probability.python.distributions import independent as tfd_independent  # noqa: E402
from tensorflow_probability.python.distributions import kullback_leibler as tfd_kl  # noqa: E402
from tensorflow_probability.python.distributions import normal as tfd_normal  # noqa: E402

import keras  # noqa: E402
from keras import ops  # noqa: E402

from keras_proba.layers.variational import (  # noqa: E402
    Conv2DFlipout,
    Conv2DReparameterization,
    DenseFlipout,
    DenseLocalReparameterization,
    DenseReparameterization,
)

PAIRS = {
    "reparameterization": (tfp_dense.DenseReparameterization, DenseReparameterization),
    "local_reparameterization": (tfp_dense.DenseLocalReparameterization, DenseLocalReparameterization),
    "flipout": (tfp_dense.DenseFlipout, DenseFlipout),
    "conv_reparameterization": (tfp_conv.Conv2DReparameterization, Conv2DReparameterization),
    "conv_flipout": (tfp_conv.Conv2DFlipout, Conv2DFlipout),
}


@pytest.fixture
def recorded_signs(monkeypatch):
    """Records the Rademacher signs drawn by the reference layers."""
    signs = []
    original = tfp_dense.tfp_random.rademacher

    def recorder(*args, **kwargs):
        result = original(*args, **kwargs)
        signs.append(result.numpy())
        return result

    monkeypatch.setattr(tfp_dense.tfp_random, "rademacher", recorder)
    return signs


def copy_weights(tf_layer, k_layer):
    by_name = {w.name.split("/")[-1].split(":")[0]: w.numpy() for w in tf_layer.weights}
    k_layer.kernel_posterior_loc.assign(by_name["kernel_posterior_loc"])
    k_layer.kernel_posterior_untransformed_scale.assign(by_name["kernel_posterior_untransformed_scale"])
    if k_layer.use_bias:
        k_layer.bias_posterior_loc.assign(by_name["bias_posterior_loc"])


def reference_normal_noise(tf_layer):
    """Recovers the standard Normal noise behind the reference's last sample."""
    posterior = tf_layer.kernel_posterior.distribution
    if tf_layer.kernel_posterior_tensor is not None:  # Reparameterization.
        return (tf_layer.kernel_posterior_tensor - posterior.loc) / posterior.scale
    affine = tf_layer.kernel_posterior_affine
    return (tf_layer.kernel_posterior_affine_tensor - affine.loc) / affine.scale


def inject_noise(k_layer, normal_noise, signs):
    """Makes the Keras 3 layer use the given noise instead of sampling."""
    normal_queue = [normal_noise]
    sign_queue = list(signs)
    k_layer._standard_normal = lambda shape, dtype: ops.cast(normal_queue.pop(0), dtype)
    # Conv signs are drawn as (batch, channels) by the reference, then broadcast spatially.
    k_layer._rademacher = lambda shape, dtype: ops.reshape(ops.cast(sign_queue.pop(0), dtype), shape)


def run_pair(kind, x, recorded_signs, build_shape=None, **kwargs):
    tf_cls, k_cls = PAIRS[kind]
    tf_layer = tf_cls(**kwargs)
    k_layer = k_cls(**kwargs)
    tf_layer.build(build_shape or x.shape)
    k_layer.build(build_shape or x.shape)
    copy_weights(tf_layer, k_layer)

    recorded_signs.clear()
    tf_out = tf_layer(x).numpy()
    inject_noise(k_layer, reference_normal_noise(tf_layer).numpy(), recorded_signs)
    k_out = ops.convert_to_numpy(k_layer(x))
    return tf_layer, k_layer, tf_out, k_out


DENSE_KINDS = ["reparameterization", "local_reparameterization", "flipout"]


@pytest.mark.parametrize("kind", DENSE_KINDS)
@pytest.mark.parametrize("activation", [None, "relu"])
def test_dense_forward_parity(kind, activation, recorded_signs):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 5)).astype("float32")
    tf_layer, k_layer, tf_out, k_out = run_pair(kind, x, recorded_signs, units=4, activation=activation)
    np.testing.assert_allclose(k_out, tf_out, rtol=1e-4, atol=1e-5)
    # Same posterior and divergence.
    np.testing.assert_allclose(
        ops.convert_to_numpy(k_layer.kernel_posterior_scale),
        tf_layer.kernel_posterior.distribution.scale.numpy(),
        rtol=1e-6,
    )
    assert len(tf_layer.losses) == len(k_layer.losses) == 1
    np.testing.assert_allclose(ops.convert_to_numpy(k_layer.losses[0]), tf_layer.losses[0].numpy(), rtol=1e-5)


def test_dense_flipout_higher_rank_parity(recorded_signs):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 5)).astype("float32")
    _, _, tf_out, k_out = run_pair("flipout", x, recorded_signs, build_shape=(None, 5), units=4)
    np.testing.assert_allclose(k_out, tf_out, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", DENSE_KINDS)
def test_dense_weights_parity(kind):
    tf_cls, k_cls = PAIRS[kind]
    tf_layer, k_layer = tf_cls(3), k_cls(3)
    tf_layer.build((None, 5))
    k_layer.build((None, 5))
    tf_weights = {w.name.split("/")[-1].split(":")[0]: tuple(w.shape) for w in tf_layer.trainable_weights}
    k_weights = {w.name: tuple(w.shape) for w in k_layer.trainable_weights}
    assert k_weights == tf_weights


def test_kl_weight_and_prior_scale_parity():
    """`kl_weight` and `kernel_prior_scale` match a custom divergence and prior in the reference."""
    weight, prior_scale = 0.01, 2.5

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        dist = tfd_normal.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(prior_scale))
        return tfd_independent.Independent(dist, reinterpreted_batch_ndims=len(shape))

    tf_layer = tfp_dense.DenseFlipout(
        4, kernel_prior_fn=prior_fn, kernel_divergence_fn=lambda q, p, _: weight * tfd_kl.kl_divergence(q, p)
    )
    k_layer = DenseFlipout(4, kernel_prior_scale=prior_scale, kl_weight=weight)
    x = np.random.default_rng(0).normal(size=(3, 6)).astype("float32")
    tf_layer(x)
    k_layer.build(x.shape)
    copy_weights(tf_layer, k_layer)
    k_layer(x)
    np.testing.assert_allclose(ops.convert_to_numpy(k_layer.losses[0]), tf_layer.losses[0].numpy(), rtol=1e-5)


CONV_CONFIGS = [
    {"filters": 3, "kernel_size": 3},
    {"filters": 3, "kernel_size": 3, "padding": "same"},
    {"filters": 4, "kernel_size": (3, 2), "strides": 2, "padding": "same"},
    {"filters": 2, "kernel_size": 3, "dilation_rate": 2},
    {"filters": 2, "kernel_size": 1, "activation": "relu"},
]


@pytest.mark.parametrize("kind", ["conv_reparameterization", "conv_flipout"])
@pytest.mark.parametrize("config", CONV_CONFIGS)
def test_conv_forward_parity(kind, config, recorded_signs):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 9, 8, 3)).astype("float32")
    tf_layer, k_layer, tf_out, k_out = run_pair(kind, x, recorded_signs, **config)
    assert k_out.shape == tf_out.shape == k_layer.compute_output_shape(x.shape)
    np.testing.assert_allclose(k_out, tf_out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ops.convert_to_numpy(k_layer.losses[0]), tf_layer.losses[0].numpy(), rtol=1e-5)


@pytest.mark.skipif(keras.backend.backend() != "tensorflow", reason="Uses tf.GradientTape on the Keras 3 layer.")
@pytest.mark.parametrize("kind", DENSE_KINDS + ["conv_reparameterization", "conv_flipout"])
def test_gradient_parity(kind, recorded_signs):
    """With the same noise, the loss gradients with respect to every posterior parameter match."""
    rng = np.random.default_rng(0)
    conv = kind.startswith("conv")
    x = rng.normal(size=(2, 6, 6, 2) if conv else (6, 5)).astype("float32")
    kwargs = {"filters": 3, "kernel_size": 3} if conv else {"units": 4}
    tf_cls, k_cls = PAIRS[kind]
    tf_layer, k_layer = tf_cls(**kwargs), k_cls(**kwargs)
    tf_layer.build(x.shape)
    k_layer.build(x.shape)
    copy_weights(tf_layer, k_layer)

    recorded_signs.clear()
    with tf.GradientTape() as tape:
        tf_loss = tf.reduce_sum(tf.square(tf_layer(x))) + tf.add_n(tf_layer.losses)
    tf_grads = dict(zip([w.name.split("/")[-1].split(":")[0] for w in tf_layer.trainable_weights],
                        tape.gradient(tf_loss, tf_layer.trainable_weights)))

    inject_noise(k_layer, tf.stop_gradient(reference_normal_noise(tf_layer)).numpy(), recorded_signs)
    with tf.GradientTape() as tape:
        k_loss = tf.reduce_sum(tf.square(k_layer(x))) + tf.add_n(k_layer.losses)
    k_grads = dict(zip([w.name for w in k_layer.trainable_weights], tape.gradient(k_loss, k_layer.trainable_weights)))

    np.testing.assert_allclose(k_loss.numpy(), tf_loss.numpy(), rtol=1e-5)
    assert set(k_grads) == set(tf_grads)
    for name in tf_grads:
        np.testing.assert_allclose(k_grads[name].numpy(), tf_grads[name].numpy(), rtol=1e-4, atol=1e-5, err_msg=name)
