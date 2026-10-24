"""Smoke tests for the classification heads. Run once per backend via `KERAS_BACKEND`.

Output shape, training logits and serialization tests are ported from TensorFlow Model Garden's `cls_head_test.py`.
"""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.classification_head import ClassificationHead, GaussianProcessClassificationHead
from keras_proba.layers.gaussian_process import RandomFeatureGaussianProcess, mean_field_logits
from keras_proba.layers.spectral_normalization import SpectralNormalization

SPEC_NORM_KWARGS = {"iteration": 1, "norm_multiplier": 1.0}
GP_LAYER_KWARGS = {"num_inducing": 512}


@pytest.mark.parametrize("inner_dim, num_weights_expected", [(0, 2), (10, 4)])
def test_pooler_layer(inner_dim, num_weights_expected):
    head = ClassificationHead(inner_dim=inner_dim, num_classes=2)
    features = np.zeros((2, 10, 10), "float32") if inner_dim else np.zeros((2, 10), "float32")
    head(features)
    assert len(head.weights) == num_weights_expected


@pytest.mark.parametrize(
    "use_gp_layer, return_covmat",
    [(True, True), (False, True)],
)
def test_sngp_output_shape(use_gp_layer, return_covmat):
    batch_size, num_classes = 32, 2
    head = GaussianProcessClassificationHead(
        inner_dim=5,
        num_classes=num_classes,
        use_spec_norm=True,
        use_gp_layer=use_gp_layer,
        **SPEC_NORM_KWARGS,
        **GP_LAYER_KWARGS,
    )
    outputs = head(np.zeros((batch_size, 10, 10), "float32"), return_covmat=return_covmat)

    if use_gp_layer and return_covmat:
        assert isinstance(outputs, tuple)
        assert ops.shape(outputs[0]) == (batch_size, num_classes)
        assert ops.shape(outputs[1]) == (batch_size, batch_size)
    else:
        assert ops.shape(outputs) == (batch_size, num_classes)


def test_sngp_train_logits():
    """Checks that the mean-field adjustment is disabled during training."""
    features = np.random.randn(5, 10, 10).astype("float32")
    # Without spectral normalization, so that the first training call's power
    # iteration does not change the kernel between the two calls.
    head = GaussianProcessClassificationHead(inner_dim=5, num_classes=2, use_spec_norm=False)

    head.temperature = None
    outputs_no_temp = ops.convert_to_numpy(head(features, training=True))
    head.temperature = 10.0
    outputs_with_temp = ops.convert_to_numpy(head(features, training=True))
    np.testing.assert_allclose(outputs_no_temp, outputs_with_temp, atol=1e-6)


def test_sngp_inference_logits_are_mean_field_adjusted():
    features = np.random.randn(6, 10).astype("float32") * 3
    head = GaussianProcessClassificationHead(inner_dim=0, num_classes=3, temperature=np.pi / 8)
    adjusted, covmat = head(features, training=False, return_covmat=True)

    head.temperature = None
    raw = head(features, training=False)
    np.testing.assert_allclose(
        ops.convert_to_numpy(adjusted),
        ops.convert_to_numpy(mean_field_logits(raw, covmat, np.pi / 8)),
        atol=1e-5,
    )


def test_sngp_sublayers():
    head = GaussianProcessClassificationHead(inner_dim=5, num_classes=2, use_spec_norm=True, use_gp_layer=True)
    head(np.zeros((2, 4, 3), "float32"))
    assert isinstance(head.dense, SpectralNormalization)
    assert head.dense.name == "pooler_dense"
    assert isinstance(head.out_proj, RandomFeatureGaussianProcess)
    # SNGP defaults from the reference head.
    assert head.dense.norm_multiplier == 0.99
    assert head.out_proj.normalize_input and not head.out_proj.scale_random_features

    plain = GaussianProcessClassificationHead(inner_dim=5, num_classes=2, use_spec_norm=False, use_gp_layer=False)
    assert isinstance(plain.dense, keras.layers.Dense)
    assert isinstance(plain.out_proj, keras.layers.Dense)


@pytest.mark.parametrize(
    "gp_kwargs, expect_covmat",
    [({"return_gp_cov": False}, False), ({"return_random_features": True}, True)],
)
def test_sngp_gp_output_variants(gp_kwargs, expect_covmat):
    head = GaussianProcessClassificationHead(inner_dim=4, num_classes=2, num_inducing=16, **gp_kwargs)
    outputs = head(np.zeros((3, 5), "float32"), training=False, return_covmat=True)
    if expect_covmat:
        assert ops.shape(outputs[0]) == (3, 2) and ops.shape(outputs[1]) == (3, 3)
    else:
        assert ops.shape(outputs) == (3, 2)


def test_reset_covariance_matrix():
    head = GaussianProcessClassificationHead(inner_dim=4, num_classes=2, num_inducing=16)
    head(np.random.randn(8, 5).astype("float32"), training=True)
    precision = head.out_proj._gp_cov_layer.precision_matrix
    assert not np.allclose(ops.convert_to_numpy(precision), np.eye(16))
    head.reset_covariance_matrix()
    np.testing.assert_allclose(ops.convert_to_numpy(precision), np.eye(16))

    # A no-op without a GP layer.
    GaussianProcessClassificationHead(inner_dim=4, num_classes=2, use_gp_layer=False).reset_covariance_matrix()


