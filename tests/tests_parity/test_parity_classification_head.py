"""Numerical parity tests comparing the Keras 3 classification heads against TensorFlow Model Garden's implementation."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
tf_keras = pytest.importorskip("tf_keras")

import keras
from keras import ops

from keras_proba.layers.classification_head import ClassificationHead, GaussianProcessClassificationHead

LAYERS_DIR = Path(__file__).resolve().parents[2] / "reference/models/official/nlp/modeling/layers"
if not LAYERS_DIR.exists():
    pytest.skip(f"Reference implementation not found at {LAYERS_DIR}", allow_module_level=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _TFWithLegacyInitializers:
    """`tf` whose `initializers` are tf_keras's, so the reference does not run on the Keras 3 backend."""

    initializers = tf_keras.initializers

    def __getattr__(self, name):
        return getattr(tf, name)


def _load_reference_cls_head():
    """Loads `cls_head.py` and its two layer dependencies without importing the whole Model Garden package.

    `official.modeling.tf_utils` is replaced by a stub providing the two helpers `cls_head.py` uses.
    """
    stub_names = [
        "official",
        "official.modeling",
        "official.modeling.tf_utils",
        "official.nlp",
        "official.nlp.modeling",
        "official.nlp.modeling.layers",
    ]
    saved = {name: sys.modules.get(name) for name in stub_names}
    try:
        for name in stub_names:
            sys.modules[name] = types.ModuleType(name)
        tf_utils = sys.modules["official.modeling.tf_utils"]
        tf_utils.get_activation = tf_keras.activations.get
        tf_utils.clone_initializer = lambda init: (
            init.__class__.from_config(init.get_config())
            if isinstance(init, tf_keras.initializers.Initializer)
            else init
        )
        sys.modules["official.modeling"].tf_utils = tf_utils

        layers_pkg = sys.modules["official.nlp.modeling.layers"]
        for dep in ["gaussian_process", "spectral_normalization"]:
            module = _load(f"official.nlp.modeling.layers.{dep}", LAYERS_DIR / f"{dep}.py")
            setattr(layers_pkg, dep, module)
        layers_pkg.spectral_normalization.tf = _TFWithLegacyInitializers()
        return _load("official.nlp.modeling.layers.cls_head", LAYERS_DIR / "cls_head.py")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for dep in ["gaussian_process", "spectral_normalization", "cls_head"]:
            sys.modules.pop(f"official.nlp.modeling.layers.{dep}", None)


tfm_cls_head = _load_reference_cls_head()

GP_KWARGS = {"num_inducing": 64, "gp_cov_momentum": 0.5}


def copy_head_weights(tf_head, k_head):
    """Copies every reference weight into the Keras 3 head, sub-layer by sub-layer."""
    if tf_head.inner_dim:
        tf_dense, k_dense = tf_head.dense, k_head.dense
        if hasattr(tf_dense, "u"):
            k_dense.u.assign(tf_dense.u.numpy())
            k_dense.v.assign(tf_dense.v.numpy())
            tf_dense, k_dense = tf_dense.layer, k_dense.layer
        k_dense.set_weights(tf_dense.get_weights())

    tf_out, k_out = tf_head.out_proj, k_head.out_proj
    if not isinstance(tf_out, tfm_cls_head.gaussian_process.RandomFeatureGaussianProcess):
        k_out.set_weights(tf_out.get_weights())
        return
    pairs = [(k_out._random_feature, tf_out._random_feature), (k_out._gp_output_layer, tf_out._gp_output_layer)]
    if tf_out.normalize_input:
        pairs.append((k_out._input_norm_layer, tf_out._input_norm_layer))
    for k_sub, tf_sub in pairs:
        k_sub.set_weights(tf_sub.get_weights())
    k_out._gp_output_bias.assign(tf_out._gp_output_bias.numpy())
    if tf_out.return_gp_cov:
        k_out._gp_cov_layer.precision_matrix.assign(tf_out._gp_cov_layer.precision_matrix.numpy())


def build_pair(tf_cls, k_cls, features, **kwargs):
    tf_head = tf_cls(**kwargs)
    k_head = k_cls(**kwargs)
    # Build both with an inference call, which does not change any state.
    tf_head(features, training=False)
    k_head(features, training=False)
    copy_head_weights(tf_head, k_head)
    return tf_head, k_head


def assert_close(k_out, tf_out, atol=1e-4):
    np.testing.assert_allclose(ops.convert_to_numpy(k_out), tf_out.numpy(), atol=atol, rtol=1e-4)


@pytest.mark.parametrize("inner_dim", [0, 6])
def test_classification_head_parity(inner_dim):
    rng = np.random.default_rng(0)
    shape = (4, 5, 8) if inner_dim else (4, 8)
    x = rng.normal(size=shape).astype("float32")
    tf_head, k_head = build_pair(
        tfm_cls_head.ClassificationHead, ClassificationHead, x, inner_dim=inner_dim, num_classes=3, cls_token_idx=1
    )
    assert_close(k_head(x), tf_head(x))
    if inner_dim:
        assert_close(k_head(x, only_project=True), tf_head(x, only_project=True))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": np.pi / 8},
        {"temperature": 2.0, "use_spec_norm": False, "normalize_input": False, "scale_random_features": True},
        {"temperature": 1.0, "use_gp_layer": False, "norm_multiplier": 0.5, "iteration": 3},
    ],
)
def test_sngp_head_parity(kwargs):
    rng = np.random.default_rng(0)
    kwargs = {"inner_dim": 6, "num_classes": 3, **GP_KWARGS, **kwargs}
    x_train = rng.normal(size=(16, 5, 8)).astype("float32")
    x_test = rng.normal(size=(10, 5, 8)).astype("float32")
    tf_head, k_head = build_pair(
        tfm_cls_head.GaussianProcessClassificationHead, GaussianProcessClassificationHead, x_train, **kwargs
    )

    # Training: raw logits, and the same covariance update.
    assert_close(k_head(x_train, training=True), tf_head(x_train, training=True))
    if kwargs.get("use_gp_layer", True):
        np.testing.assert_allclose(
            ops.convert_to_numpy(k_head.out_proj._gp_cov_layer.precision_matrix),
            tf_head.out_proj._gp_cov_layer.precision_matrix.numpy(),
            atol=1e-4,
        )

    # Inference: mean-field logits and covariance.
    tf_outputs = tf_head(x_test, training=False, return_covmat=True)
    k_outputs = k_head(x_test, training=False, return_covmat=True)
    if kwargs.get("use_gp_layer", True):
        assert isinstance(k_outputs, tuple) and len(k_outputs) == 2
        for k_out, tf_out in zip(k_outputs, tf_outputs):
            assert_close(k_out, tf_out)
    else:
        assert_close(k_outputs, tf_outputs)


def test_sngp_head_rank2_features_parity():
    """With inner_dim=0, both heads take rank-2 features directly."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 8)).astype("float32")
    tf_head, k_head = build_pair(
        tfm_cls_head.GaussianProcessClassificationHead,
        GaussianProcessClassificationHead,
        x,
        inner_dim=0,
        num_classes=2,
        temperature=1.0,
        **GP_KWARGS,
    )
    assert_close(k_head(x, training=False), tf_head(x, training=False))


def test_config_parity():
    kwargs = {"inner_dim": 5, "num_classes": 2, "norm_multiplier": 1.0, "num_inducing": 512, "temperature": 0.5}
    tf_config = tfm_cls_head.GaussianProcessClassificationHead(**kwargs).get_config()
    k_config = GaussianProcessClassificationHead(**kwargs).get_config()
    # Every reference key is present with the same value (activation/initializer are serialized differently).
    for key, value in tf_config.items():
        if key in ("activation", "initializer", "dtype", "name"):
            continue
        assert k_config[key] == value, key
    # The only extra key is the spectral normalization mode.
    assert set(k_config) - set(tf_config) <= {"mode"}
