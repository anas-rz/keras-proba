"""Serialization and reload tests for the classification heads. Run once per backend via `KERAS_BACKEND`.

Config tests are ported from TensorFlow Model Garden's `cls_head_test.py`.
"""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.classification_head import ClassificationHead, GaussianProcessClassificationHead

SPEC_NORM_KWARGS = {"iteration": 1, "norm_multiplier": 1.0}
GP_LAYER_KWARGS = {"num_inducing": 512}


def test_classification_head_serialization():
    layer = ClassificationHead(inner_dim=5, num_classes=2, activation="relu", dropout_rate=0.1)
    new_layer = ClassificationHead.from_config(layer.get_config())
    assert new_layer.get_config() == layer.get_config()


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {},
        {"temperature": 0.5, "mode": "reparameterization", "custom_random_features_activation": "relu"},
        {"custom_random_features_initializer": keras.initializers.RandomNormal(stddev=0.5, seed=1)},
    ],
)
def test_layer_serialization(extra_kwargs):
    layer = GaussianProcessClassificationHead(
        inner_dim=5,
        num_classes=2,
        use_spec_norm=True,
        use_gp_layer=True,
        **SPEC_NORM_KWARGS,
        **GP_LAYER_KWARGS,
        **extra_kwargs,
    )
    new_layer = GaussianProcessClassificationHead.from_config(layer.get_config())
    assert new_layer.get_config() == layer.get_config()


def test_sngp_kwargs_serialization():
    """Tests that SNGP-specific kwargs are added during serialization."""
    layer = GaussianProcessClassificationHead(
        inner_dim=5, num_classes=2, use_spec_norm=True, use_gp_layer=True, **SPEC_NORM_KWARGS, **GP_LAYER_KWARGS
    )
    config = layer.get_config()
    assert config["norm_multiplier"] == 1.0
    assert config["num_inducing"] == 512


@pytest.mark.parametrize("mode", ["projection", "reparameterization"])
def test_model_save_and_load(tmp_path, mode):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((6, 4))
    head = GaussianProcessClassificationHead(inner_dim=8, num_classes=3, num_inducing=32, temperature=1.0, mode=mode)
    model = keras.Model(inputs, head(inputs))
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    x = np.random.randn(32, 6, 4).astype("float32")
    model.fit(x, np.random.randint(0, 3, 32), epochs=1, batch_size=8, verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    loaded_head = loaded.layers[-1]
    assert isinstance(loaded_head, GaussianProcessClassificationHead)
    np.testing.assert_allclose(loaded.predict(x, batch_size=32, verbose=0), model.predict(x, batch_size=32, verbose=0), atol=1e-5)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(loaded_head.out_proj._gp_cov_layer.precision_matrix),
        ops.convert_to_numpy(head.out_proj._gp_cov_layer.precision_matrix),
    )
