"""Tests of the public API of `keras_proba`. Run once per backend via `KERAS_BACKEND`.

The expected names are listed explicitly, so that any change to the public API
is deliberate.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

import keras

import keras_proba

EXPECTED_API = {
    "layers": [
        "RandomFourierFeatures",
        "RandomFeatureGaussianProcess",
        "LaplaceRandomFeatureCovariance",
        "SpectralNormalization",
        "SpectralNormalizationConv2D",
        "ClassificationHead",
        "GaussianProcessClassificationHead",
        "DenseReparameterization",
        "DenseLocalReparameterization",
        "DenseFlipout",
        "Conv1DReparameterization",
        "Conv1DFlipout",
        "Conv2DReparameterization",
        "Conv2DFlipout",
        "Conv3DReparameterization",
        "Conv3DFlipout",
        "GaussianHead",
        "MixtureGaussianHead",
    ],
    "losses": ["GaussianNLL", "MixtureGaussianNLL", "gaussian_nll", "mixture_gaussian_nll"],
    "metrics": [
        "ExpectedCalibrationError",
        "BrierScore",
        "expected_calibration_error",
        "expected_calibration_error_quantiles",
        "brier_score",
        "brier_decomposition",
    ],
    "callbacks": ["ResetCovarianceCallback"],
    "postprocessing": [
        "TemperatureScaling",
        "fit_temperature",
        "temperature_scaled_nll",
        "SplitConformalRegressor",
        "SplitConformalClassifier",
        "conformal_quantile",
        "interval_coverage",
        "set_coverage",
        "average_set_size",
    ],
    "ops": [
        "mean_field_logits",
        "split_gaussian_params",
        "split_mixture_params",
        "mixture_mean_and_stddev",
        "normal_kl_divergence",
    ],
    "stats": ["percentile", "quantiles"],
}


def public_objects():
    for namespace, names in EXPECTED_API.items():
        module = getattr(keras_proba, namespace)
        for name in names:
            yield namespace, name, getattr(module, name)


def test_top_level_namespaces():
    assert sorted(keras_proba.__all__) == sorted(EXPECTED_API)
    for namespace in EXPECTED_API:
        assert inspect.ismodule(getattr(keras_proba, namespace))


@pytest.mark.parametrize("namespace", sorted(EXPECTED_API))
def test_namespace_exports(namespace):
    module = getattr(keras_proba, namespace)
    assert sorted(module.__all__) == sorted(EXPECTED_API[namespace])
    for name in module.__all__:
        assert hasattr(module, name), f"keras_proba.{namespace}.{name}"


def test_star_imports():
    for namespace in EXPECTED_API:
        scope = {}
        exec(f"from keras_proba.{namespace} import *", scope)
        assert set(EXPECTED_API[namespace]) <= set(scope)


def test_each_object_has_one_public_home():
    homes = {}
    for namespace, name, obj in public_objects():
        assert id(obj) not in homes, f"{name} is exported from both {homes[id(obj)]} and {namespace}"
        homes[id(obj)] = namespace


def test_public_objects_are_documented():
    for namespace, name, obj in public_objects():
        assert inspect.getdoc(obj), f"keras_proba.{namespace}.{name} has no docstring"
        assert "${" not in inspect.getdoc(obj), f"keras_proba.{namespace}.{name} has an unexpanded docstring"


def test_every_registered_object_is_public():
    """Every Keras-serializable class of the package is part of the public API, under its class name."""
    public = {obj for _, _, obj in public_objects()}
    registered = {key: obj for key, obj in keras.saving.get_custom_objects().items() if key.startswith("keras_proba>")}
    assert registered
    for key, obj in registered.items():
        assert obj in public, f"{key} is registered but not exported"
        assert key == f"keras_proba>{obj.__name__}"


def test_public_classes_are_serializable():
    for namespace, name, obj in public_objects():
        if inspect.isclass(obj) and issubclass(obj, (keras.layers.Layer, keras.losses.Loss, keras.metrics.Metric)):
            assert keras.saving.get_registered_name(obj) == f"keras_proba>{name}"


def test_version_matches_pyproject():
    tomllib = pytest.importorskip("tomllib") if sys.version_info >= (3, 11) else pytest.importorskip("tomli")
    with open(Path(__file__).resolve().parents[2] / "pyproject.toml", "rb") as f:
        assert keras_proba.__version__ == tomllib.load(f)["project"]["version"]


def test_end_to_end_with_public_api_only(tmp_path):
    """An SNGP classifier, calibrated and conformalized, using only public names."""
    from keras_proba import callbacks, layers, metrics, postprocessing

    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 4)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("int32")

    inputs = keras.Input((4,))
    hidden = layers.SpectralNormalization(keras.layers.Dense(16, activation="relu"), mode="reparameterization")(inputs)
    logits = layers.GaussianProcessClassificationHead(inner_dim=0, num_classes=2, num_inducing=64, temperature=1.0)(hidden)
    model = keras.Model(inputs, logits)
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics.ExpectedCalibrationError(from_logits=True), metrics.BrierScore(from_logits=True)],
    )
    model.fit(x, y, epochs=2, batch_size=32, verbose=0, callbacks=[callbacks.ResetCovarianceCallback()])

    scaling = postprocessing.TemperatureScaling()
    scaling.fit(model.predict(x[:128], verbose=0), y[:128])
    calibrated = keras.Sequential([model, scaling])
    probs = keras.ops.convert_to_numpy(keras.ops.softmax(calibrated.predict(x, verbose=0)))
    sets = postprocessing.SplitConformalClassifier(alpha=0.1).fit(y[:128], probs[:128]).predict_set(probs[128:])
    assert keras.ops.shape(sets) == (128, 2)

    path = tmp_path / "calibrated.keras"
    calibrated.save(path)
    np.testing.assert_allclose(keras.models.load_model(path).predict(x, verbose=0), calibrated.predict(x, verbose=0), atol=1e-5)
