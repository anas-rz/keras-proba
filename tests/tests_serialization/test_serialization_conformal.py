"""Serialization tests for split conformal prediction. Run once per backend via `KERAS_BACKEND`."""

import json

import numpy as np

from keras import ops

from keras_proba.postprocessing.conformal import SplitConformalClassifier, SplitConformalRegressor


def test_regressor_config_round_trip():
    rng = np.random.default_rng(0)
    y_pred, scale = rng.normal(size=(100, 2)).astype("float32"), rng.uniform(0.5, 1.5, size=(100, 2)).astype("float32")
    conformal = SplitConformalRegressor(0.1).fit(rng.normal(size=(100, 2)), y_pred, scale=scale)
    config = json.loads(json.dumps(conformal.get_config()))
    restored = SplitConformalRegressor.from_config(config)
    for original, loaded in zip(conformal.predict_interval(y_pred, scale=scale), restored.predict_interval(y_pred, scale=scale)):
        np.testing.assert_allclose(ops.convert_to_numpy(loaded), ops.convert_to_numpy(original), rtol=1e-6)


def test_classifier_config_round_trip():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 4)).astype("float32")
    # Non-randomized, so that the sets of both are deterministic.
    conformal = SplitConformalClassifier(0.2, score="aps", from_logits=True, randomized=False, seed=3)
    conformal.fit(rng.integers(0, 4, size=100), logits)
    config = json.loads(json.dumps(conformal.get_config()))
    restored = SplitConformalClassifier.from_config(config)
    assert restored.get_config() == config
    assert restored.score == "aps" and restored.from_logits and not restored.randomized and restored.seed == 3
    np.testing.assert_array_equal(
        ops.convert_to_numpy(restored.predict_set(logits)), ops.convert_to_numpy(conformal.predict_set(logits))
    )


