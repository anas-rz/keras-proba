"""Training tests for the Gaussian process layer. Run once per backend via `KERAS_BACKEND`."""

import numpy as np

import keras
from keras import ops

from keras_proba.layers.gaussian_process import RandomFeatureGaussianProcess, mean_field_logits


def make_classification_data(n=512, seed=0):
    """Two Gaussian blobs in 2D."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x = rng.normal(size=(n, 2)) * 0.5 + np.where(y[:, None] == 1, 1.5, -1.5)
    return x.astype("float32"), y.astype("int32")


def build_models(**gp_kwargs):
    """Returns a training model outputting logits and an inference model also returning covariance."""
    inputs = keras.Input(shape=(2,))
    hidden = keras.layers.Dense(32, activation="relu")(inputs)
    gp = RandomFeatureGaussianProcess(units=2, num_inducing=128, name="gp", **gp_kwargs)
    logits, covmat = gp(hidden)
    train_model = keras.Model(inputs, logits)
    train_model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return train_model, keras.Model(inputs, [logits, covmat]), gp


def test_fit_learns_and_updates_precision():
    gp_cov_likelihood = "gaussian"
    keras.utils.set_random_seed(0)
    x, y = make_classification_data()
    train_model, _, gp = build_models(gp_cov_likelihood=gp_cov_likelihood)

    precision_before = ops.convert_to_numpy(gp._gp_cov_layer.precision_matrix).copy()
    random_kernel_before = ops.convert_to_numpy(gp._random_feature.kernel).copy()

    history = train_model.fit(x, y, epochs=10, batch_size=32, verbose=0)
    assert history.history["accuracy"][-1] > 0.95

    # The precision matrix is updated by fit, but the random features stay fixed.
    assert not np.allclose(ops.convert_to_numpy(gp._gp_cov_layer.precision_matrix), precision_before)
    np.testing.assert_array_equal(ops.convert_to_numpy(gp._random_feature.kernel), random_kernel_before)


def test_predict_does_not_update_precision():
    keras.utils.set_random_seed(0)
    x, y = make_classification_data(n=128)
    train_model, infer_model, gp = build_models()
    train_model.fit(x, y, epochs=1, batch_size=32, verbose=0)

    precision = ops.convert_to_numpy(gp._gp_cov_layer.precision_matrix).copy()
    infer_model.predict(x, batch_size=32, verbose=0)
    train_model.evaluate(x, y, verbose=0)
    np.testing.assert_array_equal(ops.convert_to_numpy(gp._gp_cov_layer.precision_matrix), precision)


def test_uncertainty_is_higher_out_of_distribution():
    keras.utils.set_random_seed(0)
    x, y = make_classification_data()
    train_model, infer_model, gp = build_models(gp_cov_momentum=-1)

    class ResetCovariance(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            gp.reset_covariance_matrix()

    train_model.fit(x, y, epochs=5, batch_size=64, verbose=0, callbacks=[ResetCovariance()])

    x_ood = np.random.default_rng(1).normal(size=(64, 2)).astype("float32") * 0.5 + np.array([8.0, -8.0], "float32")
    _, cov_in = infer_model.predict(x[:64], batch_size=64, verbose=0)
    _, cov_ood = infer_model.predict(x_ood, batch_size=64, verbose=0)
    assert np.diag(cov_ood).mean() > 5 * np.diag(cov_in).mean()

    logits_ood, cov_ood = infer_model.predict(x_ood, batch_size=64, verbose=0)
    probs_raw = ops.convert_to_numpy(ops.softmax(logits_ood))
    probs_mf = ops.convert_to_numpy(ops.softmax(mean_field_logits(logits_ood, cov_ood, np.pi / 8)))
    # Mean-field adjustment moves OOD predictions toward uniform.
    assert probs_mf.max(-1).mean() < probs_raw.max(-1).mean()


def test_binary_logistic_likelihood_fit():
    keras.utils.set_random_seed(0)
    x, y = make_classification_data(n=256)
    inputs = keras.Input(shape=(2,))
    gp = RandomFeatureGaussianProcess(units=1, num_inducing=64, gp_cov_likelihood="binary_logistic")
    logits, _ = gp(inputs)
    model = keras.Model(inputs, logits)
    model.compile(optimizer=keras.optimizers.Adam(0.05), loss=keras.losses.BinaryCrossentropy(from_logits=True))

    precision_before = ops.convert_to_numpy(gp._gp_cov_layer.precision_matrix).copy()
    history = model.fit(x, y.astype("float32")[:, None], epochs=5, batch_size=32, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]
    precision_after = ops.convert_to_numpy(gp._gp_cov_layer.precision_matrix)
    assert np.all(np.isfinite(precision_after))
    assert not np.allclose(precision_after, precision_before)
