"""Training tests for RandomFourierFeatures. Run once per backend via `KERAS_BACKEND`."""

import numpy as np
import pytest

import keras
from keras import ops

from keras_proba.layers.random_fourier_features import RandomFourierFeatures


def make_regression_data(n=256, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 2)).astype("float32")
    y = (np.sin(x[:, :1]) * np.cos(x[:, 1:])).astype("float32")
    return x, y


def build_model(trainable):
    rff = RandomFourierFeatures(256, scale=1.0, trainable=trainable, name="rff")
    model = keras.Sequential([keras.Input(shape=(2,)), rff, keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    return model, rff


@pytest.mark.parametrize("trainable", [False, True])
def test_fit_reduces_loss(trainable):
    keras.utils.set_random_seed(0)
    x, y = make_regression_data()
    model, rff = build_model(trainable)

    kernel_before = ops.convert_to_numpy(rff.unscaled_kernel).copy()
    bias_before = ops.convert_to_numpy(rff.bias).copy()
    scale_before = ops.convert_to_numpy(rff.kernel_scale).copy()

    history = model.fit(x, y, epochs=30, batch_size=32, verbose=0)
    final_loss = history.history["loss"][-1]
    assert np.isfinite(final_loss)
    # Must do much better than predicting the mean.
    assert final_loss < 0.05 * y.var(), (y.var(), final_loss)

    # The random feature map itself is never trained.
    np.testing.assert_array_equal(ops.convert_to_numpy(rff.unscaled_kernel), kernel_before)
    np.testing.assert_array_equal(ops.convert_to_numpy(rff.bias), bias_before)

    scale_after = ops.convert_to_numpy(rff.kernel_scale)
    if trainable:
        assert not np.allclose(scale_after, scale_before)
        assert np.all(scale_after >= 0)
    else:
        np.testing.assert_array_equal(scale_after, scale_before)


def test_trainable_scale_is_non_negative():
    rff = RandomFourierFeatures(8, scale=1.0, trainable=True)
    rff.build((None, 2))
    assert isinstance(rff.kernel_scale.constraint, keras.constraints.NonNeg)

    # Optimizers apply the constraint after each update.
    optimizer = keras.optimizers.SGD(1.0)
    optimizer.apply_gradients([(ops.convert_to_tensor([5.0]), rff.kernel_scale)])
    np.testing.assert_array_equal(ops.convert_to_numpy(rff.kernel_scale), [0.0])


