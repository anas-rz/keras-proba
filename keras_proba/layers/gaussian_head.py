"""Multi-backend Gaussian output head for probabilistic regression in Keras 3.

The head predicts a heteroscedastic Gaussian `N(mean(x), scale(x)^2)` per
output unit [1, 2]:
    mean  = x W_mean + b_mean
    scale = softplus(x W_scale + b_scale) + min_scale

and is trained with the Gaussian negative log-likelihood:
    NLL = 0.5 * log(2 pi) + log(scale) + 0.5 * ((y - mean) / scale)^2

The output is `[mean, scale]` concatenated along the last axis, so a model can
be compiled with `loss=GaussianNLL()` and plain regression targets. Use
`split_gaussian_params` to recover the mean and scale.

[1]: David A. Nix and Andreas S. Weigend. Estimating the mean and variance of
     the target probability distribution. In _IEEE International Conference
     on Neural Networks_, 1994.
[2]: Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. Simple and
     Scalable Predictive Uncertainty Estimation using Deep Ensembles. In
     _Neural Information Processing Systems_, 2017.
     https://arxiv.org/abs/1612.01474
"""

import math

import keras
from keras import ops


def split_gaussian_params(params):
    """Splits a `GaussianHead` output into `(mean, scale)`.

    Args:
        params: Tensor of shape `(..., 2 * units)`.

    Returns:
        Tuple of `mean` and `scale` tensors, each of shape `(..., units)`.
    """
    mean, scale = ops.split(params, 2, axis=-1)
    return mean, scale


def gaussian_nll(y_true, y_pred):
    """Gaussian negative log-likelihood of `y_true` under a `GaussianHead` output.

    Args:
        y_true: Targets of shape `(..., units)`, or `(batch,)` when `units=1`.
        y_pred: `GaussianHead` output of shape `(..., 2 * units)`.

    Returns:
        Negative log-likelihood averaged over the last axis, shape `(...)`.
    """
    mean, scale = split_gaussian_params(ops.convert_to_tensor(y_pred))
    y_true = ops.cast(y_true, mean.dtype)
    if len(y_true.shape) == len(mean.shape) - 1:
        y_true = ops.expand_dims(y_true, -1)
    z = (y_true - mean) / scale
    nll = 0.5 * math.log(2.0 * math.pi) + ops.log(scale) + 0.5 * ops.square(z)
    return ops.mean(nll, axis=-1)


@keras.saving.register_keras_serializable(package="keras_proba")
class GaussianNLL(keras.losses.Loss):
    """Gaussian negative log-likelihood loss for `GaussianHead` outputs.

    Example:

    ```python
    model = keras.Sequential([keras.Input((8,)), keras.layers.Dense(64, "relu"), GaussianHead(1)])
    model.compile(optimizer="adam", loss=GaussianNLL())
    ```
    """

    def __init__(self, name="gaussian_nll", reduction="sum_over_batch_size", dtype=None):
        super().__init__(name=name, reduction=reduction, dtype=dtype)

    def call(self, y_true, y_pred):
        return gaussian_nll(y_true, y_pred)


@keras.saving.register_keras_serializable(package="keras_proba")
class GaussianHead(keras.layers.Layer):
    """Predicts the mean and scale of a Gaussian for probabilistic regression.

    Args:
        units: Number of regression targets.
        min_scale: Positive lower bound added to the predicted scale, for
            numerical stability of the likelihood.
        kernel_initializer: Initializer for both kernels.
        bias_initializer: Initializer for the mean bias.
        scale_bias_initializer: Initializer for the scale bias. The initial
            scale is roughly `softplus(bias) + min_scale`.
        kernel_regularizer: Regularizer for both kernels.
        bias_regularizer: Regularizer for both biases.

    Input shape:
        `(..., input_dim)`.

    Output shape:
        `(..., 2 * units)`: the mean followed by the scale (standard deviation).
    """

    def __init__(
        self,
        units,
        min_scale=1e-3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        scale_bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(f"`units` should be a positive integer. Received: {units}")
        if min_scale < 0:
            raise ValueError(f"`min_scale` should be non-negative. Received: {min_scale}")
        super().__init__(**kwargs)
        self.units = units
        self.min_scale = min_scale
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.scale_bias_initializer = keras.initializers.get(scale_bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        dense_kwargs = {
            "units": self.units,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "dtype": self.dtype_policy,
        }
        self.mean_dense = keras.layers.Dense(
            bias_initializer=self.bias_initializer, name="mean", **dense_kwargs
        )
        self.scale_dense = keras.layers.Dense(
            bias_initializer=self.scale_bias_initializer, name="scale", **dense_kwargs
        )
        self.mean_dense.build(input_shape)
        self.scale_dense.build(input_shape)

    def call(self, inputs):
        mean = self.mean_dense(inputs)
        scale = ops.softplus(self.scale_dense(inputs))
        scale = scale + ops.cast(self.min_scale, scale.dtype)
        return ops.concatenate([mean, scale], axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (2 * self.units,)

    def get_config(self):
        config = {
            "units": self.units,
            "min_scale": self.min_scale,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "scale_bias_initializer": keras.initializers.serialize(self.scale_bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}
