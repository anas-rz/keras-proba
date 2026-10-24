"""Multi-backend mixture density network head for probabilistic regression in Keras 3.

A mixture density network [1] predicts a mixture of `K` Gaussians per input,
which can represent multi-modal and skewed targets that a single Gaussian
(`GaussianHead`) cannot. For `D` outputs, each component has independent
dimensions:
    p(y | x) = sum_k pi_k(x) prod_d N(y_d; mean_kd(x), scale_kd(x)^2)
    pi = softmax(logits)
    scale = softplus(x W_scale + b_scale) + min_scale

The head outputs `[logits, means, scales]` concatenated along the last axis,
with shapes `K`, `K * D` and `K * D`, so a model can be compiled with
`loss=MixtureGaussianNLL()` and plain regression targets. Use
`split_mixture_params` to recover the parameters, and
`mixture_mean_and_stddev` for the predictive mean and standard deviation.

Equivalent to TensorFlow Probability's `tfp.layers.MixtureNormal` applied to a
dense layer, which gives a `tfd.MixtureSameFamily` of `tfd.Independent(Normal)`.

[1]: Christopher M. Bishop. Mixture Density Networks. Technical Report
     NCRG/94/004, Aston University, 1994.
"""

import math

import keras
from keras import ops


def _num_components(params_size, units):
    num_components, remainder = divmod(params_size, 1 + 2 * units)
    if remainder or num_components == 0:
        raise ValueError(
            f"Mixture parameters of size {params_size} do not match {units} unit(s): "
            f"expected a multiple of 1 + 2 * units = {1 + 2 * units}."
        )
    return num_components


def split_mixture_params(params, units=1):
    """Splits a `MixtureGaussianHead` output into `(logits, means, scales)`.

    Args:
        params: Tensor of shape `(..., K * (1 + 2 * units))`.
        units: Number of regression targets `D`.

    Returns:
        Tuple of `logits` of shape `(..., K)`, and `means` and `scales` of shape
        `(..., K, units)`.
    """
    params = ops.convert_to_tensor(params)
    k = _num_components(params.shape[-1], units)
    logits = params[..., :k]
    batch_shape = tuple(ops.shape(params)[:-1])
    means = ops.reshape(params[..., k : k + k * units], batch_shape + (k, units))
    scales = ops.reshape(params[..., k + k * units :], batch_shape + (k, units))
    return logits, means, scales


def _expand_targets(y_true, units, dtype):
    y_true = ops.cast(y_true, dtype)
    if units == 1 and (len(y_true.shape) == 0 or y_true.shape[-1] != 1):
        y_true = ops.expand_dims(y_true, -1)
    # Broadcast against the component axis.
    return ops.expand_dims(y_true, -2)


def mixture_gaussian_nll(y_true, y_pred, units=None):
    """Negative log-likelihood of `y_true` under a `MixtureGaussianHead` output.

    Args:
        y_true: Targets of shape `(..., units)`, or `(batch,)` when `units=1`.
        y_pred: `MixtureGaussianHead` output of shape `(..., K * (1 + 2 * units))`.
        units: Number of regression targets. Inferred from `y_true` if `None`.

    Returns:
        Negative log-likelihood of the joint `units`-dimensional target, shape
        `(...)`. Unlike `gaussian_nll`, it is summed (not averaged) over units,
        as for `tfd.Independent`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    if units is None:
        y_true_shape = ops.convert_to_tensor(y_true).shape
        units = 1 if len(y_true_shape) < len(y_pred.shape) else y_true_shape[-1]
    logits, means, scales = split_mixture_params(y_pred, units)
    y = _expand_targets(y_true, units, means.dtype)

    z = (y - means) / scales
    component_log_prob = ops.sum(-0.5 * math.log(2.0 * math.pi) - ops.log(scales) - 0.5 * ops.square(z), axis=-1)
    log_mix = ops.log_softmax(logits, axis=-1)
    return -ops.logsumexp(log_mix + component_log_prob, axis=-1)


def mixture_mean_and_stddev(params, units=1):
    """Predictive mean and standard deviation of a `MixtureGaussianHead` output.

    The variance is the law of total variance, `sum_k pi_k (scale_k^2 +
    mean_k^2) - mean^2`, so it includes the spread between components.

    Args:
        params: Tensor of shape `(..., K * (1 + 2 * units))`.
        units: Number of regression targets.

    Returns:
        Tuple of `mean` and `stddev`, each of shape `(..., units)`.
    """
    logits, means, scales = split_mixture_params(params, units)
    weights = ops.expand_dims(ops.softmax(logits, axis=-1), -1)
    mean = ops.sum(weights * means, axis=-2)
    second_moment = ops.sum(weights * (ops.square(scales) + ops.square(means)), axis=-2)
    variance = ops.maximum(second_moment - ops.square(mean), 0.0)
    return mean, ops.sqrt(variance)


@keras.saving.register_keras_serializable(package="keras_proba")
class MixtureGaussianNLL(keras.losses.Loss):
    """Negative log-likelihood loss for `MixtureGaussianHead` outputs.

    Example:

    ```python
    model = keras.Sequential([
        keras.Input((1,)),
        keras.layers.Dense(64, activation="relu"),
        MixtureGaussianHead(num_components=5),
    ])
    model.compile(optimizer="adam", loss=MixtureGaussianNLL())
    ```

    Args:
        units: Number of regression targets. Inferred from the targets if
            `None`; set it when targets of shape `(batch, units)` could be
            confused with `(batch,)` targets for `units=1`.
        name: Name of the loss.
        reduction: Keras loss reduction.
        dtype: Dtype of the loss computation.
    """

    def __init__(self, units=None, name="mixture_gaussian_nll", reduction="sum_over_batch_size", dtype=None):
        super().__init__(name=name, reduction=reduction, dtype=dtype)
        self.units = units

    def call(self, y_true, y_pred):
        return mixture_gaussian_nll(y_true, y_pred, units=self.units)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="keras_proba")
class MixtureGaussianHead(keras.layers.Layer):
    """Predicts a mixture of Gaussians for probabilistic regression (mixture density network).

    Args:
        num_components: Number of mixture components `K`.
        units: Number of regression targets `D`.
        min_scale: Non-negative lower bound added to the component scales, for
            numerical stability of the likelihood.
        kernel_initializer: Initializer for the three kernels.
        bias_initializer: Initializer for the logit and mean biases.
        scale_bias_initializer: Initializer for the scale bias. Component
            scales start near `softplus(bias) + min_scale`.
        kernel_regularizer: Regularizer for the three kernels.
        bias_regularizer: Regularizer for the three biases.

    Input shape:
        `(..., input_dim)`.

    Output shape:
        `(..., K * (1 + 2 * D))`: mixture logits, then component means, then
        component scales (standard deviations), each component's `D` values
        contiguous.
    """

    def __init__(
        self,
        num_components,
        units=1,
        min_scale=1e-3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        scale_bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        if num_components <= 0:
            raise ValueError(f"`num_components` should be a positive integer. Received: {num_components}")
        if units <= 0:
            raise ValueError(f"`units` should be a positive integer. Received: {units}")
        if min_scale < 0:
            raise ValueError(f"`min_scale` should be non-negative. Received: {min_scale}")
        super().__init__(**kwargs)
        self.num_components = num_components
        self.units = units
        self.min_scale = min_scale
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.scale_bias_initializer = keras.initializers.get(scale_bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        k, d = self.num_components, self.units
        dense_kwargs = {
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "dtype": self.dtype_policy,
        }
        self.logits_dense = keras.layers.Dense(k, bias_initializer=self.bias_initializer, name="logits", **dense_kwargs)
        self.mean_dense = keras.layers.Dense(k * d, bias_initializer=self.bias_initializer, name="mean", **dense_kwargs)
        self.scale_dense = keras.layers.Dense(
            k * d, bias_initializer=self.scale_bias_initializer, name="scale", **dense_kwargs
        )
        for layer in (self.logits_dense, self.mean_dense, self.scale_dense):
            layer.build(input_shape)

    def call(self, inputs):
        logits = self.logits_dense(inputs)
        means = self.mean_dense(inputs)
        scales = ops.softplus(self.scale_dense(inputs))
        scales = scales + ops.cast(self.min_scale, scales.dtype)
        return ops.concatenate([logits, means, scales], axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_components * (1 + 2 * self.units),)

    def get_config(self):
        config = {
            "num_components": self.num_components,
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
