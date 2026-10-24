"""Multi-backend Bayesian (variational) dense and convolutional layers in Keras 3.

Ported from TensorFlow Probability's `tensorflow_probability/python/layers/dense_variational.py`
and `conv_variational.py`, with their default distributions:

- Kernel posterior: mean-field Normal `q(W) = N(loc, scale^2)`, with
  `scale = eps + softplus(untransformed_scale)`, initialized near a point mass.
- Kernel prior: Normal `p(W) = N(0, kernel_prior_scale^2)`.
- Bias: deterministic (a point estimate), without a divergence.

Each call samples a kernel, so the forward pass is stochastic in training and
inference; average several predictions for a Monte Carlo estimate. Each call
also adds `kl_weight * KL(q(W) || p(W))` to the layer's losses. With minibatch
training on the mean loss over examples, set `kl_weight = 1 / num_train_examples`
so the KL is counted once per epoch.

Estimators:
    Reparameterization [1]: one kernel sample per batch.
    Local reparameterization [2]: samples the pre-activations directly.
    Flipout [3]: pseudo-independent kernel perturbations per example, with
        lower gradient variance at roughly twice the cost.

[1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes.
     In _International Conference on Learning Representations_, 2014.
     https://arxiv.org/abs/1312.6114
[2]: Diederik Kingma, Tim Salimans, Max Welling. Variational Dropout and the
     Local Reparameterization Trick. In _Neural Information Processing
     Systems_, 2015. https://arxiv.org/abs/1506.02557
[3]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, Roger Grosse. Flipout:
     Efficient Pseudo-Independent Weight Perturbations on Mini-Batches.
     In _International Conference on Learning Representations_, 2018.
     https://arxiv.org/abs/1803.04386
"""

import numpy as np

import keras
from keras import ops


def normal_kl_divergence(loc, scale, prior_scale):
    """`KL(N(loc, scale^2) || N(0, prior_scale^2))`, summed over all elements."""
    prior_scale = ops.cast(prior_scale, scale.dtype)
    kl = (
        ops.log(prior_scale)
        - ops.log(scale)
        + (ops.square(scale) + ops.square(loc)) / (2.0 * ops.square(prior_scale))
        - 0.5
    )
    return ops.sum(kl)


class _Variational(keras.layers.Layer):
    """Shared posterior, prior, divergence and noise handling of the variational layers."""

    def __init__(
        self,
        activation=None,
        use_bias=True,
        kernel_loc_initializer=None,
        kernel_untransformed_scale_initializer=None,
        bias_initializer=None,
        kernel_prior_scale=1.0,
        kl_weight=1.0,
        seed=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        if kernel_prior_scale <= 0:
            raise ValueError(f"`kernel_prior_scale` should be positive. Received: {kernel_prior_scale}")
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_loc_initializer = keras.initializers.get(
            kernel_loc_initializer or keras.initializers.RandomNormal(stddev=0.1)
        )
        self.kernel_untransformed_scale_initializer = keras.initializers.get(
            kernel_untransformed_scale_initializer or keras.initializers.RandomNormal(mean=-3.0, stddev=0.1)
        )
        self.bias_initializer = keras.initializers.get(bias_initializer or keras.initializers.RandomNormal(stddev=0.1))
        self.kernel_prior_scale = kernel_prior_scale
        self.kl_weight = kl_weight
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def _build_posterior(self, kernel_shape, bias_shape):
        self.kernel_posterior_loc = self.add_weight(
            name="kernel_posterior_loc", shape=kernel_shape, initializer=self.kernel_loc_initializer
        )
        self.kernel_posterior_untransformed_scale = self.add_weight(
            name="kernel_posterior_untransformed_scale",
            shape=kernel_shape,
            initializer=self.kernel_untransformed_scale_initializer,
        )
        self.bias_posterior_loc = None
        if self.use_bias:
            self.bias_posterior_loc = self.add_weight(
                name="bias_posterior_loc", shape=bias_shape, initializer=self.bias_initializer
            )

    @property
    def kernel_posterior_scale(self):
        """Standard deviation of the kernel posterior, `eps + softplus(untransformed_scale)`."""
        u = self.kernel_posterior_untransformed_scale
        eps = np.finfo(keras.backend.standardize_dtype(u.dtype)).eps
        return eps + ops.softplus(u)

    def kl_divergence(self):
        """`KL(q(W) || p(W))` of the kernel, before `kl_weight`."""
        return normal_kl_divergence(self.kernel_posterior_loc, self.kernel_posterior_scale, self.kernel_prior_scale)

    def _standard_normal(self, shape, dtype):
        return keras.random.normal(shape, dtype=dtype, seed=self.seed_generator)

    def _rademacher(self, shape, dtype):
        """Random signs in {-1, 1}."""
        return ops.cast(keras.random.randint(shape, 0, 2, seed=self.seed_generator) * 2 - 1, dtype)

    def _sample_kernel(self, dtype):
        loc = ops.cast(self.kernel_posterior_loc, dtype)
        scale = ops.cast(self.kernel_posterior_scale, dtype)
        return loc + scale * self._standard_normal(ops.shape(loc), dtype)

    def _finish(self, outputs):
        """Adds the bias, applies the activation and registers the divergence."""
        if self.use_bias:
            outputs = outputs + ops.cast(self.bias_posterior_loc, outputs.dtype)
        if self.activation is not None:
            outputs = self.activation(outputs)
        self.add_loss(self.kl_weight * self.kl_divergence())
        return outputs

    def get_config(self):
        config = {
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_loc_initializer": keras.initializers.serialize(self.kernel_loc_initializer),
            "kernel_untransformed_scale_initializer": keras.initializers.serialize(
                self.kernel_untransformed_scale_initializer
            ),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_prior_scale": self.kernel_prior_scale,
            "kl_weight": self.kl_weight,
            "seed": self.seed,
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}


def _expand_docstring(cls):
    """Fills the shared argument descriptions into a class docstring, like TFP's `docstring_util`."""
    cls.__doc__ = cls.__doc__.replace("${conv_args}", _CONV_ARGS_DOC.strip("\n")).replace(
        "${common_args}", _COMMON_ARGS_DOC.strip("\n")
    )
    return cls


_COMMON_ARGS_DOC = """
        activation: Activation function.
        use_bias: Whether to add a (deterministic) bias.
        kernel_loc_initializer: Initializer for the posterior mean of the
            kernel. Defaults to `RandomNormal(stddev=0.1)`.
        kernel_untransformed_scale_initializer: Initializer for the posterior
            scale before `softplus`. Defaults to `RandomNormal(mean=-3,
            stddev=0.1)`, a posterior close to a point mass.
        bias_initializer: Initializer for the bias. Defaults to
            `RandomNormal(stddev=0.1)`.
        kernel_prior_scale: Standard deviation of the Normal kernel prior.
        kl_weight: Multiplier of the KL divergence added to the losses, e.g.
            `1 / num_train_examples`.
        seed: Seed for the random noise.
        activity_regularizer: Regularizer for the output."""

_CONV_ARGS_DOC = """
        filters: Number of output channels.
        kernel_size: Int or tuple of ints, one per spatial dimension: the
            convolution window.
        strides: Int or tuple of ints, one per spatial dimension.
        padding: `"valid"` or `"same"`.
        data_format: `"channels_last"` or `"channels_first"`.
        dilation_rate: Int or tuple of ints, one per spatial dimension."""


class _DenseVariational(_Variational):
    def __init__(self, units, **kwargs):
        if units <= 0:
            raise ValueError(f"`units` should be a positive integer. Received: {units}")
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        in_size = input_shape[-1]
        if in_size is None:
            raise ValueError(
                f"The innermost dimension of `input_shape` must be defined, but saw: {input_shape}"
            )
        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
        self._build_posterior((in_size, self.units), (self.units,))

    def call(self, inputs):
        return self._finish(self._apply_variational_kernel(inputs))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class DenseReparameterization(_DenseVariational):
    """Bayesian dense layer with the reparameterization estimator [1].

    One kernel `W ~ q(W)` is sampled per call:
    `outputs = activation(inputs @ W + bias)`.

    Example:

    ```python
    kl_weight = 1.0 / num_train_examples
    model = keras.Sequential([
        keras.Input((num_features,)),
        DenseReparameterization(64, activation="relu", kl_weight=kl_weight),
        DenseReparameterization(num_classes, kl_weight=kl_weight),
    ])
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    ```

    Args:
        units: Dimensionality of the output.
${common_args}
    """

    def _apply_variational_kernel(self, inputs):
        return ops.matmul(inputs, self._sample_kernel(inputs.dtype))


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class DenseLocalReparameterization(_DenseVariational):
    """Bayesian dense layer with the local reparameterization estimator [2].

    Samples the pre-activations, whose distribution under `q(W)` is
    `N(inputs @ loc, (inputs^2) @ (scale^2))`, independently per example.

    Args:
        units: Dimensionality of the output.
${common_args}
    """

    def _apply_variational_kernel(self, inputs):
        loc = ops.matmul(inputs, ops.cast(self.kernel_posterior_loc, inputs.dtype))
        variance = ops.matmul(ops.square(inputs), ops.square(ops.cast(self.kernel_posterior_scale, inputs.dtype)))
        # Unlike TFP, add the smallest positive float: an all-zero input (e.g. after a
        # ReLU) gives a zero variance, where the gradient of sqrt is infinite and
        # the scale gradient becomes NaN.
        tiny = np.finfo(keras.backend.standardize_dtype(variance.dtype)).tiny
        scale = ops.sqrt(variance + tiny)
        return loc + scale * self._standard_normal(ops.shape(loc), inputs.dtype)


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class DenseFlipout(_DenseVariational):
    """Bayesian dense layer with the Flipout estimator [3].

    Samples one kernel perturbation `dW ~ N(0, scale^2)` per call, and
    decorrelates it across examples with random signs:
    `outputs = inputs @ loc + ((inputs * s_in) @ dW) * s_out`.

    Args:
        units: Dimensionality of the output.
${common_args}
    """

    def _apply_variational_kernel(self, inputs):
        dtype = inputs.dtype
        scale = ops.cast(self.kernel_posterior_scale, dtype)
        perturbation = scale * self._standard_normal(ops.shape(scale), dtype)

        input_shape = ops.shape(inputs)
        sign_input = self._rademacher(input_shape, dtype)
        sign_output = self._rademacher(tuple(input_shape[:-1]) + (self.units,), dtype)
        perturbed = ops.matmul(inputs * sign_input, perturbation) * sign_output

        return ops.matmul(inputs, ops.cast(self.kernel_posterior_loc, dtype)) + perturbed


def _normalize_tuple(value, n, name):
    if isinstance(value, int):
        return (value,) * n
    value = tuple(value)
    if len(value) != n:
        raise ValueError(f"`{name}` must be an int or a tuple of {n} ints. Received: {value}")
    return value


class _ConvVariational(_Variational):
    """Shared convolution handling; subclasses set `rank`, the number of spatial dimensions."""

    rank = None

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        **kwargs,
    ):
        if filters <= 0:
            raise ValueError(f"`filters` should be a positive integer. Received: {filters}")
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = _normalize_tuple(kernel_size, self.rank, "kernel_size")
        self.strides = _normalize_tuple(strides, self.rank, "strides")
        self.padding = padding.lower()
        if self.padding not in ("valid", "same"):
            raise ValueError(f'`padding` must be "valid" or "same". Received: {padding}')
        self.data_format = (data_format or keras.config.image_data_format()).lower()
        if self.data_format not in ("channels_last", "channels_first"):
            raise ValueError(
                f'`data_format` must be "channels_last" or "channels_first". Received: {data_format}'
            )
        self.dilation_rate = _normalize_tuple(dilation_rate, self.rank, "dilation_rate")
        self._channel_axis = -1 if self.data_format == "channels_last" else 1

    def build(self, input_shape):
        in_channels = input_shape[self._channel_axis]
        if in_channels is None:
            raise ValueError(f"The channel dimension of the inputs must be defined. Found `None`: {input_shape}")
        self.input_spec = keras.layers.InputSpec(ndim=self.rank + 2, axes={self._channel_axis: in_channels})
        self._build_posterior(self.kernel_size + (in_channels, self.filters), (self.filters,))

    def _convolution_op(self, inputs, kernel):
        return ops.conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def _channel_shape(self, batch_size, channels):
        """Shape of a per-example, per-channel tensor that broadcasts over the spatial dimensions."""
        spatial = (1,) * self.rank
        if self.data_format == "channels_last":
            return (batch_size, *spatial, channels)
        return (batch_size, channels, *spatial)

    def call(self, inputs):
        outputs = self._apply_variational_kernel(inputs)
        if self.data_format == "channels_first":
            # Add the bias on the last axis.
            return ops.moveaxis(self._finish(ops.moveaxis(outputs, 1, -1)), -1, 1)
        return self._finish(outputs)

    def compute_output_shape(self, input_shape):
        spatial = input_shape[1:-1] if self.data_format == "channels_last" else input_shape[2:]
        out_spatial = []
        for size, k, s, d in zip(spatial, self.kernel_size, self.strides, self.dilation_rate):
            if size is None:
                out_spatial.append(None)
                continue
            effective = (k - 1) * d + 1
            length = size - effective + 1 if self.padding == "valid" else size
            out_spatial.append((length + s - 1) // s)
        if self.data_format == "channels_last":
            return (input_shape[0], *out_spatial, self.filters)
        return (input_shape[0], self.filters, *out_spatial)

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
        }
        return {**super().get_config(), **config}


class _ConvReparameterization(_ConvVariational):
    def _apply_variational_kernel(self, inputs):
        return self._convolution_op(inputs, self._sample_kernel(inputs.dtype))


class _ConvFlipout(_ConvVariational):
    def _apply_variational_kernel(self, inputs):
        dtype = inputs.dtype
        scale = ops.cast(self.kernel_posterior_scale, dtype)
        perturbation = scale * self._standard_normal(ops.shape(scale), dtype)

        batch_size = ops.shape(inputs)[0]
        in_channels = inputs.shape[self._channel_axis]
        # Signs per example and channel, broadcast over the spatial dimensions.
        sign_input = self._rademacher(self._channel_shape(batch_size, in_channels), dtype)
        sign_output = self._rademacher(self._channel_shape(batch_size, self.filters), dtype)

        outputs = self._convolution_op(inputs, ops.cast(self.kernel_posterior_loc, dtype))
        return outputs + self._convolution_op(inputs * sign_input, perturbation) * sign_output


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class Conv1DReparameterization(_ConvReparameterization):
    """Bayesian 1D convolution (e.g. temporal convolution) with the reparameterization estimator [1].

    One kernel `W ~ q(W)` is sampled per call.

    Args:
${conv_args}
${common_args}
    """

    rank = 1


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class Conv1DFlipout(_ConvFlipout):
    """Bayesian 1D convolution (e.g. temporal convolution) with the Flipout estimator [3].

    Samples one kernel perturbation per call, decorrelated across examples with
    random signs per example and channel:
    `outputs = conv(inputs, loc) + conv(inputs * s_in, dW) * s_out`.

    Args:
${conv_args}
${common_args}
    """

    rank = 1


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class Conv2DReparameterization(_ConvReparameterization):
    """Bayesian 2D convolution (e.g. spatial convolution over images) with the reparameterization estimator [1].

    One kernel `W ~ q(W)` is sampled per call.

    Args:
${conv_args}
${common_args}
    """

    rank = 2


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class Conv2DFlipout(_ConvFlipout):
    """Bayesian 2D convolution (e.g. spatial convolution over images) with the Flipout estimator [3].

    Samples one kernel perturbation per call, decorrelated across examples with
    random signs per example and channel:
    `outputs = conv(inputs, loc) + conv(inputs * s_in, dW) * s_out`.

    Args:
${conv_args}
${common_args}
    """

    rank = 2


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class Conv3DReparameterization(_ConvReparameterization):
    """Bayesian 3D convolution (e.g. spatial convolution over volumes) with the reparameterization estimator [1].

    One kernel `W ~ q(W)` is sampled per call.

    Args:
${conv_args}
${common_args}
    """

    rank = 3


@keras.saving.register_keras_serializable(package="keras_proba")
@_expand_docstring
class Conv3DFlipout(_ConvFlipout):
    """Bayesian 3D convolution (e.g. spatial convolution over volumes) with the Flipout estimator [3].

    Samples one kernel perturbation per call, decorrelated across examples with
    random signs per example and channel:
    `outputs = conv(inputs, loc) + conv(inputs * s_in, dW) * s_out`.

    Args:
${conv_args}
${common_args}
    """

    rank = 3
