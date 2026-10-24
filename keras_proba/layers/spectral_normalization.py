"""Multi-backend spectral normalization wrappers in Keras 3.

Ported from TensorFlow Model Garden's
`official/nlp/modeling/layers/spectral_normalization.py`, the normalization
used by the hidden layers of the Spectral-normalized Neural Gaussian Process
(SNGP).

The wrappers bound the spectral norm of the wrapped layer's kernel W:
    sigma = v W u^T          (power iteration estimate of ||W||_2)
    W <- W * c / sigma       if c / sigma < 1, else W
where c is `norm_multiplier`. During training the normalized kernel is
assigned back to the layer and the gradient is taken with respect to it, i.e.
the kernel is projected after every update (see Alg. 1 of [1]).

Backend note: on JAX, assignments to trainable variables inside `fit` are
discarded, so the projected kernel is not persisted there. The forward pass
and its gradient still match the other backends, since the projection is
recomputed from the persisted `u`/`v` vectors on every call.

## References:

[1] Yuichi Yoshida, Takeru Miyato. Spectral Norm Regularization for Improving
    the Generalizability of Deep Learning.
    _arXiv preprint arXiv:1705.10941_, 2017. https://arxiv.org/abs/1705.10941

[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.
    Spectral normalization for generative adversarial networks.
    In _International Conference on Learning Representations_, 2018.

[3] Henry Gouk, Eibe Frank, Bernhard Pfahringer, Michael Cree.
    Regularisation of neural networks by enforcing lipschitz continuity.
    _arXiv preprint arXiv:1804.04368_, 2018. https://arxiv.org/abs/1804.04368
"""

import math

import keras
from keras import ops


def l2_normalize(x, epsilon=1e-12):
    """Normalizes `x` over all of its elements, like `tf.nn.l2_normalize(x)`."""
    return x * ops.rsqrt(ops.maximum(ops.sum(ops.square(x)), epsilon))


def _bound_spectral_norm(w, sigma, norm_multiplier):
    """Rescales `w` so that its spectral norm is not larger than `norm_multiplier`."""
    ratio = norm_multiplier / sigma
    return ops.where(ratio < 1, ratio * w, w)


def _assign_straight_through(variable, value):
    """Assigns `value` while keeping the gradient path to the original variable.

    The assigned value equals `value`, but its gradient with respect to the
    variable is the identity, matching an in-place assignment in TensorFlow.
    """
    current = variable.value
    variable.assign(current + ops.stop_gradient(value - current))


@keras.saving.register_keras_serializable(package="keras_proba")
class SpectralNormalization(keras.layers.Wrapper):
    """Implements spectral normalization for a Dense layer.

    Example:

    ```python
    dense = SpectralNormalization(keras.layers.Dense(64), norm_multiplier=0.9)
    ```

    Args:
        layer: A Keras layer with a `kernel`, typically `Dense`.
        iteration: Number of power iterations per call to estimate the
            kernel's largest singular value.
        norm_multiplier: Upper bound of the spectral norm. Under normalization,
            the singular value usually converges to this value.
        training: Whether to perform power iteration when the layer is called
            with `training=None`.
        aggregation: How the `u` and `v` vectors are aggregated across replicas
            in distributed training, e.g. `"mean"`.
        inhere_layer_name: Whether to use the name of the wrapped layer.
        **kwargs: Other keyword arguments for `keras.layers.Wrapper`.
    """

    def __init__(
        self,
        layer,
        iteration=1,
        norm_multiplier=0.95,
        training=True,
        aggregation="mean",
        inhere_layer_name=False,
        **kwargs,
    ):
        self.iteration = iteration
        self.do_power_iteration = training
        self.aggregation = aggregation
        self.norm_multiplier = norm_multiplier
        self.inhere_layer_name = inhere_layer_name

        # Set layer name.
        wrapper_name = kwargs.pop("name", None)
        if inhere_layer_name:
            wrapper_name = layer.name

        if not isinstance(layer, keras.layers.Layer):
            raise ValueError(f"`layer` must be a `keras.layers.Layer`. Observed `{layer}`")
        super().__init__(layer, name=wrapper_name, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        if not hasattr(self.layer, "kernel"):
            raise ValueError(f"{type(self.layer).__name__} object has no attribute 'kernel'")

        self.w = self.layer.kernel
        self.w_shape = list(self.w.shape)

        self.v = self.add_weight(
            shape=(1, math.prod(self.w_shape[:-1])),
            initializer=keras.initializers.RandomNormal(),
            trainable=False,
            name="v",
            dtype=self.w.dtype,
            aggregation=self.aggregation,
        )
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=keras.initializers.RandomNormal(),
            trainable=False,
            name="u",
            dtype=self.w.dtype,
            aggregation=self.aggregation,
        )

        self.update_weights()

    def call(self, inputs, training=None):
        training = self.do_power_iteration if training is None else training
        self.update_weights(training=training)
        return self.layer(inputs)

    def update_weights(self, training=True):
        """Runs power iteration and bounds the kernel's spectral norm.

        With `training=True`, `u` and `v` are updated by power iteration first.
        Otherwise the current `u` and `v` are used as is. In both cases the
        normalized kernel is assigned to the wrapped layer. Unlike the
        reference, which uses the raw kernel at inference, this keeps inference
        normalized on JAX; elsewhere it is a no-op unless the kernel changed
        since the last training call.
        """
        w = self.w.value
        w_reshaped = ops.reshape(w, (-1, self.w_shape[-1]))

        u_hat = self.u.value
        v_hat = self.v.value

        if training:
            for _ in range(self.iteration):
                v_hat = l2_normalize(ops.matmul(u_hat, ops.transpose(w_reshaped)))
                u_hat = l2_normalize(ops.matmul(v_hat, w_reshaped))
            # The power iteration vectors are running statistics.
            u_hat = ops.stop_gradient(u_hat)
            v_hat = ops.stop_gradient(v_hat)
            self.u.assign(u_hat)
            self.v.assign(v_hat)

        sigma = ops.matmul(ops.matmul(v_hat, w_reshaped), ops.transpose(u_hat))
        # Convert sigma from a 1x1 matrix to a scalar.
        sigma = ops.reshape(sigma, ())

        # Bound spectral norm to be not larger than self.norm_multiplier.
        _assign_straight_through(
            self.layer.kernel, _bound_spectral_norm(w, ops.stop_gradient(sigma), self.norm_multiplier)
        )

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = {
            "iteration": self.iteration,
            "norm_multiplier": self.norm_multiplier,
            "training": self.do_power_iteration,
            "aggregation": self.aggregation,
            "inhere_layer_name": self.inhere_layer_name,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="keras_proba")
class SpectralNormalizationConv2D(keras.layers.Wrapper):
    """Implements spectral normalization for a Conv2D layer based on [3].

    The spectral norm of the convolution as a linear operator is estimated
    by power iteration with `conv` and `conv_transpose`, so the input height
    and width must be known and divisible by the strides. Inputs are expected
    in `channels_last` format.

    Args:
        layer: A `keras.layers.Conv2D` layer.
        iteration: Number of power iterations per call.
        norm_multiplier: Upper bound of the spectral norm.
        training: Whether to perform power iteration on each call.
        aggregation: How the `u` and `v` vectors are aggregated across replicas
            in distributed training, e.g. `"mean"`.
        legacy_mode: Whether to set the leading dimension of `u` and `v` to the
            batch size. Only for backward compatibility.
        **kwargs: Other keyword arguments for `keras.layers.Wrapper`.
    """

    def __init__(
        self,
        layer,
        iteration=1,
        norm_multiplier=0.95,
        training=True,
        aggregation="mean",
        legacy_mode=False,
        **kwargs,
    ):
        self.iteration = iteration
        self.do_power_iteration = training
        self.aggregation = aggregation
        self.norm_multiplier = norm_multiplier
        self.legacy_mode = legacy_mode

        if not isinstance(layer, keras.layers.Conv2D):
            raise ValueError(f"layer must be a `keras.layers.Conv2D` instance. You passed: {layer}")
        if layer.data_format != "channels_last":
            raise ValueError(
                f'Only `data_format="channels_last"` is supported. Received: {layer.data_format}'
            )

        # Set layer attributes. Skip the suffix if the layer was already
        # wrapped, e.g. when reloading from a config.
        if not layer.name.endswith("_spec_norm"):
            layer.name += "_spec_norm"
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)

        # Shape (kernel_size_1, kernel_size_2, in_channel, out_channel).
        self.w = self.layer.kernel
        self.w_shape = list(self.w.shape)
        self.strides = self.layer.strides

        # Set the dimensions of u and v vectors.
        batch_size = input_shape[0]
        uv_dim = batch_size if self.legacy_mode else 1

        # Resolve shapes.
        in_height = input_shape[1]
        in_width = input_shape[2]
        if in_height is None or in_width is None:
            raise ValueError(
                f"The input height and width must be defined. Full input shape received: {input_shape}"
            )
        in_channel = self.w_shape[2]

        out_height = in_height // self.strides[0]
        out_width = in_width // self.strides[1]
        out_channel = self.w_shape[3]

        self.in_shape = (uv_dim, in_height, in_width, in_channel)
        self.out_shape = (uv_dim, out_height, out_width, out_channel)

        self.v = self.add_weight(
            shape=self.in_shape,
            initializer=keras.initializers.RandomNormal(),
            trainable=False,
            name="v",
            dtype=self.w.dtype,
            aggregation=self.aggregation,
        )
        self.u = self.add_weight(
            shape=self.out_shape,
            initializer=keras.initializers.RandomNormal(),
            trainable=False,
            name="u",
            dtype=self.w.dtype,
            aggregation=self.aggregation,
        )

    def call(self, inputs):
        self.update_weights()
        return self.layer(inputs)

    def _conv(self, x, w):
        return ops.conv(x, w, strides=self.strides, padding="same", data_format="channels_last")

    def update_weights(self):
        """Computes power iteration for convolutional filters based on [3]."""
        w = self.w.value

        # Initialize u, v vectors.
        u_hat = self.u.value
        v_hat = self.v.value

        if self.do_power_iteration:
            for _ in range(self.iteration):
                # Updates v.
                v_ = ops.conv_transpose(
                    u_hat, w, strides=self.strides, padding="same", data_format="channels_last"
                )
                v_hat = ops.reshape(l2_normalize(ops.reshape(v_, (1, -1))), ops.shape(v_))

                # Updates u.
                u_ = self._conv(v_hat, w)
                u_hat = ops.reshape(l2_normalize(ops.reshape(u_, (1, -1))), ops.shape(u_))
            # The power iteration vectors are running statistics.
            u_hat = ops.stop_gradient(u_hat)
            v_hat = ops.stop_gradient(v_hat)

        v_w_hat = self._conv(v_hat, w)

        sigma = ops.matmul(ops.reshape(v_w_hat, (1, -1)), ops.reshape(u_hat, (-1, 1)))
        # Convert sigma from a 1x1 matrix to a scalar.
        sigma = ops.reshape(sigma, ())

        self.u.assign(u_hat)
        self.v.assign(v_hat)

        _assign_straight_through(
            self.layer.kernel, _bound_spectral_norm(w, ops.stop_gradient(sigma), self.norm_multiplier)
        )

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = {
            "iteration": self.iteration,
            "norm_multiplier": self.norm_multiplier,
            "training": self.do_power_iteration,
            "aggregation": self.aggregation,
            "legacy_mode": self.legacy_mode,
        }
        base_config = super().get_config()
        return {**base_config, **config}
