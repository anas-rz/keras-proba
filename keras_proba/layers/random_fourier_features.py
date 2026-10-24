"""Multi-backend Random Fourier Features in Keras 3.

Ported from `tf_keras.layers.experimental.RandomFourierFeatures`, which is the
random feature layer used by the SNGP Gaussian process output layer in
TensorFlow Model Garden.

Random Fourier Features [1] approximate a shift-invariant kernel
`K(x, y) = k(x - y)` with an explicit feature map:
    phi(x) = cos(x W / scale + b)
    W_ij ~ p(w)            (spectral density of the kernel)
    b_j  ~ Uniform(0, 2 pi)
    K(x, y) ~= (2 / D) * phi(x) . phi(y)

Supported kernels:
    Gaussian:  K(x, y) = exp(-||x - y||^2 / (2 * scale^2)),  W ~ Normal(0, 1)
    Laplacian: K(x, y) = exp(-||x - y||_1 / scale),          W ~ Cauchy(0, 1)

As in the reference implementation, the output does not apply the
`sqrt(2 / D)` normalization factor.

[1]: Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel
     Machines. In _Neural Information Processing Systems_, 2007.
     https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
"""

import math

import keras
from keras import ops

_SUPPORTED_RBF_KERNEL_TYPES = ["gaussian", "laplacian"]


class _CauchyInitializer(keras.initializers.Initializer):
    """Samples from a standard Cauchy distribution via the inverse CDF."""

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        probs = keras.random.uniform(shape, minval=0.0, maxval=1.0, dtype=dtype, seed=self.seed)
        return ops.tan(math.pi * (probs - 0.5))

    def get_config(self):
        return {"seed": self.seed}


def _get_random_features_initializer(initializer):
    """Returns the initializer for the unscaled random feature kernel."""
    if isinstance(initializer, str):
        if initializer.lower() == "gaussian":
            return keras.initializers.RandomNormal(stddev=1.0)
        if initializer.lower() == "laplacian":
            return _CauchyInitializer()
        raise ValueError(
            f'Unsupported `kernel_initializer`: "{initializer}" '
            f"Expected one of: {_SUPPORTED_RBF_KERNEL_TYPES}"
        )
    return initializer


def _get_default_scale(initializer, input_dim):
    if isinstance(initializer, str) and initializer.lower() == "gaussian":
        return math.sqrt(input_dim / 2.0)
    return 1.0


@keras.saving.register_keras_serializable(package="keras_proba")
class RandomFourierFeatures(keras.layers.Layer):
    """Projects inputs into a random feature space approximating an RBF kernel.

    Typically used to "kernelize" a linear model: a linear layer trained on top
    of these features is (up to approximation) a kernel machine, e.g. kernel
    logistic regression or the random feature Gaussian process used in SNGP.

    Example:

    ```python
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        RandomFourierFeatures(output_dim=4096, scale=10.0),
        keras.layers.Dense(10, activation="softmax"),
    ])
    ```

    Args:
        output_dim: Positive integer, the number of random features `D`.
        kernel_initializer: `"gaussian"`, `"laplacian"` (case insensitive), or a
            Keras initializer for the random feature kernel `W`. The kernel is
            never trainable.
        scale: Positive float, the kernel length scale. If `None`, defaults to
            `sqrt(input_dim / 2)` for `"gaussian"` and `1.0` otherwise. If
            `trainable=True`, it is learned and this is its initial value.
        trainable: Whether `scale` is trainable. Defaults to `False`.
        name: String, name of the layer.

    Input shape:
        `(..., input_dim)` with a defined last dimension. Unlike the reference,
        inputs of rank greater than 2 are supported.

    Output shape:
        `(..., output_dim)`.
    """

    def __init__(
        self,
        output_dim,
        kernel_initializer="gaussian",
        scale=None,
        trainable=False,
        name=None,
        **kwargs,
    ):
        if output_dim <= 0:
            raise ValueError(
                "`output_dim` should be a positive integer. "
                f"Received: {output_dim}"
            )
        if isinstance(kernel_initializer, str):
            if kernel_initializer.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
                raise ValueError(
                    f"Unsupported `kernel_initializer`: {kernel_initializer} "
                    f"Expected one of: {_SUPPORTED_RBF_KERNEL_TYPES}"
                )
        else:
            kernel_initializer = keras.initializers.get(kernel_initializer)
        if scale is not None and scale <= 0.0:
            raise ValueError(
                "When provided, `scale` should be a positive float. "
                f"Received: {scale}"
            )
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        self.scale = scale

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError(
                "The rank of the input tensor should be at least 2. "
                f"Full input shape received: {input_shape}"
            )
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input tensor should be defined. "
                f"Found `None`. Full input shape received: {input_shape}"
            )
        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})

        self.unscaled_kernel = self.add_weight(
            name="unscaled_kernel",
            shape=(input_dim, self.output_dim),
            initializer=_get_random_features_initializer(self.kernel_initializer),
            trainable=False,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim,),
            initializer=keras.initializers.RandomUniform(minval=0.0, maxval=2 * math.pi),
            trainable=False,
        )

        if self.scale is None:
            self.scale = _get_default_scale(self.kernel_initializer, input_dim)
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(1,),
            initializer=keras.initializers.Constant(self.scale),
            trainable=True,
            constraint="non_neg",
        )

    def call(self, inputs):
        kernel = (1.0 / self.kernel_scale) * self.unscaled_kernel
        outputs = ops.matmul(inputs, kernel) + self.bias
        return ops.cos(outputs)

    def compute_output_shape(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError(
                "The last dimension of the input tensor should be defined. "
                f"Found `None`. Full input shape received: {input_shape}"
            )
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self):
        kernel_initializer = self.kernel_initializer
        if not isinstance(kernel_initializer, str):
            kernel_initializer = keras.initializers.serialize(kernel_initializer)
        config = {
            "output_dim": self.output_dim,
            "kernel_initializer": kernel_initializer,
            "scale": self.scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}
