"""Multi-backend random feature Gaussian process output layer in Keras 3.

Ported from TensorFlow Model Garden's
`official/nlp/modeling/layers/gaussian_process.py`, the output layer used by
the Spectral-normalized Neural Gaussian Process (SNGP).

The layer approximates a Gaussian process with random features [1]:
    phi(x) = sqrt(2 / D) * cos(x W / sqrt(kernel_scale) + b)
    logits = phi(x) beta + bias

and estimates the posterior covariance of `beta` with the Laplace method,
accumulating the precision matrix over training minibatches:
    P = ridge_penalty * I + sum_i p_i (1 - p_i) phi(x_i)^T phi(x_i)
    Cov(x*) = ridge_penalty * phi(x*) P^-1 phi(x*)^T

[1]: Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel
     Machines. In _Neural Information Processing Systems_, 2007.
     https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
"""

import math

import keras
from keras import ops

from keras_proba.layers.random_fourier_features import RandomFourierFeatures

_SUPPORTED_LIKELIHOOD = ("binary_logistic", "poisson", "gaussian")


def _serialize(obj):
    return None if obj is None else keras.saving.serialize_keras_object(obj)


@keras.saving.register_keras_serializable(package="keras_proba")
class RandomFeatureGaussianProcess(keras.layers.Layer):
    """Gaussian process layer with random feature approximation [1].

    During training, the layer updates the maximum a posteriori (MAP) logits
    and the posterior precision matrix using minibatch statistics. During
    inference, it also returns the predictive covariance, which can be used to
    compute posterior mean probabilities via `mean_field_logits`.

    Different random features can be used by setting
    `use_custom_random_features=True` and changing the initializer and
    activation of the custom random features. For example:

        MLP kernel: initializer="random_normal", activation="relu"
        RBF kernel: initializer="random_normal", activation=keras.ops.cos

    A linear kernel can be used with `gp_kernel_type="linear"` and
    `use_custom_random_features=True`.

    Example:

    ```python
    inputs = keras.Input(shape=(128,))
    logits, covmat = RandomFeatureGaussianProcess(units=10)(inputs)
    probs = keras.ops.softmax(mean_field_logits(logits, covmat, math.pi / 8))
    ```

    Args:
        units: Number of output units.
        num_inducing: Number of random Fourier features.
        gp_kernel_type: Kernel type. `"gaussian"` (RBF) by default, or
            `"linear"` when `use_custom_random_features=True`.
        gp_kernel_scale: Length scale of the kernel, i.e. for the RBF kernel
            `exp(-|x1 - x2|**2 / gp_kernel_scale)`.
        gp_output_bias: Initial value of the output bias.
        normalize_input: Whether to apply layer normalization to the input.
        gp_kernel_scale_trainable: Whether the length scale is trainable. Only
            used when `use_custom_random_features=False`.
        gp_output_bias_trainable: Whether the output bias is trainable.
        gp_cov_momentum: Discount factor of the moving average of the precision
            matrix. If not positive, minibatch precision matrices are summed
            instead, which gives the exact precision after one pass over the
            data.
        gp_cov_ridge_penalty: Ridge penalty of the initial precision matrix.
        scale_random_features: Whether to scale random features by
            `sqrt(2 / num_inducing)`.
        use_custom_random_features: Whether to use a frozen `Dense` layer for
            the random features instead of `RandomFourierFeatures`.
        custom_random_features_initializer: Initializer for the custom random
            features. Defaults to `RandomNormal(stddev=1.0)`.
        custom_random_features_activation: Activation for the custom random
            features. Defaults to `keras.ops.cos`.
        l2_regularization: L2 regularization strength on the output weights.
        gp_cov_likelihood: Likelihood used for the Laplace approximation, one of
            `"gaussian"`, `"binary_logistic"` or `"poisson"`.
        return_gp_cov: Whether to also return the GP covariance matrix. If
            `False`, no covariance is learned.
        return_random_features: Whether to also return the random features.
        **gp_output_kwargs: Additional keyword arguments to the output `Dense`
            layer.

    Call arguments:
        inputs: Tensor of shape `(batch_size, input_dim)`.
        training: Whether to update the precision matrix.

    Returns:
        `logits` of shape `(batch_size, units)` if neither `return_gp_cov` nor
        `return_random_features` is set. Otherwise a tuple of `logits`, the
        covariance of shape `(batch_size, batch_size)` (identity during
        training) and the random features of shape
        `(batch_size, num_inducing)`, in that order.
    """

    def __init__(
        self,
        units,
        num_inducing=1024,
        gp_kernel_type="gaussian",
        gp_kernel_scale=1.0,
        gp_output_bias=0.0,
        normalize_input=False,
        gp_kernel_scale_trainable=False,
        gp_output_bias_trainable=False,
        gp_cov_momentum=0.999,
        gp_cov_ridge_penalty=1.0,
        scale_random_features=True,
        use_custom_random_features=True,
        custom_random_features_initializer=None,
        custom_random_features_activation=None,
        l2_regularization=1e-6,
        gp_cov_likelihood="gaussian",
        return_gp_cov=True,
        return_random_features=False,
        dtype=None,
        name=None,
        trainable=True,
        **gp_output_kwargs,
    ):
        super().__init__(name=name, dtype=dtype, trainable=trainable)
        self.units = units
        self.num_inducing = num_inducing

        self.normalize_input = normalize_input
        self.gp_input_scale = 1.0 / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2.0 / float(num_inducing))

        self.scale_random_features = scale_random_features
        self.return_random_features = return_random_features
        self.return_gp_cov = return_gp_cov

        self.gp_kernel_type = gp_kernel_type
        self.gp_kernel_scale = gp_kernel_scale
        self.gp_output_bias = gp_output_bias
        self.gp_kernel_scale_trainable = gp_kernel_scale_trainable
        self.gp_output_bias_trainable = gp_output_bias_trainable

        self.use_custom_random_features = use_custom_random_features
        self.custom_random_features_initializer = (
            None
            if custom_random_features_initializer is None
            else keras.initializers.get(custom_random_features_initializer)
        )
        self.custom_random_features_activation = (
            None
            if custom_random_features_activation is None
            else keras.activations.get(custom_random_features_activation)
        )

        self.l2_regularization = l2_regularization
        self.gp_output_kwargs = gp_output_kwargs

        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_likelihood = gp_cov_likelihood

    def build(self, input_shape):
        if self.normalize_input:
            self._input_norm_layer = keras.layers.LayerNormalization(
                dtype=self.dtype_policy, name="gp_input_normalization"
            )
            self._input_norm_layer.build(input_shape)
            input_shape = self._input_norm_layer.compute_output_shape(input_shape)

        self._random_feature = self._make_random_feature_layer(name="gp_random_feature")
        self._random_feature.build(input_shape)
        input_shape = self._random_feature.compute_output_shape(input_shape)

        if self.return_gp_cov:
            self._gp_cov_layer = LaplaceRandomFeatureCovariance(
                momentum=self.gp_cov_momentum,
                ridge_penalty=self.gp_cov_ridge_penalty,
                likelihood=self.gp_cov_likelihood,
                dtype=self.dtype_policy,
                name="gp_covariance",
            )
            self._gp_cov_layer.build(input_shape)

        self._gp_output_layer = keras.layers.Dense(
            units=self.units,
            use_bias=False,
            kernel_regularizer=keras.regularizers.L2(self.l2_regularization),
            dtype=self.dtype_policy,
            name="gp_output_weights",
            **self.gp_output_kwargs,
        )
        self._gp_output_layer.build(input_shape)

        self._gp_output_bias = self.add_weight(
            name="gp_output_bias",
            shape=(self.units,),
            initializer=keras.initializers.Constant(self.gp_output_bias),
            trainable=self.gp_output_bias_trainable,
        )

    def _make_random_feature_layer(self, name):
        """Defines random feature layer depending on kernel type."""
        if not self.use_custom_random_features:
            return RandomFourierFeatures(
                output_dim=self.num_inducing,
                kernel_initializer=self.gp_kernel_type,
                scale=self.gp_kernel_scale,
                trainable=self.gp_kernel_scale_trainable,
                dtype=self.dtype_policy,
                name=name,
            )

        if self.gp_kernel_type.lower() == "linear":
            return keras.layers.Identity(dtype=self.dtype_policy, name=name)

        # Defaults to the Gaussian RBF kernel.
        initializer = self.custom_random_features_initializer
        if initializer is None:
            initializer = keras.initializers.RandomNormal(stddev=1.0)
        activation = self.custom_random_features_activation
        if activation is None:
            activation = ops.cos
        return keras.layers.Dense(
            units=self.num_inducing,
            use_bias=True,
            activation=activation,
            kernel_initializer=initializer,
            bias_initializer=keras.initializers.RandomUniform(minval=0.0, maxval=2.0 * math.pi),
            trainable=False,
            dtype=self.dtype_policy,
            name=name,
        )

    def reset_covariance_matrix(self):
        """Resets the covariance matrix of the GP layer.

        Useful for resetting the covariance at the beginning of a new epoch.
        """
        self._gp_cov_layer.reset_precision_matrix()

    def call(self, inputs, training=None):
        # Computes random features.
        gp_inputs = inputs
        if self.normalize_input:
            gp_inputs = self._input_norm_layer(gp_inputs)
        elif self.use_custom_random_features:
            # Supports length scale for custom random features by directly
            # rescaling the input.
            gp_inputs = gp_inputs * ops.cast(self.gp_input_scale, gp_inputs.dtype)

        gp_feature = self._random_feature(gp_inputs)

        if self.scale_random_features:
            # Scale random features by sqrt(2 / num_inducing) following [1].
            # When using the GP layer as the output layer of a neural network,
            # it is recommended to turn this off to prevent it from changing
            # the learning rate of the hidden layers.
            gp_feature = gp_feature * ops.cast(self.gp_feature_scale, gp_feature.dtype)

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        model_output = [gp_output]
        if self.return_gp_cov:
            model_output.append(self._gp_cov_layer(gp_feature, gp_output, training=training))
        if self.return_random_features:
            model_output.append(gp_feature)

        if len(model_output) == 1:
            return gp_output
        return tuple(model_output)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        output_shape = [(batch_size, self.units)]
        if self.return_gp_cov:
            output_shape.append((batch_size, batch_size))
        if self.return_random_features:
            is_identity = self.use_custom_random_features and self.gp_kernel_type.lower() == "linear"
            feature_dim = input_shape[-1] if is_identity else self.num_inducing
            output_shape.append((batch_size, feature_dim))
        if len(output_shape) == 1:
            return output_shape[0]
        return tuple(output_shape)

    def get_config(self):
        config = {
            "units": self.units,
            "num_inducing": self.num_inducing,
            "gp_kernel_type": self.gp_kernel_type,
            "gp_kernel_scale": self.gp_kernel_scale,
            "gp_output_bias": self.gp_output_bias,
            "normalize_input": self.normalize_input,
            "gp_kernel_scale_trainable": self.gp_kernel_scale_trainable,
            "gp_output_bias_trainable": self.gp_output_bias_trainable,
            "gp_cov_momentum": self.gp_cov_momentum,
            "gp_cov_ridge_penalty": self.gp_cov_ridge_penalty,
            "scale_random_features": self.scale_random_features,
            "use_custom_random_features": self.use_custom_random_features,
            "custom_random_features_initializer": _serialize(self.custom_random_features_initializer),
            "custom_random_features_activation": _serialize(self.custom_random_features_activation),
            "l2_regularization": self.l2_regularization,
            "gp_cov_likelihood": self.gp_cov_likelihood,
            "return_gp_cov": self.return_gp_cov,
            "return_random_features": self.return_random_features,
        }
        config.update({key: _serialize(value) for key, value in self.gp_output_kwargs.items()})
        base_config = super().get_config()
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="keras_proba")
class LaplaceRandomFeatureCovariance(keras.layers.Layer):
    """Computes the Gaussian process covariance using the Laplace method.

    At training time, this layer updates the Gaussian process posterior using
    model features in minibatches.

    Args:
        momentum: Discount factor of the moving average of the posterior
            precision matrix, analogous to batch normalization momentum. If not
            positive, minibatch precision matrices are summed instead, which
            gives the exact precision matrix after one pass over the data (e.g.
            in the final epoch).
        ridge_penalty: Ridge penalty of the initial precision matrix. It
            stabilizes the inverse `inv(t(X) X + s I)`, but should not be so
            large that it dominates `t(X) X`.
        likelihood: Likelihood used for the Laplace approximation, one of
            `"gaussian"`, `"binary_logistic"` or `"poisson"`.

    Call arguments:
        inputs: GP random features of shape `(batch_size, num_features)`.
        logits: Pre-activation output of shape `(batch_size, 1)`. Required for
            non-Gaussian likelihoods.
        training: Whether to update the precision matrix.

    Returns:
        Identity of shape `(batch_size, batch_size)` during training, and the
        predictive covariance of the same shape during inference.
    """

    def __init__(
        self,
        momentum=0.999,
        ridge_penalty=1.0,
        likelihood="gaussian",
        dtype=None,
        name=None,
        **kwargs,
    ):
        if likelihood not in _SUPPORTED_LIKELIHOOD:
            raise ValueError(
                f'"likelihood" must be one of {_SUPPORTED_LIKELIHOOD}, got {likelihood}.'
            )
        super().__init__(dtype=dtype, name=name, **kwargs)
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.likelihood = likelihood

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0])

    def build(self, input_shape):
        gp_feature_dim = input_shape[-1]

        # Posterior precision matrix for the GP's random feature coefficients.
        self.precision_matrix = self.add_weight(
            name="gp_precision_matrix",
            shape=(gp_feature_dim, gp_feature_dim),
            initializer=keras.initializers.Identity(self.ridge_penalty),
            trainable=False,
            aggregation="only_first_replica",
        )

    def make_precision_matrix_update_op(self, gp_feature, logits, precision_matrix):
        """Updates the precision matrix of feature weights."""
        if self.likelihood != "gaussian":
            if logits is None:
                raise ValueError(f'"logits" cannot be None when likelihood={self.likelihood}')

            if logits.shape[-1] != 1:
                raise ValueError(
                    f"likelihood={self.likelihood} only support univariate logits. "
                    f"Got logits dimension: {logits.shape[-1]}"
                )

        # The precision matrix is a running statistic, not part of the graph.
        gp_feature = ops.stop_gradient(gp_feature)
        batch_size = ops.cast(ops.shape(gp_feature)[0], gp_feature.dtype)

        # Computes batch-specific normalized precision matrix.
        if self.likelihood == "binary_logistic":
            prob = ops.sigmoid(ops.stop_gradient(logits))
            prob_multiplier = prob * (1.0 - prob)
        elif self.likelihood == "poisson":
            prob_multiplier = ops.exp(ops.stop_gradient(logits))
        else:
            prob_multiplier = 1.0

        gp_feature_adjusted = ops.sqrt(prob_multiplier) * gp_feature
        precision_matrix_minibatch = ops.matmul(ops.transpose(gp_feature_adjusted), gp_feature_adjusted)

        # Updates the population-wise precision matrix.
        if self.momentum > 0:
            # Use moving-average updates to accumulate batch-specific
            # precision matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                self.momentum * precision_matrix + (1.0 - self.momentum) * precision_matrix_minibatch
            )
        else:
            # Compute exact population-wise covariance without momentum.
            # If using this option, make sure to pass through data only once.
            precision_matrix_new = precision_matrix + precision_matrix_minibatch

        precision_matrix.assign(ops.cast(precision_matrix_new, precision_matrix.dtype))

    def reset_precision_matrix(self):
        """Resets the precision matrix to its initial value.

        Useful for resetting the covariance at the beginning of a new epoch.
        """
        dim = self.precision_matrix.shape[0]
        self.precision_matrix.assign(
            self.ridge_penalty * ops.eye(dim, dtype=self.precision_matrix.dtype)
        )

    def compute_predictive_covariance(self, gp_feature):
        """Computes the posterior predictive covariance.

        Given training random features Phi_tr of shape (num_train, num_hidden)
        and test random features Phi_ts of shape (batch_size, num_hidden), the
        predictive covariance (assuming a Gaussian likelihood) is:

            s * Phi_ts @ inv(t(Phi_tr) * Phi_tr + s * I) @ t(Phi_ts),

        where s is the ridge penalty used to stabilize the inverse.

        Args:
            gp_feature: Random features of the test data, shape
                `(batch_size, num_hidden)`.

        Returns:
            Predictive covariance matrix of shape `(batch_size, batch_size)`.
        """
        # Computes the covariance matrix of the feature coefficients.
        feature_cov_matrix = ops.inv(ops.cast(self.precision_matrix, gp_feature.dtype))

        # Computes the covariance matrix of the GP prediction.
        cov_feature_product = ops.matmul(feature_cov_matrix, ops.transpose(gp_feature)) * self.ridge_penalty
        return ops.matmul(gp_feature, cov_feature_product)

    def call(self, inputs, logits=None, training=None):
        if training:
            self.make_precision_matrix_update_op(
                gp_feature=inputs, logits=logits, precision_matrix=self.precision_matrix
            )
            # Return null estimate during training.
            return ops.eye(ops.shape(inputs)[0], dtype=inputs.dtype)
        # Return covariance estimate during inference.
        return self.compute_predictive_covariance(gp_feature=inputs)

    def get_config(self):
        config = {
            "momentum": self.momentum,
            "ridge_penalty": self.ridge_penalty,
            "likelihood": self.likelihood,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def mean_field_logits(logits, covariance_matrix=None, mean_field_factor=1.0):
    """Adjusts the model logits so their softmax approximates the posterior mean [1].

    [1]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with
         Infinitesimal Jackknife. _arXiv preprint arXiv:2006.07584_, 2020.
         https://arxiv.org/abs/2006.07584

    Args:
        logits: Float tensor of shape `(batch_size, num_classes)`.
        covariance_matrix: Covariance matrix of shape `(batch_size, batch_size)`.
            If `None`, it is assumed to be the identity.
        mean_field_factor: Scale factor of the mean-field approximation, which
            controls the influence of the posterior variance. If
            `covariance_matrix=None`, it acts as a temperature for temperature
            scaling. If `None` or negative, logits are returned unchanged.

    Returns:
        Adjusted logits of shape `(batch_size, num_classes)`.
    """
    if mean_field_factor is None or mean_field_factor < 0:
        return logits

    logits = ops.convert_to_tensor(logits)

    # Compute standard deviation.
    if covariance_matrix is None:
        variances = 1.0
    else:
        variances = ops.cast(ops.diagonal(covariance_matrix), logits.dtype)

    # Compute scaling coefficient for mean-field approximation.
    logits_scale = ops.sqrt(1.0 + variances * mean_field_factor)

    if len(logits.shape) > 1:
        # Cast logits_scale to compatible dimension.
        logits_scale = ops.expand_dims(logits_scale, axis=-1)

    return logits / logits_scale
