"""Multi-backend classification heads, including the SNGP head, in Keras 3.

Ported from TensorFlow Model Garden's `official/nlp/modeling/layers/cls_head.py`.

`GaussianProcessClassificationHead` implements the spectral-normalized neural
Gaussian process (SNGP) [1]: a spectral-normalized hidden layer followed by a
random feature Gaussian process output layer. At inference, logits are adjusted
by the predictive variance with the mean-field approximation.

[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     In _Neural Information Processing Systems_, 2020.
     https://arxiv.org/abs/2006.10108
"""

import keras

from keras_proba.layers.gaussian_process import RandomFeatureGaussianProcess, mean_field_logits
from keras_proba.layers.spectral_normalization import SpectralNormalization


def _clone_initializer(initializer):
    # Keras initializers are stateless, so reusing an instance would give
    # identical values for weights of the same shape.
    if isinstance(initializer, keras.initializers.Initializer):
        return initializer.__class__.from_config(initializer.get_config())
    return initializer


@keras.saving.register_keras_serializable(package="keras_proba")
class ClassificationHead(keras.layers.Layer):
    """Pooling head for classification tasks.

    Args:
        inner_dim: Dimensionality of the inner projection layer. If 0 or
            `None`, only the output projection layer is created.
        num_classes: Number of output classes.
        cls_token_idx: Index of the sequence position to pool when the input
            has rank 3, e.g. the `<CLS>` token of a sequence encoder.
        activation: Activation of the inner projection layer.
        dropout_rate: Dropout probability before the output projection.
        initializer: Initializer for the dense layer kernels.
        **kwargs: Keyword arguments for `keras.layers.Layer`.

    Call arguments:
        features: Tensor of shape `(batch_size, seq_len, hidden_size)`, which is
            pooled at `cls_token_idx`, or `(batch_size, hidden_size)`.
        only_project: If `True`, returns the inner projection instead of the
            logits.
        training: Whether the layer is in training mode.
    """

    def __init__(
        self,
        inner_dim,
        num_classes,
        cls_token_idx=0,
        activation="tanh",
        dropout_rate=0.0,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.activation = keras.activations.get(activation)
        self.initializer = keras.initializers.get(initializer)
        self.cls_token_idx = cls_token_idx

        if self.inner_dim:
            self.dense = keras.layers.Dense(
                units=self.inner_dim,
                activation=self.activation,
                kernel_initializer=_clone_initializer(self.initializer),
                name="pooler_dense",
            )
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

        self.out_proj = keras.layers.Dense(
            units=num_classes,
            kernel_initializer=_clone_initializer(self.initializer),
            name="logits",
        )

    def build(self, input_shape):
        shape = tuple(input_shape)
        if len(shape) == 3:
            shape = (shape[0], shape[2])  # Pooled at `cls_token_idx`.
        if self.inner_dim:
            self.dense.build(shape)
            shape = self.dense.compute_output_shape(shape)
        self.out_proj.build(shape)

    def _project(self, features, training=None):
        x = features
        if len(features.shape) == 3:
            x = features[:, self.cls_token_idx, :]  # Take the <CLS> token.
        if self.inner_dim:
            x = self.dense(x, training=training)
        return x

    def call(self, features, only_project=False, training=None):
        x = self._project(features, training=training)
        if only_project:
            return x
        x = self.dropout(x, training=training)
        return self.out_proj(x)

    def get_config(self):
        config = {
            "cls_token_idx": self.cls_token_idx,
            "dropout_rate": self.dropout_rate,
            "num_classes": self.num_classes,
            "inner_dim": self.inner_dim,
            "activation": keras.activations.serialize(self.activation),
            "initializer": keras.initializers.serialize(self.initializer),
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="keras_proba")
class GaussianProcessClassificationHead(ClassificationHead):
    """Gaussian process-based pooling head for classification (SNGP) [1].

    It applies spectral normalization to the hidden pooler layer, and replaces
    the dense output layer with a random feature Gaussian process. During
    training it returns the raw logits; at inference it returns logits adjusted
    with `mean_field_logits`, and optionally the covariance matrix.

    Remember to reset the GP covariance at the start of each epoch, e.g. with
    `keras_proba.callbacks.ResetCovarianceCallback`.

    Example:

    ```python
    inputs = keras.Input((128,))
    hidden = keras.layers.Dense(128, activation="relu")(inputs)
    head = GaussianProcessClassificationHead(inner_dim=128, num_classes=10, temperature=math.pi / 8)
    model = keras.Model(inputs, head(hidden))
    ```

    Args:
        inner_dim: Dimensionality of the inner projection layer. If 0 or
            `None`, only the output projection layer is created.
        num_classes: Number of output classes.
        cls_token_idx: Index of the sequence position to pool when the input
            has rank 3.
        activation: Activation of the inner projection layer.
        dropout_rate: Dropout probability before the output projection.
        initializer: Initializer for the dense layer kernels.
        use_spec_norm: Whether to apply spectral normalization to the pooler
            layer.
        use_gp_layer: Whether to use a Gaussian process as the output layer.
        temperature: Mean-field factor used at inference. If `None`, no
            mean-field adjustment is applied.
        **kwargs: Spectral normalization arguments (`iteration`,
            `norm_multiplier`, `mode`), `RandomFeatureGaussianProcess`
            arguments (see `extract_gp_layer_kwargs` for those supported and
            their defaults), and keyword arguments for `keras.layers.Layer`.

    Call arguments:
        features: Tensor of shape `(batch_size, seq_len, hidden_size)` or
            `(batch_size, hidden_size)`.
        training: Whether the layer is in training mode.
        return_covmat: Whether to also return the covariance matrix of shape
            `(batch_size, batch_size)` when `use_gp_layer=True`. During
            training, keep it `False` to be compatible with `model.fit()`.
    """

    def __init__(
        self,
        inner_dim,
        num_classes,
        cls_token_idx=0,
        activation="tanh",
        dropout_rate=0.0,
        initializer="glorot_uniform",
        use_spec_norm=True,
        use_gp_layer=True,
        temperature=None,
        **kwargs,
    ):
        # Collects spectral normalization and Gaussian process args from kwargs.
        self.use_spec_norm = use_spec_norm
        self.use_gp_layer = use_gp_layer
        self.spec_norm_kwargs = extract_spec_norm_kwargs(kwargs)
        self.gp_layer_kwargs = extract_gp_layer_kwargs(kwargs)
        self.temperature = temperature

        super().__init__(
            inner_dim=inner_dim,
            num_classes=num_classes,
            cls_token_idx=cls_token_idx,
            activation=activation,
            dropout_rate=dropout_rate,
            initializer=initializer,
            **kwargs,
        )

        # Applies spectral normalization to the dense pooler layer.
        if self.use_spec_norm and hasattr(self, "dense"):
            self.dense = SpectralNormalization(self.dense, inhere_layer_name=True, **self.spec_norm_kwargs)

        # Replace Dense output layer with the Gaussian process layer.
        if use_gp_layer:
            self.out_proj = RandomFeatureGaussianProcess(
                self.num_classes,
                kernel_initializer=_clone_initializer(self.initializer),
                name="logits",
                **self.gp_layer_kwargs,
            )

    def call(self, features, training=None, return_covmat=False):
        x = self._project(features, training=training)
        x = self.dropout(x, training=training)
        outputs = self.out_proj(x, training=training)

        # Extracts logits and covariance matrix from model output.
        covmat = None
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            if self.gp_layer_kwargs["return_gp_cov"]:
                covmat = outputs[1]
        else:
            logits = outputs

        # Computes the uncertainty-adjusted logits during evaluation.
        if not training:
            logits = mean_field_logits(logits, covmat, mean_field_factor=self.temperature)

        if return_covmat and covmat is not None:
            return logits, covmat
        return logits

    def reset_covariance_matrix(self):
        """Resets the covariance matrix of the Gaussian process layer."""
        if hasattr(self.out_proj, "reset_covariance_matrix"):
            self.out_proj.reset_covariance_matrix()

    def get_config(self):
        config = {"use_spec_norm": self.use_spec_norm, "use_gp_layer": self.use_gp_layer}
        config.update(self.spec_norm_kwargs)
        config.update(
            {
                key: keras.saving.serialize_keras_object(value)
                if key.startswith("custom_random_features_") and value is not None
                else value
                for key, value in self.gp_layer_kwargs.items()
            }
        )
        config["temperature"] = self.temperature
        config.update(super().get_config())
        return config


def extract_gp_layer_kwargs(kwargs):
    """Extracts Gaussian process layer configs from a given kwarg."""
    return dict(
        num_inducing=kwargs.pop("num_inducing", 1024),
        normalize_input=kwargs.pop("normalize_input", True),
        gp_cov_momentum=kwargs.pop("gp_cov_momentum", 0.999),
        gp_cov_ridge_penalty=kwargs.pop("gp_cov_ridge_penalty", 1.0),
        scale_random_features=kwargs.pop("scale_random_features", False),
        l2_regularization=kwargs.pop("l2_regularization", 1e-6),
        gp_cov_likelihood=kwargs.pop("gp_cov_likelihood", "gaussian"),
        return_gp_cov=kwargs.pop("return_gp_cov", True),
        return_random_features=kwargs.pop("return_random_features", False),
        use_custom_random_features=kwargs.pop("use_custom_random_features", True),
        custom_random_features_initializer=kwargs.pop("custom_random_features_initializer", "random_normal"),
        custom_random_features_activation=kwargs.pop("custom_random_features_activation", None),
    )


def extract_spec_norm_kwargs(kwargs):
    """Extracts spectral normalization configs from a given kwarg."""
    return dict(
        iteration=kwargs.pop("iteration", 1),
        norm_multiplier=kwargs.pop("norm_multiplier", 0.99),
        mode=kwargs.pop("mode", "projection"),
    )
