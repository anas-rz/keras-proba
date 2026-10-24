# Distance-aware models (SNGP)

The Spectral-normalized Neural Gaussian Process (SNGP) [^liu2020] makes a deterministic network
aware of the distance between a test input and its training data, so its predictions become
uncertain away from the data. It needs two changes to a standard network:

1. **Spectral normalization** of the hidden layers, so that distances in input space are
   preserved in feature space (the network is bi-Lipschitz).
2. A **Gaussian process output layer** instead of a dense layer, whose predictive variance grows
   with the distance from the training features.

The layers are ports of TensorFlow Model Garden's `official/nlp/modeling/layers`.

## The Gaussian process output layer

[`RandomFeatureGaussianProcess`](../api/layers.md#keras_proba.layers.RandomFeatureGaussianProcess)
approximates a GP with random Fourier features [^rahimi2007],
$\phi(x) = \sqrt{2 / D} \cos(x W + b)$, followed by a linear layer. During training it
accumulates the posterior precision matrix of the output weights over minibatches (the Laplace
approximation); at inference it also returns the predictive covariance:

```python
import numpy as np
import keras
from keras_proba import callbacks, layers, ops

inputs = keras.Input((16,))
logits, covmat = layers.RandomFeatureGaussianProcess(units=3, num_inducing=256)(inputs)
model = keras.Model(inputs, [logits, covmat])

x = np.random.normal(size=(8, 16)).astype("float32")
logits, covmat = model.predict(x, verbose=0)  # (8, 3) and (8, 8).

# Uncertainty-adjusted logits: the mean-field approximation of the posterior predictive.
adjusted = ops.mean_field_logits(logits, covmat, mean_field_factor=np.pi / 8)
```

The covariance is an estimate over the features seen during training. Because the features change
while the network trains, reset it at the start of every epoch, so that the final estimate only
reflects the last epoch:

```python
train_model = keras.Model(inputs, model.outputs[0])
train_model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
train_model.fit(x, np.zeros(8), epochs=2, verbose=0, callbacks=[callbacks.ResetCovarianceCallback()])
```

`ResetCovarianceCallback` finds every covariance layer in the model, including those nested in
other layers. With `gp_cov_momentum=-1`, the precision matrix is the exact sum over one epoch
instead of a moving average.

## Spectral normalization

[`SpectralNormalization`](../api/layers.md#keras_proba.layers.SpectralNormalization) wraps a
`Dense` layer (and
[`SpectralNormalizationConv2D`](../api/layers.md#keras_proba.layers.SpectralNormalizationConv2D)
a `Conv2D`) and bounds the spectral norm of its kernel by `norm_multiplier`, estimated with power
iteration. It has two modes:

| `mode` | Training rule | Backends |
|---|---|---|
| `"projection"` (default) | The kernel is projected to the norm bound before each step, as in the reference. | Identical to the reference on TensorFlow and PyTorch. On JAX, assignments to trainable variables inside `fit` are discarded, so the projected kernel is not persisted between steps; a warning is shown. |
| `"reparameterization"` | The layer is called with $W \cdot c / \sigma(W)$ and the gradient flows through $\sigma$, as in PyTorch's `spectral_norm`. The kernel variable is never modified. | Identical on every backend. |

In both modes, inference uses a normalized kernel on every backend.

## The SNGP classification head

[`GaussianProcessClassificationHead`](../api/layers.md#keras_proba.layers.GaussianProcessClassificationHead)
combines both: a spectral-normalized pooling layer and a GP output layer. It returns raw logits
during training (compatible with `model.fit`) and mean-field adjusted logits at inference
(`temperature` is the mean-field factor). Pass `return_covmat=True` to also get the covariance.

!!! warning "The defaults are tuned for BERT features"
    The head's defaults follow the reference, which was designed for BERT encoders: a `tanh`
    pooler, layer normalization of the GP input and random features with a standard deviation of
    0.05. On small dense inputs, these hide the distance from the data: in our tests,
    out-of-distribution confidence stayed at 0.77 to 0.98. For such inputs, use the setup of
    the SNGP paper: a distance-preserving backbone and

    ```python
    head = layers.GaussianProcessClassificationHead(
        inner_dim=0,                  # No tanh pooler.
        num_classes=2,
        normalize_input=False,        # Keep the input magnitude.
        scale_random_features=True,
        custom_random_features_initializer=keras.initializers.RandomNormal(stddev=1.0),
        temperature=np.pi / 8,
        mode="reparameterization",
    )
    ```

## A complete recipe

A distance-preserving backbone (a frozen random projection and residual spectral-normalized
blocks) with the head:

```python
from keras_proba import layers

def build_sngp(num_classes, hidden=64, blocks=2):
    inputs = keras.Input((2,))
    h = keras.layers.Dense(hidden, trainable=False)(inputs)
    for _ in range(blocks):
        block = layers.SpectralNormalization(
            keras.layers.Dense(hidden, activation="relu"), norm_multiplier=0.9, mode="reparameterization"
        )
        h = h + block(h)
    logits = layers.GaussianProcessClassificationHead(
        inner_dim=0,
        num_classes=num_classes,
        num_inducing=1024,
        normalize_input=False,
        scale_random_features=True,
        custom_random_features_initializer=keras.initializers.RandomNormal(stddev=1.0),
        temperature=np.pi / 8,
        gp_cov_momentum=-1,
    )(h)
    return keras.Model(inputs, logits)

model = build_sngp(num_classes=2)
model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

On a two-moons dataset this reaches about 99% confidence in distribution, and 0.55 to 0.73 far
from the data (0.5 is uniform), on every backend.

[^liu2020]: Jeremiah Liu et al. *Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness.* NeurIPS 2020. [arXiv:2006.10108](https://arxiv.org/abs/2006.10108)
[^rahimi2007]: Ali Rahimi, Benjamin Recht. *Random Features for Large-Scale Kernel Machines.* NeurIPS 2007.
