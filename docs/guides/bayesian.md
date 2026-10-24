# Bayesian layers

Bayesian layers learn a distribution over their weights instead of a point estimate, with
variational inference [^blundell2015]. Their predictions are stochastic: averaging several of them
gives the predictive mean, and their spread is the model's (epistemic) uncertainty.

The layers are ports of TensorFlow Probability's `tfp.layers`, with its default distributions:

- **Posterior**: a mean-field Normal $q(W) = \mathcal{N}(\mu, \sigma^2)$ with
  $\sigma = \epsilon + \mathrm{softplus}(\rho)$, initialized close to a point mass.
- **Prior**: $p(W) = \mathcal{N}(0, s^2)$ with `kernel_prior_scale` $= s$ (default 1).
- **Bias**: a point estimate.

Each call samples a kernel and adds `kl_weight * KL(q(W) || p(W))` to the layer's losses, which
`model.fit` adds to the training loss.

## Estimators

| Layers | Estimator | Notes |
|---|---|---|
| `DenseReparameterization`, `Conv{1,2,3}DReparameterization` | Reparameterization [^kingma2014] | One kernel sample per call, shared by the batch. |
| `DenseLocalReparameterization` | Local reparameterization [^kingma2015] | Samples the pre-activations directly, independently per example. |
| `DenseFlipout`, `Conv{1,2,3}DFlipout` | Flipout [^wen2018] | Pseudo-independent perturbations per example with random signs: lower gradient variance, about twice the compute. |

## Setting `kl_weight`

With the default mean loss over examples, the KL term must be counted once per epoch: set
`kl_weight = 1 / num_train_examples`. With `kl_weight=1` (the TFP default), the prior dominates
and the posterior stays close to it.

```python
import numpy as np
import keras
from keras_proba import layers

rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, size=(512, 1)).astype("float32")
y = (np.sin(2 * x) + rng.normal(size=(512, 1)) * 0.1).astype("float32")

kl_weight = 1.0 / len(x)
model = keras.Sequential([
    keras.Input((1,)),
    layers.DenseFlipout(64, activation="relu", kl_weight=kl_weight),
    layers.DenseFlipout(64, activation="relu", kl_weight=kl_weight),
    layers.DenseFlipout(1, kl_weight=kl_weight),
])
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
model.fit(x, y, epochs=20, batch_size=64, verbose=0)
```

## Monte Carlo prediction

Every `predict` call draws fresh weights, on every backend, so the predictive distribution is a
loop away:

```python
x_test = np.linspace(-4, 4, 50, dtype="float32")[:, None]
samples = np.stack([model.predict(x_test, verbose=0) for _ in range(30)])
mean, epistemic_std = samples.mean(0), samples.std(0)  # The spread grows outside [-2, 2].
```

Pass `seed=` to a layer for reproducible samples.

## Convolutions

`Conv1D`, `Conv2D` and `Conv3D` variants take the usual `filters`, `kernel_size`, `strides`,
`padding` (`"valid"` or `"same"`), `data_format` and `dilation_rate`:

```python
model = keras.Sequential([
    keras.Input((28, 28, 1)),
    layers.Conv2DFlipout(16, 3, padding="same", activation="relu", kl_weight=1e-4),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    layers.DenseFlipout(10, kl_weight=1e-4),
])
```

!!! note "Differences from TFP"
    - The posterior and prior are fixed to TFP's defaults and configured with plain arguments,
      instead of arbitrary `tfd` distribution functions.
    - `kl_weight` replaces passing a custom divergence function to scale the KL term.
    - `DenseLocalReparameterization` adds the smallest positive float under its square root: in
      TFP, an all-zero input (e.g. after a ReLU) makes the scale gradient NaN.

[^blundell2015]: Charles Blundell et al. *Weight Uncertainty in Neural Networks.* ICML 2015.
[^kingma2014]: Diederik Kingma, Max Welling. *Auto-Encoding Variational Bayes.* ICLR 2014.
[^kingma2015]: Diederik Kingma, Tim Salimans, Max Welling. *Variational Dropout and the Local Reparameterization Trick.* NeurIPS 2015.
[^wen2018]: Yeming Wen et al. *Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches.* ICLR 2018.
