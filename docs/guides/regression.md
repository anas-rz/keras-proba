# Probabilistic regression

A regression model trained with mean squared error predicts a single value. A probabilistic
regression head predicts a distribution per input, so the model also learns how noisy each
target is (aleatoric uncertainty), and it is trained by maximum likelihood.

## Gaussian head

[`GaussianHead`](../api/layers.md#keras_proba.layers.GaussianHead) predicts a mean and a standard
deviation per output [^nix1994] [^lakshminarayanan2017]:
$y \sim \mathcal{N}(\mu(x), \sigma(x)^2)$ with $\sigma = \mathrm{softplus}(\cdot) + $ `min_scale`.
Its output is `[mean, scale]` concatenated on the last axis, so it trains with plain targets and
[`GaussianNLL`](../api/losses.md#keras_proba.losses.GaussianNLL):

```python
import numpy as np
import keras
from keras_proba import layers, losses, ops

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, size=(2048, 1)).astype("float32")
noise_std = 0.05 + 0.3 * np.abs(x)  # The noise grows with |x|.
y = (np.sin(x) + rng.normal(size=x.shape) * noise_std).astype("float32")

model = keras.Sequential([
    keras.Input((1,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    layers.GaussianHead(1),
])
model.compile(optimizer=keras.optimizers.Adam(0.005), loss=losses.GaussianNLL())
model.fit(x, y, epochs=20, batch_size=128, verbose=0)

mean, scale = ops.split_gaussian_params(model.predict(x[:5], verbose=0))
```

The predicted `scale` tracks the true noise level, and `mean ± 1.96 * scale` is a 95% interval.
For intervals with a guaranteed coverage, pass `mean` and `scale` to a
[conformal regressor](conformal.md#regression).

## Mixture density network

A single Gaussian cannot represent targets with several modes, e.g. $y = \pm x^2$.
[`MixtureGaussianHead`](../api/layers.md#keras_proba.layers.MixtureGaussianHead) predicts a mixture
of `num_components` Gaussians [^bishop1994]:

$$p(y \mid x) = \sum_k \pi_k(x) \prod_d \mathcal{N}(y_d; \mu_{kd}(x), \sigma_{kd}(x)^2)$$

```python
sign = np.where(rng.uniform(size=x.shape) < 0.5, -1.0, 1.0)
y_bimodal = (sign * x**2 + rng.normal(size=x.shape) * 0.1).astype("float32")

model = keras.Sequential([
    keras.Input((1,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    layers.MixtureGaussianHead(num_components=3),
])
model.compile(optimizer=keras.optimizers.Adam(0.005), loss=losses.MixtureGaussianNLL())
model.fit(x, y_bimodal, epochs=20, batch_size=128, verbose=0)

params = model.predict(x[:5], verbose=0)
logits, means, scales = ops.split_mixture_params(params)  # (5, 3), (5, 3, 1), (5, 3, 1).
mean, stddev = ops.mixture_mean_and_stddev(params)  # Including the spread between components.
```

!!! tip "Training mixtures"
    Mixtures can get stuck covering only some of the modes. In our tests, `tanh` hidden layers at
    a learning rate of 0.005 saturated and missed a mode in about half the runs, while ReLU
    features recovered both modes in every run. Prefer ReLU-like features or a small learning
    rate.

The mixture NLL sums the log-densities of a multi-output target (its joint likelihood), while
`GaussianNLL` averages them over outputs; they agree for one output and one component.

[^nix1994]: David A. Nix, Andreas S. Weigend. *Estimating the mean and variance of the target probability distribution.* ICNN 1994.
[^lakshminarayanan2017]: Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
[^bishop1994]: Christopher M. Bishop. *Mixture Density Networks.* Technical report, Aston University, 1994.
