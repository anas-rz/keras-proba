# Conformal prediction

Split conformal prediction [^vovk2005] [^angelopoulos2021] turns any trained model into one with
prediction intervals (regression) or prediction sets (classification) that contain the true label
with probability at least $1 - \alpha$. The only assumption is that calibration and test
examples are exchangeable, e.g. drawn from the same distribution.

1. Compute a nonconformity score $s_i$ for each of $n$ held-out calibration examples.
2. Take $\hat{q}$, the $k$-th smallest score, with $k = \lceil (n + 1)(1 - \alpha) \rceil$.
3. For a test input, return every label whose score is at most $\hat{q}$.

The expected coverage is then exactly $k / (n + 1)$ for continuous scores, between $1 - \alpha$
and $1 - \alpha + 1 / (n + 1)$.

!!! note "The exact order statistic"
    `keras_proba` uses the $k$-th smallest score. The widely used
    `numpy.quantile(scores, k / n, method="higher")` selects the next larger one, which over-covers.

## Regression

[`SplitConformalRegressor`](../api/postprocessing.md#keras_proba.postprocessing.SplitConformalRegressor)
uses absolute residuals $|y - \hat{y}|$, giving intervals of constant width:

```python
import numpy as np
import keras
from keras_proba import layers, losses, ops, postprocessing

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, size=(4000, 1)).astype("float32")
y = (np.sin(x) + rng.normal(size=x.shape) * (0.1 + 0.4 * np.abs(x))).astype("float32")
(x_train, y_train), (x_cal, y_cal), (x_test, y_test) = [(x[s], y[s]) for s in (slice(0, 2000), slice(2000, 3000), slice(3000, None))]

model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(64, activation="relu"), layers.GaussianHead(1)])
model.compile(optimizer=keras.optimizers.Adam(0.005), loss=losses.GaussianNLL())
model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=0)
mean_cal, scale_cal = ops.split_gaussian_params(model.predict(x_cal, verbose=0))
mean_test, scale_test = ops.split_gaussian_params(model.predict(x_test, verbose=0))

conformal = postprocessing.SplitConformalRegressor(alpha=0.1).fit(y_cal, mean_cal)
lower, upper = conformal.predict_interval(mean_test)
print(float(postprocessing.interval_coverage(y_test, lower, upper)[0]))  # ~0.9
```

With a per-example `scale`, the score is the normalized residual $|y - \hat{y}| / \sigma$, and the
intervals $\hat{y} \pm \hat{q} \sigma$ adapt to each input: narrow where the noise is low, wide where
it is high. The scale can come from a `GaussianHead`, a `MixtureGaussianHead`
(`ops.mixture_mean_and_stddev`) or the spread of Monte Carlo samples of a Bayesian model:

```python
conformal = postprocessing.SplitConformalRegressor(alpha=0.1).fit(y_cal, mean_cal, scale=scale_cal)
lower, upper = conformal.predict_interval(mean_test, scale=scale_test)
```

Each output dimension gets its own threshold, so coverage holds for each dimension.

## Classification

[`SplitConformalClassifier`](../api/postprocessing.md#keras_proba.postprocessing.SplitConformalClassifier)
returns boolean prediction sets of shape `(n, num_classes)`, with two scores:

| `score` | Sets | Properties |
|---|---|---|
| `"lac"` | Classes with probability at least $1 - \hat{q}$ | Exact coverage and the smallest sets on average; can be empty for ambiguous inputs. |
| `"aps"` [^romano2020] | The most probable classes until their mass reaches $\hat{q}$ | Set size adapts to each input's difficulty. Randomized by default for exact coverage (set `seed=` for reproducibility). |

```python
rng = np.random.default_rng(1)
x = rng.normal(size=(3000, 2)).astype("float32")
y = (x @ rng.normal(size=(2, 5)) + rng.gumbel(size=(3000, 5))).argmax(-1)
classifier = keras.Sequential([keras.Input((2,)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(5, activation="softmax")])
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
classifier.fit(x[:2000], y[:2000], epochs=10, verbose=0)
probs_cal, probs_test = classifier.predict(x[2000:2500], verbose=0), classifier.predict(x[2500:], verbose=0)

conformal = postprocessing.SplitConformalClassifier(alpha=0.1, score="aps", seed=0).fit(y[2000:2500], probs_cal)
sets = conformal.predict_set(probs_test)
print(float(postprocessing.set_coverage(y[2500:], sets)), float(postprocessing.average_set_size(sets)))
```

!!! warning "Non-randomized APS"
    `randomized=False` gives the classic non-randomized APS sets, which are never empty but
    conservative. On a hard 5-class task (63% accuracy) with a 90% target, they covered 99.6% of
    labels with 3.7 classes per set, against 90.0% and 2.2 classes for randomized APS.

## Saving a fitted predictor

```python
config = conformal.get_config()  # JSON-compatible, including the fitted threshold.
restored = postprocessing.SplitConformalClassifier.from_config(config)
```

[^vovk2005]: Vladimir Vovk, Alexander Gammerman, Glenn Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
[^angelopoulos2021]: Anastasios N. Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
[^romano2020]: Yaniv Romano, Matteo Sesia, Emmanuel J. Candès. *Classification with Valid and Adaptive Coverage.* NeurIPS 2020. [arXiv:2006.02544](https://arxiv.org/abs/2006.02544)
