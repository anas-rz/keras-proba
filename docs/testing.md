# Testing

The test suite runs once per backend:

```bash
pip install -e ".[dev]"
KERAS_BACKEND=jax pytest          # Or torch, tensorflow.
```

## Test categories

| Folder | What it checks |
|---|---|
| `tests/tests_parity` | Numerical parity with the original implementation, given the same weights and random draws. |
| `tests/tests_smoke` | Shapes, analytic values, invariants, error handling and the public API. |
| `tests/tests_training` | That each method does its job when trained: out-of-distribution uncertainty, improved calibration, conformal coverage of exactly $k / (n + 1)$, ... |
| `tests/tests_serialization` | `get_config` / `from_config` and `model.save` / `keras.models.load_model` round trips. |

## Parity references

| Component | Reference |
|---|---|
| `RandomFourierFeatures` | `tf_keras.layers.experimental.RandomFourierFeatures` |
| Gaussian process, spectral normalization, SNGP head | TensorFlow Model Garden, `official/nlp/modeling/layers` |
| Bayesian layers, mixture head, calibration metrics, quantiles | TensorFlow Probability |
| `GaussianNLL` | `torch.nn.GaussianNLLLoss` |
| Temperature scaling | Model Garden's object detection `calibration_builder` (applying $T$); SciPy's bounded minimizer (fitting $T$) |
| Conformal prediction | NumPy implementations of the published procedures |

The Model Garden and TensorFlow Probability parity tests load the reference sources from a local
`reference/` folder (not distributed with the package) and are skipped when it is absent, as in
CI. `RandomFourierFeatures` parity only needs `tf_keras`.

## Differences from the references

Some deviations are deliberate, each covered by a test:

- **Brier decomposition**: TFP's `brier_decomposition` indexes the average label distribution by
  the predicted class in its resolution term. `keras_proba` uses the textbook formula, for which
  uncertainty − resolution + reliability reproduces the Brier score exactly when all predictions
  with the same argmax are identical; TFP's can be off by up to 0.39.
- **Quantile-bucket ECE**: bucket edges are data points, and TFP compares log-probabilities with
  edges that went through `exp` and `log`, so a point on an edge can fall in either bucket
  depending on float rounding. `keras_proba` compares in probability space, where an edge equals
  its data point exactly.
- **Local reparameterization**: an all-zero input gives a NaN scale gradient in TFP; a tiny
  constant under the square root keeps it finite.
- **Spectral normalization on JAX**: see the [SNGP guide](guides/sngp.md#spectral-normalization).
- **Robustness**: the GP and SNGP head unpack every output combination
  (`return_gp_cov=False`, `return_random_features=True`) and accept rank-2 features, where the
  reference crashes.
