# Keras Probability: Uncertainty Awareness in Deep Learning

Documentation: https://anas-rz.github.io/keras-proba/

Install from PyPI:

```bash
pip install keras-proba
```

Keras 3 needs a backend; install one with an extra, e.g. `pip install "keras-proba[jax]"` (`[torch]`, `[jax]` or `[tensorflow]`).

## Layers

- [ ] `RandomFourierFeatures`
- [ ] `GaussianProcess` output layer
- [ ] `SpectralNormalization`
- [ ] `GaussianHead` for probabilistic regression

## Post-processing

- [ ] Temperature scaling for classification calibration
- [ ] Split conformal prediction for classification and regression

## Models and Examples

- [ ] SNGP classifier example
- [ ] SNGP regressor example
- [ ] Calibrated classifier example
- [ ] Conformal prediction example

## Tests

- [ ] Parity tests with original implementation
- [ ] Training tests across backends
- [ ] Smoke tests across backends
- [ ] Serialization and reload tests