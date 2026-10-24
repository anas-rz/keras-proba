# Layers: `keras_proba.layers`

Uncertainty-aware Keras 3 layers. Every layer works with the TensorFlow, JAX and PyTorch backends and is serializable with `model.save`.

## Distance-aware (SNGP)

::: keras_proba.layers.RandomFourierFeatures
    options:
      heading_level: 3

::: keras_proba.layers.RandomFeatureGaussianProcess
    options:
      heading_level: 3

::: keras_proba.layers.LaplaceRandomFeatureCovariance
    options:
      heading_level: 3

::: keras_proba.layers.SpectralNormalization
    options:
      heading_level: 3

::: keras_proba.layers.SpectralNormalizationConv2D
    options:
      heading_level: 3

::: keras_proba.layers.ClassificationHead
    options:
      heading_level: 3

::: keras_proba.layers.GaussianProcessClassificationHead
    options:
      heading_level: 3

## Bayesian (variational)

::: keras_proba.layers.DenseReparameterization
    options:
      heading_level: 3

::: keras_proba.layers.DenseLocalReparameterization
    options:
      heading_level: 3

::: keras_proba.layers.DenseFlipout
    options:
      heading_level: 3

::: keras_proba.layers.Conv1DReparameterization
    options:
      heading_level: 3

::: keras_proba.layers.Conv1DFlipout
    options:
      heading_level: 3

::: keras_proba.layers.Conv2DReparameterization
    options:
      heading_level: 3

::: keras_proba.layers.Conv2DFlipout
    options:
      heading_level: 3

::: keras_proba.layers.Conv3DReparameterization
    options:
      heading_level: 3

::: keras_proba.layers.Conv3DFlipout
    options:
      heading_level: 3

## Probabilistic regression heads

::: keras_proba.layers.GaussianHead
    options:
      heading_level: 3

::: keras_proba.layers.MixtureGaussianHead
    options:
      heading_level: 3
