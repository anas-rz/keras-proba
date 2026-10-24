"""Callback that resets Gaussian process covariance estimates during training."""

import keras

from keras_proba.layers.gaussian_process import LaplaceRandomFeatureCovariance


class ResetCovarianceCallback(keras.callbacks.Callback):
    """Resets the precision matrix of every GP covariance layer at the start of each epoch.

    `RandomFeatureGaussianProcess` accumulates its posterior precision matrix
    over training minibatches. Since the feature extractor changes during
    training, the estimate should be recomputed in each epoch, so that the
    final covariance only reflects the last epoch's features [1].

    Example:

    ```python
    model.fit(x, y, epochs=10, callbacks=[ResetCovarianceCallback()])
    ```

    [1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
         Deterministic Deep Learning via Distance Awareness.
         In _Neural Information Processing Systems_, 2020.
         https://arxiv.org/abs/2006.10108
    """

    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model._flatten_layers(include_self=False, recursive=True):
            if isinstance(layer, LaplaceRandomFeatureCovariance) and layer.built:
                layer.reset_precision_matrix()
