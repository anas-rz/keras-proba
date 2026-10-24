"""Numerical parity tests comparing the Gaussian NLL against `torch.nn.GaussianNLLLoss`.

There is no Gaussian regression head in the TensorFlow Model Garden reference,
so the loss is checked against PyTorch's implementation instead.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import keras
from keras import ops

from keras_proba.layers.gaussian_head import GaussianHead, GaussianNLL, gaussian_nll, split_gaussian_params


def torch_nll(y, mean, scale, reduction="mean"):
    loss = torch.nn.GaussianNLLLoss(full=True, eps=0.0, reduction=reduction)
    return loss(torch.tensor(mean), torch.tensor(y), torch.tensor(scale**2)).numpy()


def test_loss_parity():
    units = 3
    rng = np.random.default_rng(0)
    y = rng.normal(size=(16, units)).astype("float32")
    mean = rng.normal(size=(16, units)).astype("float32")
    scale = rng.uniform(0.1, 3.0, size=(16, units)).astype("float32")
    y_pred = np.concatenate([mean, scale], axis=-1)

    expected = torch_nll(y, mean, scale)
    actual = ops.convert_to_numpy(GaussianNLL()(y, y_pred))
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Per-element values, averaged over units, match the functional form.
    per_element = torch_nll(y, mean, scale, reduction="none").mean(-1)
    np.testing.assert_allclose(ops.convert_to_numpy(gaussian_nll(y, y_pred)), per_element, rtol=1e-5)


def test_loss_parity_with_sample_weight():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(8, 1)).astype("float32")
    mean = rng.normal(size=(8, 1)).astype("float32")
    scale = rng.uniform(0.5, 2.0, size=(8, 1)).astype("float32")
    weights = rng.uniform(size=(8,)).astype("float32")

    per_sample = torch_nll(y, mean, scale, reduction="none")[:, 0]
    expected = np.sum(per_sample * weights) / len(weights)
    actual = GaussianNLL()(y, np.concatenate([mean, scale], -1), sample_weight=weights)
    np.testing.assert_allclose(ops.convert_to_numpy(actual), expected, rtol=1e-5)


def test_head_forward_parity():
    """The head matches the same computation written in PyTorch."""
    rng = np.random.default_rng(0)
    head = GaussianHead(2, min_scale=0.01)
    x = rng.normal(size=(5, 4)).astype("float32")
    mean, scale = split_gaussian_params(head(x))

    w_mean, b_mean = (torch.tensor(ops.convert_to_numpy(w)) for w in head.mean_dense.weights)
    w_scale, b_scale = (torch.tensor(ops.convert_to_numpy(w)) for w in head.scale_dense.weights)
    xt = torch.tensor(x)
    torch_mean = xt @ w_mean + b_mean
    torch_scale = torch.nn.functional.softplus(xt @ w_scale + b_scale) + 0.01
    np.testing.assert_allclose(ops.convert_to_numpy(mean), torch_mean.numpy(), atol=1e-6)
    np.testing.assert_allclose(ops.convert_to_numpy(scale), torch_scale.numpy(), atol=1e-6)
