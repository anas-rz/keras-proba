"""Multi-backend percentiles and quantiles in Keras 3.

Ported from TensorFlow Probability's `tensorflow_probability/python/stats/quantiles.py`
(`percentile` and `quantiles`). The requested percentiles `q` must be concrete
(Python or NumPy values), since the gathered indices are computed statically.
"""

import numpy as np

import keras
from keras import ops

_ALLOWED_INTERPOLATIONS = ("linear", "lower", "higher", "nearest", "midpoint")


def _normalize_axis(axis, ndims):
    if axis is None:
        return None
    axis = [axis] if isinstance(axis, int) else list(axis)
    axis = sorted({a % ndims for a in axis})
    return axis


def percentile(x, q, axis=None, interpolation=None, keepdims=False):
    """Computes the `q`-th percentile(s) of `x`.

    Given a vector `x`, the `q`-th percentile is the value `q / 100` of the way
    from the minimum to the maximum in a sorted copy of `x`. The two nearest
    neighbors and `interpolation` determine the result when the normalized rank
    does not match a sample exactly. Compare to `numpy.percentile`.

    Example:

    ```python
    x = [1.0, 2.0, 3.0, 4.0]
    percentile(x, q=30.0)                          # 2.0
    percentile(x, q=30.0, interpolation="linear")  # 1.9
    percentile(x, q=[30.0, 70.0], interpolation="lower")  # [1.0, 3.0]
    percentile([[1.0, 2.0], [3.0, 4.0]], q=100.0, axis=[0])  # [3.0, 4.0]
    ```

    Args:
        x: Numeric tensor with at least one dimension.
        q: Scalar or 1-D percentile(s) in `[0, 100]`, as concrete values.
        axis: Int or list of ints: the axes indexing samples. If `None`, all
            dimensions are sample dimensions and the result is scalar per `q`.
        interpolation: One of `"nearest"` (default), `"linear"`, `"lower"`,
            `"higher"` or `"midpoint"`, used when a percentile lies between
            samples `i < j`: `i + (j - i) * fraction`, `i`, `j`, the nearest of
            the two, or `(i + j) / 2`. `"linear"` and `"midpoint"` require a
            floating point `x`.
        keepdims: Whether to keep the reduced axes with size 1.

    Returns:
        Tensor whose leading `rank(q)` dimensions index the percentiles,
        followed by the dimensions of `x` not in `axis`.
    """
    x = ops.convert_to_tensor(x)
    if interpolation is None:
        interpolation = "nearest"
    if interpolation not in _ALLOWED_INTERPOLATIONS:
        raise ValueError(
            f"Argument `interpolation` must be in {_ALLOWED_INTERPOLATIONS}. Found {interpolation}"
        )
    is_float = keras.backend.is_float_dtype(x.dtype)
    if interpolation in ("linear", "midpoint") and not is_float:
        raise TypeError(f"{interpolation} interpolation not allowed with dtype {x.dtype}")

    # In float64, like TFP, so the selected indices match exactly.
    q = np.asarray(q, dtype="float64")
    if q.ndim > 1:
        raise ValueError(f"`q` must be a scalar or a vector. Received shape {q.shape}")

    ndims = len(x.shape)
    axis = _normalize_axis(axis, ndims)
    if axis is None:
        y = ops.reshape(x, (-1,))
    else:
        # Move the sample axes to the end and flatten them.
        other = [d for d in range(ndims) if d not in axis]
        y = ops.transpose(x, other + axis)
        y = ops.reshape(y, tuple(x.shape[d] for d in other) + (-1,))

    n = y.shape[-1]
    if n is None:
        raise ValueError("The number of samples must be statically known.")
    sorted_y = ops.sort(y, axis=-1)
    frac = q / 100.0
    exact = (n - 1) * frac

    def gather(indices):
        indices = np.clip(indices.astype("int64"), 0, n - 1)
        return ops.take(sorted_y, indices, axis=-1)

    if interpolation == "lower":
        result = gather(np.floor(exact))
    elif interpolation == "higher":
        result = gather(np.ceil(exact))
    elif interpolation == "nearest":
        result = gather(np.round(exact))  # Round half to even, like `tf.round`.
    elif interpolation == "midpoint":
        result = 0.5 * (gather(np.floor(exact)) + gather(np.ceil(exact)))
    else:
        # Same weighting as TFP: larger * (1 - fraction) + smaller * fraction.
        fraction = ops.cast(np.ceil(exact) - exact, y.dtype)
        result = gather(np.ceil(exact)) * (1 - fraction) + gather(np.floor(exact)) * fraction

    if is_float:
        # A NaN anywhere in the samples makes the percentile NaN.
        has_nan = ops.any(ops.isnan(y), axis=-1)
        has_nan = ops.reshape(has_nan, tuple(ops.shape(has_nan)) + (1,) * q.ndim)
        result = ops.where(has_nan, ops.cast(np.nan, result.dtype), result)

    if keepdims:
        if axis is None:
            result = ops.reshape(result, (1,) * ndims + q.shape)
        else:
            for d in axis:
                result = ops.expand_dims(result, d)

    # Move the percentile dimensions to the front.
    if q.ndim:
        result = ops.moveaxis(result, -1, 0)
    return result


def quantiles(x, num_quantiles, axis=None, interpolation=None, keepdims=False):
    """Computes `num_quantiles + 1` cut points of `x` along `axis`.

    The cut points `(c0, ..., cn)` divide the samples into `num_quantiles`
    intervals `[c0, c1), ..., [c_{n-1}, cn]` with roughly equal numbers of
    samples; `c0` is the minimum and `cn` the maximum.

    Example:

    ```python
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    quantiles(x, num_quantiles=4)                          # [0., 2., 5., 8., 10.]
    quantiles(x, num_quantiles=4, interpolation="linear")  # [0., 2.5, 5., 7.5, 10.]
    ```

    Args:
        x: Numeric tensor with at least one dimension.
        num_quantiles: Number of intervals.
        axis: Int or list of ints: the axes indexing samples. If `None`, all
            dimensions are sample dimensions.
        interpolation: See `percentile`.
        keepdims: Whether to keep the reduced axes with size 1.

    Returns:
        Tensor of shape `(num_quantiles + 1, ...)`, where the trailing shape is
        that of `x` without `axis` (unless `keepdims`).
    """
    q = np.linspace(0.0, 100.0, num_quantiles + 1)
    return percentile(x, q, axis=axis, interpolation=interpolation, keepdims=keepdims)
