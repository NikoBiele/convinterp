"""convinterp: convolution interpolation with exact kernels, a Python interface to a Rust core."""

import numpy as np

from ._core import Interpolant1D, InterpolantND, evaluate_polynomial

__all__ = ["convolution_interpolation", "Interpolation", "evaluate_polynomial"]


class Interpolation:
    """A convolution interpolant. Call it with one coordinate per axis: numbers or arrays.

    Create one with `convolution_interpolation`.
    """

    def __init__(self, core, n_dims):
        # the compiled Rust interpolant that does the work, and its number of axes
        self._core = core
        self._n_dims = n_dims

    def __call__(self, *coords):
        # one coordinate per axis
        if len(coords) != self._n_dims:
            raise TypeError(f"expected {self._n_dims} coordinates (one per axis), got {len(coords)}")
        # numbers, lists or arrays in float64, broadcast against each other to one common shape
        arrays = np.broadcast_arrays(*[np.asarray(c, dtype=np.float64) for c in coords])
        shape = arrays[0].shape
        if self._n_dims == 1:
            # the 1D core takes a flat, contiguous array of points
            values = self._core.evaluate(np.ascontiguousarray(arrays[0].ravel()))
        else:
            # the N-D core takes one row per point: shape (number of points, number of axes)
            values = self._core.evaluate(np.column_stack([a.ravel() for a in arrays]))
        # numbers in give a number out; arrays in give an array of the broadcast shape out
        return float(values[0]) if shape == () else values.reshape(shape)


def convolution_interpolation(knots, values, kernel="auto", bc="detect", derivative=0):
    """Interpolate data on a uniform grid in any number of dimensions, or differentiate it.

    Parameters
    ----------
    knots : array_like, or tuple of array_like
        For 1D data, the uniformly spaced, increasing knots. For N-D data, a tuple of N such
        arrays, one per axis of `values`.
    values : array_like
        The data at the knots: a 1D array, or an N-D array whose axis d runs along knots[d].
    kernel : str, default "auto"
        The interpolation kernel: "a0" (nearest neighbour), "a1" (linear), "a3", "a4", "a5",
        "a7", or the high-order kernels "b5", "b7", "b9", "b11", "b13". "auto" chooses by the
        number of dimensions N, as the Julia package does: "b7" for N ≤ 2, "b5" for N = 3, "a4"
        for N = 4 and 5, "a3" beyond (a stencil has (2·eqs)^N coefficients, so higher
        dimensions use narrower kernels).
    bc : str, (str, str), or sequence of (str, str), default "detect"
        Boundary conditions: one for every boundary, one (left, right) pair for every axis, or
        one (left, right) pair per axis. Each is "poly" (polynomial extrapolation), "linear",
        "quadratic", or "detect" (polynomial where the data allow it, otherwise linear).
    derivative : int or sequence of int, default 0
        Derivative order: one order for every axis, or one per axis. 0 is the interpolant
        itself, 1 its first derivative, and so on. Available up to order 1 for "a3" to "a7", 3
        for "b5", 5 for "b7", 6 for "b9" and 7 for "b11" and "b13".

    Returns
    -------
    Interpolation
        Call it with one coordinate per axis, inside the data range: numbers, or arrays that
        broadcast against each other.

    Example
    -------
    >>> import numpy as np
    >>> from convinterp import convolution_interpolation
    >>> x = np.linspace(0.0, 2 * np.pi, 50)                            # uniform grid
    >>> itp = convolution_interpolation(x, np.sin(x))                  # interpolant of the data
    >>> round(itp(1.0), 6)                                             # sin(1)
    0.841471
    >>> d_itp = convolution_interpolation(x, np.sin(x), derivative=1)  # its first derivative
    >>> round(d_itp(1.0), 6)                                           # cos(1)
    0.540302
    >>> y = np.linspace(0.0, 1.0, 30)                                  # a second axis
    >>> data = np.sin(x)[:, None] * np.exp(y)[None, :]                 # data on the 2D grid
    >>> itp2 = convolution_interpolation((x, y), data)                 # 2D interpolant
    >>> round(itp2(1.0, 0.5), 6)                                       # sin(1)·exp(0.5)
    1.387351
    >>> dxy = convolution_interpolation((x, y), data, derivative=(1, 1))  # ∂²/∂x∂y
    >>> round(dxy(1.0, 0.5), 6)                                        # cos(1)·exp(0.5)
    0.890808
    """
    # the data as a float64 array; its number of axes decides 1D or N-D
    values = np.ascontiguousarray(values, dtype=np.float64)
    n_dims = values.ndim
    if n_dims == 0:
        raise ValueError("values must have at least one axis")
    # a tuple holding a single knot array is accepted for 1D data too
    if n_dims == 1 and isinstance(knots, tuple) and len(knots) == 1:
        knots = knots[0]

    kernel = _default_kernel(n_dims) if kernel == "auto" else kernel
    if not isinstance(kernel, str):
        raise ValueError("one kernel name for all axes is supported (a kernel per axis: not yet)")
    bcs = _per_axis_bcs(bc, n_dims)
    orders = _per_axis_orders(derivative, n_dims)

    if n_dims == 1:
        # the 1D core, with its own specialized evaluator
        x = np.ascontiguousarray(knots, dtype=np.float64)
        (bc_left, bc_right), = bcs
        return Interpolation(Interpolant1D(x, values, kernel, bc_left, bc_right, orders[0]), 1)

    # N-D: one contiguous float64 knot array per axis
    if not isinstance(knots, (tuple, list)) or len(knots) != n_dims:
        raise ValueError(f"{n_dims}-dimensional data need a tuple of {n_dims} knot arrays")
    knots = [np.ascontiguousarray(k, dtype=np.float64) for k in knots]
    return Interpolation(InterpolantND(knots, values, kernel, bcs, orders), n_dims)


def _default_kernel(n_dims):
    # the Julia package's default by dimension: narrower kernels in higher dimensions
    if n_dims <= 2:
        return "b7"
    if n_dims == 3:
        return "b5"
    if n_dims <= 5:
        return "a4"
    return "a3"


def _per_axis_bcs(bc, n_dims):
    # one name for every boundary
    if isinstance(bc, str):
        return [(bc, bc)] * n_dims
    bc = tuple(bc)
    # one (left, right) pair of names for every axis
    if len(bc) == 2 and all(isinstance(b, str) for b in bc):
        return [bc] * n_dims
    # one (left, right) pair per axis
    if len(bc) == n_dims and all(len(pair) == 2 for pair in bc):
        return [tuple(pair) for pair in bc]
    raise ValueError(f"bc must be a name, a (left, right) pair, or {n_dims} such pairs")


def _per_axis_orders(derivative, n_dims):
    # one order for every axis
    if isinstance(derivative, (int, np.integer)):
        return [int(derivative)] * n_dims
    # one order per axis
    orders = [int(d) for d in derivative]
    if len(orders) != n_dims:
        raise ValueError(f"derivative must be one order, or {n_dims} orders (one per axis)")
    return orders