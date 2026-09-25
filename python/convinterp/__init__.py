"""convinterp: convolution interpolation with exact kernels, a Python interface to a Rust core."""

import numpy as np

from ._core import Interpolant1D, evaluate_polynomial

__all__ = ["convolution_interpolation", "Interpolation", "evaluate_polynomial"]


class Interpolation:
    """A convolution interpolant. Call it with a number or an array of points.

    Create one with `convolution_interpolation`.
    """

    def __init__(self, core):
        # the compiled Rust interpolant that does the work
        self._core = core

    def __call__(self, x):
        # accept a number, a list or an array; compute in float64
        points = np.asarray(x, dtype=np.float64)
        # the Rust core takes a flat, contiguous array of points
        values = self._core.evaluate(np.ascontiguousarray(points.ravel()))
        # a number in gives a number out; an array in gives an array of the same shape out
        return float(values[0]) if points.ndim == 0 else values.reshape(points.shape)


def convolution_interpolation(x, values, kernel="b7", bc="detect"):
    """Interpolate data on a uniform 1D grid.

    Parameters
    ----------
    x : array_like
        Uniformly spaced, increasing knots.
    values : array_like
        The data at the knots.
    kernel : str, default "b7"
        The interpolation kernel: "a0" (nearest neighbour), "a1" (linear), "a3", "a4", "a5",
        "a7", or the high-order kernels "b5", "b7", "b9", "b11", "b13".
    bc : str or (str, str), default "detect"
        Boundary condition at both ends, or separately (left, right): "poly" (polynomial
        extrapolation), "linear", "quadratic", or "detect" (polynomial where the data allow it,
        otherwise linear).

    Returns
    -------
    Interpolation
        Call it with a number or an array of points inside the data range.

    Example
    -------
    >>> import numpy as np
    >>> from convinterp import convolution_interpolation
    >>> x = np.linspace(0.0, 2 * np.pi, 50)             # uniform grid
    >>> itp = convolution_interpolation(x, np.sin(x))   # interpolant of the data
    >>> round(itp(1.0), 6)                              # evaluate at one point
    0.841471
    """
    # one boundary condition for both ends, or a (left, right) pair
    bc_left, bc_right = (bc, bc) if isinstance(bc, str) else bc
    # contiguous float64 arrays, as the Rust core expects
    x = np.ascontiguousarray(x, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    return Interpolation(Interpolant1D(x, values, kernel, bc_left, bc_right))