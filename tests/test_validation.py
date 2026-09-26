import numpy as np
import pytest

from convinterp import convolution_interpolation

# Kernels with ghost values beyond the boundaries (:a0 and :a1 have none)
GHOST_KERNELS = ["a3", "a4", "a5", "a7", "b5", "b7", "b9", "b11", "b13"]


def test_non_finite_knots_1d():
    # A NaN inside the knots, and an infinite end knot, are rejected instead of accepted
    y = np.array([0.0, 1.0, 2.0, 3.0])                       # data for four knots
    for x in [np.array([0.0, 1.0, np.nan, 3.0]),             # NaN inside the grid
              np.array([0.0, 1.0, 2.0, np.inf])]:            # infinite last knot
        with pytest.raises(ValueError, match="finite"):
            convolution_interpolation(x, y)


def test_non_finite_knots_nd():
    # The same on the second axis of 2D data; the message names the axis
    x = np.linspace(0.0, 1.0, 10)                            # a valid first axis
    y = np.array([0.0, 1.0, np.nan, 3.0])                    # NaN on the second axis
    with pytest.raises(ValueError, match="axis 1"):
        convolution_interpolation((x, y), np.zeros((10, 4)))


def test_quadratic_needs_three_values_1d():
    # bc="quadratic" with only 2 data values: a clear error for every kernel with ghost values,
    # on either boundary (the right one checked separately with a (left, right) pair)
    x = np.array([0.0, 1.0])                                 # two knots
    y = np.array([0.0, 1.0])                                 # two values
    for kernel in GHOST_KERNELS:
        with pytest.raises(ValueError, match="needs 3 values"):
            convolution_interpolation(x, y, kernel=kernel, bc="quadratic")
        with pytest.raises(ValueError, match="needs 3 values"):
            convolution_interpolation(x, y, kernel=kernel, bc=("linear", "quadratic"))


def test_two_values_still_work():
    # With 2 data values, :a0 and :a1 (no ghost values) and bc="linear" still work: the
    # interpolant of a straight line through (0, 0) and (1, 1) is exact at its midpoint
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    assert convolution_interpolation(x, y, kernel="a1", bc="quadratic")(0.5) == pytest.approx(0.5)
    for kernel in GHOST_KERNELS:
        itp = convolution_interpolation(x, y, kernel=kernel, bc="linear")
        assert itp(0.5) == pytest.approx(0.5), f"kernel {kernel}"


def test_quadratic_needs_three_values_nd():
    # 2D data with only 2 values along the first axis: bc="quadratic" is rejected
    x = np.array([0.0, 1.0])                                 # two knots on the first axis
    y = np.linspace(0.0, 1.0, 10)                            # ten on the second
    with pytest.raises(ValueError, match="needs 3 values"):
        convolution_interpolation((x, y), np.zeros((2, 10)), bc="quadratic")