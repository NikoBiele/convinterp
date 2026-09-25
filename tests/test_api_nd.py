import numpy as np
import pytest

from convinterp import convolution_interpolation
from convinterp._core import InterpolantND   # the Rust class, for exact comparisons

# A 2D grid with different lengths per axis, and smooth data on it
X = np.linspace(0.0, 2.0, 17)
Y = np.linspace(-1.0, 1.0, 13)
DATA = np.sin(1.0 + 2.0 * X[:, None] + 3.0 * Y[None, :])
# random points inside the data range, as separate coordinate arrays
RNG = np.random.default_rng(7)
PX = RNG.uniform(X[0], X[-1], 50)
PY = RNG.uniform(Y[0], Y[-1], 50)


def core_values(kernel, bcs, orders, px, py):
    # the Rust class called directly, at the points (px[i], py[i])
    itp = InterpolantND([X, Y], DATA, kernel, bcs, orders)
    return itp.evaluate(np.column_stack([px, py]))


def test_numbers_give_a_number():
    # one number per axis gives a Python float
    itp = convolution_interpolation((X, Y), DATA)
    result = itp(0.3, 0.2)
    assert isinstance(result, float)
    assert result == core_values("b7", [("detect", "detect")] * 2, [0, 0], [0.3], [0.2])[0]


def test_arrays_broadcast():
    itp = convolution_interpolation((X, Y), DATA)
    expected = core_values("b7", [("detect", "detect")] * 2, [0, 0], PX, PY)
    # two arrays of the same shape give an array of that shape
    np.testing.assert_array_equal(itp(PX, PY), expected)
    # a column against a row gives the whole table of combinations, shape (50, 50)
    table = itp(PX[:, None], PY[None, :])
    assert table.shape == (50, 50)
    np.testing.assert_array_equal(np.diag(table), expected)
    # a number against an array gives an array of the array's shape
    np.testing.assert_array_equal(
        itp(0.3, PY), core_values("b7", [("detect", "detect")] * 2, [0, 0], np.full(50, 0.3), PY))


def test_boundary_condition_forms():
    # one name, one (left, right) pair, and one pair per axis describe the same boundaries
    expected = core_values("b5", [("poly", "linear")] * 2, [0, 0], PX, PY)
    for bc in [("poly", "linear"), [("poly", "linear"), ("poly", "linear")]]:
        itp = convolution_interpolation((X, Y), DATA, kernel="b5", bc=bc)
        np.testing.assert_array_equal(itp(PX, PY), expected, err_msg=f"bc {bc}")
    same = core_values("b5", [("poly", "poly")] * 2, [0, 0], PX, PY)
    itp = convolution_interpolation((X, Y), DATA, kernel="b5", bc="poly")
    np.testing.assert_array_equal(itp(PX, PY), same)
    # different pairs per axis reach the core in order
    per_axis = [("poly", "linear"), ("quadratic", "detect")]
    itp = convolution_interpolation((X, Y), DATA, kernel="b5", bc=per_axis)
    np.testing.assert_array_equal(itp(PX, PY), core_values("b5", per_axis, [0, 0], PX, PY))


def test_derivative_forms():
    # one order means that order on every axis; a tuple gives one order per axis
    bcs = [("detect", "detect")] * 2
    itp = convolution_interpolation((X, Y), DATA, kernel="b7", derivative=1)
    np.testing.assert_array_equal(itp(PX, PY), core_values("b7", bcs, [1, 1], PX, PY))
    itp = convolution_interpolation((X, Y), DATA, kernel="b7", derivative=(2, 0))
    np.testing.assert_array_equal(itp(PX, PY), core_values("b7", bcs, [2, 0], PX, PY))


def test_auto_kernel_by_dimension():
    # "auto" is "b7" in 1D and 2D, and "b5" in 3D, as in the Julia package
    z = np.linspace(0.0, 1.0, 9)
    data_3d = DATA[:, :, None] * np.exp(z)[None, None, :]
    px, py, pz = PX[:10], PY[:10], RNG.uniform(0.0, 1.0, 10)
    auto = convolution_interpolation((X, Y, z), data_3d)
    explicit = convolution_interpolation((X, Y, z), data_3d, kernel="b5")
    np.testing.assert_array_equal(auto(px, py, pz), explicit(px, py, pz))
    auto_2d = convolution_interpolation((X, Y), DATA)
    explicit_2d = convolution_interpolation((X, Y), DATA, kernel="b7")
    np.testing.assert_array_equal(auto_2d(PX, PY), explicit_2d(PX, PY))


def test_one_d_accepts_a_tuple_of_one_knot_array():
    # (x,) and x mean the same for 1D data
    values = np.sin(X)
    np.testing.assert_array_equal(convolution_interpolation((X,), values)(PX),
                                  convolution_interpolation(X, values)(PX))


def test_errors():
    itp = convolution_interpolation((X, Y), DATA)
    # the wrong number of coordinates
    with pytest.raises(TypeError):
        itp(0.3)
    # knots not given as one array per axis
    with pytest.raises(ValueError):
        convolution_interpolation(X, DATA)
    # a malformed boundary condition, and the wrong number of derivative orders
    with pytest.raises(ValueError):
        convolution_interpolation((X, Y), DATA, bc=["poly", "poly", "poly"])
    with pytest.raises(ValueError):
        convolution_interpolation((X, Y), DATA, derivative=(1, 0, 0))
    # a kernel per axis is not supported yet
    with pytest.raises(ValueError):
        convolution_interpolation((X, Y), DATA, kernel=("b5", "b7"))
    # a point outside the data range (reported by the Rust core)
    with pytest.raises(ValueError):
        itp(0.3, 5.0)