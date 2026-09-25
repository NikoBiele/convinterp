import numpy as np
import pytest

from convinterp import convolution_interpolation

KERNELS = ["a0", "a1", "a3", "a4", "a5", "a7", "b5", "b7", "b9", "b11", "b13"]
X = np.linspace(0.0, 2.0, 40)                 # uniform grid
Y = np.sin(3.0 * X) + 0.5 * X                 # smooth data


def test_reproduces_data_at_knots():
    # Every kernel except nearest neighbour interpolates: the data come back at the knots
    for kernel in KERNELS[1:]:
        itp = convolution_interpolation(X, Y, kernel=kernel)
        np.testing.assert_allclose(itp(X), Y, rtol=0, atol=1e-12, err_msg=kernel)


def test_scalar_and_array_calls():
    # A number in gives a number out; an array in gives an array of the same shape out
    itp = convolution_interpolation(X, Y)
    assert isinstance(itp(1.0), float)
    assert itp(np.ones((3, 4))).shape == (3, 4)


def test_errors():
    itp = convolution_interpolation(X, Y)
    # points outside the data range
    with pytest.raises(ValueError):
        itp(2.5)
    # nonuniform knots
    with pytest.raises(ValueError):
        convolution_interpolation(X**2, Y)
    # unknown kernel and boundary condition
    with pytest.raises(ValueError):
        convolution_interpolation(X, Y, kernel="b6")
    with pytest.raises(ValueError):
        convolution_interpolation(X, Y, bc="cubic")


@pytest.fixture(scope="module")
def julia_interpolation():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # Build the Julia package's interpolant and evaluate it at the given points
    return jl.seval("""
        (x, y, kernel, bc, points) -> begin
            itp = ConvolutionInterpolations.convolution_interpolation(
                collect(x), collect(y); kernel=Symbol(kernel), bc=Symbol(bc))
            [itp(p) for p in points]
        end""")


def test_matches_julia(julia_interpolation):
    # Random points across the whole range, including close to both boundaries
    points = np.sort(np.random.default_rng(1).uniform(X[0], X[-1], 500))
    for kernel in KERNELS:
        for bc in ["detect", "poly", "linear"]:
            itp = convolution_interpolation(X, Y, kernel=kernel, bc=bc)
            expected = np.asarray(julia_interpolation(X, Y, kernel, bc, points))
            # near the boundaries the ghost values carry rounding of up to ~1e-11 (see
            # test_coefficients), so allow 1e-10; a real mismatch would be far larger
            np.testing.assert_allclose(itp(points), expected, rtol=0, atol=1e-10,
                                       err_msg=f"kernel {kernel}, bc {bc}")