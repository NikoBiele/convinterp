import numpy as np
import pytest

from convinterp import convolution_interpolation

# The highest derivative order of each kernel with derivatives
MAX_DERIVATIVE = {"a3": 1, "a4": 1, "a5": 1, "a7": 1,
                  "b5": 3, "b7": 5, "b9": 6, "b11": 7, "b13": 7}
X = np.linspace(0.0, 2.0, 40)                 # uniform grid
H = X[1] - X[0]                               # its spacing
Y = np.sin(3.0 * X) + 0.5 * X                 # smooth data


def test_exact_on_reproduced_polynomials():
    # :b5 reproduces quintics exactly, so with polynomial boundaries its derivatives
    # up to order 3 equal the quintic's derivatives exactly (up to rounding)
    x = np.linspace(0.0, 1.0, 21)
    p = np.polynomial.Polynomial([1.0, -2.0, 0.5, 3.0, -1.0, 0.7])   # a quintic
    points = np.linspace(0.0, 1.0, 101)
    for d in range(4):
        itp = convolution_interpolation(x, p(x), kernel="b5", bc="poly", derivative=d)
        expected = p.deriv(d)(points)
        np.testing.assert_allclose(itp(points), expected, rtol=1e-8, atol=1e-8, err_msg=f"order {d}")


def test_errors():
    # an order above the kernel's highest, a kernel without derivatives, and an integral order
    # above the kernel's highest (the integrals themselves are tested in test_integrals.py)
    with pytest.raises(ValueError):
        convolution_interpolation(X, Y, kernel="b5", derivative=4)
    with pytest.raises(ValueError):
        convolution_interpolation(X, Y, kernel="a1", derivative=1)
    with pytest.raises(ValueError):
        convolution_interpolation(X, Y, kernel="b5", derivative=-7)


@pytest.fixture(scope="module")
def julia_derivative():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # Build the Julia package's derivative interpolant and evaluate it at the given points
    return jl.seval("""
        (x, y, kernel, bc, d, points) -> begin
            itp = ConvolutionInterpolations.convolution_interpolation(
                collect(x), collect(y); kernel=Symbol(kernel), bc=Symbol(bc), derivative=d)
            [itp(p) for p in points]
        end""")


def test_matches_julia(julia_derivative):
    # Every kernel and derivative order, at random points across the whole range
    points = np.sort(np.random.default_rng(2).uniform(X[0], X[-1], 300))
    for kernel, max_order in MAX_DERIVATIVE.items():
        for d in range(1, max_order + 1):
            for bc in ["detect", "poly"]:
                itp = convolution_interpolation(X, Y, kernel=kernel, bc=bc, derivative=d)
                expected = np.asarray(julia_derivative(X, Y, kernel, bc, d, points))
                # a derivative of order d multiplies rounding by (1/h)^d; near the boundaries the
                # ghost values already differ by up to ~1e-11 (summation order, see
                # test_coefficients), so the tolerance scales the same way
                tolerance = 1e-10 * (1.0 / H) ** d
                np.testing.assert_allclose(itp(points), expected, rtol=0, atol=tolerance,
                                           err_msg=f"kernel {kernel}, order {d}, bc {bc}")