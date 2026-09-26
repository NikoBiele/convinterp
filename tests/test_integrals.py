import numpy as np
import pytest

from convinterp import convolution_interpolation

# The highest integral order m (derivative order −m) of each kernel, as in ConvolutionInterpolations.jl
MAX_INTEGRAL = {"a0": 2, "a1": 2, "a3": 4, "a4": 4, "a5": 4, "a7": 4,
                "b5": 6, "b7": 8, "b9": 8, "b11": 8, "b13": 8}
X = np.linspace(0.0, 2.0, 40)                 # uniform grid
Y = np.sin(3.0 * X) + 0.5 * X                 # smooth data


def test_zero_at_first_knot():
    # Every integral is anchored at zero at the first knot, for every kernel and order
    for kernel, m_max in MAX_INTEGRAL.items():
        for m in range(1, m_max + 1):
            itp = convolution_interpolation(X, Y, kernel=kernel, derivative=-m)
            assert abs(itp(X[0])) < 1e-13, f"kernel {kernel}, order {-m}: {itp(X[0])}"


def test_exact_on_reproduced_polynomials():
    # A kernel that reproduces a polynomial exactly also integrates it exactly: its m-fold
    # integral equals the polynomial's, anchored (with all lower integrals) at the first knot.
    # :a0 reproduces constants, :a1 straight lines, :b5 (with polynomial boundaries) quintics.
    x = np.linspace(0.0, 1.0, 21)
    points = np.linspace(0.0, 1.0, 101)
    cases = [("a0", np.polynomial.Polynomial([1.7])),                              # a constant
             ("a1", np.polynomial.Polynomial([1.0, -2.0])),                        # a straight line
             ("b5", np.polynomial.Polynomial([1.0, -2.0, 0.5, 3.0, -1.0, 0.7]))]   # a quintic
    for kernel, p in cases:
        for m in range(1, MAX_INTEGRAL[kernel] + 1):
            itp = convolution_interpolation(x, p(x), kernel=kernel, bc="poly", derivative=-m)
            expected = p.integ(m, lbnd=x[0])(points)       # each integral zero at x[0]
            np.testing.assert_allclose(itp(points), expected, rtol=1e-10, atol=1e-12,
                                       err_msg=f"kernel {kernel}, order {-m}")


def test_errors():
    # One integral order beyond each kernel's highest
    for kernel, m_max in MAX_INTEGRAL.items():
        with pytest.raises(ValueError):
            convolution_interpolation(X, Y, kernel=kernel, derivative=-(m_max + 1))


@pytest.fixture(scope="module")
def julia_integral():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # Build the Julia package's integral interpolant (derivative = −m) and evaluate it at the points
    return jl.seval("""
        (x, y, kernel, bc, d, points) -> begin
            itp = ConvolutionInterpolations.convolution_interpolation(
                collect(x), collect(y); kernel=Symbol(kernel), bc=Symbol(bc), derivative=d)
            [itp(p) for p in points]
        end""")


def test_matches_julia(julia_integral):
    # Every kernel and integral order, at random points across the whole range and at both ends
    rng = np.random.default_rng(3)
    points = np.sort(np.concatenate([[X[0], X[-1]], rng.uniform(X[0], X[-1], 300)]))
    for kernel, m_max in MAX_INTEGRAL.items():
        for m in range(1, m_max + 1):
            for bc in ["detect", "poly"]:
                itp = convolution_interpolation(X, Y, kernel=kernel, bc=bc, derivative=-m)
                expected = np.asarray(julia_integral(X, Y, kernel, bc, -m, points))
                # integration smooths rounding rather than amplifying it; the ghost values near
                # the boundaries differ by up to ~1e-11 (summation order, see test_coefficients),
                # and the tails are evaluated by Horner's scheme rather than Julia's powers
                np.testing.assert_allclose(itp(points), expected, rtol=0, atol=1e-10,
                                           err_msg=f"kernel {kernel}, order {-m}, bc {bc}")