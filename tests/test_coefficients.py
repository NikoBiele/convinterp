import numpy as np
import pytest

from convinterp._core import extended_coefficients   # internal function, tested directly

# Kernels with ghost values, and every boundary condition
GHOST_KERNELS = ["a3", "a4", "a5", "a7", "b5", "b7", "b9", "b11", "b13"]
BOUNDARIES = ["poly", "linear", "quadratic", "detect"]


def test_no_ghosts_for_a0_a1():
    # :a0 and :a1 use the data values unchanged
    values = np.array([1.0, 3.0, 2.0, 5.0])
    for kernel in ["a0", "a1"]:
        np.testing.assert_array_equal(extended_coefficients(values, kernel, "detect", "detect"), values)


def test_poly_reproduces_polynomials():
    # :poly extrapolates polynomials of the kernel's degree exactly: cubic data for :a3,
    # which has one ghost value per side (eqs − 1 with eqs = 2)
    x = np.arange(-1.0, 11.0)                     # one ghost position per side, and the data
    cubic = 0.5 * x**3 - x**2 + 2.0 * x - 1.0
    c = extended_coefficients(cubic[1:-1], "a3", "poly", "poly")
    np.testing.assert_allclose(c, cubic, rtol=1e-12, atol=1e-10)


@pytest.fixture(scope="module")
def julia_coefficients():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # The package's own 1D construction (grid spacing 1; it doesn't affect the ghost values)
    return jl.seval(
        "(values, kernel, bl, br) -> ConvolutionInterpolations.create_convolutional_coefs("
        "collect(values), (1.0,), "
        "(ConvolutionInterpolations.get_equations_for_degree(Symbol(kernel)),), "
        "((Symbol(bl), Symbol(br)),), (Symbol(kernel),), Val(false))"
    )


def test_matches_julia(julia_coefficients):
    # Smooth data (where :detect accepts :poly) and rough data (where it falls back to :linear)
    x = np.linspace(0.0, 3.0, 40)
    datasets = [np.sin(x) + 0.3 * x, np.where(x < 0.1, 5.0, 0.0) + np.cos(7.0 * x)]
    for values in datasets:
        for kernel in GHOST_KERNELS:
            for left in BOUNDARIES:
                for right in BOUNDARIES:
                    expected = np.asarray(julia_coefficients(values, kernel, left, right))
                    np.testing.assert_allclose(
                        extended_coefficients(values, kernel, left, right), expected,
                        # ghost values of the wide kernels come from heavy cancellation (:b13's
                        # matrix has entries up to ~3e5), so summation order shows up near 1e-11
                        rtol=0, atol=1e-10,
                        err_msg=f"kernel {kernel}, boundaries {left}/{right}",
                    )