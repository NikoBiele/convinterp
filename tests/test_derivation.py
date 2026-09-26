from fractions import Fraction

import numpy as np
import pytest

from convinterp import convolution_interpolation
from convinterp.derivation import (KERNEL_SETTINGS, derive_kernel, kernel_function,
                                   kernel_properties)


@pytest.fixture(scope="module")
def derived():
    # every kernel of convinterp, derived once for all tests in this file (about 20 seconds)
    pytest.importorskip("sympy")
    return {name: derive_kernel(*settings) for name, settings in KERNEL_SETTINGS.items()}


def test_keys_kernel():
    # Keys (1981): 1 − 5/2 s² + 3/2 s³ on [0, 1], and 2 − 4s + 5/2 s² − 1/2 s³ on [1, 2]
    pytest.importorskip("sympy")
    keys = derive_kernel(pieces=2, degree=3, continuity=1, reproduce=2)
    assert keys == ((Fraction(1), Fraction(0), Fraction(-5, 2), Fraction(3, 2)),
                    (Fraction(2), Fraction(-4), Fraction(5, 2), Fraction(-1, 2)))


def test_properties(derived):
    # each derived kernel has exactly the properties it was derived for
    for name, (pieces, degree, continuity, reproduce) in KERNEL_SETTINGS.items():
        assert kernel_properties(derived[name]) == {
            "pieces": pieces, "degree": degree, "continuity": continuity, "reproduces": reproduce,
        }, name


def test_matches_compiled_kernels(derived):
    # the derived kernels are the ones convinterp evaluates: interpolating a unit impulse
    # (1 at one knot, 0 at all others) reproduces the kernel itself
    knots = np.arange(-15.0, 16.0)                                   # wide enough for every kernel
    impulse = (knots == 0.0).astype(float)
    for name, kernel in derived.items():
        pieces = len(kernel)
        s = np.linspace(-pieces - 0.5, pieces + 0.5, 4001)          # the support and a little beyond
        compiled = convolution_interpolation(knots, impulse, kernel=name, bc="poly")
        np.testing.assert_allclose(compiled(s), kernel_function(kernel)(s), rtol=0, atol=1e-13,
                                   err_msg=name)


def test_errors():
    # contradictory conditions, and conditions that leave the kernel undetermined
    pytest.importorskip("sympy")
    with pytest.raises(ValueError, match="contradictory"):
        derive_kernel(pieces=2, degree=3, continuity=1, reproduce=3)
    with pytest.raises(ValueError, match="free parameter"):
        derive_kernel(pieces=3, degree=5, continuity=1, reproduce=2)


def test_properties_of_the_linear_kernel():
    # the hat function 1 − s on [0, 1]: continuous but not differentiable, reproduces lines
    # (needs no SymPy: kernel_properties works with fractions only)
    hat = ((Fraction(1), Fraction(-1)),)
    assert kernel_properties(hat) == {"pieces": 1, "degree": 1, "continuity": 0, "reproduces": 1}


def test_kernel_function_numbers_and_arrays():
    # a number in gives a Python float; an array in gives an array of the same shape
    hat = kernel_function(((Fraction(1), Fraction(-1)),))
    assert isinstance(hat(0.25), float) and hat(0.25) == 0.75
    np.testing.assert_array_equal(hat(np.array([[-0.5, 0.0], [1.0, 2.0]])), [[0.5, 1.0], [0.0, 0.0]])


@pytest.fixture(scope="module")
def julia_coefficients():
    # the exact rational coefficients stored in ConvolutionInterpolations.jl, as "p/q" strings
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    return jl.seval("""
        name -> begin
            d = getfield(ConvolutionInterpolations, Symbol(name, "_coefs"))
            [[string(numerator(c)) * "/" * string(denominator(c)) for c in d[Symbol("eq", i)]]
             for i in 1:length(d)]
        end""")


def test_identical_to_julia(derived, julia_coefficients):
    # the b-series kernels are the unique solutions of their conditions, and exactly the kernels
    # stored in the Julia package, found there by a long manual search
    for name in ["b5", "b7", "b9", "b11", "b13"]:
        stored = tuple(tuple(Fraction(str(c)) for c in row) for row in julia_coefficients(name))
        assert derived[name] == stored, name