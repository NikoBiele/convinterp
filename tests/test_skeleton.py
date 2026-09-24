import numpy as np
import pytest

from convinterp import evaluate_polynomial

# The test polynomial 1 − 2x + 0.5x² + 3x³, and points to evaluate it at
COEFS = np.array([1.0, -2.0, 0.5, 3.0])
X = np.linspace(-2.0, 2.0, 101)


def test_matches_numpy():
    # NumPy's own polynomial evaluation as the reference
    expected = np.polynomial.polynomial.polyval(X, COEFS)
    np.testing.assert_allclose(evaluate_polynomial(COEFS, X), expected, rtol=1e-14, atol=1e-14)


def test_matches_julia():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    # Julia's evalpoly as the reference: the pattern later tests against the Julia package follow
    expected = np.array([jl.evalpoly(float(x), tuple(COEFS)) for x in X])
    np.testing.assert_allclose(evaluate_polynomial(COEFS, X), expected, rtol=1e-14, atol=1e-14)