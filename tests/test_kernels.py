import numpy as np
import pytest

from convinterp._core import kernel_weights   # internal function, tested directly

# Every kernel with exported column tables, and positions tau = 1 − t across the cell
KERNELS = ["a1", "a3", "a4", "a5", "a7", "b5", "b7", "b9", "b11", "b13"]
TAUS = np.linspace(0.0, 1.0, 11)


def test_partition_of_unity():
    # Interpolation weights sum to one at every position, for every kernel
    for kernel in KERNELS:
        for tau in TAUS:
            assert abs(sum(kernel_weights(kernel, 0, tau)) - 1.0) < 1e-13


def test_unknown_kernel_raises():
    # A kernel or order without a table gives a Python ValueError, not a crash
    with pytest.raises(ValueError):
        kernel_weights("b6", 0, 0.5)


@pytest.fixture(scope="module")
def julia_weights():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # A Julia function returning the package's own weights as a vector
    return jl.seval(
        "(kernel, order, tau) -> collect(ConvolutionInterpolations._kernel_weights("
        "Val(Symbol(kernel)), Val(order), tau))"
    )


def test_matches_julia(julia_weights):
    # The same weights as ConvolutionInterpolations.jl, to rounding, for every kernel and position
    for kernel in KERNELS:
        for tau in TAUS:
            expected = np.asarray(julia_weights(kernel, 0, float(tau)))
            np.testing.assert_allclose(kernel_weights(kernel, 0, tau), expected, rtol=0, atol=1e-15)


# Highest integral order m (derivative order −m) per kernel, as in ConvolutionInterpolations.jl
MAX_INTEGRAL = {"a1": 2, "a3": 4, "a4": 4, "a5": 4, "a7": 4,
                "b5": 6, "b7": 8, "b9": 8, "b11": 8, "b13": 8}


def test_integral_tables_end_at_max_order():
    # Every integral order up to the kernel's maximum has a table, and the next one has none
    for kernel, m_max in MAX_INTEGRAL.items():
        for m in range(1, m_max + 1):
            kernel_weights(kernel, -m, 0.5)                  # raises if the table is missing
        with pytest.raises(ValueError):
            kernel_weights(kernel, -(m_max + 1), 0.5)


def test_integral_weights_match_julia(julia_weights):
    # Integral orders −m: Julia evaluates Kₘ at t with the columns in reversed order; the Rust
    # tables hold the same weights at tau = 1 − t in eager order (sign (−1)^m folded in at export)
    for kernel, m_max in MAX_INTEGRAL.items():
        for m in range(1, m_max + 1):
            for t in TAUS:
                # Julia's weights at t, reversed into eager order
                expected = np.asarray(julia_weights(kernel, -m, float(t)))[::-1]
                # the Rust weights at tau = 1 − t
                got = kernel_weights(kernel, -m, 1.0 - t)
                np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12,
                                           err_msg=f"{kernel}, order {-m}, t = {t}")