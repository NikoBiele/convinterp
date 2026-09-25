import numpy as np
import pytest

from convinterp._core import extended_coefficients_nd   # internal function, tested directly

# Kernels with ghost values
GHOST_KERNELS = ["a3", "a4", "a5", "a7", "b5", "b7", "b9", "b11", "b13"]

# Smooth data with a different length along every axis, so a mixed-up axis order changes the shape
X = np.linspace(0.0, 3.0, 15)
Y = np.linspace(0.0, 2.0, 12)
Z = np.linspace(0.0, 1.0, 10)
SMOOTH_2D = np.sin(X)[:, None] * np.cos(Y)[None, :] + 0.3 * X[:, None]
# the same with a jump near the left end of axis 0, where :detect falls back to linear
ROUGH_2D = SMOOTH_2D + np.where(X < 0.3, 5.0, 0.0)[:, None]
SMOOTH_3D = np.sin(X[:11])[:, None, None] * np.cos(Y[:10])[None, :, None] + np.exp(Z[:9])[None, None, :]

# One (left, right) pair of boundary conditions per axis: the same everywhere, and mixed
BCS_2D = [
    [("detect", "detect"), ("detect", "detect")],
    [("poly", "poly"), ("poly", "poly")],
    [("poly", "linear"), ("quadratic", "detect")],
    [("linear", "quadratic"), ("detect", "poly")],
]
BCS_3D = [
    [("detect", "detect"), ("detect", "detect"), ("detect", "detect")],
    [("poly", "linear"), ("quadratic", "detect"), ("detect", "poly")],
]


@pytest.fixture(scope="module")
def julia_coefficients_nd():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # The package's own N-D construction (grid spacing 1; it doesn't affect the ghost values).
    # `bcs` is one string such as "poly/linear,detect/detect": a left/right pair per axis.
    return jl.seval("""
        (values, kernel, bcs) -> begin
            v = collect(values)
            N = ndims(v)
            k = Symbol(kernel)
            pairs = Tuple(Tuple(Symbol.(split(p, "/"))) for p in split(bcs, ","))
            ConvolutionInterpolations.create_convolutional_coefs(
                v, ntuple(_ -> 1.0, N),
                ntuple(_ -> ConvolutionInterpolations.get_equations_for_degree(k), N),
                pairs, ntuple(_ -> k, N), Val(false))
        end""")


def test_no_ghosts_for_a0_a1():
    # :a0 and :a1 use the data values unchanged, in any number of dimensions
    bcs = [("detect", "detect")] * 3
    for kernel in ["a0", "a1"]:
        np.testing.assert_array_equal(extended_coefficients_nd(SMOOTH_3D, kernel, bcs), SMOOTH_3D)


def test_errors():
    # a wrong number of boundary condition pairs, and an unknown boundary condition
    with pytest.raises(ValueError):
        extended_coefficients_nd(SMOOTH_2D, "b5", [("poly", "poly")])
    with pytest.raises(ValueError):
        extended_coefficients_nd(SMOOTH_2D, "b5", [("poly", "poly"), ("poly", "cubic")])


def test_matches_julia(julia_coefficients_nd):
    # Both sides compute every ghost value with the same compensated sum, in the same order, so
    # they agree bit for bit: inside the data, on the faces, on the edges and in the corners
    cases = [(SMOOTH_2D, bcs) for bcs in BCS_2D] + [(ROUGH_2D, bcs) for bcs in BCS_2D] \
          + [(SMOOTH_3D, bcs) for bcs in BCS_3D]
    for values, bcs in cases:
        bc_string = ",".join(f"{left}/{right}" for left, right in bcs)
        for kernel in GHOST_KERNELS:
            got = extended_coefficients_nd(values, kernel, bcs)
            expected = np.asarray(julia_coefficients_nd(values, kernel, bc_string))
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{values.ndim}D, kernel {kernel}, boundaries {bc_string}"
            )