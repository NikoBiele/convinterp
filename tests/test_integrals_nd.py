import numpy as np
import pytest

from convinterp import convolution_interpolation

# Kernels: highest integral order m (derivative order −m) and highest derivative order
MAX_INTEGRAL = {"a0": 2, "a1": 2, "a3": 4, "a4": 4, "a5": 4, "a7": 4,
                "b5": 6, "b7": 8, "b9": 8, "b11": 8, "b13": 8}
MAX_DERIVATIVE = {"a0": 0, "a1": 0, "a3": 1, "a4": 1, "a5": 1, "a7": 1,
                  "b5": 3, "b7": 5, "b9": 6, "b11": 7, "b13": 7}

# Grids as (start, stop, number of knots) per axis, with a different length on every axis
GRID_2D = [(0.0, 2.0, 17), (-1.0, 1.0, 13)]
GRID_3D = [(0.0, 1.0, 11), (-0.5, 0.5, 10), (0.0, 2.0, 9)]
GRID_4D = [(0.0, 1.0, 6), (0.0, 1.0, 5), (-1.0, 0.0, 5), (0.0, 2.0, 6)]


def knots_of(grid):
    # one vector of uniform knots per axis
    return [np.linspace(start, stop, n) for start, stop, n in grid]


def smooth_data(knots):
    # a smooth, non-polynomial, non-separable function sampled on the grid
    mesh = np.meshgrid(*knots, indexing="ij")      # coordinate arrays: axis d's coordinate varies along axis d
    return np.sin(1.0 + sum(c * m for c, m in zip((2.0, 3.0, 1.5, 1.0), mesh)))


def random_points(grid, count, seed):
    # uniformly random points in the whole box, near the boundaries and corners too
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(start, stop, count) for start, stop, _ in grid])


def apply_order(p, order, anchor):
    # a polynomial's derivative (order > 0), itself (0), or its integral anchored at `anchor` (< 0)
    return p.integ(-order, lbnd=anchor) if order < 0 else p.deriv(order)


@pytest.fixture(scope="module")
def julia_nd():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # The Julia interpolant on the same grid, evaluated at every row of `points`. The grid comes
    # as a flat array (start, stop, number of knots per axis), the boundary conditions as one
    # string such as "detect/detect,poly/linear", and the orders as an integer array.
    return jl.seval("""
        (grid, values, kernel, bcs, derivs, points) -> begin
            N = ndims(values)
            knots = ntuple(d -> range(grid[3d-2], grid[3d-1]; length=Int(grid[3d])), N)
            pairs = Tuple(Tuple(Symbol.(split(p, "/"))) for p in split(bcs, ","))
            itp = ConvolutionInterpolations.convolution_interpolation(
                knots, collect(values); kernel=Symbol(kernel), bc=pairs,
                derivative=Tuple(Int.(derivs)))
            [itp(points[i, :]...) for i in 1:size(points, 1)]
        end""")


def compare_with_julia(julia_nd, grid, kernel, bcs, orders, count, seed):
    # Evaluate our interpolant and Julia's at the same random points, and compare
    knots = knots_of(grid)
    values = smooth_data(knots)
    points = random_points(grid, count, seed)
    itp = convolution_interpolation(tuple(knots), values, kernel=kernel, bc=bcs, derivative=orders)
    got = itp(*points.T)                                              # one coordinate array per axis
    flat_grid = np.array([v for axis in grid for v in axis], dtype=float)
    bc_string = ",".join(f"{left}/{right}" for left, right in bcs)
    expected = np.asarray(julia_nd(flat_grid, values, kernel, bc_string, np.array(orders), points))
    # integration smooths rounding; each derivative order multiplies it by 1/h (as in the
    # derivative tests), so the tolerance scales with the derivative orders only
    h = [(stop - start) / (n - 1) for start, stop, n in grid]
    tolerance = 1e-10 * np.prod([(1.0 / h[d]) ** max(o, 0) for d, o in enumerate(orders)])
    np.testing.assert_allclose(got, expected, rtol=0, atol=tolerance,
                               err_msg=f"kernel {kernel}, orders {orders}, bc {bcs}")


def test_exact_on_reproduced_polynomials_2d():
    # :b5 reproduces quintics along each axis, so with polynomial boundaries the product of two
    # quintics is integrated (and differentiated) exactly, each integral anchored at the first knot
    p = np.polynomial.Polynomial([1.0, -2.0, 0.5, 3.0, -1.0, 0.7])   # a quintic in x
    q = np.polynomial.Polynomial([0.3, 1.0, -1.5, 0.2, 0.8, -0.4])   # a quintic in y
    grid = [(0.0, 1.0, 21), (-0.5, 0.5, 17)]
    knots = knots_of(grid)
    values = p(knots[0])[:, None] * q(knots[1])[None, :]             # the product on the grid
    points = random_points(grid, 200, seed=5)
    for orders in [(-1, -1), (-2, 0), (0, -3), (-1, 1), (2, -2), (-6, -6)]:
        itp = convolution_interpolation(tuple(knots), values, kernel="b5", bc="poly",
                                        derivative=orders)
        expected = (apply_order(p, orders[0], knots[0][0])(points[:, 0])
                    * apply_order(q, orders[1], knots[1][0])(points[:, 1]))
        np.testing.assert_allclose(itp(points[:, 0], points[:, 1]), expected,
                                   rtol=1e-8, atol=1e-10, err_msg=f"orders {orders}")


def test_zero_on_the_anchor_face():
    # An integral along axis 0 is zero wherever x lies on the first knot of axis 0, for every y
    knots = knots_of(GRID_2D)
    values = smooth_data(knots)
    y = np.linspace(GRID_2D[1][0], GRID_2D[1][1], 25)                # points along the whole face
    for orders in [(-1, 0), (-2, 1), (-3, -1)]:
        itp = convolution_interpolation(tuple(knots), values, kernel="b7", derivative=orders)
        face = itp(np.full_like(y, knots[0][0]), y)
        assert np.max(np.abs(face)) < 1e-12, f"orders {orders}: {np.max(np.abs(face))}"


def test_matches_julia_2d(julia_nd):
    # Every kernel: integrals along one and both axes, the kernel's highest integral order, and
    # an integral mixed with the highest derivative, with different boundary conditions per axis
    bcs = [("detect", "detect"), ("poly", "linear")]
    for kernel, m_max in MAX_INTEGRAL.items():
        d_max = MAX_DERIVATIVE[kernel]
        cases = [(-1, -1), (0, -1), (-m_max, 0), (-2, -m_max)]
        if d_max >= 1:
            cases.append((-1, d_max))
        for orders in cases:
            compare_with_julia(julia_nd, GRID_2D, kernel, bcs, orders, count=150, seed=7)


def test_matches_julia_3d(julia_nd):
    # Integral, derivative and value axes mixed in three dimensions
    bcs = [("detect", "detect"), ("poly", "linear"), ("linear", "poly")]
    for kernel, orders in [("b5", (-1, -1, -1)), ("b5", (-2, 0, 1)), ("b5", (-3, -1, 0)),
                           ("a3", (-1, 0, -1)), ("a3", (1, -2, -4))]:
        compare_with_julia(julia_nd, GRID_3D, kernel, bcs, orders, count=100, seed=11)


def test_four_integral_axes_match_julia(julia_nd):
    # With more than 3 integral axes Julia sums directly over the grid, while convinterp uses
    # region tails: both compute the same thing. A tiny grid keeps Julia's direct sum cheap.
    bcs = [("poly", "poly")] * 4
    for orders in [(-1, -1, -1, -1), (-2, -1, -1, -2)]:
        compare_with_julia(julia_nd, GRID_4D, "a3", bcs, orders, count=50, seed=13)


def test_memory_guard():
    # A 100³ grid integrated 6 times along every axis would need about 2.9 GiB of tails: rejected
    # before anything is allocated, with a message naming the memory needed
    knots = [np.linspace(0.0, 1.0, 100)] * 3                         # three axes of 100 knots
    values = np.zeros((100, 100, 100))                               # 8 MB of data
    with pytest.raises(ValueError, match="GiB"):
        convolution_interpolation(tuple(knots), values, kernel="b5", derivative=(-6, -6, -6))


def test_errors_nd():
    # An integral order beyond the kernel's highest, on one axis only
    knots = knots_of(GRID_2D)
    values = smooth_data(knots)
    with pytest.raises(ValueError, match="axis 1"):
        convolution_interpolation(tuple(knots), values, kernel="b5", derivative=(-1, -7))