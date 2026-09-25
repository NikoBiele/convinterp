import numpy as np
import pytest

from convinterp._core import InterpolantND   # the Rust class, tested directly until the Python API wraps it

# Kernels and their highest derivative order (0 for :a0 and :a1)
MAX_DERIVATIVE = {"a0": 0, "a1": 0, "a3": 1, "a4": 1, "a5": 1, "a7": 1,
                  "b5": 3, "b7": 5, "b9": 6, "b11": 7, "b13": 7}

# Grids as (start, stop, number of knots) per axis, with a different length on every axis
GRID_2D = [(0.0, 2.0, 17), (-1.0, 1.0, 13)]
GRID_3D = [(0.0, 1.0, 11), (-0.5, 0.5, 10), (0.0, 2.0, 9)]


def knots_of(grid):
    # one vector of uniform knots per axis
    return [np.linspace(start, stop, n) for start, stop, n in grid]


def smooth_data(knots):
    # a smooth, non-polynomial function sampled on the grid
    mesh = np.meshgrid(*knots, indexing="ij")      # coordinate arrays: axis d's coordinate varies along axis d
    return np.sin(1.0 + sum(c * m for c, m in zip((2.0, 3.0, 1.5), mesh)))


def random_points(grid, count, seed):
    # uniformly random points in the whole box, near the boundaries and corners too
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(start, stop, count) for start, stop, _ in grid])


def derivative_orders(max_order, n_dims):
    # values, a first derivative along each single axis, the mixed first derivative along all
    # axes, and the kernel's highest order along axis 0
    orders = [(0,) * n_dims]
    if max_order >= 1:
        orders += [tuple(int(d == axis) for d in range(n_dims)) for axis in range(n_dims)]
        orders.append((1,) * n_dims)
    if max_order >= 2:
        orders.append((max_order,) + (0,) * (n_dims - 1))
    return orders


@pytest.fixture(scope="module")
def julia_nd():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import ConvolutionInterpolations")
    # The Julia interpolant on the same grid, evaluated at every row of `points`. The grid comes
    # as a flat array (start, stop, number of knots per axis), the boundary conditions as one
    # string such as "detect/detect,poly/linear", and the derivative orders as an integer array.
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


def test_exact_on_reproduced_polynomials():
    # :b5 reproduces quintics along each axis, so with polynomial boundaries the product of two
    # quintics, and its derivatives with orders per axis, are reproduced exactly (up to rounding)
    p = np.polynomial.Polynomial([1.0, -2.0, 0.5, 3.0, -1.0, 0.7])   # a quintic in x
    q = np.polynomial.Polynomial([0.3, 1.0, -1.5, 0.2, 0.8, -0.4])   # a quintic in y
    grid = [(0.0, 1.0, 21), (0.0, 1.0, 17)]
    knots = knots_of(grid)
    values = p(knots[0])[:, None] * q(knots[1])[None, :]             # the product on the grid
    points = random_points(grid, 200, seed=5)
    for orders in [(0, 0), (1, 0), (0, 2), (1, 1), (3, 2)]:
        itp = InterpolantND(knots, values, "b5", [("poly", "poly")] * 2, list(orders))
        expected = p.deriv(orders[0])(points[:, 0]) * q.deriv(orders[1])(points[:, 1])
        np.testing.assert_allclose(itp.evaluate(points), expected, rtol=1e-8, atol=1e-8,
                                   err_msg=f"orders {orders}")


def test_errors():
    knots = knots_of(GRID_2D)
    values = smooth_data(knots)
    bcs = [("detect", "detect")] * 2
    # a wrong number of derivative orders, an order above the kernel's highest, and knots that
    # don't match the data along an axis
    with pytest.raises(ValueError):
        InterpolantND(knots, values, "b5", bcs, [0])
    with pytest.raises(ValueError):
        InterpolantND(knots, values, "b5", bcs, [0, 4])
    with pytest.raises(ValueError):
        InterpolantND([knots[0], knots[1][:-1]], values, "b5", bcs, [0, 0])
    # a point outside the data range, and points with the wrong number of coordinates
    itp = InterpolantND(knots, values, "b5", bcs, [0, 0])
    with pytest.raises(ValueError):
        itp.evaluate(np.array([[0.5, 2.0]]))
    with pytest.raises(ValueError):
        itp.evaluate(np.array([[0.5, 0.0, 0.0]]))


def test_matches_julia(julia_nd):
    for grid, n_points in [(GRID_2D, 300), (GRID_3D, 200)]:
        n_dims = len(grid)
        knots = knots_of(grid)
        values = smooth_data(knots)
        points = random_points(grid, n_points, seed=n_dims)
        flat_grid = np.array([v for axis in grid for v in axis], dtype=float)
        spacing = [(stop - start) / (n - 1) for start, stop, n in grid]
        # the same boundary condition everywhere, and a different one on each side and axis
        bcs_options = [[("detect", "detect")] * n_dims,
                       [("poly", "linear")] + [("quadratic", "detect")] * (n_dims - 1)]
        for kernel, max_order in MAX_DERIVATIVE.items():
            for bcs in bcs_options:
                bc_string = ",".join(f"{left}/{right}" for left, right in bcs)
                for orders in derivative_orders(max_order, n_dims):
                    itp = InterpolantND(knots, values, kernel, bcs, list(orders))
                    expected = np.asarray(julia_nd(flat_grid, values, kernel, bc_string,
                                                   np.array(orders), points))
                    # the ghost values agree bit for bit (test_coefficients_nd); what remains is
                    # the evaluators' summation order, which a derivative multiplies by 1/h per order
                    tolerance = 1e-11 * np.prod([(1.0 / h) ** d for h, d in zip(spacing, orders)])
                    np.testing.assert_allclose(
                        itp.evaluate(points), expected, rtol=0, atol=tolerance,
                        err_msg=f"{n_dims}D, kernel {kernel}, boundaries {bc_string}, orders {orders}",
                    )