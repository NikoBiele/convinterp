"""Time 1D evaluation: convinterp (Rust) versus ConvolutionInterpolations.jl, same data and points.

Build Rust in release mode first:  maturin develop --release
Run from the project folder:       python benchmarks/benchmark_1d.py
Prints only; writes nothing to disk.
"""
#%%

import time
import numpy as np
from juliacall import Main as jl

from convinterp import convolution_interpolation

jl.seval("import ConvolutionInterpolations")

# Julia: sum the interpolant over all points in a compiled loop (a function, so the loop is
# specialized on the interpolant's type), and time it, best of 5, in nanoseconds per point
jl.seval("""
function _sum_over(itp, points)
    s = 0.0
    for p in points
        s += itp(p)
    end
    return s
end""")
julia_ns_per_point = jl.seval("""
(x, y, kernel, points) -> begin
    itp = ConvolutionInterpolations.convolution_interpolation(
        collect(x), collect(y); kernel=Symbol(kernel))
    pts = collect(points)
    _sum_over(itp, pts)                      # warm-up: compile
    best = minimum(@elapsed(_sum_over(itp, pts)) for _ in 1:5)
    return best / length(pts) * 1e9
end""")


def rust_ns_per_point(x, y, kernel, points):
    """Time the Rust core over all points in one call, best of 5, in nanoseconds per point."""
    itp = convolution_interpolation(x, y, kernel=kernel)
    itp._core.evaluate(points)               # warm-up
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        itp._core.evaluate(points)
        best = min(best, time.perf_counter() - start)
    return best / len(points) * 1e9

#%%
# The same data and points for both packages
x = np.linspace(0.0, 2.0 * np.pi, 1000)
y = np.sin(x)
points = np.random.default_rng(1).uniform(x[0], x[-1], 1_000_000)

print(f"{'kernel':<8}{'Rust [ns]':>12}{'Julia [ns]':>12}{'ratio':>9}")
for kernel in ["a1", "a3", "b5", "b7", "b13"]:
    rust = rust_ns_per_point(x, y, kernel, points)
    julia = julia_ns_per_point(x, y, kernel, points)
    print(f"{kernel:<8}{rust:>12.1f}{julia:>12.1f}{rust / julia:>9.2f}")
# %%
