# convinterp

High-order convolution interpolation on uniform grids in any number of dimensions, with
derivatives, for Python. The core is written in Rust; the kernels are exact polynomials.

convinterp is a port of the Julia package
[ConvolutionInterpolations.jl](https://github.com/NikoBiele/ConvolutionInterpolations.jl), by the
same author, and is tested against it.

**Documentation: [convinterp.org](https://convinterp.org)**, with a user guide and accuracy and
speed comparisons with SciPy.

**Status: alpha.** Interpolation and derivatives on uniform grids work in any dimension.
Integrals and scattered data are coming.

## Installation

```
pip install convinterp
```

## Example

```python
import numpy as np
from convinterp import convolution_interpolation

# 1D: data on a uniform grid
x = np.linspace(0.0, 2 * np.pi, 50)                     # 50 uniformly spaced knots
itp = convolution_interpolation(x, np.sin(x))           # the interpolant of sin(x)
print(itp(1.0))                                         # ≈ sin(1) = 0.841471

# derivatives: one order for the whole interpolant
d_itp = convolution_interpolation(x, np.sin(x), derivative=1)
print(d_itp(1.0))                                       # ≈ cos(1) = 0.540302

# 2D: one knot array per axis, and data whose axis d runs along knots[d]
y = np.linspace(0.0, 1.0, 30)                           # knots of the second axis
data = np.sin(x)[:, None] * np.exp(y)[None, :]          # sin(x)·exp(y) on the 50 × 30 grid
itp2 = convolution_interpolation((x, y), data)          # the 2D interpolant
print(itp2(1.0, 0.5))                                   # ≈ sin(1)·exp(0.5) = 1.387351

# a mixed derivative, with an order per axis: ∂²/∂x∂y
dxy = convolution_interpolation((x, y), data, derivative=(1, 1))
print(dxy(1.0, 0.5))                                    # ≈ cos(1)·exp(0.5) = 0.890808

# arrays of points broadcast like NumPy arrays: here a 3 × 4 table of values
xs = np.array([[0.5], [1.0], [1.5]])                    # a column of x coordinates
ys = np.array([[0.1, 0.2, 0.3, 0.4]])                   # a row of y coordinates
print(itp2(xs, ys).shape)                               # (3, 4)
```

## Kernels

`kernel="auto"` (the default) chooses by the number of dimensions: `"b7"` in 1D and 2D, `"b5"` in
3D, and narrower kernels beyond. You can choose any of:

| kernel | order of accuracy | highest derivative |
|---|---|---|
| `"a0"` | nearest neighbour | none |
| `"a1"` | linear | none |
| `"a3"`, `"a4"`, `"a5"`, `"a7"` | cubic and higher | 1 |
| `"b5"`, `"b7"`, `"b9"`, `"b11"`, `"b13"` | high order | 3 to 7 |

## Boundary conditions

The data are extended beyond each boundary by extrapolation: `bc="detect"` (the default) uses
polynomial extrapolation where the data near the boundary allow it and linear otherwise. You can
also choose `"poly"`, `"linear"` or `"quadratic"`, separately for each side and each axis.

## Declaration of AI Assistance

The Rust core and much of the Python code of convinterp were written with substantial assistance
from Claude (Anthropic). The mathematical methods, the kernels and the reference implementation
come from the author's Julia package
[ConvolutionInterpolations.jl](https://github.com/NikoBiele/ConvolutionInterpolations.jl), and the
port is validated against it: the test suite compares convinterp with the Julia package across
kernels, boundary conditions, derivative orders and dimensions, with the boundary coefficients
agreeing bit for bit. The author has reviewed and is responsible for all code.

## License

MIT