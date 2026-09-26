//! An N-D convolution interpolant on a uniform grid, as in ConvolutionInterpolations.jl (the
//! eager N-D evaluator), for values and derivatives, with a derivative order per axis.

use numpy::ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use wide::f64x4;

use crate::coefficients::{self, Boundary};
use crate::interpolant::{horner_weights, pad_rows};
use crate::kernels;

/// One axis of the grid, and the kernel table of this axis's derivative order
struct GridAxis {
    /// Position of the first extended knot, and the grid spacing
    x0: f64,
    h: f64,
    /// The data range along this axis; points outside it are rejected
    x_first: f64,
    x_last: f64,
    /// Number of extended coefficients along this axis (data plus ghost values)
    len: usize,
    /// Distance in the flat coefficient vector between neighbours along this axis
    stride: usize,
    /// The column table of this axis's derivative order, as 4-wide vectors padded with zero
    /// columns to a multiple of four (empty for nearest neighbour)
    padded_rows: Vec<f64x4>,
}

/// An N-D interpolant. `#[pyclass]` makes it a Python class; its fields stay private to Rust.
#[pyclass]
pub struct InterpolantND {
    /// The extended coefficients as one flat vector in C order: the last axis is contiguous
    coefs: Vec<f64>,
    /// The grid axes, in the order of the data array's axes
    axes: Vec<GridAxis>,
    /// Stencil half-width of the kernel
    eqs: usize,
    /// Nearest neighbour (:a0), which needs no weights
    nearest: bool,
    /// Number of stencil columns K of the kernel (2·eqs), the same on every axis
    columns: usize,
    /// Factor applied to every result: the product of (−1/h)^derivative over the axes
    scale: f64,
}

#[pymethods]
impl InterpolantND {
    /// Construct from one vector of uniform knots per axis, the N-D data `values`, a kernel name,
    /// one (left, right) pair of boundary conditions per axis, and one derivative order per axis.
    #[new]
    fn new(
        knots: Vec<PyReadonlyArray1<'_, f64>>,
        values: PyReadonlyArrayDyn<'_, f64>,
        kernel: &str,
        bcs: Vec<(String, String)>,
        derivatives: Vec<i32>,
    ) -> PyResult<Self> {
        let values = values.as_array();
        let shape: Vec<usize> = values.shape().to_vec();
        let n_dims = shape.len();
        if n_dims == 0 {
            return Err(PyValueError::new_err("the data must have at least one axis"));
        }
        if knots.len() != n_dims || bcs.len() != n_dims || derivatives.len() != n_dims {
            return Err(PyValueError::new_err(format!(
                "{n_dims}-dimensional data need {n_dims} knot vectors, boundary condition pairs \
                 and derivative orders (got {}, {} and {})",
                knots.len(),
                bcs.len(),
                derivatives.len()
            )));
        }

        // The kernel: its highest derivative order and its stencil half-width
        let max_order = kernels::max_derivative(kernel)
            .ok_or_else(|| PyValueError::new_err(format!("unknown kernel {kernel:?}")))?;
        let nearest = kernel == "a0";
        let eqs = coefficients::stencil_half_width(kernel).map_err(|e| PyValueError::new_err(e))?;

        // Per axis: the grid (spacing and data range) and the table of its derivative order
        let mut grid: Vec<(f64, f64, f64)> = Vec::with_capacity(n_dims); // (h, x_first, x_last)
        let mut tables: Vec<Vec<f64x4>> = Vec::with_capacity(n_dims);
        let mut columns = 2; // nearest neighbour: one coefficient on each side
        for d in 0..n_dims {
            let x = knots[d].as_slice()?;
            if x.len() != shape[d] {
                return Err(PyValueError::new_err(format!(
                    "axis {d}: {} knots for {} values",
                    x.len(),
                    shape[d]
                )));
            }
            let h = uniform_spacing(x, d)?;
            grid.push((h, x[0], x[x.len() - 1]));

            let order = derivatives[d];
            if order < 0 {
                return Err(PyValueError::new_err(
                    "antiderivatives (negative derivative orders) are not supported yet",
                ));
            }
            if order > max_order {
                return Err(PyValueError::new_err(format!(
                    "axis {d}: kernel {kernel:?} supports derivatives up to order {max_order}, \
                     got {order}"
                )));
            }
            if nearest {
                tables.push(Vec::new());
            } else {
                let table = kernels::column_table(kernel, order).ok_or_else(|| {
                    PyValueError::new_err(format!("no table for kernel {kernel:?}, order {order}"))
                })?;
                columns = table.columns;
                tables.push(pad_rows(table));
            }
        }

        // The boundary conditions, and the extended coefficients
        let bcs = bcs
            .iter()
            .map(|(l, r)| Ok((Boundary::parse(l)?, Boundary::parse(r)?)))
            .collect::<Result<Vec<_>, String>>()
            .map_err(|e| PyValueError::new_err(e))?;
        let c = coefficients::extended_coefficients_nd(values.view(), kernel, &bcs)
            .map_err(|e| PyValueError::new_err(e))?;
        let lens: Vec<usize> = c.shape().to_vec();
        // `iter` visits the elements in C order whatever the memory layout, so the flat vector
        // has the last axis contiguous
        let coefs: Vec<f64> = c.iter().copied().collect();

        // Strides of the flat vector: 1 along the last axis; before it, the product of the
        // lengths of all later axes
        let mut strides = vec![1usize; n_dims];
        for d in (0..n_dims - 1).rev() {
            strides[d] = strides[d + 1] * lens[d + 1];
        }

        let axes: Vec<GridAxis> = tables
            .into_iter()
            .enumerate()
            .map(|(d, padded_rows)| {
                let (h, x_first, x_last) = grid[d];
                GridAxis {
                    x0: x_first - (eqs - 1) as f64 * h,
                    h,
                    x_first,
                    x_last,
                    len: lens[d],
                    stride: strides[d],
                    padded_rows,
                }
            })
            .collect();

        // d/dx = (1/h)·d/du on every axis, and the columns are functions of tau = 1 − t
        let scale: f64 = grid
            .iter()
            .zip(&derivatives)
            .map(|(&(h, _, _), &order)| (-1.0 / h).powi(order))
            .product();

        Ok(InterpolantND { coefs, axes, eqs, nearest, columns, scale })
    }

    /// Evaluate at every row of `points`, an array of shape (number of points, number of axes)
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points = points.as_array();
        if points.ncols() != self.axes.len() {
            return Err(PyValueError::new_err(format!(
                "points must have {} coordinates each, got {}",
                self.axes.len(),
                points.ncols()
            )));
        }
        // Choose the specialized evaluator once per call: one compiled copy per column count K,
        // with B = ⌈K/4⌉ vectors per table row
        let values = if self.nearest {
            self.evaluate_nearest(points)?
        } else {
            match self.columns {
                2 => self.evaluate_columns::<2, 1>(points)?,
                4 => self.evaluate_columns::<4, 1>(points)?,
                6 => self.evaluate_columns::<6, 2>(points)?,
                8 => self.evaluate_columns::<8, 2>(points)?,
                10 => self.evaluate_columns::<10, 3>(points)?,
                12 => self.evaluate_columns::<12, 3>(points)?,
                14 => self.evaluate_columns::<14, 4>(points)?,
                16 => self.evaluate_columns::<16, 4>(points)?,
                18 => self.evaluate_columns::<18, 5>(points)?,
                20 => self.evaluate_columns::<20, 5>(points)?,
                k => {
                    return Err(PyValueError::new_err(format!(
                        "no evaluator for a kernel with {k} columns"
                    )))
                }
            }
        };
        Ok(values.into_pyarray(py))
    }
}

// Methods only Rust can call: a separate `impl` block without #[pymethods]
impl InterpolantND {
    /// Evaluate a kernel with K stencil columns, stored as B 4-wide vectors per table row, at
    /// every point. K and B are compile-time constants: one specialized copy per kernel size.
    fn evaluate_columns<const K: usize, const B: usize>(
        &self,
        points: ArrayView2<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        let n_dims = self.axes.len();
        // Per axis: the padded table rows, as arrays of B vectors each
        let rows: Vec<&[[f64x4; B]]> =
            self.axes.iter().map(|axis| axis.padded_rows.as_chunks::<B>().0).collect();
        // Per axis, reused for every point: the K weights of the stencil
        let mut weights: Vec<[f64; K]> = vec![[0.0; K]; n_dims];
        let mut out: Vec<f64> = Vec::with_capacity(points.nrows());
        for point in points.rows() {
            // position of the stencil's first coefficient in the flat vector
            let mut base = 0usize;
            for (d, axis) in self.axes.iter().enumerate() {
                let x = point[d];
                check_range(axis, d, x)?;
                let (start, t) = locate(axis, x, self.eqs);
                base += start * axis.stride;
                // the weights at tau = 1 − t, as B vectors of four
                let w = horner_weights::<B>(rows[d], 1.0 - t);
                // unpacked into 4·B plain numbers, of which the first K are the stencil's weights
                let lanes: [[f64; 4]; B] = w.map(|v| v.to_array());
                weights[d].copy_from_slice(&lanes.as_flattened()[..K]);
            }
            // the tensor sum, contracted one axis at a time; (−1/h)^d factors applied at the end
            out.push(self.contract::<K>(0, base, &weights) * self.scale);
        }
        Ok(out)
    }

    /// The stencil sum over axes `axis`, `axis` + 1, … starting at `offset` in the flat vector:
    /// Σ_k w[axis][k] · (the same sum over the remaining axes, at offset + k·stride). On the last
    /// axis it is a dot product of K contiguous coefficients with that axis's weights.
    fn contract<const K: usize>(&self, axis: usize, offset: usize, weights: &[[f64; K]]) -> f64 {
        let w = &weights[axis];
        let mut sum = 0.0_f64;
        if axis + 1 == weights.len() {
            // the K coefficients along the last axis, as a fixed-size array (one bounds check)
            let c: &[f64; K] = self.coefs[offset..offset + K].try_into().unwrap();
            for k in 0..K {
                sum = c[k].mul_add(w[k], sum);
            }
        } else {
            let stride = self.axes[axis].stride;
            for k in 0..K {
                // a function calling itself: the remaining axes, one step further along this one
                let inner = self.contract::<K>(axis + 1, offset + k * stride, weights);
                sum = w[k].mul_add(inner, sum);
            }
        }
        sum
    }

    /// Evaluate the nearest-neighbour kernel (:a0) at every point
    fn evaluate_nearest(&self, points: ArrayView2<'_, f64>) -> PyResult<Vec<f64>> {
        let mut out: Vec<f64> = Vec::with_capacity(points.nrows());
        for point in points.rows() {
            let mut index = 0usize;
            for (d, axis) in self.axes.iter().enumerate() {
                let x = point[d];
                check_range(axis, d, x)?;
                let (start, t) = locate(axis, x, self.eqs);
                // the left knot for t < ½, the right one otherwise (as in Julia), on every axis
                let nearest = if t < 0.5 { start } else { start + 1 };
                index += nearest * axis.stride;
            }
            out.push(self.coefs[index]);
        }
        Ok(out)
    }
}

/// The spacing of uniform knots along `axis`, with checks that they increase and are uniform
fn uniform_spacing(x: &[f64], axis: usize) -> PyResult<f64> {
    let n = x.len();
    if n < 2 {
        return Err(PyValueError::new_err(format!("axis {axis}: need at least 2 knots")));
    }
    // grid spacing from the full span
    let h = (x[n - 1] - x[0]) / (n - 1) as f64;
    if !(h > 0.0) {
        return Err(PyValueError::new_err(format!("axis {axis}: knots must be increasing")));
    }
    for (k, &xk) in x.iter().enumerate() {
        if (xk - (x[0] + k as f64 * h)).abs() > 1e-8 * h {
            return Err(PyValueError::new_err(format!(
                "axis {axis}: knots must be uniformly spaced"
            )));
        }
    }
    Ok(h)
}

/// Reject coordinates outside the data range, like Julia's default `Throw` extrapolation
#[inline]
fn check_range(axis: &GridAxis, d: usize, x: f64) -> PyResult<()> {
    if x < axis.x_first || x > axis.x_last {
        return Err(PyValueError::new_err(format!(
            "coordinate {x} on axis {d} lies outside the data range [{}, {}]",
            axis.x_first, axis.x_last
        )));
    }
    Ok(())
}

/// Along one axis: the first stencil coefficient of the cell containing x, and x's position t in
/// that cell (the same computation as the 1D interpolant)
#[inline]
fn locate(axis: &GridAxis, x: f64, eqs: usize) -> (usize, f64) {
    // position in index units; cell i (0-based), clamped so the stencil stays inside
    let u = (x - axis.x0) / axis.h;
    let lowest = (eqs - 1) as isize;
    let highest = (axis.len - eqs - 1) as isize;
    let i = (u.floor() as isize).clamp(lowest, highest) as usize;
    (i + 1 - eqs, u - i as f64)
}