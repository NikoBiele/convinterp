//! A 1D convolution interpolant on a uniform grid, as in ConvolutionInterpolations.jl
//! (the eager 1D evaluator).

use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use wide::f64x4;

use crate::coefficients::{self, Boundary};
use crate::kernels::{self, ColumnTable};

/// A 1D interpolant. `#[pyclass]` makes it a Python class; its fields stay private to Rust.
#[pyclass]
pub struct Interpolant1D {
    /// The data values extended by eqs − 1 ghost values beyond each boundary
    coefs: Vec<f64>,
    /// Position of the first extended knot, and the grid spacing
    x0: f64,
    h: f64,
    /// Stencil half-width of the kernel
    eqs: usize,
    /// The data range; points outside it are rejected
    x_first: f64,
    x_last: f64,
    /// Nearest neighbour (:a0), which needs no weights
    nearest: bool,
    /// Number of stencil columns K of the kernel (2·eqs)
    columns: usize,
    /// The kernel's column table as 4-wide vectors, padded with zero columns to a multiple of four:
    /// each row is B = ⌈K/4⌉ consecutive vectors, highest power of tau first
    padded_rows: Vec<f64x4>,
}

#[pymethods]
impl Interpolant1D {
    /// Construct from uniform knots `x`, data `values`, a kernel name and the two boundary
    /// conditions. `#[new]` makes this the Python constructor, Interpolant1D(...).
    #[new]
    fn new(
        x: PyReadonlyArray1<'_, f64>,
        values: PyReadonlyArray1<'_, f64>,
        kernel: &str,
        bc_left: &str,
        bc_right: &str,
    ) -> PyResult<Self> {
        let x = x.as_slice()?;
        let values = values.as_slice()?;
        let n = x.len();
        if n < 2 || values.len() != n {
            return Err(PyValueError::new_err(
                "need at least 2 knots, and as many values as knots",
            ));
        }

        // Grid spacing from the full span, and a check that the knots are uniform
        let h = (x[n - 1] - x[0]) / (n - 1) as f64;
        if !(h > 0.0) {
            return Err(PyValueError::new_err("knots must be increasing"));
        }
        for (k, &xk) in x.iter().enumerate() {
            if (xk - (x[0] + k as f64 * h)).abs() > 1e-8 * h {
                return Err(PyValueError::new_err(
                    "knots must be uniformly spaced (nonuniform data: use fit_scattered)",
                ));
            }
        }

        // The kernel: nearest neighbour, or a column table converted to padded 4-wide vectors
        let (nearest, columns, padded_rows) = if kernel == "a0" {
            (true, 2, Vec::new())
        } else {
            let table = kernels::column_table(kernel, 0).ok_or_else(|| {
                PyValueError::new_err(format!("unknown kernel {kernel:?}"))
            })?;
            (false, table.columns, pad_rows(table))
        };

        // The extended coefficients
        let left = Boundary::parse(bc_left).map_err(|e| PyValueError::new_err(e))?;
        let right = Boundary::parse(bc_right).map_err(|e| PyValueError::new_err(e))?;
        let coefs = coefficients::extended_coefficients(values, kernel, left, right)
            .map_err(|e| PyValueError::new_err(e))?;
        let eqs = coefficients::stencil_half_width(kernel).map_err(|e| PyValueError::new_err(e))?;

        Ok(Interpolant1D {
            coefs,
            x0: x[0] - (eqs - 1) as f64 * h,
            h,
            eqs,
            x_first: x[0],
            x_last: x[n - 1],
            nearest,
            columns,
            padded_rows,
        })
    }

    /// Evaluate at every point of the NumPy array `points`
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points = points.as_array();
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
impl Interpolant1D {
    /// Evaluate a kernel with K stencil columns, stored as B 4-wide vectors per table row, at
    /// every point. K and B are compile-time constants: one specialized copy per kernel size.
    fn evaluate_columns<const K: usize, const B: usize>(
        &self,
        points: ArrayView1<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        // The padded table rows, as arrays of B vectors each
        let (rows, _) = self.padded_rows.as_chunks::<B>();
        let mut out: Vec<f64> = Vec::with_capacity(points.len());
        for &x in points.iter() {
            self.check_range(x)?;
            let (start, t) = self.locate(x);
            // the weights at tau = 1 − t, as B vectors of four
            let w = horner_weights::<B>(rows, 1.0 - t);
            // unpacked into 4·B plain numbers, of which the first K are the stencil's weights
            let lanes: [[f64; 4]; B] = w.map(|v| v.to_array());
            let weights = lanes.as_flattened();
            // the K coefficients of the stencil, as a fixed-size array (one bounds check)
            let c: &[f64; K] = self.coefs[start..start + K].try_into().unwrap();
            let mut sum = 0.0_f64;
            for k in 0..K {
                sum = c[k].mul_add(weights[k], sum);
            }
            out.push(sum);
        }
        Ok(out)
    }

    /// Evaluate the nearest-neighbour kernel (:a0) at every point
    fn evaluate_nearest(&self, points: ArrayView1<'_, f64>) -> PyResult<Vec<f64>> {
        let mut out: Vec<f64> = Vec::with_capacity(points.len());
        for &x in points.iter() {
            self.check_range(x)?;
            let (start, t) = self.locate(x);
            // the left coefficient for t < ½, the right one otherwise (as in Julia)
            out.push(if t < 0.5 { self.coefs[start] } else { self.coefs[start + 1] });
        }
        Ok(out)
    }

    /// Reject points outside the data range, like Julia's default `Throw` extrapolation
    #[inline]
    fn check_range(&self, x: f64) -> PyResult<()> {
        if x < self.x_first || x > self.x_last {
            return Err(PyValueError::new_err(format!(
                "point {x} lies outside the data range [{}, {}]",
                self.x_first, self.x_last
            )));
        }
        Ok(())
    }

    /// The first stencil coefficient of the cell containing x, and x's position t in that cell
    #[inline]
    fn locate(&self, x: f64) -> (usize, f64) {
        let n_coefs = self.coefs.len();
        // position in index units; cell i (0-based), clamped so the stencil stays inside
        let u = (x - self.x0) / self.h;
        let lowest = (self.eqs - 1) as isize;
        let highest = (n_coefs - self.eqs - 1) as isize;
        let i = (u.floor() as isize).clamp(lowest, highest) as usize;
        (i + 1 - self.eqs, u - i as f64)
    }
}

/// The rows of a column table as 4-wide vectors, padded with zero columns to a multiple of four
fn pad_rows(table: &ColumnTable) -> Vec<f64x4> {
    let k = table.columns;
    let blocks = (k + 3) / 4;
    let mut padded = Vec::with_capacity(table.rows.len() / k * blocks);
    for row in table.rows.chunks_exact(k) {
        for b in 0..blocks {
            // four consecutive columns of this row; zero beyond the last column
            let lanes: [f64; 4] =
                std::array::from_fn(|l| row.get(4 * b + l).copied().unwrap_or(0.0));
            padded.push(f64x4::new(lanes));
        }
    }
    padded
}

/// The weights at tau as B vectors of four: Horner's scheme over the rows, where every step is
/// one explicit 4-wide fused multiply-add per vector, whatever the kernel size
#[inline]
fn horner_weights<const B: usize>(rows: &[[f64x4; B]], tau: f64) -> [f64x4; B] {
    let tau = f64x4::splat(tau);
    let mut w = [f64x4::splat(0.0); B];
    for row in rows {
        for b in 0..B {
            // w ← w·tau + row, four columns at once
            w[b] = w[b].mul_add(tau, row[b]);
        }
    }
    w
}