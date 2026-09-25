//! A 1D convolution interpolant on a uniform grid, as in ConvolutionInterpolations.jl
//! (the eager 1D evaluator).

use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::coefficients::{self, Boundary};
use crate::kernels::{self, ColumnTable};

/// How the weights of a kernel are computed
#[derive(Clone, Copy)]
enum Weights {
    /// Nearest neighbour (:a0): all weight on the nearer of the two coefficients
    NearestNeighbour,
    /// Every other kernel: the exact column polynomials
    Table(&'static ColumnTable),
}

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
    /// How the kernel weights are computed
    weights: Weights,
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

        // The weights and the extended coefficients
        let weights = if kernel == "a0" {
            Weights::NearestNeighbour
        } else {
            Weights::Table(kernels::column_table(kernel, 0).ok_or_else(|| {
                PyValueError::new_err(format!("unknown kernel {kernel:?}"))
            })?)
        };
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
            weights,
        })
    }

    /// Evaluate at every point of the NumPy array `points`
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points = points.as_array();
        // Choose the specialized evaluator once per call: one compiled copy per column count K
        let values = match self.weights {
            Weights::NearestNeighbour => self.evaluate_nearest(points)?,
            Weights::Table(table) => match table.columns {
                2 => self.evaluate_columns::<2>(table, points)?,
                4 => self.evaluate_columns::<4>(table, points)?,
                6 => self.evaluate_columns::<6>(table, points)?,
                8 => self.evaluate_columns::<8>(table, points)?,
                10 => self.evaluate_columns::<10>(table, points)?,
                12 => self.evaluate_columns::<12>(table, points)?,
                14 => self.evaluate_columns::<14>(table, points)?,
                16 => self.evaluate_columns::<16>(table, points)?,
                18 => self.evaluate_columns::<18>(table, points)?,
                20 => self.evaluate_columns::<20>(table, points)?,
                k => {
                    return Err(PyValueError::new_err(format!(
                        "no evaluator for a kernel with {k} columns"
                    )))
                }
            },
        };
        Ok(values.into_pyarray(py))
    }
}

// Methods only Rust can call: a separate `impl` block without #[pymethods]
impl Interpolant1D {
    /// Evaluate a kernel with K stencil columns at every point. `<const K: usize>` makes K a
    /// compile-time constant: Rust compiles a separate, fully specialized copy for each K used.
    fn evaluate_columns<const K: usize>(
        &self,
        table: &ColumnTable,
        points: ArrayView1<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        // The table rows as fixed-size arrays of K coefficients each
        let (rows, _) = table.rows.as_chunks::<K>();
        let mut out: Vec<f64> = Vec::with_capacity(points.len());
        for &x in points.iter() {
            self.check_range(x)?;
            let (start, t) = self.locate(x);
            // the weights of the K stencil columns at tau = 1 − t
            let w = horner_weights::<K>(rows, 1.0 - t);
            // the K coefficients of the stencil, as a fixed-size array: a single bounds check here,
            // and none in the loop below, since every index is known to be < K
            let c: &[f64; K] = self.coefs[start..start + K].try_into().unwrap();
            let mut sum = 0.0_f64;
            for k in 0..K {
                sum = c[k].mul_add(w[k], sum);
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

/// The weights of all K columns at tau: Horner's scheme over the rows, for all columns at once.
/// With K known at compile time, the compiler can unroll this and use SIMD instructions.
#[inline]
fn horner_weights<const K: usize>(rows: &[[f64; K]], tau: f64) -> [f64; K] {
    let mut w = [0.0_f64; K];
    for row in rows {
        for k in 0..K {
            w[k] = w[k].mul_add(tau, row[k]);
        }
    }
    w
}