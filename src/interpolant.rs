//! A 1D convolution interpolant on a uniform grid, as in ConvolutionInterpolations.jl
//! (the eager 1D evaluator), for values, derivatives and integrals of any order.

use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use wide::f64x4;

use crate::coefficients::{self, Boundary};
use crate::kernels::{self, AnchorTable, ColumnTable};

/// r! for r = 0 … 8, exact in f64: enough for every integral order the kernels provide (m ≤ 8)
const FACTORIALS: [f64; 9] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0];

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
    /// Nearest neighbour (:a0), which needs no weight tables
    nearest: bool,
    /// Number of stencil columns K of the kernel (2·eqs)
    columns: usize,
    /// The column table of the requested order (derivative, or integral for negative orders), as
    /// 4-wide vectors padded with zero columns to a multiple of four: each row is B = ⌈K/4⌉
    /// consecutive vectors
    padded_rows: Vec<f64x4>,
    /// Factor applied to every result: (−1/h)^derivative for values and derivatives (exactly 1 for
    /// values), h^m for integrals of order m
    scale: f64,
    /// Integral order m (derivative order −m); 0 for values and derivatives
    integral: usize,
    /// Integrals only: the anchoring Taylor values K_{m−r}(eqs − j), 2·eqs − 1 rows of m values
    taylor: Vec<f64>,
    /// Integrals only: the left tail, one row of m values per extended coefficient l, holding the
    /// coefficients of t^0 … t^(m−1) of the contribution of all coefficients up to l, for a point
    /// in the cell whose stencil starts just right of l
    tails: Vec<f64>,
    /// Integrals only (not :a0): the column table of K_m with each column's far-field term
    /// (t + eqs − k − 1)^(m−1) / (2·(m−1)!) folded in, padded as `padded_rows`. In a cell whose
    /// stencil lies entirely beyond the anchor's reach (start ≥ 2·eqs − 1) these are the anchored
    /// weights themselves, so almost every cell needs no per-coefficient correction.
    far_rows: Vec<f64x4>,
}

#[pymethods]
impl Interpolant1D {
    /// Construct from uniform knots `x`, data `values`, a kernel name, the two boundary conditions
    /// and the derivative order: 0 for values, positive for derivatives, −m for the m-fold integral
    /// (anchored at zero, with all lower integrals, at the first knot). `#[new]` makes this the
    /// Python constructor.
    #[new]
    fn new(
        x: PyReadonlyArray1<'_, f64>,
        values: PyReadonlyArray1<'_, f64>,
        kernel: &str,
        bc_left: &str,
        bc_right: &str,
        derivative: i32,
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
                return Err(PyValueError::new_err("knots must be uniformly spaced"));
            }
        }

        // The order must be one the kernel provides: derivatives up to max_order, integrals up
        // to max_integral
        let max_order = kernels::max_derivative(kernel)
            .ok_or_else(|| PyValueError::new_err(format!("unknown kernel {kernel:?}")))?;
        let max_integral = kernels::max_integral(kernel)
            .ok_or_else(|| PyValueError::new_err(format!("unknown kernel {kernel:?}")))?;
        if derivative > max_order {
            return Err(PyValueError::new_err(format!(
                "kernel {kernel:?} supports derivatives up to order {max_order}, got {derivative}"
            )));
        }
        if -derivative > max_integral {
            return Err(PyValueError::new_err(format!(
                "kernel {kernel:?} supports integrals up to order {max_integral} \
                 (derivative = -{max_integral}), got derivative = {derivative}"
            )));
        }
        // the integral order m, or 0 for values and derivatives
        let integral = if derivative < 0 { (-derivative) as usize } else { 0 };

        // The kernel: nearest neighbour, or the column table of the requested order, converted
        // to padded 4-wide vectors; for integrals also the table with the far field folded in
        let (nearest, columns, padded_rows, far_rows) = if kernel == "a0" {
            (true, 2, Vec::new(), Vec::new())
        } else {
            let table = kernels::column_table(kernel, derivative).ok_or_else(|| {
                PyValueError::new_err(format!("no table for kernel {kernel:?}, order {derivative}"))
            })?;
            let far_rows = if integral > 0 {
                pad_values(&far_field_rows(table, integral), table.columns)
            } else {
                Vec::new()
            };
            (false, table.columns, pad_rows(table), far_rows)
        };

        // The extended coefficients
        let left = Boundary::parse(bc_left).map_err(|e| PyValueError::new_err(e))?;
        let right = Boundary::parse(bc_right).map_err(|e| PyValueError::new_err(e))?;
        let coefs = coefficients::extended_coefficients(values, kernel, left, right)
            .map_err(|e| PyValueError::new_err(e))?;
        let eqs = coefficients::stencil_half_width(kernel).map_err(|e| PyValueError::new_err(e))?;

        // Integrals: the anchoring Taylor values and the left tail, built once from the
        // coefficients; nothing for values and derivatives
        let (taylor, tails) = if integral > 0 {
            let anchor = kernels::anchor_table(kernel, integral as i32).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "no anchoring table for kernel {kernel:?}, integral order {integral}"
                ))
            })?;
            (anchor.taylor.to_vec(), left_tails(&coefs, anchor, eqs))
        } else {
            (Vec::new(), Vec::new())
        };

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
            // integrals: ∫ dx = h·∫ du, once per order; derivatives: d/dx = (1/h)·d/du, and the
            // columns are functions of tau = 1 − t, hence (−1/h)^d
            scale: if integral > 0 {
                h.powi(integral as i32)
            } else {
                (-1.0 / h).powi(derivative)
            },
            integral,
            taylor,
            tails,
            far_rows,
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
        let values = if self.integral > 0 {
            // integrals of any order, every kernel (including :a0, with K = 2)
            match self.columns {
                2 => self.evaluate_integral::<2, 1>(points)?,
                4 => self.evaluate_integral::<4, 1>(points)?,
                6 => self.evaluate_integral::<6, 2>(points)?,
                8 => self.evaluate_integral::<8, 2>(points)?,
                10 => self.evaluate_integral::<10, 3>(points)?,
                12 => self.evaluate_integral::<12, 3>(points)?,
                14 => self.evaluate_integral::<14, 4>(points)?,
                16 => self.evaluate_integral::<16, 4>(points)?,
                18 => self.evaluate_integral::<18, 5>(points)?,
                20 => self.evaluate_integral::<20, 5>(points)?,
                k => {
                    return Err(PyValueError::new_err(format!(
                        "no evaluator for a kernel with {k} columns"
                    )))
                }
            }
        } else if self.nearest {
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
            // (−1/h)^d for derivatives; exactly 1 for values
            out.push(sum * self.scale);
        }
        Ok(out)
    }

    /// Evaluate the m-fold integral (m = self.integral) at every point, for a kernel with K
    /// stencil columns stored as B 4-wide vectors per table row, as Julia's generic integral
    /// evaluator does in 1D:
    ///   in the stencil       — K_m weight minus its Taylor polynomial at the anchor
    ///   left of the stencil  — the precomputed tail, a polynomial in t
    ///   right of the stencil — exactly zero
    /// scaled by h^m. Beyond the anchor's reach (almost every cell) the Taylor polynomial is the
    /// far field, already folded into `far_rows`, so those cells cost one table evaluation.
    fn evaluate_integral<const K: usize, const B: usize>(
        &self,
        points: ArrayView1<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        // The padded table rows of K_m, and of K_m with the far field folded in, as arrays of
        // B vectors each (both empty for :a0)
        let (rows, _) = self.padded_rows.as_chunks::<B>();
        let (far_rows, _) = self.far_rows.as_chunks::<B>();
        let m = self.integral;
        let eqs = self.eqs;
        // divisor of the far field (u − j)^(m−1) / (2·(m−1)!)
        let far = 2.0 * FACTORIALS[m - 1];
        let mut out: Vec<f64> = Vec::with_capacity(points.len());
        for &x in points.iter() {
            self.check_range(x)?;
            let (start, t) = self.locate(x);
            // the K coefficients of the stencil, as a fixed-size array (one bounds check)
            let c: &[f64; K] = self.coefs[start..start + K].try_into().unwrap();
            let mut sum = 0.0_f64;

            if !self.nearest && start >= 2 * eqs - 1 {
                // the whole stencil is beyond the anchor's reach: the far-field table at
                // tau = 1 − t gives the anchored weights directly, unpacked from B vectors of four
                let w = horner_weights::<B>(far_rows, 1.0 - t);
                let lanes: [[f64; 4]; B] = w.map(|v| v.to_array());
                let weights = lanes.as_flattened();
                for k in 0..K {
                    sum = c[k].mul_add(weights[k], sum);
                }
            } else {
                // near the anchor (the first 2·eqs − 1 cells), and :a0 everywhere: K_m of the
                // K stencil coefficients, corrected coefficient by coefficient
                let raw: [f64; K] = if self.nearest {
                    // :a0 in closed form (K = 2)
                    let w = a0_integral_weights(m, t);
                    std::array::from_fn(|k| w[k])
                } else {
                    // the column table at tau = 1 − t, unpacked from B vectors of four
                    let w = horner_weights::<B>(rows, 1.0 - t);
                    let lanes: [[f64; 4]; B] = w.map(|v| v.to_array());
                    let flat = lanes.as_flattened();
                    std::array::from_fn(|k| flat[k])
                };
                // position relative to the anchor (the first data knot), in cells
                let v = start as f64 + t;
                for k in 0..K {
                    let j = start + k; // index of this coefficient
                    let weight = if j >= 2 * eqs - 1 {
                        // beyond the anchor's reach: its Taylor polynomial is the far field
                        // −(u − j)^(m−1) / (2·(m−1)!), with u − j = t + eqs − k − 1
                        let s = t + (eqs as isize - k as isize - 1) as f64;
                        raw[k] + int_power(s, m - 1) / far
                    } else {
                        // within reach: subtract Σ_r K_{m−r}(eqs − j)·v^r / r! from the table
                        let row = &self.taylor[j * m..(j + 1) * m];
                        let mut acc = 0.0_f64;
                        let mut vr = 1.0_f64;
                        for r in 0..m {
                            acc += row[r] * vr / FACTORIALS[r];
                            vr *= v;
                        }
                        raw[k] - acc
                    };
                    sum = c[k].mul_add(weight, sum);
                }
            }

            // every coefficient left of the stencil: the tail of the coefficient just left of it,
            // a polynomial in t evaluated by Horner's scheme (none when the stencil starts at 0)
            if start >= 1 {
                let tail = &self.tails[(start - 1) * m..start * m];
                let mut acc = 0.0_f64;
                for k in (0..m).rev() {
                    acc = acc.mul_add(t, tail[k]);
                }
                sum += acc;
            }

            // h^m
            out.push(sum * self.scale);
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

/// s^p for a small non-negative integer p, by repeated multiplication (inlined, unlike powi with
/// a runtime exponent, which becomes a library call)
#[inline]
fn int_power(s: f64, p: usize) -> f64 {
    let mut result = 1.0_f64;
    for _ in 0..p {
        result *= s;
    }
    result
}

/// K_m of the two stencil coefficients of :a0 (left, right) at position t in the cell, in closed
/// form as in Julia: K₁(s) = clamp(s, −½, ½), and K₂(s) = s²/2 + 1/8 for |s| ≤ ½, |s|/2 beyond
#[inline]
fn a0_integral_weights(m: usize, t: f64) -> [f64; 2] {
    if m == 1 {
        // left coefficient: K₁(t); right coefficient: K₁(t − 1)
        if t > 0.5 { [0.5, t - 1.0] } else { [t, -0.5] }
    } else {
        // left coefficient: K₂(t); right coefficient: K₂(t − 1)
        let left = if t <= 0.5 { t * t / 2.0 + 0.125 } else { t / 2.0 };
        let right = if t > 0.5 { (t - 1.0) * (t - 1.0) / 2.0 + 0.125 } else { (1.0 - t) / 2.0 };
        [left, right]
    }
}

/// binomial(n, k) as f64, exact for the small arguments used here
fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The rows of the integral table of order m with each column's far-field term folded in: column
/// k gains (t + eqs − k − 1)^(m−1) / (2·(m−1)!), which in tau = 1 − t is
/// (eqs − k − tau)^(m−1) / (2·(m−1)!) = Σ_p binomial(m−1, p)·(eqs − k)^(m−1−p)·(−tau)^p / (2·(m−1)!).
/// Same layout as the table: rows from the highest power of tau down, `columns` values each.
fn far_field_rows(table: &ColumnTable, m: usize) -> Vec<f64> {
    let k_cols = table.columns;
    let eqs = k_cols / 2;
    let degree = table.rows.len() / k_cols - 1;
    let far = 2.0 * FACTORIALS[m - 1];
    let mut rows = table.rows.to_vec();
    for k in 0..k_cols {
        // eqs − k, an exact small integer
        let a = eqs as f64 - k as f64;
        for p in 0..m {
            // coefficient of tau^p: binomial(m−1, p)·(eqs − k)^(m−1−p)·(−1)^p, exact in f64
            let sign = if p % 2 == 1 { -1.0 } else { 1.0 };
            let coef = sign * binomial(m - 1, p) * int_power(a, m - 1 - p);
            // the row of tau^p (rows run from the highest power down)
            rows[(degree - p) * k_cols + k] += coef / far;
        }
    }
    rows
}

/// The left tail of integral order m, as Julia's _left_tail_polynomial in 1D: row l holds the
/// coefficients of t^0 … t^(m−1) of Σ_{j ≤ l} c_j·W_j(t + l + eqs), the contribution of all
/// coefficients up to l for a point in the cell whose stencil starts at l + 1. Coefficients far
/// from the anchor enter with (t + eqs)^(m−1) / (m−1)!, those near it with the exact entry
/// polynomials of the anchor table. Built by the shifted prefix sum
/// Λ(l) = S·Λ(l − 1) + c_l·entry_l, where S re-expands a polynomial in t around the next cell.
fn left_tails(coefs: &[f64], anchor: &AnchorTable, eqs: usize) -> Vec<f64> {
    let m = anchor.order;
    let n = coefs.len();
    // entry polynomial far from the anchor: binomial(m−1, k)·eqs^(m−1−k) / (m−1)!
    let far_entry: Vec<f64> = (0..m)
        .map(|k| binomial(m - 1, k) * (eqs as f64).powi((m - 1 - k) as i32) / FACTORIALS[m - 1])
        .collect();
    let mut tails = vec![0.0_f64; n * m];
    for l in 0..n {
        // the exact entry polynomial near the anchor, the far one beyond it
        let entry: &[f64] = if l < anchor.rows {
            &anchor.entries[l * m..(l + 1) * m]
        } else {
            &far_entry
        };
        for k in 0..m {
            let mut acc = entry[k] * coefs[l];
            if l > 0 {
                // Taylor shift of the previous row by one cell: Σ_{q ≥ k} binomial(q, k)·Λ_q(l − 1)
                for q in k..m {
                    acc = acc + binomial(q, k) * tails[(l - 1) * m + q];
                }
            }
            tails[l * m + k] = acc;
        }
    }
    tails
}

/// The rows of a column table as 4-wide vectors, padded with zero columns to a multiple of four
pub(crate) fn pad_rows(table: &ColumnTable) -> Vec<f64x4> {
    pad_values(table.rows, table.columns)
}

/// Rows of `k` values each (highest power of tau first) as 4-wide vectors, padded with zero
/// columns to a multiple of four
fn pad_values(rows: &[f64], k: usize) -> Vec<f64x4> {
    let blocks = (k + 3) / 4;
    let mut padded = Vec::with_capacity(rows.len() / k * blocks);
    for row in rows.chunks_exact(k) {
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
pub(crate) fn horner_weights<const B: usize>(rows: &[[f64x4; B]], tau: f64) -> [f64x4; B] {
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