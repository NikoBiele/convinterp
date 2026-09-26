//! An N-D convolution interpolant on a uniform grid, as in ConvolutionInterpolations.jl (the
//! eager N-D evaluator), for values, derivatives and integrals, with an order per axis.

use numpy::ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use wide::f64x4;

use crate::coefficients::{self, Boundary};
use crate::integral::IntegralAxis;
use crate::interpolant::{horner_weights, pad_rows};
use crate::kernels;

/// The largest memory the integral tails may take, in bytes (2 GiB)
const MAX_TAIL_BYTES: f64 = 2.0 * 1024.0 * 1024.0 * 1024.0;

/// One axis of the grid, and the kernel data of this axis's order
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
    /// Values and derivatives: the column table of this axis's order, as 4-wide vectors padded
    /// with zero columns to a multiple of four (empty for nearest neighbour and integral axes)
    padded_rows: Vec<f64x4>,
    /// Integral axes (negative order): the tables and anchoring data of the integral
    integral: Option<IntegralAxis>,
}

/// One region of an integral evaluation: the coefficients left of the stencil along the integral
/// axes of the region, and within the stencil along all other axes (Julia: _build_region_tails)
struct Region {
    /// Per axis: whether it belongs to the region
    in_region: Vec<bool>,
    /// The axes of the region, in increasing order
    axes: Vec<usize>,
    /// Number of moments per coefficient: the product of the integral orders of the region's axes
    moments: usize,
    /// The tail, `moments` values per extended coefficient (position·moments + moment): the
    /// coefficient of Π t_d^k_d over the region's axes, the last axis's power varying fastest
    tails: Vec<f64>,
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
    /// Nearest neighbour (:a0), which needs no weight tables
    nearest: bool,
    /// Number of stencil columns K of the kernel (2·eqs), the same on every axis
    columns: usize,
    /// Factor applied to every result: the product over the axes of (−1/h)^q for values and
    /// derivatives of order q, and h^m for integrals of order m
    scale: f64,
    /// Integrals: one region per non-empty subset of the integral axes (empty without integrals)
    regions: Vec<Region>,
}

#[pymethods]
impl InterpolantND {
    /// Construct from one vector of uniform knots per axis, the N-D data `values`, a kernel name,
    /// one (left, right) pair of boundary conditions per axis, and one order per axis: 0 for
    /// values, positive for derivatives, −m for the m-fold integral (anchored at zero, with all
    /// lower integrals, at the axis's first knot).
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

        // The kernel: its highest derivative and integral orders, and its stencil half-width
        let max_order = kernels::max_derivative(kernel)
            .ok_or_else(|| PyValueError::new_err(format!("unknown kernel {kernel:?}")))?;
        let max_integral = kernels::max_integral(kernel)
            .ok_or_else(|| PyValueError::new_err(format!("unknown kernel {kernel:?}")))?;
        let nearest = kernel == "a0";
        let eqs = coefficients::stencil_half_width(kernel).map_err(|e| PyValueError::new_err(e))?;

        // Per axis: the grid (spacing and data range), and the table of its order or its
        // integral data
        let mut grid: Vec<(f64, f64, f64)> = Vec::with_capacity(n_dims); // (h, x_first, x_last)
        let mut tables: Vec<Vec<f64x4>> = Vec::with_capacity(n_dims);
        let mut integrals: Vec<Option<IntegralAxis>> = Vec::with_capacity(n_dims);
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
            if order > max_order {
                return Err(PyValueError::new_err(format!(
                    "axis {d}: kernel {kernel:?} supports derivatives up to order {max_order}, \
                     got {order}"
                )));
            }
            if -order > max_integral {
                return Err(PyValueError::new_err(format!(
                    "axis {d}: kernel {kernel:?} supports integrals up to order {max_integral} \
                     (derivative = -{max_integral}), got derivative = {order}"
                )));
            }
            // the column count K, from the table of this order (none for nearest neighbour)
            if !nearest {
                let table = kernels::column_table(kernel, order).ok_or_else(|| {
                    PyValueError::new_err(format!("no table for kernel {kernel:?}, order {order}"))
                })?;
                columns = table.columns;
                // values and derivatives keep the padded table; integral axes have their own
                tables.push(if order < 0 { Vec::new() } else { pad_rows(table) });
            } else {
                tables.push(Vec::new());
            }
            integrals.push(if order < 0 {
                Some(IntegralAxis::new(kernel, (-order) as usize).map_err(|e| PyValueError::new_err(e))?)
            } else {
                None
            });
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

        // Integrals: the regions and their tails, after checking that they fit in memory
        let regions = build_regions(&coefs, &integrals, &lens, &strides)?;

        // Per axis: d/dx = (1/h)·d/du, with the columns functions of tau = 1 − t, hence (−1/h)^q;
        // ∫ dx = h·∫ du, once per integral order
        let scale: f64 = grid
            .iter()
            .zip(&derivatives)
            .map(|(&(h, _, _), &order)| {
                if order < 0 { h.powi(-order) } else { (-1.0 / h).powi(order) }
            })
            .product();

        let axes: Vec<GridAxis> = tables
            .into_iter()
            .zip(integrals)
            .enumerate()
            .map(|(d, (padded_rows, integral))| {
                let (h, x_first, x_last) = grid[d];
                GridAxis {
                    x0: x_first - (eqs - 1) as f64 * h,
                    h,
                    x_first,
                    x_last,
                    len: lens[d],
                    stride: strides[d],
                    padded_rows,
                    integral,
                }
            })
            .collect();

        Ok(InterpolantND { coefs, axes, eqs, nearest, columns, scale, regions })
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
        let has_integral = self.axes.iter().any(|axis| axis.integral.is_some());
        let values = if has_integral {
            // integrals along at least one axis, every kernel (including :a0, with K = 2)
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

    /// Evaluate with integrals along at least one axis, for a kernel with K stencil columns stored
    /// as B 4-wide vectors per table row, as Julia's generic integral evaluator: the stencil sum
    /// with anchored weights on the integral axes (region ∅), plus for every region the tail of
    /// the coefficients left of the stencil along the region's axes, summed over the stencil
    /// along all other axes; then the scale.
    fn evaluate_integral<const K: usize, const B: usize>(
        &self,
        points: ArrayView2<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        let n_dims = self.axes.len();
        // Per axis: the padded table rows of values and derivatives, as arrays of B vectors each
        let rows: Vec<&[[f64x4; B]]> =
            self.axes.iter().map(|axis| axis.padded_rows.as_chunks::<B>().0).collect();
        // Per axis, reused for every point: the K stencil weights, the stencil start, and t
        let mut weights: Vec<[f64; K]> = vec![[0.0; K]; n_dims];
        let mut starts: Vec<usize> = vec![0; n_dims];
        let mut ts: Vec<f64> = vec![0.0; n_dims];
        // Per region, reused for every point: the powers Π t_d^k_d of its moments
        let mut powers: Vec<Vec<f64>> =
            self.regions.iter().map(|region| vec![0.0; region.moments]).collect();
        let mut out: Vec<f64> = Vec::with_capacity(points.nrows());
        for point in points.rows() {
            // position of the stencil's first coefficient in the flat vector
            let mut base = 0usize;
            for (d, axis) in self.axes.iter().enumerate() {
                let x = point[d];
                check_range(axis, d, x)?;
                let (start, t) = locate(axis, x, self.eqs);
                starts[d] = start;
                ts[d] = t;
                base += start * axis.stride;
                weights[d] = if let Some(integral) = &axis.integral {
                    // integral axis: the anchored weights
                    integral.weights::<K, B>(start, t)
                } else if self.nearest {
                    // :a0 value axis: the left coefficient for t < ½, the right one otherwise
                    let w = if t < 0.5 { [1.0, 0.0] } else { [0.0, 1.0] };
                    std::array::from_fn(|k| w[k])
                } else {
                    // value or derivative axis: the table at tau = 1 − t, unpacked from B vectors
                    let w = horner_weights::<B>(rows[d], 1.0 - t);
                    let lanes: [[f64; 4]; B] = w.map(|v| v.to_array());
                    let flat = lanes.as_flattened();
                    std::array::from_fn(|k| flat[k])
                };
            }

            // region ∅: every axis within the stencil
            let mut sum = self.contract::<K>(0, base, &weights);

            // every other region: its axes left of the stencil
            for (region, pw) in self.regions.iter().zip(powers.iter_mut()) {
                // empty unless every axis of the region has coefficients left of its stencil
                if region.axes.iter().any(|&d| starts[d] == 0) {
                    continue;
                }
                // the powers of the positions, the region's last axis varying fastest
                fill_powers(region, &self.axes, &ts, pw);
                // the tail sits one step left of the stencil start on the region's axes
                let offset = base - region.axes.iter().map(|&d| self.axes[d].stride).sum::<usize>();
                sum += self.contract_region::<K>(region, 0, offset, &weights, pw);
            }

            // h^m and (−1/h)^q factors applied at the end
            out.push(sum * self.scale);
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

    /// The sum of one region over axes `axis`, `axis` + 1, … starting at `offset`: on the region's
    /// axes the position is fixed (the tail, weight 1); on every other axis Σ_k w[axis][k] · (the
    /// same sum over the remaining axes, at offset + k·stride). At the end, the region's moments
    /// at that position dotted with the powers of the positions.
    fn contract_region<const K: usize>(
        &self,
        region: &Region,
        axis: usize,
        offset: usize,
        weights: &[[f64; K]],
        powers: &[f64],
    ) -> f64 {
        if axis == weights.len() {
            // every axis placed: the tail polynomial at this position
            let m = region.moments;
            let tail = &region.tails[offset * m..(offset + 1) * m];
            let mut acc = 0.0_f64;
            for q in 0..m {
                acc = tail[q].mul_add(powers[q], acc);
            }
            return acc;
        }
        if region.in_region[axis] {
            // an axis of the region: its tail position is already in `offset`
            return self.contract_region::<K>(region, axis + 1, offset, weights, powers);
        }
        let w = &weights[axis];
        let stride = self.axes[axis].stride;
        let mut sum = 0.0_f64;
        for k in 0..K {
            // the remaining axes, one step further along this one
            let inner = self.contract_region::<K>(region, axis + 1, offset + k * stride, weights, powers);
            sum = w[k].mul_add(inner, sum);
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

/// The regions of an integral evaluation, one per non-empty subset of the integral axes (bit b of
/// the mask ↔ the b-th integral axis, as in Julia), each with its tail built one axis at a time
/// along the region's axes. Checks first that all tails together fit in MAX_TAIL_BYTES: they take
/// (Π_d (1 + m_d) − 1) values per coefficient. No integral axes: no regions.
fn build_regions(
    coefs: &[f64],
    integrals: &[Option<IntegralAxis>],
    lens: &[usize],
    strides: &[usize],
) -> PyResult<Vec<Region>> {
    let n_dims = integrals.len();
    // the integral axes, in increasing order
    let integral_axes: Vec<usize> = (0..n_dims).filter(|&d| integrals[d].is_some()).collect();
    if integral_axes.is_empty() {
        return Ok(Vec::new());
    }

    // the memory of all tails, computed in f64 so that it cannot overflow
    let values_per_coefficient: f64 = integral_axes
        .iter()
        .map(|&d| 1.0 + integrals[d].as_ref().map_or(0, |i| i.order) as f64)
        .product::<f64>()
        - 1.0;
    let bytes = values_per_coefficient * coefs.len() as f64 * 8.0;
    if bytes > MAX_TAIL_BYTES {
        return Err(PyValueError::new_err(format!(
            "the integral tails would need {:.1} GiB of memory (limit {:.0} GiB): integrate along \
             fewer axes, use lower integral orders, or use a smaller grid",
            bytes / (1024.0 * 1024.0 * 1024.0),
            MAX_TAIL_BYTES / (1024.0 * 1024.0 * 1024.0)
        )));
    }

    let mut regions = Vec::with_capacity((1usize << integral_axes.len()) - 1);
    for mask in 1..(1usize << integral_axes.len()) {
        // the integral axes whose bit is set, in increasing order
        let axes: Vec<usize> = integral_axes
            .iter()
            .enumerate()
            .filter(|&(b, _)| (mask >> b) & 1 == 1)
            .map(|(_, &d)| d)
            .collect();
        let mut in_region = vec![false; n_dims];
        for &d in &axes {
            in_region[d] = true;
        }
        // the tail along the first axis of the region, from the coefficients (1 value each) …
        let first = integrals[axes[0]].as_ref().unwrap();
        let mut tails = first.extend_tails(coefs, 1, lens[axes[0]], strides[axes[0]]);
        let mut moments = first.order;
        // … then along each further axis, multiplying the moments by its order
        for &d in &axes[1..] {
            let integral = integrals[d].as_ref().unwrap();
            tails = integral.extend_tails(&tails, moments, lens[d], strides[d]);
            moments *= integral.order;
        }
        regions.push(Region { in_region, axes, moments, tails });
    }
    Ok(regions)
}

/// The powers Π t_d^k_d of a region's moments at the positions `ts`, the region's last axis
/// varying fastest (as its tails are stored): built one axis at a time, each moment of the axes
/// so far followed by t_d^0 … t_d^(m_d − 1)
fn fill_powers(region: &Region, axes: &[GridAxis], ts: &[f64], powers: &mut [f64]) {
    powers[0] = 1.0;
    let mut filled = 1usize;
    for &d in &region.axes {
        let m = axes[d].integral.as_ref().map_or(1, |i| i.order);
        let t = ts[d];
        // from the last moment down, so that no moment is overwritten before it is read
        for i in (0..filled).rev() {
            let value = powers[i];
            let mut p = 1.0_f64;
            for k in 0..m {
                powers[i * m + k] = value * p;
                p *= t;
            }
        }
        filled *= m;
    }
}

/// The spacing of uniform knots along `axis`, with checks that they are finite, increase and are
/// uniform
fn uniform_spacing(x: &[f64], axis: usize) -> PyResult<f64> {
    let n = x.len();
    if n < 2 {
        return Err(PyValueError::new_err(format!("axis {axis}: need at least 2 knots")));
    }
    // every knot must be a finite number: a NaN would slip through the checks below, since
    // every comparison with NaN is false
    if x.iter().any(|xk| !xk.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "axis {axis}: knots must be finite numbers (no NaN or infinity)"
        )));
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