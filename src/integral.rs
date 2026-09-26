//! Integrals of any order along one axis, shared by the 1D and N-D interpolants, as in
//! ConvolutionInterpolations.jl: the anchored stencil weights of an integral axis, and the left
//! tails (the contribution of all coefficients left of the stencil) built along it.

use wide::f64x4;

use crate::coefficients;
use crate::interpolant::{horner_weights, pad_rows, pad_values};
use crate::kernels::{self, AnchorTable, ColumnTable};

/// r! for r = 0 … 8, exact in f64: enough for every integral order the kernels provide (m ≤ 8)
const FACTORIALS: [f64; 9] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0];

/// Everything about one axis integrated m times: its kernel tables and anchoring data. The
/// integral is anchored at zero, with all lower integrals, at the first data knot of the axis.
pub(crate) struct IntegralAxis {
    /// The integral order m (derivative order −m)
    pub(crate) order: usize,
    /// Stencil half-width of the kernel
    eqs: usize,
    /// Nearest neighbour (:a0): its weights are computed in closed form, without tables
    nearest: bool,
    /// The column table of K_m as padded 4-wide vectors (empty for :a0)
    padded_rows: Vec<f64x4>,
    /// The same table with each column's far-field term (t + eqs − k − 1)^(m−1) / (2·(m−1)!)
    /// folded in: in a cell whose stencil lies entirely beyond the anchor's reach
    /// (start ≥ 2·eqs − 1) these are the anchored weights themselves (empty for :a0)
    far_rows: Vec<f64x4>,
    /// The exact anchoring data: Taylor values and near-anchor tail entries
    anchor: &'static AnchorTable,
}

impl IntegralAxis {
    /// The integral data of a kernel for integral order m (the caller checks that the kernel
    /// provides this order)
    pub(crate) fn new(kernel: &str, m: usize) -> Result<IntegralAxis, String> {
        let eqs = coefficients::stencil_half_width(kernel)?;
        let anchor = kernels::anchor_table(kernel, m as i32)
            .ok_or_else(|| format!("no anchoring table for kernel {kernel:?}, integral order {m}"))?;
        let (nearest, padded_rows, far_rows) = if kernel == "a0" {
            (true, Vec::new(), Vec::new())
        } else {
            let table = kernels::column_table(kernel, -(m as i32))
                .ok_or_else(|| format!("no table for kernel {kernel:?}, order -{m}"))?;
            (false, pad_rows(table), pad_values(&far_field_rows(table, m), table.columns))
        };
        // the order as the anchor table records it (the table exported for order m)
        Ok(IntegralAxis { order: anchor.order, eqs, nearest, padded_rows, far_rows, anchor })
    }

    /// The anchored weights of the K stencil coefficients start … start + K − 1 for a point at
    /// position t in its cell, in eager order, as Julia's _anchored_stencil_weights: K_m of each
    /// coefficient minus the Taylor polynomial of K_m at the anchor. K and B = ⌈K/4⌉ are
    /// compile-time constants, as in the evaluators.
    #[inline]
    pub(crate) fn weights<const K: usize, const B: usize>(&self, start: usize, t: f64) -> [f64; K] {
        let m = self.order;
        let eqs = self.eqs;

        // The whole stencil beyond the anchor's reach (almost every cell): the far-field table
        // at tau = 1 − t gives the anchored weights directly, unpacked from B vectors of four
        if !self.nearest && start >= 2 * eqs - 1 {
            let (rows, _) = self.far_rows.as_chunks::<B>();
            let w = horner_weights::<B>(rows, 1.0 - t);
            let lanes: [[f64; 4]; B] = w.map(|v| v.to_array());
            let flat = lanes.as_flattened();
            return std::array::from_fn(|k| flat[k]);
        }

        // Near the anchor (the first 2·eqs − 1 cells), and :a0 everywhere: K_m of each
        // coefficient, corrected coefficient by coefficient
        let raw: [f64; K] = if self.nearest {
            // :a0 in closed form (K = 2)
            let w = a0_integral_weights(m, t);
            std::array::from_fn(|k| w[k])
        } else {
            // the column table at tau = 1 − t, unpacked from B vectors of four
            let (rows, _) = self.padded_rows.as_chunks::<B>();
            let w = horner_weights::<B>(rows, 1.0 - t);
            let lanes: [[f64; 4]; B] = w.map(|v| v.to_array());
            let flat = lanes.as_flattened();
            std::array::from_fn(|k| flat[k])
        };
        // position relative to the anchor (the first data knot), in cells
        let v = start as f64 + t;
        // divisor of the far field (u − j)^(m−1) / (2·(m−1)!)
        let far = 2.0 * FACTORIALS[m - 1];
        let taylor = self.anchor.taylor;
        std::array::from_fn(|k| {
            let j = start + k; // index of this coefficient
            if j >= 2 * eqs - 1 {
                // beyond the anchor's reach: its Taylor polynomial is the far field
                // −(u − j)^(m−1) / (2·(m−1)!), with u − j = t + eqs − k − 1
                let s = t + (eqs as isize - k as isize - 1) as f64;
                raw[k] + int_power(s, m - 1) / far
            } else {
                // within reach: subtract Σ_r K_{m−r}(eqs − j)·v^r / r! from the table
                let row = &taylor[j * m..(j + 1) * m];
                let mut acc = 0.0_f64;
                let mut vr = 1.0_f64;
                for r in 0..m {
                    acc += row[r] * vr / FACTORIALS[r];
                    vr *= v;
                }
                raw[k] - acc
            }
        })
    }

    /// The left tail along this axis of a flat array `input` whose entries carry `moments_in`
    /// values each (entry p, value q at p·moments_in + q), for an axis of `len` positions and
    /// `stride` between neighbours (in entries). The result carries moments_in·m values per entry,
    /// at (p·moments_in + q)·m + k: the coefficient of t^k of Σ_{j ≤ l} value_j·W_j(t + l + eqs)
    /// along this axis, where l is the position of p along it, the contribution of all
    /// coefficients up to l for a point in the cell whose stencil starts at l + 1 (Julia's
    /// _left_tail_polynomial). Coefficients far from the anchor enter with (t + eqs)^(m−1)/(m−1)!,
    /// those near it with the exact entry polynomials of the anchor table. Built by the shifted
    /// prefix sum Λ(l) = S·Λ(l − 1) + value_l·entry_l, where S re-expands a polynomial in t
    /// around the next cell. With moments_in = 1 and stride 1 this is the 1D tail; applied along
    /// one axis after another it builds the tail of an N-D region.
    pub(crate) fn extend_tails(
        &self,
        input: &[f64],
        moments_in: usize,
        len: usize,
        stride: usize,
    ) -> Vec<f64> {
        let m = self.order;
        let positions = input.len() / moments_in;
        // entry polynomial far from the anchor: binomial(m−1, k)·eqs^(m−1−k) / (m−1)!
        let far_entry: Vec<f64> = (0..m)
            .map(|k| {
                binomial(m - 1, k) * int_power(self.eqs as f64, m - 1 - k) / FACTORIALS[m - 1]
            })
            .collect();
        // Taylor shift by one cell: shift[k·m + q] = binomial(q, k) for q ≥ k (zero below, unused)
        let shift: Vec<f64> = (0..m * m)
            .map(|i| {
                let (k, q) = (i / m, i % m);
                if q >= k { binomial(q, k) } else { 0.0 }
            })
            .collect();

        let mut out = vec![0.0_f64; positions * moments_in * m];
        // entries in increasing order, so the neighbour one step back along the axis is done
        for p in 0..positions {
            // position of this entry along the axis
            let l = (p / stride) % len;
            // the exact entry polynomial near the anchor, the far one beyond it
            let entry: &[f64] = if l < self.anchor.rows {
                &self.anchor.entries[l * m..(l + 1) * m]
            } else {
                &far_entry
            };
            for q_in in 0..moments_in {
                let value = input[p * moments_in + q_in];
                // where this entry's m output values go, and where its neighbour's are
                let row = (p * moments_in + q_in) * m;
                let previous = if l > 0 { Some(((p - stride) * moments_in + q_in) * m) } else { None };
                for k in 0..m {
                    let mut acc = entry[k] * value;
                    if let Some(previous) = previous {
                        // Taylor shift of the neighbour's tail: Σ_{q ≥ k} binomial(q, k)·Λ_q(l − 1)
                        for q in k..m {
                            acc = acc + shift[k * m + q] * out[previous + q];
                        }
                    }
                    out[row + k] = acc;
                }
            }
        }
        out
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

/// binomial(n, k) as f64 for k ≤ n, exact for the small arguments used here
fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
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