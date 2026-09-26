//! Coefficients of an interpolant: the data values, extended by ghost values beyond each
//! boundary, as ConvolutionInterpolations.jl does it (create_convolutional_coefs), in 1D and N-D.

use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMut1, Axis, IxDyn, Slice};

use crate::kernels::{self, GhostMatrix};

/// Boundary condition at one end of the data.
/// `#[derive(...)]` asks the compiler to generate standard behaviour: copying, printing, comparing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    /// Polynomial extrapolation, of the degree the kernel reproduces
    Poly,
    /// Linear extrapolation
    Linear,
    /// Quadratic extrapolation
    Quadratic,
    /// Polynomial if the data near the boundary allow it, otherwise linear
    Detect,
}

impl Boundary {
    /// Parse a boundary condition name: "poly", "linear", "quadratic" or "detect".
    /// Returns an error message for anything else.
    pub fn parse(name: &str) -> Result<Boundary, String> {
        match name {
            "poly" => Ok(Boundary::Poly),
            "linear" => Ok(Boundary::Linear),
            "quadratic" => Ok(Boundary::Quadratic),
            "detect" => Ok(Boundary::Detect),
            _ => Err(format!(
                "unknown boundary condition {name:?} (expected poly, linear, quadratic or detect)"
            )),
        }
    }
}

/// Stencil half-width `eqs` of a kernel: each point is weighted with eqs coefficients on each side
pub fn stencil_half_width(kernel: &str) -> Result<usize, String> {
    if kernel == "a0" {
        return Ok(1); // nearest neighbour: no column table, one coefficient on each side
    }
    kernels::column_table(kernel, 0)
        .map(|table| table.columns / 2)
        .ok_or_else(|| format!("unknown kernel {kernel:?}"))
}

/// The data values extended by eqs − 1 ghost values beyond each boundary
pub fn extended_coefficients(
    values: &[f64],
    kernel: &str,
    left: Boundary,
    right: Boundary,
) -> Result<Vec<f64>, String> {
    let eqs = stencil_half_width(kernel)?;
    let n = values.len();
    let n_ghost = eqs - 1;

    // n_ghost ghost values, the data, n_ghost ghost values
    let mut c: Vec<f64> = vec![0.0; n + 2 * n_ghost];
    c[n_ghost..n_ghost + n].copy_from_slice(values);
    if n_ghost == 0 {
        return Ok(c); // :a0 and :a1 need no ghost values
    }

    let poly = kernels::poly_ghost_matrix(kernel)
        .ok_or_else(|| format!("no polynomial ghost matrix for kernel {kernel:?}"))?;
    // the vector seen as a 1D array view, so the same line routine serves 1D and N-D
    let mut line = ArrayViewMut1::from(&mut c[..]);
    fill_ghosts(&mut line, n, n_ghost, poly, left, right)?;
    Ok(c)
}

/// The N-D data array extended by eqs − 1 ghost values beyond each boundary of every axis, with
/// boundary conditions `bcs[axis] = (left, right)`. As in Julia, the axes are extended one after
/// the other, each along every line of the array built so far: extending axis 1 also runs along
/// the ghost layers of axis 0, which fills the corners.
pub fn extended_coefficients_nd(
    values: ArrayViewD<'_, f64>,
    kernel: &str,
    bcs: &[(Boundary, Boundary)],
) -> Result<ArrayD<f64>, String> {
    let eqs = stencil_half_width(kernel)?;
    let n_ghost = eqs - 1;
    let data_shape: Vec<usize> = values.shape().to_vec();
    if bcs.len() != data_shape.len() {
        return Err(format!(
            "{} boundary condition pairs given for {}-dimensional data",
            bcs.len(),
            data_shape.len()
        ));
    }

    // an array of zeros with n_ghost extra positions at both ends of every axis
    let shape: Vec<usize> = data_shape.iter().map(|&n| n + 2 * n_ghost).collect();
    let mut c = ArrayD::<f64>::zeros(IxDyn(&shape));
    // the data go in the middle: along every axis, positions n_ghost … n_ghost + n − 1
    c.slice_each_axis_mut(|ax| Slice::from(n_ghost..n_ghost + data_shape[ax.axis.index()]))
        .assign(&values);
    if n_ghost == 0 {
        return Ok(c); // :a0 and :a1 need no ghost values
    }

    let poly = kernels::poly_ghost_matrix(kernel)
        .ok_or_else(|| format!("no polynomial ghost matrix for kernel {kernel:?}"))?;
    for (axis, &(left, right)) in bcs.iter().enumerate() {
        let n = data_shape[axis];
        // `lanes_mut` visits every line of the array along `axis`, as a mutable 1D view
        for mut line in c.lanes_mut(Axis(axis)) {
            fill_ghosts(&mut line, n, n_ghost, poly, left, right)?;
        }
    }
    Ok(c)
}

/// Fill the ghost values at both ends of one line: `line` holds n_ghost ghost positions, the n
/// data values, and n_ghost ghost positions (Julia: apply_boundary_conditions_for_dim!, for one
/// line). Only the ghost positions are written. The :detect test reads the mean-centered values;
/// the ghost values are computed from the values themselves (see `ghost_value`).
fn fill_ghosts(
    line: &mut ArrayViewMut1<'_, f64>,
    n: usize,
    n_ghost: usize,
    poly: &'static GhostMatrix,
    left: Boundary,
    right: Boundary,
) -> Result<(), String> {
    let ns = poly.cols; // data values the polynomial extrapolation uses
    let few_points = n < ns;
    let eqs = n_ghost + 1;
    // number of values next to each boundary that are read: the detect test needs ns + 3
    let m = n.min((ns + 3).max(eqs));

    // ---- left boundary: the first m values, nearest to the boundary first ----
    let raw: Vec<f64> = (0..m).map(|d| line[n_ghost + d]).collect();
    // a mean-centered copy, for the :detect test only
    let mut centered = raw.clone();
    center(&mut centered);
    let g = choose_matrix(poly, left, few_points, &centered, 0, 1);
    // the matrix reads g.cols values next to the boundary; with fewer data values along this
    // line (e.g. :quadratic with 2 values) it cannot be applied, as in Julia
    // (fill_ghost_points_polynomial!)
    if g.cols > m {
        return Err(format!("boundary condition needs {} values along the line, got {m}", g.cols));
    }
    for j in 1..=n_ghost {
        // ghost j lies j positions left of the first data value
        line[n_ghost - j] = ghost_value(g, j, |d| raw[d]);
    }

    // ---- right boundary: the last m values, in increasing order ----
    let raw: Vec<f64> = (0..m).map(|d| line[n_ghost + n - m + d]).collect();
    let mut centered = raw.clone();
    center(&mut centered);
    let g = choose_matrix(poly, right, few_points, &centered, m - 1, -1);
    // the same check for the right boundary, whose condition may differ from the left one
    if g.cols > m {
        return Err(format!("boundary condition needs {} values along the line, got {m}", g.cols));
    }
    for j in 1..=n_ghost {
        // ghost j lies j positions right of the last data value; nearest data value first
        line[n_ghost + n - 1 + j] = ghost_value(g, j, |d| raw[m - 1 - d]);
    }
    Ok(())
}

/// Subtract the mean from every value (in place), returning the mean
fn center(slice: &mut [f64]) -> f64 {
    let mean = slice.iter().sum::<f64>() / slice.len() as f64;
    for v in slice.iter_mut() {
        *v -= mean;
    }
    mean
}

/// Ghost value j (1-based): row j of the matrix times the values y(0), y(1), …, nearest to the
/// boundary first. `y` is a closure: a small function passed as an argument.
/// The sum is the compensated dot product Dot2 (Ogita, Rump & Oishi 2005): as accurate as if
/// computed in twice the precision and then rounded. The terms are added in the same fixed order
/// as the Julia package (_compensated_row_dot), so both give the same bits.
fn ghost_value(g: &GhostMatrix, j: usize, y: impl Fn(usize) -> f64) -> f64 {
    // the matrix must provide a row for ghost value j (checked in development builds)
    debug_assert!(j <= g.rows, "ghost matrix has {} rows, ghost value {} requested", g.rows, j);
    let row = &g.values[(j - 1) * g.cols..j * g.cols];
    let mut s = 0.0_f64; // running sum of the products
    let mut e = 0.0_f64; // running sum of the rounding errors of the products and additions
    for (d, &weight) in row.iter().enumerate() {
        let (p, ep) = two_prod(weight, y(d)); // product, and its exact rounding error
        let (s_new, es) = two_sum(s, p); // new sum, and its exact rounding error
        s = s_new;
        e += ep + es;
    }
    s + e
}

/// Error-free transformation of a product: a·b = p + e exactly, using one fused multiply-add
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Error-free transformation of a sum: a + b = s + e exactly
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    (s, (a - (s - z)) + (b - z))
}

/// The ghost matrix for one boundary: the kernel's polynomial matrix if the boundary condition
/// asks for it (and :detect accepts it), otherwise the linear or quadratic matrix
fn choose_matrix(
    poly: &'static GhostMatrix,
    bc: Boundary,
    few_points: bool,
    y: &[f64],
    base: usize,
    step: isize,
) -> &'static GhostMatrix {
    let use_poly = !few_points
        && match bc {
            Boundary::Poly => true,
            Boundary::Detect => accept_polynomial(y, base, step, poly.cols),
            _ => false,
        };
    if use_poly {
        poly
    } else {
        match bc {
            Boundary::Quadratic => &kernels::QUADRATIC_GHOST,
            _ => &kernels::LINEAR_GHOST, // :linear, and the fallback of :poly and :detect
        }
    }
}

/// The :detect test: accept polynomial extrapolation of degree k − 1 when the k-th differences of
/// the centered data stay below the local data range (Julia: bc_accept_polynomial)
fn accept_polynomial(y: &[f64], base: usize, step: isize, k: usize) -> bool {
    let n_avail = y.len();
    if n_avail < k + 1 {
        return false;
    }
    let nwin = n_avail.min(k + 3); // three overlapping windows of k + 1 values
    guard_diff(y, base, step, nwin, k, 1.0)
}

/// Every window of k + 1 consecutive values within the first `nwin` (read from `base` in
/// direction `step`) must have |Δ^k| ≤ kappa·range + rounding allowance (Julia: bc_guard_diff)
fn guard_diff(y: &[f64], base: usize, step: isize, nwin: usize, k: usize, kappa: f64) -> bool {
    if nwin < k + 1 {
        return false;
    }
    // the d-th value from the boundary; `as` converts between integer types
    let at = |d: usize| y[(base as isize + step * d as isize) as usize];

    // range and largest magnitude of the values read
    let (mut lo, mut hi, mut amax) = (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64);
    for d in 0..nwin {
        let v = at(d);
        lo = lo.min(v);
        hi = hi.max(v);
        amax = amax.max(v.abs());
    }
    let tol = kappa * (hi - lo) + f64::EPSILON * amax * (1u64 << k.min(30)) as f64;

    // k-th difference weights (−1)^d·C(k, d), e.g. 1, −4, 6, −4, 1 for k = 4
    let weights = binomial_differences(k);
    for start in 0..(nwin - k) {
        let mut s = 0.0_f64;
        for d in 0..=k {
            s = weights[d].mul_add(at(start + k - d), s);
        }
        if s.abs() > tol {
            return false;
        }
    }
    true
}

/// The weights (−1)^d·C(k, d) for d = 0 … k, by the recurrence C(k, d) = C(k, d−1)·(k−d+1)/d
fn binomial_differences(k: usize) -> Vec<f64> {
    let mut w = vec![1.0_f64; k + 1];
    for d in 1..=k {
        w[d] = -w[d - 1] * (k + 1 - d) as f64 / d as f64;
    }
    w
}