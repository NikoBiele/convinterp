// The Rust core of convinterp.

// The generated kernel tables (src/kernels.rs) and the coefficient construction
// (src/coefficients.rs) become modules of this crate
mod coefficients;
mod integral;
mod interpolant;
mod interpolant_nd;
mod kernels;

use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Evaluate the polynomial with coefficients `coefs` (coefs[k] multiplies x^k) at `x`,
/// by Horner's scheme.
fn horner(coefs: &[f64], x: f64) -> f64 {
    coefs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Python-visible: evaluate the polynomial at every point of the NumPy array `x`.
#[pyfunction]
fn evaluate_polynomial<'py>(
    py: Python<'py>,
    coefs: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let coefs = coefs.as_slice()?;
    let values: Vec<f64> = x.as_array().iter().map(|&xi| horner(coefs, xi)).collect();
    Ok(values.into_pyarray(py))
}

/// Weights of all columns of a kernel table at `tau`: Horner's scheme over the rows, for all
/// columns at once (w ← w·tau + row), exactly as the Julia package evaluates them.
fn column_weights(table: &kernels::ColumnTable, tau: f64) -> Vec<f64> {
    let k = table.columns;
    // one accumulator per column, starting at zero; the type annotation tells Rust these are f64
    let mut w: Vec<f64> = vec![0.0; k];
    // `chunks_exact(k)` splits the flat coefficient list into rows of k coefficients each
    for row in table.rows.chunks_exact(k) {
        // walk the accumulators and the row's coefficients side by side
        for (wc, &rc) in w.iter_mut().zip(row) {
            // w ← w·tau + row, as one fused multiply-add like Julia's muladd;
            // `*wc` is the accumulator that the mutable reference `wc` points to
            *wc = (*wc).mul_add(tau, rc);
        }
    }
    w
}

/// Python-visible (for testing): the weights of all stencil columns of `kernel` (e.g. "b5")
/// for derivative order `order`, at `tau` = 1 − t.
#[pyfunction]
fn kernel_weights(kernel: &str, order: i32, tau: f64) -> PyResult<Vec<f64>> {
    // Look up the table; `ok_or_else` turns "not found" into a Python ValueError
    let table = kernels::column_table(kernel, order).ok_or_else(|| {
        PyValueError::new_err(format!("no kernel table for :{kernel} with order {order}"))
    })?;
    Ok(column_weights(table, tau))
}

/// Python-visible (for testing): the data values extended by ghost values beyond each boundary,
/// for `kernel` and the boundary conditions `bc_left` and `bc_right` ("poly", "linear",
/// "quadratic" or "detect").
#[pyfunction]
fn extended_coefficients<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    kernel: &str,
    bc_left: &str,
    bc_right: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Parse the boundary conditions; an unknown name becomes a Python ValueError
    let left = coefficients::Boundary::parse(bc_left).map_err(|e| PyValueError::new_err(e))?;
    let right = coefficients::Boundary::parse(bc_right).map_err(|e| PyValueError::new_err(e))?;
    let c = coefficients::extended_coefficients(values.as_slice()?, kernel, left, right)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(c.into_pyarray(py))
}

/// Python-visible (for testing): the N-D data array extended by ghost values beyond each
/// boundary of every axis, for `kernel` and one (left, right) pair of boundary condition names
/// per axis, e.g. [("poly", "poly"), ("detect", "linear")] for 2D data.
#[pyfunction]
fn extended_coefficients_nd<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<'py, f64>,
    kernel: &str,
    bcs: Vec<(String, String)>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    // Parse every pair; `collect` into a Result stops at the first unknown name
    let bcs = bcs
        .iter()
        .map(|(l, r)| Ok((coefficients::Boundary::parse(l)?, coefficients::Boundary::parse(r)?)))
        .collect::<Result<Vec<_>, String>>()
        .map_err(|e| PyValueError::new_err(e))?;
    // `as_array` gives a view of the NumPy array (any number of dimensions), without copying
    let c = coefficients::extended_coefficients_nd(values.as_array(), kernel, &bcs)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(c.into_pyarray(py))
}

/// The module definition: what Python sees when it imports convinterp._core.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_weights, m)?)?;
    m.add_function(wrap_pyfunction!(extended_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(extended_coefficients_nd, m)?)?;
    m.add_class::<interpolant::Interpolant1D>()?;
    m.add_class::<interpolant_nd::InterpolantND>()?;
    Ok(())
}