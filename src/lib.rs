// The Rust core of convinterp.

// NEW: the generated kernel tables (src/kernels.rs) become a module of this crate
mod kernels;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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

// NEW ------------------------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------------------------

/// The module definition: what Python sees when it imports convinterp._core.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_weights, m)?)?; // NEW
    Ok(())
}