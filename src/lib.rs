// The Rust core. For now: one function, evaluating a polynomial at many points.

// `use` imports names from other crates (Rust's word for packages), like Julia's `using`.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Evaluate the polynomial with coefficients `coefs` (coefs[k] multiplies x^k) at `x`,
/// by Horner's scheme: start from the highest coefficient, then repeatedly multiply by x
/// and add the next lower coefficient.
///
/// `&[f64]` is a *slice*: a borrowed view of a contiguous run of f64 values, without copying.
fn horner(coefs: &[f64], x: f64) -> f64 {
    // .iter() walks the coefficients, .rev() from the highest down, and .fold(start, step)
    // carries an accumulator along: acc ← acc·x + c. The last expression is the return value.
    coefs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Python-visible function: evaluate the polynomial at every point of the NumPy array `x`.
///
/// `#[pyfunction]` is a *macro attribute*: it generates the glue that makes this function
/// callable from Python, including converting the arguments and the result.
///
/// `'py` is a *lifetime*: it tells the compiler that these arrays are only valid while we hold
/// Python's interpreter lock (`py`). The compiler checks this, so we can't misuse them.
#[pyfunction]
fn evaluate_polynomial<'py>(
    py: Python<'py>,
    coefs: PyReadonlyArray1<'py, f64>,   // a read-only 1D NumPy array of f64
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // as_slice() fails if the array isn't contiguous in memory; `?` then returns that error
    // to Python as an exception, instead of crashing
    let coefs = coefs.as_slice()?;
    // Evaluate at every point, collecting the results into a new Rust vector (Vec<f64>)
    let values: Vec<f64> = x.as_array().iter().map(|&xi| horner(coefs, xi)).collect();
    // Hand the vector to Python as a new NumPy array (no copy: NumPy takes ownership)
    Ok(values.into_pyarray(py))
}

/// The module definition: what Python sees when it imports convinterp._core.
/// The function name must match the last part of `module-name` in pyproject.toml.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_polynomial, m)?)?;
    Ok(())
}