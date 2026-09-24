"""convinterp: convolution interpolation with exact kernels, a Python interface to a Rust core."""

# Re-export the compiled function, so users write
#     from convinterp import evaluate_polynomial
# instead of reaching into the private compiled module _core
from ._core import evaluate_polynomial

# The names `from convinterp import *` brings in: the public interface
__all__ = ["evaluate_polynomial"]