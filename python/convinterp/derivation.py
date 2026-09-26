"""Derivation and exact verification of convolution kernels.

A convolution kernel k(s) here is an even, piecewise polynomial function: on each unit interval
[i, i + 1], i = 0, …, M − 1, it is a polynomial of degree p, and it is zero for |s| ≥ M. The
kernels a3, a4 and b5 to b13 of convinterp are each the unique solution of a linear system of
conditions on these polynomials (see the theory page of the documentation):

1. Interpolation: k(0) = 1 and k(i) = 0 at every other integer i.
2. Symmetry: k is even, so its odd derivatives vanish at s = 0 (up to the continuity order).
3. Continuity: derivatives up to order d are continuous at every knot, and vanish at the end of
   the support, s = M.
4. Polynomial reproduction: Σ_j k(t − j) j^m = t^m for all t and m = 0, …, R, so that
   interpolating samples of a polynomial of degree ≤ R returns that polynomial exactly.

`derive_kernel` solves this system in exact rational arithmetic (it needs SymPy:
``pip install convinterp[derivation]``). `kernel_properties` checks a kernel's continuity and
polynomial reproduction exactly, and `kernel_function` evaluates a kernel for plotting; these two
need only the Python standard library and NumPy.

A kernel is given as a tuple of M tuples: row i holds the coefficients of the polynomial on
[i, i + 1] in the variable s, lowest power first, as fractions.Fraction.
"""

from fractions import Fraction
from math import comb

import numpy as np

# The settings that produce convinterp's kernels as unique solutions:
# (pieces M, degree p, continuity d, reproduction degree R)
KERNEL_SETTINGS = {
    "a3": (2, 3, 1, 2),     # Keys (1981), the classic cubic convolution kernel
    "a4": (3, 3, 1, 3),     # Keys' 4th-order cubic kernel
    "b5": (5, 5, 3, 5),
    "b7": (6, 7, 5, 6),
    "b9": (7, 9, 7, 6),
    "b11": (8, 11, 9, 6),
    "b13": (9, 13, 11, 6),
}


def derive_kernel(pieces, degree, continuity, reproduce):
    """Solve the kernel conditions exactly and return the kernel's coefficients.

    Parameters
    ----------
    pieces : int
        Number of unit intervals M on each side of zero; the kernel is zero for |s| ≥ M.
    degree : int
        Polynomial degree p of every piece.
    continuity : int
        Highest derivative order d that is continuous everywhere, including at s = M.
    reproduce : int
        Highest polynomial degree R that the kernel reproduces exactly.

    Returns
    -------
    tuple of tuple of Fraction
        Row i holds the coefficients of the polynomial on [i, i + 1], lowest power first.

    Raises
    ------
    ValueError
        If the conditions contradict each other (no such kernel exists), or if they leave free
        parameters (the kernel is not unique; the message says how many).

    Example
    -------
    >>> from convinterp.derivation import derive_kernel
    >>> keys = derive_kernel(pieces=2, degree=3, continuity=1, reproduce=2)   # Keys' cubic kernel
    >>> [str(c) for c in keys[0]]                                              # 1 − 5/2 s² + 3/2 s³
    ['1', '0', '-5/2', '3/2']
    """
    import sympy as sp                                               # needed only here

    M, p, d, R = pieces, degree, continuity, reproduce
    # the unknown coefficients: A[i, j] multiplies s^j on the piece [i, i + 1]
    A = sp.Matrix(M, p + 1, lambda i, j: sp.Symbol(f"A{i}_{j}"))
    s, t = sp.symbols("s t")
    P = [sum(A[i, j] * s**j for j in range(p + 1)) for i in range(M)]   # the polynomial pieces

    equations = []
    # 1. interpolation: k(0) = 1, and zero at every other integer
    equations.append(P[0].subs(s, 0) - 1)
    for i in range(M):
        if i > 0:
            equations.append(P[i].subs(s, i))                        # left end of piece i
        equations.append(P[i].subs(s, i + 1))                        # right end of piece i
    # 2. and 3. symmetry and continuity of the derivatives of order 1 … d
    for n in range(1, d + 1):
        D = [sp.diff(Pi, s, n) for Pi in P]                          # n-th derivative of each piece
        if n % 2 == 1:
            equations.append(D[0].subs(s, 0))                        # odd derivatives vanish at 0
        for k in range(1, M):
            equations.append(D[k - 1].subs(s, k) - D[k].subs(s, k))  # continuous at knot k
        equations.append(D[M - 1].subs(s, M))                        # vanishes at the support end
    # 4. reproduction: Σ_j k(t − j) j^m = t^m for t in [0, 1), as an identity in t
    for m in range(R + 1):
        total = 0
        for j in range(-(M - 1), M + 1):                             # every j with t − j in (−M, M)
            if j <= 0:
                total += P[-j].subs(s, t - j) * sp.Integer(j) ** m   # t − j in [|j|, |j| + 1)
            else:
                total += P[j - 1].subs(s, j - t) * sp.Integer(j) ** m  # |t − j| in (j − 1, j]
        equations += sp.Poly(sp.expand(total - t**m), t).all_coeffs()  # every power of t must match

    solutions = sp.linsolve(equations, list(A))                      # exact rational linear algebra
    if not solutions:
        raise ValueError("the conditions are contradictory: no such kernel exists")
    (values,) = solutions
    free = set().union(*(sp.sympify(v).free_symbols for v in values))
    if free:
        raise ValueError(f"the conditions leave {len(free)} free parameter(s): the kernel is not unique")
    return tuple(tuple(Fraction(int(sp.numer(values[i * (p + 1) + j])),
                                int(sp.denom(values[i * (p + 1) + j])))
                       for j in range(p + 1))
                 for i in range(M))


def _polynomial_value(coefs, x):
    # value of Σ_j coefs[j] x^j, exactly for fractions
    return sum(c * x**j for j, c in enumerate(coefs))


def _polynomial_derivative(coefs, n):
    # coefficients of the n-th derivative of Σ_j coefs[j] x^j
    for _ in range(n):
        coefs = [j * coefs[j] for j in range(1, len(coefs))]
    return coefs


def _kernel_value(kernel, x):
    # exact kernel value at a rational x
    x = abs(x)
    i = int(x)                                                       # the piece containing |x|
    return _polynomial_value(kernel[i], x) if i < len(kernel) else Fraction(0)


def kernel_properties(kernel):
    """The continuity and reproduction degree of a kernel, computed exactly.

    Parameters
    ----------
    kernel : sequence of sequences of numbers convertible to Fraction
        Row i holds the coefficients of the polynomial on [i, i + 1], lowest power first.

    Returns
    -------
    dict
        ``"pieces"``, ``"degree"``, ``"continuity"`` (the highest derivative order that is
        continuous everywhere; −1 if the kernel itself is discontinuous) and ``"reproduces"``
        (the highest polynomial degree reproduced exactly; −1 if not even constants).
    """
    P = [[Fraction(c) for c in row] for row in kernel]               # exact coefficients
    M, p = len(P), len(P[0]) - 1

    continuity = -1
    for n in range(p + 1):
        D = [_polynomial_derivative(row, n) for row in P]
        ok = not (n % 2 == 1 and _polynomial_value(D[0], 0) != 0)   # even kernel: odd derivatives vanish at 0
        ok = ok and all(_polynomial_value(D[k - 1], k) == _polynomial_value(D[k], k) for k in range(1, M))
        ok = ok and _polynomial_value(D[M - 1], M) == 0              # zero beyond the support
        if not ok:
            break
        continuity = n

    # reproduction of degree m: Σ_j k(t − j) j^m = t^m. Both sides are polynomials in t of degree
    # at most max(p, m) on [0, 1), so checking that many + 1 distinct points proves the identity.
    reproduces = -1
    for m in range(2 * p + 2):
        points = [Fraction(q, max(p, m) + 2) for q in range(max(p, m) + 1)]   # distinct t in [0, 1)
        if all(sum(_kernel_value(P, t - j) * Fraction(j) ** m for j in range(-M, M + 2)) == t**m
               for t in points):
            reproduces = m
        else:
            break
    return {"pieces": M, "degree": p, "continuity": continuity, "reproduces": reproduces}


def kernel_function(kernel):
    """A NumPy function evaluating the kernel in floating point, for plotting and comparisons.

    Each piece is first rewritten exactly in the local coordinate u = |s| − i on [0, 1), so the
    floating-point evaluation avoids the cancellation that large powers of s would cause.
    """
    rows = []
    for i, row in enumerate(kernel):
        coefs = [Fraction(c) for c in row]                           # the piece in powers of s
        local = [Fraction(0)] * len(coefs)                           # the same piece in powers of u
        for j, c in enumerate(coefs):                                # s^j = (u + i)^j, expanded exactly
            for k in range(j + 1):
                local[k] += c * comb(j, k) * Fraction(i) ** (j - k)
        rows.append(np.array([float(c) for c in local]))             # rounded once, at the end

    def k(s):
        s = np.abs(np.asarray(s, dtype=np.float64))                  # the kernel is even
        out = np.zeros_like(s)
        for i, coefs in enumerate(rows):
            on_piece = (s >= i) & (s < i + 1)                        # points on piece i
            out[on_piece] = np.polynomial.polynomial.polyval(s[on_piece] - i, coefs)
        return out if out.ndim else float(out)

    return k