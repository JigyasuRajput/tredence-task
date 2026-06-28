"""Finite-difference gradient checking: float64, central differences, honest tolerances.

Run `python -m engine.gradcheck` for a readable per-op report.
"""

import sys

import numpy as np

from .losses import softmax_cross_entropy
from .tensor import Tensor

# Central differences in float64 land near 1e-9 relative error for these ops, so
# 1e-6 is a tight, honest bar — a failure here means the op is wrong, not the check.
DEFAULT_EPS = 1e-6
DEFAULT_TOL = 1e-6


def numeric_gradient(f, x, eps=DEFAULT_EPS):
    """Central-difference gradient of scalar function f at array x."""
    x = np.array(x, dtype=np.float64)  # mutable ndarray, even for 0-d
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f(x)
        x[idx] = orig - eps
        minus = f(x)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    """Max relative error between two arrays, stable where both are near zero."""
    a, b = np.asarray(a), np.asarray(b)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def grad_check(build, params, eps=DEFAULT_EPS):
    """Compare analytic grads (via backward) to finite differences through the full graph.

    `build(*params)` returns a scalar-loss Tensor. Returns the worst relative error.
    """
    build(*params).backward()
    worst = 0.0
    for p in params:
        analytic = p.grad.copy()

        def scalar_of(x, p=p):
            saved = p.data
            p.data = x
            value = float(build(*params).data)
            p.data = saved
            return value

        numeric = numeric_gradient(scalar_of, p.data.copy(), eps)
        worst = max(worst, relative_error(analytic, numeric))
    return worst


def standard_cases():
    """Every op as a scalar objective, shared by the test suite and the CLI report."""

    def p(shape, seed):
        return Tensor(np.random.default_rng(seed).standard_normal(shape), requires_grad=True)

    def obj(out, seed):
        # A fixed random covector weights each output unevenly, so a bug in a
        # reduction axis or a broadcast can't hide behind a uniform upstream.
        c = np.random.default_rng(seed).standard_normal(out.shape)
        return (out * Tensor(c)).sum()

    div_b = p((3,), 17)
    div_b.data = div_b.data + 3.0  # keep the denominator away from zero
    targets = np.array([0, 2, 1, 0, 3])

    return [
        ("add (broadcast)", lambda a, b: obj(a + b, 1), [p((4, 3), 10), p((3,), 11)]),
        ("sub (broadcast)", lambda a, b: obj(a - b, 2), [p((4, 3), 12), p((4, 1), 13)]),
        ("mul (broadcast)", lambda a, b: obj(a * b, 3), [p((4, 3), 14), p((3,), 15)]),
        ("div (broadcast)", lambda a, b: obj(a / b, 4), [p((4, 3), 16), div_b]),
        ("matmul", lambda a, b: obj(a @ b, 5), [p((4, 3), 18), p((3, 5), 19)]),
        ("sum (axis)", lambda a: obj(a.sum(axis=0), 6), [p((4, 3), 20)]),
        ("mean (axis)", lambda a: obj(a.mean(axis=1), 7), [p((4, 3), 21)]),
        ("relu", lambda a: obj(a.relu(), 8), [p((4, 5), 22)]),
        ("tanh", lambda a: obj(a.tanh(), 9), [p((4, 5), 23)]),
        ("softmax cross-entropy", lambda z: softmax_cross_entropy(z, targets), [p((5, 4), 24)]),
    ]


def run_report(eps=DEFAULT_EPS, tol=DEFAULT_TOL):
    """Print each op's worst relative error and whether it passes; return overall ok."""
    print(f"{'operation':24s} {'max rel error':>14s}   result")
    print("-" * 50)
    all_ok = True
    for name, build, params in standard_cases():
        err = grad_check(build, params, eps=eps)
        ok = err < tol
        all_ok = all_ok and ok
        print(f"{name:24s} {err:14.2e}   {'PASS' if ok else 'FAIL'}")
    print("-" * 50)
    print("all ops pass" if all_ok else "SOME OPS FAILED")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_report() else 1)
