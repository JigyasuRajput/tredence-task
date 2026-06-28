"""Finite-difference helpers shared by the op tests (central differences, float64)."""

import numpy as np


def numeric_grad(f, x, eps=1e-6):
    """Central-difference gradient of scalar f at array x."""
    x = np.array(x, dtype=np.float64)  # ensure a mutable ndarray, even for 0-d
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


def check_single_op(build, inputs, seed=0, atol=1e-6):
    """Compare one op's analytic gradient to finite differences.

    `build(*inputs)` returns the op's output Tensor; inputs must require grad.
    Only valid for a single op — we drive that op's backward closure, not a graph.
    """
    rng = np.random.default_rng(seed)
    out = build(*inputs)
    upstream = rng.standard_normal(out.shape)
    out.grad = np.array(upstream, dtype=np.float64)
    out._backward()
    for t in inputs:
        def scalar_of(x, t=t):
            saved = t.data
            t.data = x
            value = float(np.sum(upstream * build(*inputs).data))
            t.data = saved
            return value

        ng = numeric_grad(scalar_of, t.data.copy())
        assert np.allclose(t.grad, ng, atol=atol), f"grad mismatch for shape {t.shape}"
