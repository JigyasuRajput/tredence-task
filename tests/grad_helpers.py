"""Per-op test helper that drives a single backward closure in isolation."""

import numpy as np

from engine.gradcheck import numeric_gradient as numeric_grad


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
