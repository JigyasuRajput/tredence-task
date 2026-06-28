"""Forward values and finite-difference gradients for the elementwise ops."""

import numpy as np

from engine import Tensor


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


def check_binary(op_tensor, op_numpy, a_shape, b_shape, seed=0):
    """Drive one binary op's backward closure and compare to finite differences."""
    rng = np.random.default_rng(seed)
    a = Tensor(rng.standard_normal(a_shape), requires_grad=True)
    b = Tensor(rng.standard_normal(b_shape), requires_grad=True)
    # b is kept away from zero so division stays well-conditioned.
    b.data = b.data + 2.0
    upstream = rng.standard_normal(np.broadcast_shapes(a_shape, b_shape))

    out = op_tensor(a, b)
    assert np.allclose(out.data, op_numpy(a.data, b.data))

    out.grad = upstream
    out._backward()

    ga = numeric_grad(lambda av: np.sum(upstream * op_numpy(av, b.data)), a.data.copy())
    gb = numeric_grad(lambda bv: np.sum(upstream * op_numpy(a.data, bv)), b.data.copy())
    assert np.allclose(a.grad, ga, atol=1e-6), "grad wrt a mismatch"
    assert np.allclose(b.grad, gb, atol=1e-6), "grad wrt b mismatch"


OPS = [
    (lambda a, b: a + b, lambda a, b: a + b),
    (lambda a, b: a - b, lambda a, b: a - b),
    (lambda a, b: a * b, lambda a, b: a * b),
    (lambda a, b: a / b, lambda a, b: a / b),
]


def test_same_shape_grads():
    for op_t, op_np in OPS:
        check_binary(op_t, op_np, (4, 3), (4, 3))


def test_broadcast_row_vector():
    # (N, D) op (D,) — the bias-style broadcast; grad to the vector must sum over N.
    for op_t, op_np in OPS:
        check_binary(op_t, op_np, (4, 3), (3,))


def test_broadcast_column_and_scalar():
    for op_t, op_np in OPS:
        check_binary(op_t, op_np, (4, 3), (4, 1))
        check_binary(op_t, op_np, (4, 3), ())


def test_broadcast_grad_keeps_operand_shape():
    a = Tensor(np.ones((5, 2)), requires_grad=True)
    b = Tensor(np.ones(2), requires_grad=True)
    out = a + b
    out.grad = np.ones((5, 2))
    out._backward()
    assert b.grad.shape == (2,)
    # Each of b's entries was added into all 5 rows, so the summed grad is 5.
    assert np.allclose(b.grad, 5.0)


def test_negation_and_scalar_reflected_ops():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert np.allclose((-x).data, [-1.0, -2.0, -3.0])
    assert np.allclose((2.0 - x).data, [1.0, 0.0, -1.0])
    assert np.allclose((6.0 / x).data, [6.0, 3.0, 2.0])
    assert np.allclose((2.0 + x).data, [3.0, 4.0, 5.0])
    assert np.allclose((2.0 * x).data, [2.0, 4.0, 6.0])
