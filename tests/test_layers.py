"""Composable layers: shapes, parameter collection, and backprop through a stack."""

import numpy as np

from engine import Tensor, softmax_cross_entropy
from nn import Linear, Parameter, ReLU, Sequential, Tanh


def test_linear_forward_shape():
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    out = layer(Tensor(np.random.default_rng(1).standard_normal((5, 4))))
    assert out.shape == (5, 3)


def test_linear_exposes_weight_and_bias():
    params = Linear(4, 3, rng=np.random.default_rng(0)).parameters()
    assert len(params) == 2
    assert all(isinstance(p, Parameter) for p in params)
    assert sorted(p.shape for p in params) == [(3,), (4, 3)]


def test_sequential_forward_and_param_collection():
    rng = np.random.default_rng(1)
    net = Sequential(Linear(4, 8, rng=rng), ReLU(), Linear(8, 3, rng=rng))
    out = net(Tensor(rng.standard_normal((6, 4))))
    assert out.shape == (6, 3)
    assert len(net.parameters()) == 4  # two linears, activations hold none


def test_backprop_reaches_every_parameter():
    rng = np.random.default_rng(2)
    net = Sequential(Linear(4, 8, rng=rng), ReLU(), Linear(8, 3, rng=rng))
    x = Tensor(rng.standard_normal((6, 4)))
    softmax_cross_entropy(net(x), rng.integers(0, 3, size=6)).backward()
    for p in net.parameters():
        assert p.grad.shape == p.shape
        assert np.any(p.grad != 0.0)


def test_activation_modules_match_the_tensor_ops():
    x = Tensor([-1.0, 0.5, 2.0])
    assert np.allclose(ReLU()(x).data, x.relu().data)
    assert np.allclose(Tanh()(x).data, x.tanh().data)


def test_masked_weight_in_a_layer_zeros_its_gradient():
    rng = np.random.default_rng(3)
    layer = Linear(4, 3, rng=rng)
    layer.weight.mask[0, 0] = 0.0
    layer.weight.apply_mask()
    x = Tensor(rng.standard_normal((5, 4)))
    softmax_cross_entropy(layer(x), rng.integers(0, 3, size=5)).backward()
    assert layer.weight.grad[0, 0] == 0.0
