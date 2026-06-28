"""Composable layers: a base module, a linear layer, activations, and a sequential container."""

import numpy as np

from .init import init_weight
from .parameter import Parameter


class Module:
    """Base class: callable, and knows how to gather its trainable parameters."""

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        """Collect Parameters held directly or inside child modules / lists of them."""
        found = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                found.append(value)
            elif isinstance(value, Module):
                found.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        found.append(item)
                    elif isinstance(item, Module):
                        found.extend(item.parameters())
        return found


class Linear(Module):
    """Affine layer y = x @ W + b; the weight runs through its mask so pruning is real."""

    def __init__(self, in_features, out_features, rng=None, init="he"):
        rng = rng if rng is not None else np.random.default_rng()
        self.weight = Parameter(init_weight(in_features, out_features, rng, init))
        self.bias = Parameter(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.masked() + self.bias


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    """Runs a list of layers front to back."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
