"""Build the small MLP used for the digits task."""

from nn import Linear, ReLU, Sequential


def build_mlp(rng, in_features=64, hidden=(128,), n_classes=10):
    """He-init ReLU hidden layers, Xavier-init linear head for the softmax."""
    layers = []
    prev = in_features
    for width in hidden:
        layers.append(Linear(prev, width, rng=rng, init="he"))
        layers.append(ReLU())
        prev = width
    layers.append(Linear(prev, n_classes, rng=rng, init="xavier"))
    return Sequential(*layers)
