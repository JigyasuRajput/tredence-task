# AQUA — The Self-Pruning Network

A small neural-network library written from scratch in pure Python + NumPy, used
to train a classifier that prunes its own connections during training — and
honest evidence that the pruned model is actually cheaper.

No deep-learning frameworks are used in the core: the autodiff engine, backward
pass, optimizer, pruning, and training loop are all hand-written. **scikit-learn
appears only to load a standard dataset — never for models, training, or
gradients.**

## Layout

| Directory | What lives here |
|-----------|-----------------|
| `engine/` | Reverse-mode autodiff: the tensor type, ops, `backward()`. |
| `nn/`     | Layers, the MLP, and the optimizers (SGD + Adam). |
| `prune/`  | Masking, importance criteria, and the pruning schedule. |
| `train/`  | Dataset loading (data only), training loop, and the sweeps. |
| `tests/`  | Gradient checks and the masked-weight correctness test. |
| `utils/`  | Reproducibility helpers. |

## Install

```bash
uv sync          # preferred
# or
pip install -r requirements.txt
```

## Run

```bash
uv run pytest    # the test suite (gradient checks live here)
```

The training, self-pruning, and Pareto-sweep commands are documented alongside
their code as each part lands.
