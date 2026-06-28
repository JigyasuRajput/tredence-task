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

## The self-pruning loop

How `train_pruned` actually runs: the masked forward `x @ (W*M)`, the two distinct
gradients (the update grad that is zero on dead connections vs the would-be-dense
revival signal), and exactly where masks flip and dead optimizer state is zeroed.

```mermaid
flowchart TD
    A([train_pruned: for each epoch]) --> B[for each minibatch]

    subgraph TRAIN["Per minibatch"]
        direction TB
        B --> C["Masked forward<br/>logits = x @ (W * M) + b<br/>(Parameter.masked builds W*M inside the graph)"]
        C --> D["loss = softmax_cross_entropy(logits, y)"]
        D --> E["loss.backward()"]
        E --> F["UPDATE grad: w.grad = dL/d(W*M) * M<br/>exactly 0 on dead connections, used by the optimizer"]
        E -. also computed .-> G["REVIVAL signal: dense_grad = dL/d(W*M)<br/>nonzero on dead; NOT read by train_pruned"]
        F --> H["optimizer.step()"]
        H --> H1["1: update moments from w.grad, then w.data -= lr * update"]
        H1 --> H2["2: apply_mask -> w.data *= M   (MASK RE-ASSERTED)"]
        H2 --> H3["3: m,v *= M (Adam) / velocity *= M (SGD)   (DEAD OPTIMIZER STATE ZEROED)"]
    end

    H3 --> I{more minibatches?}
    I -- yes --> B
    I -- no --> J{epoch < prune_window?}

    subgraph PRUNE["End of epoch, only inside the pruning window"]
        direction TB
        J -- yes --> K["_full_batch_grads: one fwd+back over ALL train data<br/>refreshes w.grad as a stable saliency signal"]
        K --> L["target = cubic_sparsity(epoch+1, prune_window, final_sparsity)"]
        L --> M["pruner.prune_to(target, saliency)"]
        M --> N["saliency = |w.data * w.grad|, global rank, keep the n largest"]
        N --> O["set new w.mask, then apply_mask zeros newly-dead weights   (MASKS FLIP)"]
    end

    J -- no --> P
    O --> P["record loss / test_acc / sparsity"]
    P --> A

    G -. regrowth path .-> R["(bonus, NOT in train_pruned)<br/>prune_and_grow grows dead conns by |dense_grad|;<br/>reset_dead_state -> revived weight and moments start at 0"]

    classDef flip fill:#ffe6cc,stroke:#d79b00,color:#000;
    classDef grad fill:#dae8fc,stroke:#6c8ebf,color:#000;
    classDef sep fill:#f5f5f5,stroke:#999999,stroke-dasharray:4 3,color:#000;
    class H2,H3,O flip;
    class F,G grad;
    class R sep;
```

## Install

```bash
uv sync          # preferred
# or
pip install -r requirements.txt
```

## Run

```bash
uv run pytest                              # the whole test suite
uv run pytest tests/test_gradcheck.py      # gradient-check every op vs finite differences
uv run pytest tests/test_masked_weight.py  # masked-weight correctness (pruned weights stay zero)
uv run python -m engine.gradcheck          # per-op finite-difference report
uv run python -m train.train_mlp           # train the MLP, save the learning curve
uv run python -m train.prune_run           # self-prune to a target sparsity
uv run python -m train.pareto_run          # multi-seed sparsity sweep + Pareto plot
```

Every run fixes its seeds, so the committed numbers in `results/` reproduce from a
clean clone.

## Dataset

sklearn's `load_digits` — 1,797 8×8 handwritten digits, 10 classes. Small and real:
it trains in seconds, so a multi-seed sparsity sweep is cheap to reproduce, yet the
~9.5k-weight MLP is large enough that 90% pruning is meaningful. sklearn hands over
the raw arrays only; the shuffle, split, and standardization are NumPy.

## Results

Digits, 64-128-10 MLP, mean ± std test accuracy over 5 seeds (`results/part4_pareto.json`):

| target sparsity | saliency | magnitude |
|----------------:|:--------:|:---------:|
| 0%  | 0.972 ± 0.009 | 0.972 ± 0.009 |
| 50% | 0.977 ± 0.006 | 0.973 ± 0.009 |
| 75% | 0.978 ± 0.005 | 0.972 ± 0.005 |
| 90% | 0.973 ± 0.002 | 0.976 ± 0.004 |
| 95% | 0.962 ± 0.007 | 0.972 ± 0.005 |

**Falsifiable claim.** Saliency (|w·∂L/∂w|) pruning beats magnitude pruning at
moderate sparsity (50–75%, ≈ +0.5 accuracy points) but the two **cross over**:
at 95% sparsity magnitude wins by 0.95 points (paired over 5 seeds; standard error
0.29 points, so the gap is over 2 standard errors and not noise). So the
gradient-based criterion helps until the budget gets tight, where the magnitude
baseline is the stronger choice on this task. Re-running `train.pareto_run`
reproduces these numbers exactly.

The cost is genuine, not dense-times-zero: at 90% sparsity the model keeps 947 of
9,472 weights and the weight matmuls do 10× fewer multiply-adds — `active_params`,
`total_params`, and `mac_reduction` committed in `results/part3_pruning.json` — with a
sparse-aware forward (`train/cost.py`) that matches the dense output. This is a
FLOP / active-parameter reduction, **not** a wall-clock speedup at this scale: a NumPy
scatter over live connections does fewer operations but does not beat optimized dense
BLAS — realizing latency gains needs structured sparsity or a real sparse kernel (see
DESIGN §4).
