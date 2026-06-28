# DESIGN

Design notes and the reasoning the live defense will probe. Sections are filled
in as the corresponding parts land.

## 1. The importance criterion

We prune the connections whose removal increases the loss the least.

**Derivation.** Removing connection `i` sets its weight `w_i → 0`, i.e. a
perturbation `Δw_i = −w_i`. A first-order Taylor expansion of the loss around the
current weights gives

```
L(w + Δw) ≈ L(w) + g_i · Δw_i,    where g_i = ∂L/∂w_i.
```

So the loss change from removing connection `i` is `ΔL_i ≈ g_i · (−w_i) = −w_i g_i`,
and its magnitude is `|ΔL_i| ≈ |w_i · g_i|`. We therefore score each connection by
the **saliency** `s_i = |w_i · g_i|` and prune the smallest scores — the connections
whose removal perturbs the loss least. (`prune/importance.py::saliency`.)

**Why this beats magnitude-only.** Magnitude `|w_i|` assumes every large weight
matters and every small one doesn't, ignoring how sensitive the loss actually is to
that weight. Saliency scales the magnitude by the loss gradient, so a small weight on
a high-sensitivity path can outrank a large weight the loss barely depends on. At a
true minimum `g_i → 0` and the first-order term vanishes (magnitude would then
dominate), but during training `g_i` is informative — which is exactly when we prune.

**Practical notes.** We take the absolute value because we care about the size of the
loss change, not its sign. The gradient `g_i` is estimated from gradients accumulated
over a full pass rather than one noisy minibatch, so the score is stable. Ranking is
**global** across layers (`prune/pruner.py`), and the budget is reached gradually via
a cubic ramp (`prune/schedule.py`) so the network can recover between cuts.

## 2. The gradient of a masked weight

The forward pass uses the effective weight `eff = W ⊙ mask` (`Parameter.masked`).
Backprop therefore gives

```
∂L/∂W = (∂L/∂eff) ⊙ mask.
```

So for a **live** connection (mask = 1) the gradient is the ordinary gradient, and
for a **dead** connection (mask = 0) it is **exactly zero**. That is the right answer
for the *update*: a pruned connection did not participate in the forward pass, so it
should receive no gradient and the optimizer must not move it. This is what
`weight.grad` holds, and it is the only thing the optimizer reads.

**Two different gradients, kept separate.** There is a second, distinct quantity:

1. **Update gradient** `∂L/∂W = (∂L/∂eff) ⊙ mask` — zero on dead connections. Drives
   the optimizer. (`weight.grad`)
2. **Would-be-dense gradient** `∂L/∂eff` — the gradient a connection *would* have
   received if it were active, evaluated on the current sparse activations. Non-zero on
   dead connections. Used **only** to decide which dead connections to revive.
   (`weight.dense_grad()`, read from the `masked()` product's grad.)

The optimizer never reads `dense_grad()`, so the revival signal can never move a pruned
weight — dead weights stay frozen, and the would-be-dense gradient only informs
*topology* changes (regrowth), never the *update*.

**Optimizer state on revival.** Each optimizer step re-asserts the mask and multiplies
the moment buffers by it (`reset_dead_state` does the same on demand after regrowth), so
dead connections always carry zero momentum / zero Adam moments. A revived connection
therefore starts from weight 0 **and** zero moments — a clean start. This is what avoids
the classic failure mode: a stale, tiny second moment left in Adam's denominator would
produce an enormous first step and destabilise training. We reuse Adam's **global**
timestep `t` for bias correction rather than a per-connection local counter; this is
simpler and standard, with the understood caveat that a revived connection's effective
bias correction reflects the global step rather than a fresh `t = 1`. The resulting first
step is bounded (we verify this in the tests), so the small difference is an acceptable
trade for not carrying a per-connection step count.

## 3. Where the engine bottlenecks

The engine is built for correctness and clarity, not throughput. The hot paths, and
how I'd fix each:

- **Python-per-op overhead.** Every operation allocates a new `Tensor`, builds a
  closure, and stores parent tuples; `backward()` then walks the graph in a Python
  loop calling those closures one at a time. For a tiny MLP this is invisible, but it
  dominates as the graph grows. *Fix:* fuse elementwise chains, reuse pre-allocated
  gradient buffers instead of allocating per step, and lower the hot kernels to a
  compiled path (Numba/Cython) once correctness is locked.
- **Re-deriving the graph every step.** `backward()` rebuilds the topological order
  by recursion on each call. For a fixed architecture the order never changes. *Fix:*
  build the topo order once and cache it; switch the recursion to an explicit stack to
  avoid Python's recursion limit on deep graphs.
- **float64 everywhere.** Required for honest gradient checks, but it doubles memory
  traffic versus float32, and these ops are memory-bound, not compute-bound. *Fix:*
  train in float32 (or bfloat16) and keep float64 only for the gradient-check suite.
- **The mask multiply runs even when nothing is pruned.** `masked()` always inserts a
  full `W ⊙ mask` op and node. *Fix:* skip it when the mask is all ones, or fold the
  mask into the weight buffer and apply the gradient correction directly.
- **The sparse forward's scatter.** `train/cost.py` uses `np.add.at`, a slow Python
  scatter. It proves the FLOP reduction is real but is not how you'd serve. *Fix:* a
  CSR/CSC sparse matmul (a compiled sparse kernel), or structured sparsity so dense
  BLAS stays efficient (see §4).

The honest one-liner: at this scale the measurable cost is Python op dispatch and
memory traffic, not arithmetic — so the wins come from fewer/fused allocations and a
lower-precision, compiled hot path, not from a cleverer matmul.

## 4. Serving a self-pruned model at scale

The platform cares about cost per served model, and pruning's payoff there is real but
needs honesty about *where* it shows up.

**Export, don't ship the trainer.** Freeze the model to an inference artifact: the
architecture, biases, and each layer's weights in a sparse format (CSR/CSC) plus the
mask, versioned. No optimizer, no autograd graph — the mask is frozen, so there is no
drift to worry about. Validate the artifact by checking the sparse forward matches the
dense forward bit-for-bit on a fixed batch (we already do exactly this in the tests).

**Be honest about wall-clock.** Unstructured 90% sparsity does **not** speed up a dense
GPU/BLAS kernel — multiplying by zeros costs the same. To turn sparsity into latency you
need either (a) a genuine sparse kernel (cuSPARSE / sparse BLAS), which wins only at high
sparsity, or (b) **structured** sparsity — 2:4 semi-structured on recent NVIDIA, or
block sparsity — which keeps dense kernels efficient. Absent that, the realized win is
**memory and density**, not per-request latency. I'd state that to stakeholders rather
than quote a FLOP count as if it were a speedup.

**Where pruning actually pays in multi-tenant serving:**
- **Density / cost.** Smaller resident footprint (sparse storage + int8 quantization on
  top) means more models per host — directly the "four-times-smaller" lever the platform
  cares about. More models stay hot in memory, cutting cold starts.
- **Routing & affinity.** A router maps `(tenant, model, version)` to a replica that
  already has those weights loaded, via consistent hashing, to maximize cache hits; cold
  models lazy-load into an LRU weight cache.
- **Batching.** Dynamic-batch requests *per model* to amortize kernel launches; keep
  latency-sensitive and bulk traffic in separate pools with their own SLOs.
- **Isolation & fairness.** Per-tenant quotas, rate limits, and priority so one tenant
  can't starve others; bin-pack models onto hosts by memory and QPS.
- **Safety & observability.** Canary a new pruned version against the dense baseline,
  watch per-tenant latency and accuracy, and roll back on regression. Autoscale on
  latency/QPS SLOs.

So the pitch I'd make: pruning buys **model density and memory cost** today (reliable,
measurable), and buys **latency** only once paired with structured sparsity or a real
sparse kernel — and I'd ship it with the export-time correctness check that the sparse
path equals the dense one.
