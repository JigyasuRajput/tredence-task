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

_The hot paths, and how they would be optimized._

## 4. Serving a self-pruned model at scale

_How this would ship inside a multi-tenant inference service at thousands of
requests per second._
