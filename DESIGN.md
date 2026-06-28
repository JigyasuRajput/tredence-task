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

_What the engine computes as the gradient of a pruned weight, why that is the
right choice, and how the revival signal is kept separate from the update._

## 3. Where the engine bottlenecks

_The hot paths, and how they would be optimized._

## 4. Serving a self-pruned model at scale

_How this would ship inside a multi-tenant inference service at thousands of
requests per second._
