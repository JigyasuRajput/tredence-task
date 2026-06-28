# DESIGN

Design notes and the reasoning the live defense will probe. Sections are filled
in as the corresponding parts land.

## 1. The importance criterion

_How a connection earns the right to survive, and why the criterion approximates
the loss increase from removing it._

## 2. The gradient of a masked weight

_What the engine computes as the gradient of a pruned weight, why that is the
right choice, and how the revival signal is kept separate from the update._

## 3. Where the engine bottlenecks

_The hot paths, and how they would be optimized._

## 4. Serving a self-pruned model at scale

_How this would ship inside a multi-tenant inference service at thousands of
requests per second._
