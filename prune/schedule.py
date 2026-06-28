"""Gradual pruning schedule: how much sparsity to target at each pruning step."""


def cubic_sparsity(step, total_steps, final_sparsity, initial_sparsity=0.0):
    """Cubic ramp (Zhu & Gupta): prune quickly at first, then taper toward the budget.

    Gradual pruning beats one-shot because the network adapts and recovers between
    cuts; the cubic shape removes the easy, clearly-unimportant connections early and
    slows as it nears the target, where each further cut costs more accuracy.
    """
    if total_steps <= 0:
        return final_sparsity
    t = min(max(step, 0), total_steps) / total_steps
    return final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - t) ** 3
