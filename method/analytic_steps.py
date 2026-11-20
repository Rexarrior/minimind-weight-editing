"""Numerical kernels from the R5 MiniMind2 fit, expressed in NumPy."""

import numpy as np


def fit_conditional_token(design, log_ratio):
    """Fit one token in the first transfer. Column zero is the intercept."""
    count, width = design.shape
    if count < 40:
        return np.zeros(width, dtype=design.dtype)
    penalty = np.full(width, 0.15 * count + 70, dtype=design.dtype)
    penalty[0] = 0.02 * count + 8
    return np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ log_ratio)


def recover_margins(keys, residuals, embedding_diffs, metric, margin):
    """Return an additive (residual width x MLP width) real-valued correction.

    Each row corresponds to one accumulated (teacher-forced position, rival)
    constraint. Pass the updated residual estimated from the decoded candidate
    weights, the replay key, and E[target] - E[rival] for that constraint.
    """
    dual_keys = np.linalg.solve(metric, keys.T).T
    gram = (embedding_diffs @ embedding_diffs.T) * (dual_keys @ keys.T)
    scale = np.sqrt(np.maximum(np.diag(gram), 1e-6))
    current = (residuals * embedding_diffs).sum(axis=1)
    deficits = np.maximum(margin - current, 0) / scale
    gram = gram / scale[:, None] / scale[None, :]
    regularized = gram + 0.04 * np.eye(len(scale), dtype=gram.dtype)
    factors = np.linalg.solve(regularized, deficits) / scale
    return (embedding_diffs.T * factors) @ dual_keys


def retained_blocks(parent, candidate, delta):
    """True selects the original quantized block. Inputs are decoded weights."""
    if parent.shape != candidate.shape or parent.shape != delta.shape:
        raise ValueError("All three arrays must have the same shape.")
    unedited_error = np.square(delta).reshape(-1, 256).mean(axis=1)
    edited_error = np.square(candidate - parent - delta).reshape(-1, 256).mean(axis=1)
    return unedited_error <= edited_error
