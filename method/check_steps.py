"""Small deterministic checks of the three real-valued kernels."""

import numpy as np

from analytic_steps import fit_conditional_token, recover_margins, retained_blocks


def main():
    rng = np.random.default_rng(123)
    design = np.column_stack([np.ones(80), rng.normal(size=(80, 4))])
    target = rng.normal(size=80)
    fitted = fit_conditional_token(design, target)
    penalties = [0.02 * 80 + 8] + [0.15 * 80 + 70] * 4
    extended_design = np.vstack([design, np.diag(np.sqrt(penalties))])
    expected, *_ = np.linalg.lstsq(extended_design, np.r_[target, np.zeros(5)], rcond=None)
    np.testing.assert_allclose(fitted, expected)
    np.testing.assert_array_equal(fit_conditional_token(design[:39], target[:39]), np.zeros(5))

    # Two independent constraints with different state/output widths.
    keys = np.eye(2, 3)
    diffs = np.eye(2, 4)
    residuals = np.zeros((2, 4))
    metric = np.diag([2.0, 3.0, 4.0])
    change = recover_margins(keys, residuals, diffs, metric, margin=1.4)
    achieved = ((keys @ change.T + residuals) * diffs).sum(axis=1)
    np.testing.assert_allclose(achieved, np.full(2, 1.4 / 1.04))

    parent = np.zeros((3, 256))
    proposed = np.ones_like(parent)
    decoded = np.array([np.full(256, value) for value in (0.9, 3.0, 2.0)])
    np.testing.assert_array_equal(retained_blocks(parent, decoded, proposed), [False, True, True])
    print("Three analytical kernels passed.")


if __name__ == "__main__":
    main()
