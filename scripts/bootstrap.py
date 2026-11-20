"""Reproduce paired R5 intervals, using NumPy and the published per-item data."""
import json

import numpy as np

from audit import EVIDENCE, REFERENCES, read


def bootstrap():
    tiny, blimp, copa = [read(name) for name in ("tinystories", "blimp", "copa")]
    groups = sorted({r["paradigm"] for r in blimp})
    token_counts = np.array([int(r["tokens"]) for r in tiny])
    group_id = np.array([groups.index(r["paradigm"]) for r in blimp])
    successes = {}
    nll = {}
    answers = {}
    for m in (*REFERENCES, "r5"):
        nll[m] = np.array([float(r[m + "_nll"]) for r in tiny])
        successes[m] = np.bincount(group_id, weights=[float(r[m + "_margin"]) > 0 for r in blimp])
        answers[m] = np.array([int(float(r[m + "_margin"]) < 0) == int(r["gold_index"]) for r in copa])

    rng = np.random.default_rng(11391)
    intervals = []
    for ref in REFERENCES:
        draws = []
        delta = nll[ref] - nll["r5"]
        for _ in range(100):
            stories = rng.integers(len(tiny), size=(40, len(tiny)))
            ppl = np.exp(delta[stories].sum(axis=1) / token_counts[stories].sum(axis=1))
            paradigms = rng.integers(len(groups), size=(40, len(groups)))
            blimp_ratio = successes["r5"][paradigms].sum(axis=1) / successes[ref][paradigms].sum(axis=1)
            questions = rng.integers(len(copa), size=(40, len(copa)))
            copa_ratio = answers["r5"][questions].sum(axis=1) / answers[ref][questions].sum(axis=1)
            index = (ppl * blimp_ratio * copa_ratio) ** (1 / 3)
            draws.append(np.column_stack((ppl, blimp_ratio, copa_ratio, index)))
        low, high = np.quantile(np.vstack(draws), [0.025, 0.975], axis=0)
        intervals.append({
            "baseline": ref, "lower": low.tolist(), "upper": high.tolist(),
            "stories_lower_nll": int(np.sum(delta > 0)),
            "blimp_categories_up": int(np.sum(successes["r5"] > successes[ref])),
        })
    return intervals


if __name__ == "__main__":
    reference = json.loads((EVIDENCE / "r5-intervals.json").read_text())
    for actual, expected in zip(bootstrap(), reference["intervals"], strict=True):
        for name in ("baseline", "stories_lower_nll", "blimp_categories_up"):
            if actual[name] != expected[name]:
                raise ValueError(f"Unexpected {name}")
        for name in ("lower", "upper"):
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-10, atol=1e-12)
        print(f"R5 vs {actual['baseline']} index 95% interval: "
              f"[{actual['lower'][-1]:.4f}, {actual['upper'][-1]:.4f}]")
