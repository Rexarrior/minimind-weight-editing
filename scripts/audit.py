"""Recompute the published measures from per-example scores. Requires Python 3.10+."""
import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
MODELS = ("original", "r3", "r4", "r5")
REFERENCES = ("original", "r4")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    with gzip.open(EVIDENCE / f"{name}.csv.gz", "rt", newline="") as source:
        return list(csv.DictReader(source))


def finite(row, key):
    value = float(row[key])
    check(math.isfinite(value), f"Non-finite {key}")
    return value


def transitions(old, new):
    return {
        "gained": sum(b and not a for a, b in zip(old, new, strict=True)),
        "lost": sum(a and not b for a, b in zip(old, new, strict=True)),
    }


def compare(predictions):
    return {
        "correct": {m: sum(predictions[m]) for m in MODELS},
        "r5_vs": {ref: transitions(predictions[ref], predictions["r5"]) for ref in REFERENCES},
    }


def forced_choice(rows, task):
    answers = {m: [] for m in MODELS}
    for row in rows:
        for model in MODELS:
            margin = finite(row, model + "_margin")
            if task == "blimp":
                correct = margin > 0
            else:
                gold = int(row["gold_index"])
                check(gold in (0, 1), "Invalid COPA gold index")
                correct = int(margin < 0) == gold
            answers[model].append(correct)
    return compare(answers)


def stories(rows):
    counts = [int(r["tokens"]) for r in rows]
    check(all(n > 0 for n in counts), "Invalid token count")
    nll = {m: [finite(r, m + "_nll") for r in rows] for m in MODELS}
    return {
        "items": len(rows),
        "tokens": sum(counts),
        "ppl": {m: round(math.exp(math.fsum(v) / sum(counts)), 8) for m, v in nll.items()},
        "r5_lower_loss": {ref: sum(a > b for a, b in zip(nll[ref], nll["r5"])) for ref in REFERENCES},
    }


def subsets(rows, expected):
    observed = Counter(r["selection"] for r in rows)
    check(observed == expected, f"Unexpected selection counts {observed}")
    return {"full": rows, "outside_selection": [r for r in rows if r["selection"] == "other"]}


def internal(rows):
    by_set = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    check({k: len(v) for k, v in by_set.items()} == {
        "historical360": 360, "additional32": 32, "postfreeze48": 48
    }, "Unexpected internal set sizes")
    for k, v in by_set.items():
        check({int(r["ordinal"]) for r in v} == set(range(len(v))), f"Bad ordinal in {k}")
    for group in ("strict96", "copy72"):
        by_set[group] = [r for r in rows if r["group"] == group]
    output = {}
    for name, part in by_set.items():
        scored = [r for r in part if int(r["scorable"])]
        answers = {m: [bool(int(r[m + "_exact"])) for r in scored] for m in ("r4", "r5")}
        output[name] = {
            "items": len(part), "scorable": len(scored),
            "correct": {m: sum(v) for m, v in answers.items()},
            "r5_vs_r4": transitions(answers["r4"], answers["r5"]),
            "detector_hits": {d: {m: sum(int(r[f"{m}_{d}"]) for r in part) for m in ("r4", "r5")}
                              for d in ("word_loop", "character_loop", "empty")},
        }
    return output


def compute():
    tiny, blimp, copa = [read(k) for k in ("tinystories", "blimp", "copa")]
    check({int(r["story_id"]) for r in tiny} == set(range(21990)), "TinyStories IDs")
    check(len({(r["paradigm"], r["pair_id"]) for r in blimp}) == 67000, "BLiMP IDs")
    groups = Counter(r["paradigm"] for r in blimp)
    check(len(groups) == 67 and set(groups.values()) == {1000}, "BLiMP paradigms")
    check(len(copa) == len({int(r["test_id"]) for r in copa}) == len({int(r["original_copa_id"]) for r in copa}) == 500, "COPA IDs")
    ts = subsets(tiny, {"screen": 100, "validation": 1000, "other": 20890})
    bs = subsets(blimp, {"screen": 335, "validation": 3350, "other": 63315})
    ts = {k: stories(rows) for k, rows in ts.items()}
    bs = {k: {"pairs": len(rows), **forced_choice(rows, "blimp")} for k, rows in bs.items()}
    cs = {"items": len(copa), **forced_choice(copa, "copa")}
    ppl = ts["full"]["ppl"]
    grammar = bs["full"]["correct"]
    causal = cs["correct"]
    # Ratios use the same denominator within each forced-choice benchmark.
    index = {m: round(((ppl["original"] / ppl[m]) * (grammar[m] / grammar["original"]) *
                       (causal[m] / causal["original"])) ** (1 / 3), 8) for m in MODELS}
    return {"tinystories": ts, "blimp": bs, "copa": cs, "index": index, "internal": internal(read("internal"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print all paired and subset results")
    args = parser.parse_args()
    result = compute()
    expected = json.loads((EVIDENCE / "summary.json").read_text())
    check(result == expected, "Scores disagree with evidence/summary.json")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Verified. Model          TinyStories PPL    BLiMP /67000   COPA /500    Index")
        for m in MODELS:
            print(f"{m:>18} {result['tinystories']['full']['ppl'][m]:17.4f}"
                  f"{result['blimp']['full']['correct'][m]:16}{result['copa']['correct'][m]:12}"
                  f"{result['index'][m]:9.4f}")
