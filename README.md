# Analytical weight editing in MiniMind2

Can direct changes to a small language model's weights stop repetitive generation, preserve correct short answers, and improve its predictions on English text? This repository accompanies a case study of MiniMind2 Q4. It contains the English paper, the analytical methods, and the measurements behind the comparisons.

[Read the paper (PDF)](paper/minimind-analytical-repair.pdf) · [LaTeX source](paper/main.tex) · [GGUF checkpoints on Hugging Face](https://huggingface.co/rexarrior/MiniMind2-English-Analytical-Repair-GGUF)

## Results

The early edits improved short answers while sharply worsening story prediction. Later edits recovered that loss. R4 scored highest on the full grammatical-pair benchmark among the four checkpoints below. R5 had the lowest story perplexity. Causal-choice accuracy stayed close to chance throughout.

| Checkpoint | Strict English (of 96) | Exact copying (of 72) | TinyStories perplexity ↓ | BLiMP (of 67,000) | COPA test (of 500) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Q4, tokenizer aligned | 0 | 0 | 24.524 | 39,658 | 252 |
| R3 | 7 | 47 | 214.478 | 36,571 | 253 |
| R4 | 9 | 48 | 17.615 | 40,210 | 253 |
| R5 pilot | 12 | 52 | 11.981 | 39,900 | 251 |

The historical short-answer checks and subsets of the public benchmarks informed model selection. R5 preserved every exact R4 success on the 392 development prompts. On 48 prompts written after the checkpoint was frozen, 40 had reference answers. Exact matches rose from 4 to 5, with two gains and one loss. See the paper for held-out-by-ID slices, paired changes and the limits of those comparisons.

## Materials

- [Analytical methods](method/) — implementation details for the weight changes.
- [Evidence](evidence/) — archived measurements for the published comparisons.
- [Scoring and analysis scripts](scripts/) — prepare benchmark inputs, score a GGUF and recompute paired results.
- [Hugging Face model repository](https://huggingface.co/rexarrior/MiniMind2-English-Analytical-Repair-GGUF) — checkpoint files, model card and file hashes.

All checkpoint comparisons use the MiniMind tokenizer alignment described in the paper. The edits use calibration text, synthetic examples and an external English model. No gradient fine-tuning was used in this study.

## Reproduce the tables

From the repository root, run the following command with Python 3.10 or later. It reads the archived per-item scores and checks the totals and paired changes against the released summary.

```sh
python3 scripts/audit.py
```

The [evidence guide](evidence/README.md) describes how to rebuild the scorer inputs from upstream datasets, check their hashes and repeat the model scoring. The [method guide](method/README.md) includes the numerical checks and a byte-level comparison of the distributed R4 and R5 files.

To build the paper with a LaTeX installation that includes Latin Modern, microtype, natbib and hyperref:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -jobname=minimind-analytical-repair main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=minimind-analytical-repair main.tex
```
