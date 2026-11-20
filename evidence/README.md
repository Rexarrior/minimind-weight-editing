# Data and replication

The four columns in each external benchmark file correspond to the same four checkpoints. “Original” denotes the author's Q4 parameters with the aligned tokenizer metadata. Each score was measured on the serialized GGUF. For inference the standalone file contains the complete model.

## Check the results

Run the following commands from the repository root. The main audit requires Python 3.10 or later and the standard library. It reads every scored example, checks IDs and denominators, recomputes the values and compares them with `evidence/summary.json`.

```sh
python3 scripts/audit.py
python3 scripts/audit.py --json
```

With NumPy installed, `python3 scripts/bootstrap.py` reproduces the archived 95% percentile intervals. It uses 4,000 paired draws per comparison with seed 11391. Sampling units are complete stories, BLiMP paradigms and COPA questions. Each draw resamples the two checkpoints on the same units within a benchmark. Columns in `r5-intervals.json` are the inverse perplexity ratio, BLiMP accuracy ratio, COPA accuracy ratio and their geometric mean, in that order. These intervals describe variability under resampling of the measured units. They do not quantify uncertainty from model selection.

## Files and units

| File | Contents |
| --- | --- |
| `tinystories.csv.gz` | One row per validation story in file order, indexed from zero. `tokens` is the number of scored text tokens. Each `*_nll` is their summed negative natural log likelihood. Corpus PPL is the exponential of total NLL divided by total tokens. |
| `blimp.csv.gz` | One row per `(paradigm, pair_id)`. Each margin equals the summed NLL of the ungrammatical sentence minus the summed NLL of the grammatical one. A positive margin is correct. Ties fail. |
| `copa.csv.gz` | One row per SuperGLUE test question with the original COPA ID. The margin is summed conditional NLL of choice 2 minus that of choice 1. A nonnegative margin selects index 0. The gold label is zero-based. Original labels were matched to the SuperGLUE examples by the normalized premise and both alternatives and checked against `gold.test`. |
| `internal.csv.gz` | Paired R4 and R5 results on 360 historical prompts, 32 additional development prompts and 48 post-freeze prompts. Each set has a zero-based row ordinal. Blank exact scores denote open questions without gold answers. The strict-English 96 and copy 72 are subsets of the 360. Detector hits measure word loops, character loops and empty outputs. These cannot establish whether an open answer makes sense. |
| `checkpoints.csv` | Recorded aggregate counts and perplexities for all 11 measured checkpoints and controls. Column names specify the denominator. Blank cells are unreported measurements. `limit22` counts generations reaching the token cap on 22 probes. `tiny_ppl_1000`, `blimp_3350`, `blimp335`, `copa_dev100`, `copa_train400` and `tiny_screen100` describe development subsets. |
| `r5-edited-tensors.json` | SHA-256 of the R4 and R5 files, total file size, and byte differences by tensor. These are byte counts, including quantization block changes. |
| `dataset-sources.json` | Canonical benchmark download URLs and SHA-256 digests of the data files. The archive contains the original COPA gold labels. |

Rows marked `screen` and `validation` were used in model development (100 and 1,000 TinyStories stories, 335 and 3,350 BLiMP pairs). The two subsets within each benchmark are disjoint. The audit reports results on the remaining 20,890 stories and 63,315 pairs. Earlier rounds saw aggregate results for these public tasks. On COPA the 400 training and 100 validation questions were available during development. The 500-row file here is the test split. The R5 bytes were frozen before this final measurement. The 48 post-freeze internal prompts are a small diagnostic set informed by prior observations.

## Rerun the likelihood scorer

Download `tiny.txt`, `blimp.zip` and `test.parquet` using the URLs for those names in [dataset-sources.json](dataset-sources.json). The preparation script checks their SHA-256 and verifies that the token stream for each benchmark agrees byte-for-byte with the scored input. It uses the author's 6,400-token tokenizer from a fixed revision. [Tokenizer download](https://huggingface.co/jingyaogong/MiniMind2/resolve/d18103e50a735dfb8ab801183be734bc635dfc22/tokenizer.json)

```sh
python3 -m pip install tokenizers==0.23.3 pyarrow==26.0.1
python3 scripts/prepare_inputs.py tinystories tiny.txt tokenizer.json tinystories.bin
python3 scripts/prepare_inputs.py blimp blimp.zip tokenizer.json blimp.bin
python3 scripts/prepare_inputs.py copa test.parquet tokenizer.json copa.bin
```

For the original-model column, download `Q4-MiniMind2.gguf` from the author's GGUF repository and apply the one-field metadata correction. [Fixed upstream revision](https://huggingface.co/jingyaogong/MiniMind2-gguf/resolve/97d25a81aa75a75c74f13ef419307afc63c9e7e9/Q4-MiniMind2.gguf)

```sh
python3 scripts/align_tokenizer.py Q4-MiniMind2.gguf aligned-original.gguf
```

The native reference scorer uses llama.cpp, ggml and Apple's Accelerate library. The archived run used an Apple M2 Pro with Metal, an effective batch capacity of 2,048 tokens and at most 64 independent sequences per batch. The following commands build the scorer against Homebrew with llama.cpp 0.5.0 and ggml 0.25.1. Other builds or backends may differ numerically. Supply each checkpoint in turn as `model.gguf`. Each output line contains the number of scored tokens and their summed NLL.

```sh
LLAMA_PREFIX=$(brew --prefix llama.cpp)
GGML_PREFIX=$(brew --prefix ggml)
clang++ -std=c++17 -O3 -I"$LLAMA_PREFIX/include" -I"$GGML_PREFIX/include" \
  -L"$LLAMA_PREFIX/lib" -L"$GGML_PREFIX/lib" scripts/score.cpp \
  -lllama -lggml -lggml-base -framework Accelerate -o score
./score model.gguf tinystories.bin tinystories.scores 2048 64
./score model.gguf blimp.bin blimp.scores 2048 64
./score model.gguf copa.bin copa.scores 2048 64
```

Each independent sequence starts with BOS ID 1. TinyStories and BLiMP score every following text token, including the first. No EOS is appended. COPA uses the premise without its final punctuation followed by “ because” for a cause or “ therefore” for an effect. The first letter of each continuation is lowercased. Only continuation tokens contribute to its score. Both forced-choice measures compare summed losses without length normalization. The recorded NLL margins preserve the precision of the archived text scores. Replaying the full measurement requires the corresponding GGUF and the upstream data. Recomputing the reported summaries and intervals requires only the numerical files in this release.

For the release check, we regenerated all three complete token streams and compared their SHA-256 digests with the research archive. A second build of the scorer then replayed an original-Q4 subset. The 15 TinyStories sums matched at the stored precision. All 32 COPA and 32 BLiMP decisions matched. The maximum absolute margin change was below 8 × 10⁻¹⁵ on COPA and 0.028 on BLiMP when the examples were packed into different batches.
