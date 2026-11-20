# How the R5 pilot was edited

The starting file was the released R4 checkpoint. Calibration used English text, a frozen teacher and historical MiniMind2 generations. At inference the model uses its own weights in a single GGUF. The functions in [analytic_steps.py](analytic_steps.py) implement three individual operations from the experiment. The paper describes the full sequence and the benchmark protocol.

## From calibration data to the last MLP

1. Draw stories from a 64 MiB prefix of TinyStories training text with seed 7182818, excluding normalized exact matches to validation and to an earlier calibration slice. Estimate the R4 hidden-state second moment on 2,000 stories (517,611 next-token positions). The first teacher fit used the first **1,080** of those stories. For each matching byte position, map next-token probabilities of the frozen TinyStories-33M teacher to the MiniMind vocabulary and fill unmapped probability mass from MiniMind.
2. For the selected first fit, use 192 whitened directions of the final SwiGLU activation plus an intercept. For each sampled vocabulary class, regress the centered, clipped teacher-mixture versus student log-probability difference on the activation coordinates. In the first pass, 24 candidate token IDs are sampled per aligned position. A class needs at least 40 sampled rows. The intercept and slope have different ridge penalties. The function `fit_conditional_token` takes the sampled rows for a *single* class. Map the resulting token-specific slopes back through the effective output embeddings and regularize against dialogue activations. Apply 0.85 times this down-projection correction together with 1.0 times the separately computed n-gram correction.
3. Quantize the proposed update to `blk.15.ffn_down.weight`. For each block of 256 values, compare mean squared error to the proposed floating-point weights under the old and new encodings. Copy the original block back when its error is smaller or tied. The function `retained_blocks` returns this selection mask.
4. Decode 392 development prompts. Track exact R4 successes, empty output and two types of repetition. At failed prompts, teacher-force the **R4** trajectory and measure the margin between the R4 next token and a competing token. The function `recover_margins` computes one increment to the last MLP from the accumulated constraints. Its metric is the story-state second moment plus 0.6 times the sum of the grammar and COPA development second moments, plus a diagonal stabilizer equal to 0.012 times the mean story diagonal. Keys and residuals are divided by the terminal residual RMS. The effective embedding includes the final RMSNorm gain. After applying and quantizing the accumulated increments, replay all prompts every three steps. The first round passed at step 15.
5. Collect states from that protected model on the next 2,200 training stories (positions 2,001–4,200 in the sampled sequence). Fit a second token-specific correction with up to 512 directions. The effective target mixture weights are 0.87 for the teacher including the unmapped-mass fallback, 0.08 for the student, and 0.05 for the observed next token. Apply 0.4 of that correction (guard coefficient 0.12), plus 0.18 of a correction estimated from 4,994 matched-versus-shuffled causal continuation pairs. Multiply the input RMSNorm gain for attention in layers 8 through 15 by 1.06. Repeat the margin and quantization procedure. The number of failing prompts was five at step 0, four at step 3 and zero at step 6.

Guard increments use a local linear estimate of the residual. The margin parameter is 1.4 for steps 0–8 and 2.8 thereafter. Rival constraints are added when the target's lead is less than 0.65 times that margin at an active replay position. The solver uses the normalized constraint Gram matrix with a ridge of 0.04. Actual greedy generation from the serialized GGUF determines whether a candidate passes.

## Run the checks

The numerical example needs Python 3.11+ and NumPy. It compares the token fit to an independent least-squares calculation and tests the margin correction and block rule on small arrays.

```sh
python -m pip install numpy
python method/check_steps.py
```

Given the published R4 and R5 files, the second script checks SHA-256 identities, lists every changed tensor with its differing byte count, and verifies all 6,144 edited attention-norm entries against the expected scale. It reads both models without modifying them. Install the Python GGUF reader from llama.cpp or its `gguf` package, then supply the two filenames.

```sh
python -m pip install gguf
python method/check_weights.py /path/to/R4.gguf /path/to/R5-pilot.gguf
```

The files differ in nine tensors. The last MLP down projection accounts for 1,258,329 of the 1,276,633 changed bytes. The norm vectors account for the remainder.

Generating the update from scratch also requires activation traces, aligned teacher distributions, sufficient statistics and the historical guard trajectories. The release provides the numerical examples, final checkpoints and archived evaluation evidence. These scripts reproduce the selected operations and compare the actual files. The numerical examples use synthetic inputs.

[Paper](../paper/minimind-analytical-repair.pdf) · [Checkpoints](https://huggingface.co/rexarrior/MiniMind2-English-Analytical-Repair-GGUF) · [Frozen teacher revision](https://huggingface.co/roneneldan/TinyStories-33M/tree/2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8)
