# Research notes: sentiment probes in GPT-2 small

My working log for this project. Every number here comes from the saved files in `results/`.

## Question

After which block of GPT-2 small can a linear classifier read sentiment from the last word of a sentence, and how much of that goes beyond word counting?

## Status

- Experiments, controls, statistics and the negation extension are done.
- Paper written (3 pages plus references), compiles on Overleaf with pdfLaTeX.
- Left to do: read the paper end to end and reword anything I can't explain out loud; push the repo to GitHub.

## Setup

- Windows 11, 8 GB RAM, CPU only. Python 3.13.3.
- torch 2.15.0+cpu, transformer-lens 4.4.0, transformers 5.24.0, scikit-learn 1.9.1, datasets 5.2.0.
- GPT-2 small loaded with `TransformerBridge.boot_transformers("gpt2")` plus `enable_compatibility_mode()`. `HookedTransformer` no longer exists in TransformerLens 4.
- Untrained control: `GPT2Model(GPT2Config())` from Hugging Face, final LayerNorm removed, seed k in run k.
- Seed 0 everywhere in run 0; run k uses seed k.

## Conventions

- 13 points per sentence: point 0 = before any block (`blocks.0.hook_resid_pre`), point L+1 = after block L (`blocks.L.hook_resid_post`).
- Only the last token is probed. The start token is identical for every sentence, so it carries nothing.

## Data

- SST-2 from Hugging Face (`stanfordnlp/sst2`). Test set = the 872 validation sentences (444 positive, 428 negative). Official test labels are hidden.
- Training pool: train split with at least 6 words, whitespace stripped, duplicates removed, no overlap with the test set (38,717 examples).
- Each run: 1,000 positive + 1,000 negative examples. Five runs, five disjoint samples.
- 94% of test sentences end in a full stop, only 29% of training examples do.

## Method

- Probe: standardise features (training data only), then logistic regression.
- Regularisation C picked by 5-fold stratified cross-validation on the training sample, from 0.0001 to 10.
- Baselines: majority class (50.9%), shuffled training labels, bag of words (binary word indicators, space-split), untrained GPT-2.
- Statistics: mean and standard deviation over 5 runs; paired bootstrap over test sentences (10,000 resamples) for comparisons; Bonferroni cutoff 0.05 / 12 = 0.0042.

## Main results (test accuracy, mean over 5 runs)

| point | trained GPT-2 | std | untrained GPT-2 | std |
|---|---|---|---|---|
| 0 | 0.494 | 0.015 | 0.500 | 0.014 |
| 1 | 0.672 | 0.024 | 0.570 | 0.020 |
| 2 | 0.717 | 0.015 | 0.568 | 0.016 |
| 3 | 0.711 | 0.005 | 0.558 | 0.019 |
| 4 | 0.739 | 0.015 | 0.548 | 0.034 |
| 5 | 0.780 | 0.003 | 0.561 | 0.014 |
| 6 | 0.787 | 0.006 | 0.556 | 0.012 |
| 7 | 0.842 | 0.009 | 0.550 | 0.016 |
| 8 | 0.848 | 0.005 | 0.554 | 0.014 |
| 9 | 0.852 | 0.014 | 0.556 | 0.015 |
| 10 | 0.841 | 0.007 | 0.556 | 0.016 |
| 11 | 0.837 | 0.010 | 0.556 | 0.019 |
| 12 | 0.834 | 0.008 | 0.561 | 0.011 |

- Bag of words: 0.725 (std 0.015).
- Shuffled labels (run 0): 0.471 to 0.537 at every point. No leakage.
- Best point per run: 9, 8, 7, 11, 8. No single best block.

## Paired comparisons (percentage points)

| comparison | difference | 95% range | p |
|---|---|---|---|
| block 9 vs block 7 | +1.0 | -0.3 to +2.3 | 0.119 |
| block 9 vs block 12 | +1.7 | +0.7 to +2.8 | 0.001 |
| block 1 vs bag of words | -5.4 | -7.8 to -3.0 | <0.001 |
| block 4 vs bag of words | +1.3 | -1.3 to +3.8 | 0.305 |
| block 5 vs bag of words | +5.4 | +3.0 to +7.9 | <0.001 |
| block 7 vs bag of words | +11.6 | +9.2 to +14.0 | <0.001 |
| block 1, trained vs untrained | +10.1 | +7.7 to +12.5 | <0.001 |
| block 9, trained vs untrained | +29.6 | +27.0 to +32.2 | <0.001 |

Block 9 vs 11 was also checked: +1.5, p = 0.0018.

## Full-stop control

- Trained only on examples ending in a full stop (560 to 584 per run) vs random sets of the same size.
- Full-stop training wins at blocks 1, 4, 7, 9 by 2.3 to 4.8 points (all p < 0.001).
- With about 570 examples it matches the 2,000-example probe (0.848 vs 0.852 at block 9).
- So the train/test mismatch pulls the main results down slightly. It can't separate "ends in a full stop" from "is a full sentence".

## Negation test

- 40 pairs from the template "{subject} was {adjective} ." vs "{subject} was not {adjective} .".
- Subjects: the movie, the acting, the plot, the script, the ending. Positive: good, great, funny, charming. Negative: bad, boring, dull, awful.
- Probes applied without retraining. Shift measured in standard deviations of the probe score on the test set.
- Plain sentences classified correctly 91.5% to 96.5% at blocks 3 to 7.
- "not good": score drops at every block, prediction flips in 92% of pairs at block 5 and 91% at block 7.
- "not bad": score drops at blocks 1 to 4 (like bag of words), zero at blocks 5 and 6, rises from block 7 (+1.31). Flips to positive in at most 31% of pairs.
- Interaction (shift for bad minus shift for good): about 0 at block 1, peak 3.41 at block 7, 2.07 at block 12. Bag of words exactly 0, untrained model near 0.
- Bag of words learned a negative weight for "not" in all 5 runs.

## Limitations

- A probe shows information is readable, not that GPT-2 uses it.
- One model size, one dataset, last token only.
- Five runs is a rough estimate of spread; the test set never changes.
- Trained model loaded with centred weights, untrained model without.
- Negation test: one template, 40 pairs; interaction measure defined after seeing the results.

## Problems hit

- `HookedTransformer` import failed in TransformerLens 4.0. Switched to `TransformerBridge` with compatibility mode.
- First majority baseline was computed on the train split (55.8%) instead of the test set (50.9%).
- Three points first chose the smallest C in the grid. Widened the grid; fixed.
- Figure legends covered data. Moved them below the plots.
