"""
Step 10: paired bootstrap comparisons on the test set.
1. Refit every probe from the 5 runs with the C already chosen in Step 9 (no new CV),
   and record which test sentences each probe gets right.
2. Average correctness over the 5 runs for each sentence.
3. Bootstrap the 872 test sentences to get a 95% interval for each accuracy difference.
Output: results/paired_bootstrap.csv
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from probe_utils import file_paths, make_probe

start = time.time()
SAMPLES = [0, 1, 2, 3, 4]
N_POINTS = 13
N_BOOT = 10_000
SEED = 0

long = pd.read_csv("results/resamples_long.csv")    # chosen C per (sample, model, point)
bow_runs = pd.read_csv("results/resamples_bow.csv")
y_test = np.load("data/labels_test.npy")
X_test_pre = np.load("data/hidden_test.npy")
test_text = pd.read_csv("data/probe_test.csv")["sentence"]
n_test = len(y_test)


def make_bow(C):
    return make_pipeline(CountVectorizer(binary=True, token_pattern=r"\S+"),
                         LogisticRegression(C=C, max_iter=5000, random_state=0))


# ---- 1. Per-sentence correctness for every probe in every run ----
# correct[name] has shape [5 runs, 872 sentences], 1 = right, 0 = wrong
correct = {}
for k in SAMPLES:
    p = file_paths(k)
    y_train = np.load(p["labels"])
    data = {"pretrained": (np.load(p["pre_train"]), X_test_pre),
            "random": (np.load(p["rand_train"]), np.load(p["rand_test"]))}
    for model, (X_tr, X_te) in data.items():
        for point in range(N_POINTS):
            row = long[(long["sample"] == k) & (long["model"] == model) & (long["point"] == point)]
            probe = make_probe(row["C"].item()).fit(X_tr[:, point], y_train)
            right = (probe.predict(X_te[:, point]) == y_test).astype(float)
            correct.setdefault((model, point), np.zeros((len(SAMPLES), n_test)))[k] = right

            # Sanity check: this refit must reproduce the Step 9 accuracy exactly.
            assert abs(right.mean() - row["test_acc"].item()) < 1e-12, (k, model, point)

    C_bow = bow_runs.loc[bow_runs["sample"] == k, "C"].item()
    bow = make_bow(C_bow).fit(pd.read_csv(p["text"])["sentence"], y_train)
    right = (bow.predict(test_text) == y_test).astype(float)
    correct.setdefault(("bow", None), np.zeros((len(SAMPLES), n_test)))[k] = right
    print(f"run {k} refit done after {time.time() - start:.0f} s")
print("all refits match Step 9 accuracies exactly")

# Average over runs: one number per sentence, e.g. 0.8 = right in 4 of 5 runs
per_sentence = {name: arr.mean(axis=0) for name, arr in correct.items()}

# ---- 2. Paired bootstrap ----
rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, n_test, size=(N_BOOT, n_test))   # 10,000 pretend test sets


def paired_bootstrap(a, b):
    """a, b: per-sentence scores for the same 872 sentences. Returns diff, CI and p-value."""
    diff = a - b                              # positive where A did better on that sentence
    observed = diff.mean()                    # = accuracy of A minus accuracy of B
    boot = diff[boot_idx].mean(axis=1)       # the difference on each pretend test set
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided p-value: how often the bootstrap lands on the other side of zero, doubled
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return observed, lo, hi, min(p, 1.0)


comparisons = [
    ("pretrained 9 vs pretrained 7", ("pretrained", 9), ("pretrained", 7)),
    ("pretrained 9 vs pretrained 8", ("pretrained", 9), ("pretrained", 8)),
    ("pretrained 9 vs pretrained 10", ("pretrained", 9), ("pretrained", 10)),
    ("pretrained 9 vs pretrained 11", ("pretrained", 9), ("pretrained", 11)),
    ("pretrained 9 vs pretrained 12", ("pretrained", 9), ("pretrained", 12)),
    ("pretrained 1 vs bag of words", ("pretrained", 1), ("bow", None)),
    ("pretrained 4 vs bag of words", ("pretrained", 4), ("bow", None)),
    ("pretrained 5 vs bag of words", ("pretrained", 5), ("bow", None)),
    ("pretrained 6 vs bag of words", ("pretrained", 6), ("bow", None)),
    ("pretrained 7 vs bag of words", ("pretrained", 7), ("bow", None)),
    ("pretrained 1 vs random 1", ("pretrained", 1), ("random", 1)),
    ("pretrained 9 vs random 9", ("pretrained", 9), ("random", 9)),
]
bonferroni = 0.05 / len(comparisons)

rows = []
for label, a, b in comparisons:
    diff, lo, hi, p = paired_bootstrap(per_sentence[a], per_sentence[b])
    rows.append({"comparison": label, "acc_A": per_sentence[a].mean(), "acc_B": per_sentence[b].mean(),
                 "diff": diff, "ci_low": lo, "ci_high": hi, "p": p,
                 "excludes_0": lo > 0 or hi < 0, "passes_bonferroni": p < bonferroni})

table = pd.DataFrame(rows)
table.to_csv("results/paired_bootstrap.csv", index=False)
pd.set_option("display.width", 140)
print(f"\nBonferroni cutoff for {len(comparisons)} comparisons: p < {bonferroni:.4f}")
print(table.round(4).to_string(index=False))
print(f"\nruntime: {time.time() - start:.0f} s")