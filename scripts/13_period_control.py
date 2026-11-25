"""
Step 11: does the train/test mismatch in last tokens matter?
For each of the 5 runs:
  period  = training examples whose last GPT-2 token is a period, balanced by class
  matched = random examples of the SAME size and balance, any last token
Both are probed at all 13 points (pretrained GPT-2) with C chosen by CV, scored on the full test set.
Output: results/period_control_long.csv, results/period_control_bootstrap.csv, figures/period_control.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast

from probe_utils import file_paths, fit_with_cv, make_probe, paired_bootstrap

start = time.time()
SAMPLES = [0, 1, 2, 3, 4]
N_POINTS = 13
TEST_POINTS = [1, 4, 7, 9]           # chosen before seeing results
PERIOD_TOKENS = {"Ġ.", "."}          # GPT-2 period with and without a leading space

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
y_test = np.load("data/labels_test.npy")
X_test = np.load("data/hidden_test.npy")
n_test = len(y_test)


def last_token(sentence):
    return tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"][-1])


def cv_fit_predict(X_tr, y_tr, X_te, seed):
    """Choose C by CV, refit, return (C, cv_acc, per-sentence correctness on test)."""
    C, cv_acc, _, _ = fit_with_cv(make_probe, X_tr, y_tr, X_te, y_test, seed=seed)
    probe = make_probe(C).fit(X_tr, y_tr)
    return C, cv_acc, (probe.predict(X_te) == y_test).astype(float)


rows = []
correct = {}   # (condition, point) -> array [5 runs, 872 sentences]
for k in SAMPLES:
    p = file_paths(k)
    text = pd.read_csv(p["text"])["sentence"]
    y = np.load(p["labels"])
    X = np.load(p["pre_train"])

    # ---- 1. Which training examples end in a period token? ----
    is_period = text.map(last_token).isin(PERIOD_TOKENS).to_numpy()

    # ---- 2. Balanced period-only subset and a size-matched random subset ----
    rng = np.random.default_rng(k)
    per_pos = np.where(is_period & (y == 1))[0]
    per_neg = np.where(is_period & (y == 0))[0]
    n_per_class = min(len(per_pos), len(per_neg))
    period_idx = np.concatenate([rng.choice(per_pos, n_per_class, replace=False),
                                 rng.choice(per_neg, n_per_class, replace=False)])
    matched_idx = np.concatenate([rng.choice(np.where(y == 1)[0], n_per_class, replace=False),
                                  rng.choice(np.where(y == 0)[0], n_per_class, replace=False)])
    print(f"run {k}: {is_period.sum()} of {len(y)} end in a period "
          f"({(is_period & (y == 1)).sum()} pos, {(is_period & (y == 0)).sum()} neg) "
          f"-> {2 * n_per_class} examples per condition; "
          f"matched subset that ends in a period: {is_period[matched_idx].mean():.2f}")

    # ---- 3. Probe every point in both conditions ----
    for condition, idx in [("period", period_idx), ("matched", matched_idx)]:
        for point in range(N_POINTS):
            C, cv_acc, right = cv_fit_predict(X[idx, point], y[idx], X_test[:, point], seed=k)
            rows.append({"sample": k, "condition": condition, "point": point, "n_train": len(idx),
                         "C": C, "cv_acc": cv_acc, "test_acc": right.mean()})
            correct.setdefault((condition, point), np.zeros((len(SAMPLES), n_test)))[k] = right
    print(f"   done after {time.time() - start:.0f} s")

long = pd.DataFrame(rows)
long.to_csv("results/period_control_long.csv", index=False)

# ---- 4. Summary over the 5 runs, next to the full 2,000-example probe from Step 9 ----
summary = long.groupby(["condition", "point"])[["cv_acc", "test_acc"]].mean().unstack("condition")
summary.columns = [f"{metric}_{cond}" for metric, cond in summary.columns]
full = pd.read_csv("results/resamples_summary.csv")
summary["test_full_2000"] = full[full["model"] == "pretrained"].set_index("point")["mean"]
print("\nmean over 5 runs:")
print(summary.round(3).to_string())

# ---- 5. Paired bootstrap: period vs matched at the pre-chosen points ----
bonferroni = 0.05 / len(TEST_POINTS)
boot_rows = []
for point in TEST_POINTS:
    a = correct[("period", point)].mean(axis=0)
    b = correct[("matched", point)].mean(axis=0)
    diff, lo, hi, pval = paired_bootstrap(a, b)
    boot_rows.append({"point": point, "period": a.mean(), "matched": b.mean(), "diff": diff,
                      "ci_low": lo, "ci_high": hi, "p": pval, "passes_bonferroni": pval < bonferroni})
boot = pd.DataFrame(boot_rows)
boot.to_csv("results/period_control_bootstrap.csv", index=False)
print(f"\nperiod minus matched (Bonferroni cutoff p < {bonferroni:.4f}):")
print(boot.round(4).to_string(index=False))

# ---- 6. Plot ----
fig, ax = plt.subplots(figsize=(6.5, 4))
points = np.arange(N_POINTS)
ax.plot(points, summary["test_full_2000"], color="#257de0", lw=2, marker="o", ms=5,
        label="All examples (2,000)")
for cond, color, label in [("period", "#f47230", "Period-ending only"),
                           ("matched", "#00b894", "Size-matched random")]:
    s = long[long["condition"] == cond].groupby("point")["test_acc"].agg(["mean", "std"])
    ax.errorbar(points, s["mean"], yerr=s["std"], color=color, lw=2, marker="o", ms=5,
                capsize=3, label=label)
ax.set_xticks(points)
ax.set_xticklabels(["emb"] + [str(i) for i in range(1, N_POINTS)])
ax.set_xlabel("Residual stream point (emb = embeddings, L+1 = after layer L)")
ax.set_ylabel("Test accuracy (mean ± std, 5 runs)")
ax.set_ylim(0.4, 1.0)
ax.set_title("Training on period-ending examples only", loc="left")
ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/period_control.png", dpi=200)
print(f"\nsaved figures/period_control.png, runtime {time.time() - start:.0f} s")