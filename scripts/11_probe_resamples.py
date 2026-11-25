"""
Step 9c: probe all 13 points for pretrained and random-init GPT-2, plus bag of words,
in 5 runs (training sample k, random-init seed k, CV seed k). Report mean and spread.
Output: results/resamples_long.csv, results/resamples_summary.csv,
        results/resamples_bow.csv, figures/probe_resamples.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from probe_utils import fit_with_cv, make_probe

start = time.time()
SAMPLES = [0, 1, 2, 3, 4]
N_POINTS = 13

y_test = np.load("data/labels_test.npy")
X_test_pre = np.load("data/hidden_test.npy")              # pretrained test features, same for every run
test_text = pd.read_csv("data/probe_test.csv")["sentence"]
baseline = max(y_test.mean(), 1 - y_test.mean())


def make_bow(C):
    return make_pipeline(CountVectorizer(binary=True, token_pattern=r"\S+"),
                         LogisticRegression(C=C, max_iter=5000, random_state=0))


def file_paths(k):
    """Sample 0 uses the original file names; samples 1-4 use the _s{k} names."""
    if k == 0:
        return {"text": "data/probe_train.csv", "labels": "data/labels_train.npy",
                "pre_train": "data/hidden_train.npy",
                "rand_train": "data/hidden_random_train.npy", "rand_test": "data/hidden_random_test.npy"}
    return {"text": f"data/probe_train_s{k}.csv", "labels": f"data/labels_train_s{k}.npy",
            "pre_train": f"data/hidden_train_s{k}.npy",
            "rand_train": f"data/hidden_random_train_s{k}.npy",
            "rand_test": f"data/hidden_random_test_s{k}.npy"}


# ---- 1. Run every sample ----
rows, bow_rows = [], []
for k in SAMPLES:
    p = file_paths(k)
    y_train = np.load(p["labels"])
    models = {
        "pretrained": (np.load(p["pre_train"]), X_test_pre),
        "random": (np.load(p["rand_train"]), np.load(p["rand_test"])),
    }
    for name, (X_tr, X_te) in models.items():
        for point in range(N_POINTS):
            C, cv_acc, train_acc, test_acc = fit_with_cv(
                make_probe, X_tr[:, point], y_train, X_te[:, point], y_test, seed=k)
            rows.append({"sample": k, "model": name, "point": point, "C": C,
                         "cv_acc": cv_acc, "train_acc": train_acc, "test_acc": test_acc})

    train_text = pd.read_csv(p["text"])["sentence"]
    C, _, _, bow_acc = fit_with_cv(make_bow, train_text, y_train, test_text, y_test, seed=k)
    bow_rows.append({"sample": k, "C": C, "test_acc": bow_acc})
    print(f"sample {k} done after {time.time() - start:.0f} s, bag of words test = {bow_acc:.3f}")

long = pd.DataFrame(rows)
bow = pd.DataFrame(bow_rows)
long.to_csv("results/resamples_long.csv", index=False)
bow.to_csv("results/resamples_bow.csv", index=False)

# ---- 2. Reproducibility check: sample 0 must equal Steps 7 and 8 exactly ----
s0 = long[long["sample"] == 0]
step7 = pd.read_csv("results/controls_by_layer.csv")["real_test"].to_numpy()
step8 = pd.read_csv("results/random_init_by_layer.csv")["test_acc"].to_numpy()
print("\nsample 0 vs Step 7 (pretrained), max difference:",
      np.abs(s0[s0["model"] == "pretrained"]["test_acc"].to_numpy() - step7).max())
print("sample 0 vs Step 8 (random), max difference:",
      np.abs(s0[s0["model"] == "random"]["test_acc"].to_numpy() - step8).max())

# ---- 3. Summary: mean, standard deviation, min, max over the 5 runs ----
summary = (long.groupby(["model", "point"])["test_acc"]
           .agg(["mean", "std", "min", "max"]).reset_index())
summary.to_csv("results/resamples_summary.csv", index=False)
for name in ["pretrained", "random"]:
    print(f"\n{name} (test accuracy over {len(SAMPLES)} runs):")
    print(summary[summary["model"] == name].drop(columns="model").round(3).to_string(index=False))

print(f"\nbag of words: mean {bow['test_acc'].mean():.3f}, std {bow['test_acc'].std():.3f}, "
      f"runs {bow['test_acc'].round(3).tolist()}")

# Which point is best in each run? Tests whether "peak at point 9" holds up.
pre = long[long["model"] == "pretrained"]
best = pre.loc[pre.groupby("sample")["test_acc"].idxmax(), ["sample", "point", "test_acc"]]
print("\nbest pretrained point in each run:")
print(best.round(3).to_string(index=False))

# ---- 4. Plot mean +/- std ----
fig, ax = plt.subplots(figsize=(6.5, 4))
for name, color, label in [("pretrained", "#257de0", "Pretrained GPT-2"),
                           ("random", "#f47230", "Random-init GPT-2")]:
    s = summary[summary["model"] == name]
    ax.errorbar(s["point"], s["mean"], yerr=s["std"], color=color, lw=2,
                marker="o", ms=5, capsize=3, label=label)
bow_mean, bow_std = bow["test_acc"].mean(), bow["test_acc"].std()
ax.axhspan(bow_mean - bow_std, bow_mean + bow_std, color="#00b894", alpha=0.15, lw=0)
ax.axhline(bow_mean, color="#00b894", lw=2, ls="--", label=f"Bag of words ({bow_mean:.3f})")
ax.axhline(baseline, color="#888888", lw=1.5, ls=":", label=f"Majority class ({baseline:.3f})")

ax.set_xticks(range(N_POINTS))
ax.set_xticklabels(["emb"] + [str(i) for i in range(1, N_POINTS)])
ax.set_xlabel("Residual stream point (emb = embeddings, L+1 = after layer L)")
ax.set_ylabel("Test accuracy (mean ± std, 5 runs)")
ax.set_ylim(0.4, 1.0)
ax.set_title("Probe accuracy over 5 training samples", loc="left")
ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/probe_resamples.png", dpi=200)
print(f"\nsaved figures/probe_resamples.png, total runtime {time.time() - start:.0f} s")