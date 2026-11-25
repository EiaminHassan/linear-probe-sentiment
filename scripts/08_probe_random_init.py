"""
Step 8b: run the same CV-tuned probe on the random-init hidden states,
and compare with the pretrained probe, bag of words and the majority baseline.
Output: results/random_init_by_layer.csv, figures/pretrained_vs_random.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probe_utils import ci95, fit_with_cv, make_probe

start = time.time()

# ---- 1. Load random-model states, labels, and earlier results ----
X_train = np.load("data/hidden_random_train.npy")
X_test = np.load("data/hidden_random_test.npy")
y_train = np.load("data/labels_train.npy")
y_test = np.load("data/labels_test.npy")
n_points, n_test = X_train.shape[1], len(y_test)

pretrained = pd.read_csv("results/controls_by_layer.csv")   # from Step 7
bow_test = pretrained["bow_test"].iloc[0]
baseline = max(y_test.mean(), 1 - y_test.mean())

# ---- 2. Probe every point of the random model ----
rows = []
for point in range(n_points):
    C, cv_acc, train_acc, test_acc = fit_with_cv(make_probe, X_train[:, point], y_train,
                                                 X_test[:, point], y_test)
    rows.append({"point": point, "C": C, "cv_acc": cv_acc,
                 "train_acc": train_acc, "test_acc": test_acc})
    gap = pretrained["real_test"].iloc[point] - test_acc
    print(f"point {point:2d} | random: C={C:<6} cv={cv_acc:.3f} train={train_acc:.3f} "
          f"test={test_acc:.3f} ± {ci95(test_acc, n_test):.3f} | pretrained - random = {gap:+.3f}")

results = pd.DataFrame(rows)
results.to_csv("results/random_init_by_layer.csv", index=False)

# ---- 3. Plot ----
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(pretrained["point"], pretrained["real_test"], yerr=ci95(pretrained["real_test"], n_test),
            color="#257de0", lw=2, marker="o", ms=5, capsize=3, label="Pretrained GPT-2")
ax.errorbar(results["point"], results["test_acc"], yerr=ci95(results["test_acc"], n_test),
            color="#f47230", lw=2, marker="o", ms=5, capsize=3, label="Random-init GPT-2")
ax.axhline(bow_test, color="#00b894", lw=2, ls="--", label=f"Bag of words ({bow_test:.3f})")
ax.axhline(baseline, color="#888888", lw=1.5, ls=":", label=f"Majority class ({baseline:.3f})")

ax.set_xticks(range(n_points))
ax.set_xticklabels(["emb"] + [str(i) for i in range(1, n_points)])
ax.set_xlabel("Residual stream point (emb = embeddings, L+1 = after layer L)")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.4, 1.0)
ax.set_title("Pretrained vs random-init GPT-2", loc="left")
ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/pretrained_vs_random.png", dpi=200)

print(f"\nsaved results/random_init_by_layer.csv and figures/pretrained_vs_random.png")
print(f"runtime: {time.time() - start:.0f} seconds")