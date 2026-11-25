"""
Step 6: train a probe at every point (0 = embeddings, L+1 = after layer L).
C is chosen by 5-fold cross-validation on the training set only;
the test set is scored once per point.
Output: results/probe_by_layer.csv and figures/probe_accuracy_by_layer.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]   # smaller C = stronger regularisation

# ---- 1. Load data ----
X_train = np.load("data/hidden_train.npy")
y_train = np.load("data/labels_train.npy")
X_test = np.load("data/hidden_test.npy")
y_test = np.load("data/labels_test.npy")
n_points = X_train.shape[1]   # 13

baseline = max(y_test.mean(), 1 - y_test.mean())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_probe(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, random_state=SEED),
    )


# ---- 2. Loop over every point ----
rows = []
for point in range(n_points):
    Xtr = X_train[:, point]
    Xte = X_test[:, point]

    # Cross-validation on the training set only: one mean score per C value.
    cv_scores = {C: cross_val_score(make_probe(C), Xtr, y_train, cv=cv).mean() for C in C_VALUES}
    best_C = max(cv_scores, key=cv_scores.get)

    # Retrain on all 2,000 training sentences with the chosen C, then score once on test.
    probe = make_probe(best_C).fit(Xtr, y_train)
    train_acc = probe.score(Xtr, y_train)
    test_acc = probe.score(Xte, y_test)
    ci95 = 1.96 * np.sqrt(test_acc * (1 - test_acc) / len(y_test))

    rows.append({"point": point, "best_C": best_C, "cv_acc": cv_scores[best_C],
                 "train_acc": train_acc, "test_acc": test_acc, "ci95": ci95})
    print(f"point {point:2d}  C={best_C:<6}  cv={cv_scores[best_C]:.3f}  "
          f"train={train_acc:.3f}  test={test_acc:.3f} ± {ci95:.3f}")

results = pd.DataFrame(rows)
results.to_csv("results/probe_by_layer.csv", index=False)

# ---- 3. Plot ----
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(results["point"], results["train_acc"], color="#f47230", lw=2,
        marker="o", ms=5, label="Train accuracy")
ax.errorbar(results["point"], results["test_acc"], yerr=results["ci95"], color="#257de0",
            lw=2, marker="o", ms=5, capsize=3, label="Test accuracy (95% CI)")
ax.axhline(baseline, color="#888888", lw=1.5, ls="--",
           label=f"Majority baseline ({baseline:.3f})")

ax.set_xticks(range(n_points))
ax.set_xticklabels(["emb"] + [str(i) for i in range(1, n_points)])
ax.set_xlabel("Residual stream point (emb = embeddings, L+1 = after layer L)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.4, 1.02)
ax.set_title("Sentiment probe accuracy across GPT-2 small", loc="left")
ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig("figures/probe_accuracy_by_layer.png", dpi=200)
print("\nsaved results/probe_by_layer.csv and figures/probe_accuracy_by_layer.png")