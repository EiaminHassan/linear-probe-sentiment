"""
Step 7: controls that use only saved data.
  A. GPT-2 probe again, with a wider C grid (adds 0.0001).
  B. Shuffled-label control: train on randomly permuted labels, test on true labels.
  C. Bag-of-words baseline: logistic regression on which words appear, no GPT-2.
Output: results/controls_by_layer.csv, figures/probe_with_controls.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
C_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# ---- 1. Load data ----
X_train = np.load("data/hidden_train.npy")
y_train = np.load("data/labels_train.npy")
X_test = np.load("data/hidden_test.npy")
y_test = np.load("data/labels_test.npy")
train_text = pd.read_csv("data/probe_train.csv")["sentence"]
test_text = pd.read_csv("data/probe_test.csv")["sentence"]
n_points = X_train.shape[1]
n_test = len(y_test)
baseline = max(y_test.mean(), 1 - y_test.mean())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def fit_with_cv(make_model, X_tr, y_tr, X_te, y_te):
    """Pick C by 5-fold CV on the training data, refit, score once on test."""
    scores = {C: cross_val_score(make_model(C), X_tr, y_tr, cv=cv).mean() for C in C_VALUES}
    best_C = max(scores, key=scores.get)
    model = make_model(best_C).fit(X_tr, y_tr)
    return best_C, model.score(X_tr, y_tr), model.score(X_te, y_te)


def make_probe(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, random_state=SEED))


def make_bow(C):
    # binary=True: 1 if the word appears, 0 if not. token_pattern splits on spaces only.
    return make_pipeline(CountVectorizer(binary=True, token_pattern=r"\S+"),
                         LogisticRegression(C=C, max_iter=5000, random_state=SEED))


def ci95(acc):
    return 1.96 * np.sqrt(acc * (1 - acc) / n_test)


# ---- 2. Shuffled labels: same sentences, labels in random order ----
rng = np.random.default_rng(SEED)
y_shuffled = rng.permutation(y_train)
print("shuffled labels still balanced:", np.bincount(y_shuffled))
print("share of shuffled labels that match the real ones:", (y_shuffled == y_train).mean().round(3))

# ---- 3. Loop over points: real probe and shuffled-label probe ----
rows = []
for point in range(n_points):
    C_real, tr_real, te_real = fit_with_cv(make_probe, X_train[:, point], y_train,
                                           X_test[:, point], y_test)
    C_shuf, tr_shuf, te_shuf = fit_with_cv(make_probe, X_train[:, point], y_shuffled,
                                           X_test[:, point], y_test)
    rows.append({"point": point,
                 "real_C": C_real, "real_train": tr_real, "real_test": te_real,
                 "shuf_C": C_shuf, "shuf_train": tr_shuf, "shuf_test": te_shuf})
    print(f"point {point:2d} | real: C={C_real:<6} train={tr_real:.3f} test={te_real:.3f}"
          f" | shuffled: C={C_shuf:<6} train={tr_shuf:.3f} test={te_shuf:.3f}")

results = pd.DataFrame(rows)

# ---- 4. Bag-of-words baseline (one number, no layers) ----
C_bow, tr_bow, te_bow = fit_with_cv(make_bow, train_text, y_train, test_text, y_test)
print(f"\nbag-of-words: C={C_bow} train={tr_bow:.3f} test={te_bow:.3f} ± {ci95(te_bow):.3f}")
results["bow_test"] = te_bow
results.to_csv("results/controls_by_layer.csv", index=False)

# ---- 5. Plot: test accuracy only, with controls ----
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(results["point"], results["real_test"], yerr=ci95(results["real_test"]),
            color="#257de0", lw=2, marker="o", ms=5, capsize=3, label="GPT-2 probe")
ax.errorbar(results["point"], results["shuf_test"], yerr=ci95(results["shuf_test"]),
            color="#f47230", lw=2, marker="o", ms=5, capsize=3, label="Shuffled-label probe")
ax.axhline(te_bow, color="#00b894", lw=2, ls="--", label=f"Bag of words ({te_bow:.3f})")
ax.axhline(baseline, color="#888888", lw=1.5, ls=":", label=f"Majority class ({baseline:.3f})")

ax.set_xticks(range(n_points))
ax.set_xticklabels(["emb"] + [str(i) for i in range(1, n_points)])
ax.set_xlabel("Residual stream point (emb = embeddings, L+1 = after layer L)")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.4, 1.0)
ax.set_title("Sentiment probe against controls", loc="left")
ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/probe_with_controls.png", dpi=200)
print("saved results/controls_by_layer.csv and figures/probe_with_controls.png")