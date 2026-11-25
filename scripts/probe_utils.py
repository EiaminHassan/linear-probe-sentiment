"""
Shared probe code used by the later scripts.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
C_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]


def make_probe(C, seed=SEED):
    """Standardise features, then logistic regression with regularisation strength C."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, random_state=seed))


def fit_with_cv(make_model, X_tr, y_tr, X_te, y_te, seed=SEED):
    """Pick C by 5-fold stratified CV on training data, refit, score once on test.
    Returns (best_C, cv_acc, train_acc, test_acc)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {C: cross_val_score(make_model(C), X_tr, y_tr, cv=cv).mean() for C in C_VALUES}
    best_C = max(scores, key=scores.get)
    model = make_model(best_C).fit(X_tr, y_tr)
    return best_C, scores[best_C], model.score(X_tr, y_tr), model.score(X_te, y_te)


def ci95(acc, n):
    """Normal-approximation 95% interval half-width for an accuracy measured on n examples."""
    return 1.96 * np.sqrt(acc * (1 - acc) / n)


def file_paths(k):
    """File names for run k. Run 0 uses the original names, runs 1-4 use _s{k}."""
    if k == 0:
        return {"text": "data/probe_train.csv", "labels": "data/labels_train.npy",
                "pre_train": "data/hidden_train.npy",
                "rand_train": "data/hidden_random_train.npy",
                "rand_test": "data/hidden_random_test.npy"}
    return {"text": f"data/probe_train_s{k}.csv", "labels": f"data/labels_train_s{k}.npy",
            "pre_train": f"data/hidden_train_s{k}.npy",
            "rand_train": f"data/hidden_random_train_s{k}.npy",
            "rand_test": f"data/hidden_random_test_s{k}.npy"}


def paired_bootstrap(a, b, n_boot=10_000, seed=SEED):
    """a, b: per-sentence scores on the same test sentences.
    Returns (difference in means, 95% low, 95% high, two-sided p)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diff = a - b
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return diff.mean(), lo, hi, min(p, 1.0)