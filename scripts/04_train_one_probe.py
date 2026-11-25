"""
Step 5: train a logistic regression probe on the hidden states at one point
and compare it with the majority-class baseline.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0

# ---- 1. Load the saved hidden states and labels ----
X_train = np.load("data/hidden_train.npy")   # [2000, 13, 768]
y_train = np.load("data/labels_train.npy")   # [2000]
X_test = np.load("data/hidden_test.npy")     # [872, 13, 768]
y_test = np.load("data/labels_test.npy")     # [872]

# ---- 2. Baseline: always guess the most common class in the test set ----
share_positive = y_test.mean()
baseline = max(share_positive, 1 - share_positive)
print(f"majority-class baseline on test: {baseline:.3f}")


# ---- 3. A function that trains and scores one probe ----
def train_probe(point, C=1.0):
    """Fit scaler + logistic regression on one point. Return the probe and two accuracies."""
    probe = make_pipeline(
        StandardScaler(),                     # learns mean/spread from training data only
        LogisticRegression(C=C, max_iter=2000, random_state=SEED),
    )
    probe.fit(X_train[:, point], y_train)    # X_train[:, point] has shape [2000, 768]
    train_acc = probe.score(X_train[:, point], y_train)
    test_acc = probe.score(X_test[:, point], y_test)
    return probe, train_acc, test_acc


# ---- 4. Run it at point 0 (embeddings only) and point 6 (middle) ----
for point in [0, 6]:
    probe, train_acc, test_acc = train_probe(point)
    print(f"\npoint {point}: train accuracy = {train_acc:.3f}, test accuracy = {test_acc:.3f}")

    # Confusion matrix: rows = true label, columns = predicted label.
    preds = probe.predict(X_test[:, point])
    print("confusion matrix [[true neg, false pos], [false neg, true pos]]:")
    print(confusion_matrix(y_test, preds))