"""
Checks that the shared helper code reproduces saved results exactly.
Run after any refactor. Expected output: all differences 0.0.
"""
import numpy as np
import pandas as pd

from extract_utils import extract_pretrained, extract_random, load_pretrained, make_random_model
from probe_utils import fit_with_cv, make_probe

N_CHECK = 5
train_text = pd.read_csv("data/probe_train.csv")["sentence"].tolist()[:N_CHECK]
test_text = pd.read_csv("data/probe_test.csv")["sentence"].tolist()[:N_CHECK]

# 1. Extraction, pretrained model
model = load_pretrained()
for name, sents, path in [("train", train_text, "data/hidden_train.npy"),
                          ("test", test_text, "data/hidden_test.npy")]:
    diff = np.abs(extract_pretrained(model, sents) - np.load(path)[:N_CHECK]).max()
    print(f"pretrained extraction ({name}), max difference: {diff}")

# 2. Extraction, random-init model with seed 0
random_model = make_random_model(seed=0)
diff = np.abs(extract_random(random_model, test_text) - np.load("data/hidden_random_test.npy")[:N_CHECK]).max()
print(f"random-init extraction (test), max difference: {diff}")

# 3. One probe, compared with the Step 7 result at point 7
X_train, y_train = np.load("data/hidden_train.npy"), np.load("data/labels_train.npy")
X_test, y_test = np.load("data/hidden_test.npy"), np.load("data/labels_test.npy")
_, _, _, test_acc = fit_with_cv(make_probe, X_train[:, 7], y_train, X_test[:, 7], y_test, seed=0)
expected = pd.read_csv("results/controls_by_layer.csv").loc[7, "real_test"]
print(f"probe at point 7: {test_acc:.4f}, saved {expected:.4f}, difference {abs(test_acc - expected)}")