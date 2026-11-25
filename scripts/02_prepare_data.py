"""
Step 3: build a balanced training set and a test set from SST-2.
Output: data/probe_train.csv and data/probe_test.csv (columns: sentence, label).
"""
import pandas as pd
from datasets import load_dataset

SEED = 0
N_PER_CLASS = 1000   # training sentences per class
MIN_WORDS = 6        # drop very short phrase fragments from the training pool

# ---- 1. Download SST-2 and look at the raw splits ----
ds = load_dataset("stanfordnlp/sst2")
print(ds)

train = ds["train"].to_pandas()
val = ds["validation"].to_pandas()
test = ds["test"].to_pandas()

for name, df in [("train", train), ("validation", val), ("test", test)]:
    print(f"{name}: {len(df)} rows, label counts = {df['label'].value_counts().to_dict()}")

# Show a few raw examples with repr() so hidden spaces are visible.
print("\nRaw training examples:")
for s in train["sentence"].sample(5, random_state=SEED):
    print("   ", repr(s))

# ---- 2. Clean: strip whitespace and count words ----
for df in (train, val):
    df["sentence"] = df["sentence"].str.strip()
    df["n_words"] = df["sentence"].str.split().str.len()

print("\nWords per example, train split:")
print(train["n_words"].describe().round(1))
print("\nWords per example, validation split:")
print(val["n_words"].describe().round(1))

# ---- 3. Filter the training pool ----
pool = train[train["n_words"] >= MIN_WORDS]
pool = pool.drop_duplicates(subset="sentence")
# Remove any training sentence that also appears in the test set (leakage check).
test_sentences = set(val["sentence"].str.lower())
overlap = pool["sentence"].str.lower().isin(test_sentences)
print(f"\nTraining sentences that also appear in the test set: {overlap.sum()}")
pool = pool[~overlap]
print("Pool size after filtering:", len(pool), pool["label"].value_counts().to_dict())

# ---- 4. Balanced random sample, then shuffle ----
probe_train = pool.groupby("label", group_keys=False).sample(n=N_PER_CLASS, random_state=SEED)
probe_train = probe_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
probe_test = val.reset_index(drop=True)

# ---- 5. Save and summarise ----
probe_train[["sentence", "label"]].to_csv("data/probe_train.csv", index=False)
probe_test[["sentence", "label"]].to_csv("data/probe_test.csv", index=False)

print("\nSaved probe_train:", len(probe_train), probe_train["label"].value_counts().to_dict())
print("Saved probe_test: ", len(probe_test), probe_test["label"].value_counts().to_dict())
print("\nCleaned training examples:")
for s, y in probe_train[["sentence", "label"]].head(5).values:
    print(f"    label={y}  {s!r}")