"""
Step 9a: build 4 new balanced training samples (1,000 pos + 1,000 neg each),
with no sentence shared between any two samples or with the original sample.
Same filtering as Step 3. Output: data/probe_train_s1.csv ... data/probe_train_s4.csv
"""
import pandas as pd
from datasets import load_dataset

SEED = 0
N_PER_CLASS = 1000
MIN_WORDS = 6
NEW_SAMPLES = [1, 2, 3, 4]   # sample 0 is the original data/probe_train.csv

# ---- 1. Rebuild the filtered pool exactly as in Step 3 ----
ds = load_dataset("stanfordnlp/sst2")
train = ds["train"].to_pandas()
val = ds["validation"].to_pandas()
for df in (train, val):
    df["sentence"] = df["sentence"].str.strip()
train["n_words"] = train["sentence"].str.split().str.len()

pool = train[train["n_words"] >= MIN_WORDS].drop_duplicates(subset="sentence")
test_sentences = set(val["sentence"].str.lower())
pool = pool[~pool["sentence"].str.lower().isin(test_sentences)]

# ---- 2. Remove the 2,000 sentences already used in sample 0 ----
original = pd.read_csv("data/probe_train.csv")
pool = pool[~pool["sentence"].isin(set(original["sentence"]))]
print("pool after removing sample 0:", len(pool), pool["label"].value_counts().to_dict())

# ---- 3. Shuffle each class once, then cut consecutive blocks -> disjoint samples ----
shuffled = pool.sample(frac=1, random_state=SEED + 1)
neg = shuffled[shuffled["label"] == 0]
pos = shuffled[shuffled["label"] == 1]

for k in NEW_SAMPLES:
    lo, hi = (k - 1) * N_PER_CLASS, k * N_PER_CLASS
    sample = pd.concat([neg.iloc[lo:hi], pos.iloc[lo:hi]])
    sample = sample.sample(frac=1, random_state=k).reset_index(drop=True)   # mix the two classes
    sample[["sentence", "label"]].to_csv(f"data/probe_train_s{k}.csv", index=False)
    print(f"sample {k}: {len(sample)} rows, {sample['label'].value_counts().to_dict()}")

# ---- 4. Check: no sentence appears in two samples ----
sets = [set(original["sentence"])]
sets += [set(pd.read_csv(f"data/probe_train_s{k}.csv")["sentence"]) for k in NEW_SAMPLES]
shared = sum(len(a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
print("sentences shared between any two samples:", shared)