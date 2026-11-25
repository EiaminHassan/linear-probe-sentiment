"""
Step 12: negation minimal pairs.
Apply the probes trained in Step 9 (5 runs x 13 points, pretrained and random-init),
and the bag-of-words models, to pairs like "the plot was good ." / "the plot was not good .".
Measure how far adding "not" moves the probe score, and how often it flips the prediction.
Output: data/negation_pairs.csv, results/negation_long.csv, figures/negation_shift.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformer_lens.model_bridge import TransformerBridge
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

from probe_utils import file_paths, make_probe

start = time.time()
torch.set_grad_enabled(False)
SAMPLES = [0, 1, 2, 3, 4]
N_POINTS = 13

# ---- 1. Build the minimal pairs from a template ----
SUBJECTS = ["the movie", "the acting", "the plot", "the script", "the ending"]
POSITIVE = ["good", "great", "funny", "charming"]
NEGATIVE = ["bad", "boring", "dull", "awful"]

rows = []
for subject in SUBJECTS:
    for base, adjectives in [("positive", POSITIVE), ("negative", NEGATIVE)]:
        for adj in adjectives:
            rows.append({"subject": subject, "adjective": adj, "base": base,
                         "affirmative": f"{subject} was {adj} .",
                         "negated": f"{subject} was not {adj} ."})
pairs = pd.DataFrame(rows)
pairs.to_csv("data/negation_pairs.csv", index=False)
print(len(pairs), "pairs:", pairs["base"].value_counts().to_dict())
print(pairs[["affirmative", "negated"]].head(3).to_string(index=False))

aff_text = pairs["affirmative"].tolist()
neg_text = pairs["negated"].tolist()
is_pos_base = (pairs["base"] == "positive").to_numpy()

# ---- 2. Extract last-token hidden states (same methods as Steps 4 and 8) ----
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
bridge.enable_compatibility_mode()
bridge.eval()
HOOK_NAMES = ["blocks.0.hook_resid_pre"] + [f"blocks.{l}.hook_resid_post" for l in range(12)]
HOOK_SET = set(HOOK_NAMES)


def extract_pretrained(sentences):
    states = np.zeros((len(sentences), N_POINTS, 768), dtype=np.float32)
    for i, s in enumerate(sentences):
        _, cache = bridge.run_with_cache(s, names_filter=lambda name: name in HOOK_SET)
        for point, hook in enumerate(HOOK_NAMES):
            states[i, point] = cache[hook][0, -1].cpu().numpy()
    return states


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def make_random_model(seed):
    torch.manual_seed(seed)
    model = GPT2Model(GPT2Config())
    model.ln_f = torch.nn.Identity()
    return model.eval()


def extract_random(model, sentences):
    states = np.zeros((len(sentences), N_POINTS, 768), dtype=np.float32)
    for i, s in enumerate(sentences):
        ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer(s)["input_ids"]])
        out = model(ids, output_hidden_states=True)
        for point, h in enumerate(out.hidden_states):
            states[i, point] = h[0, -1].numpy()
    return states


pre_aff, pre_neg = extract_pretrained(aff_text), extract_pretrained(neg_text)
rand_aff, rand_neg = {}, {}
for k in SAMPLES:
    m = make_random_model(seed=k)          # same random model as run k in Step 9
    rand_aff[k], rand_neg[k] = extract_random(m, aff_text), extract_random(m, neg_text)
print(f"extraction done after {time.time() - start:.0f} s")


# ---- 3. Scoring helpers ----
def make_bow(C):
    return make_pipeline(CountVectorizer(binary=True, token_pattern=r"\S+"),
                         LogisticRegression(C=C, max_iter=5000, random_state=0))


def score_pairs(model, X_aff, X_neg, X_test_input):
    """Shift in probe score from adding 'not', scaled by the score spread on the SST test set.
    Also returns predicted labels for the plain and negated sentences."""
    scale = model.decision_function(X_test_input).std()
    d_aff = model.decision_function(X_aff)
    d_neg = model.decision_function(X_neg)
    return (d_neg - d_aff) / scale, (d_aff > 0).astype(int), (d_neg > 0).astype(int)


def summarise(shift, pred_aff, pred_neg):
    expected_aff = is_pos_base.astype(int)        # "good" sentences should be predicted 1
    return {
        "aff_acc": (pred_aff == expected_aff).mean(),
        "shift_pos_base": shift[is_pos_base].mean(),     # good -> not good: want below 0
        "shift_neg_base": shift[~is_pos_base].mean(),    # bad -> not bad: want above 0
        "flip_pos_base": ((pred_aff == 1) & (pred_neg == 0))[is_pos_base].mean(),
        "flip_neg_base": ((pred_aff == 0) & (pred_neg == 1))[~is_pos_base].mean(),
    }


# ---- 4. Apply every Step 9 probe, plus bag of words, in every run ----
long = pd.read_csv("results/resamples_long.csv")
bow_runs = pd.read_csv("results/resamples_bow.csv")
X_test_pre = np.load("data/hidden_test.npy")
test_text = pd.read_csv("data/probe_test.csv")["sentence"]

results = []
for k in SAMPLES:
    p = file_paths(k)
    y_train = np.load(p["labels"])
    setups = [("pretrained", np.load(p["pre_train"]), pre_aff, pre_neg, X_test_pre),
              ("random", np.load(p["rand_train"]), rand_aff[k], rand_neg[k], np.load(p["rand_test"]))]
    for model_name, X_tr, A, N, X_te in setups:
        for point in range(N_POINTS):
            row = long[(long["sample"] == k) & (long["model"] == model_name) & (long["point"] == point)]
            probe = make_probe(row["C"].item()).fit(X_tr[:, point], y_train)
            out = summarise(*score_pairs(probe, A[:, point], N[:, point], X_te[:, point]))
            results.append({"sample": k, "model": model_name, "point": point, **out})

    C_bow = bow_runs.loc[bow_runs["sample"] == k, "C"].item()
    bow = make_bow(C_bow).fit(pd.read_csv(p["text"])["sentence"], y_train)
    vocab = bow.named_steps["countvectorizer"].vocabulary_
    missing = sorted({w for s in aff_text + neg_text for w in s.split()} - set(vocab))
    not_weight = bow.named_steps["logisticregression"].coef_[0][vocab["not"]]
    print(f"run {k}: words missing from BoW vocabulary: {missing}; BoW weight of 'not' = {not_weight:+.3f}")
    out = summarise(*score_pairs(bow, aff_text, neg_text, test_text))
    results.append({"sample": k, "model": "bow", "point": -1, **out})

res = pd.DataFrame(results)
res.to_csv("results/negation_long.csv", index=False)

# ---- 5. Summaries (mean over 5 runs) ----
metrics = ["aff_acc", "shift_pos_base", "shift_neg_base", "flip_pos_base", "flip_neg_base"]
summary = res.groupby(["model", "point"])[metrics].mean()
for name in ["pretrained", "random"]:
    print(f"\n{name} (mean over 5 runs):")
    print(summary.loc[name].round(3).to_string())
print("\nbag of words (mean over 5 runs):")
print(summary.loc["bow"].round(3).to_string(index=False))

# ---- 6. Plot: one panel per direction ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
panels = [(axes[0], "shift_pos_base", '"good" to "not good" (should go down)'),
          (axes[1], "shift_neg_base", '"bad" to "not bad" (should go up)')]
for ax, col, title in panels:
    for model_name, color, label in [("pretrained", "#257de0", "Pretrained GPT-2"),
                                     ("random", "#f47230", "Random-init GPT-2")]:
        s = res[res["model"] == model_name].groupby("point")[col].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], color=color, lw=2,
                    marker="o", ms=5, capsize=3, label=label)
    bow_mean = res.loc[res["model"] == "bow", col].mean()
    ax.axhline(bow_mean, color="#00b894", lw=2, ls="--", label=f"Bag of words ({bow_mean:+.2f})")
    ax.axhline(0, color="#888888", lw=1)
    ax.set_xticks(range(N_POINTS))
    ax.set_xticklabels(["emb"] + [str(i) for i in range(1, N_POINTS)])
    ax.set_xlabel("Residual stream point")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)
axes[0].set_ylabel("Score change from adding 'not'\n(test-set standard deviations)")
fig.tight_layout()
fig.savefig("figures/negation_shift.png", dpi=200)
print(f"\nsaved figures/negation_shift.png, runtime {time.time() - start:.0f} s")