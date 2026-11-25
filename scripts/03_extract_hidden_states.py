"""
Step 4: run every sentence through GPT-2 and save the LAST token's hidden state
at all 13 points (point 0 = embeddings, point L+1 = after layer L).
Output: data/hidden_{split}.npy  shape [n_sentences, 13, 768]
        data/labels_{split}.npy  shape [n_sentences]
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge

SEED = 0
torch.manual_seed(SEED)
torch.set_grad_enabled(False)

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
model.enable_compatibility_mode()
model.eval()

N_LAYERS = model.cfg.n_layers
D_MODEL = model.cfg.d_model
HOOK_NAMES = ["blocks.0.hook_resid_pre"] + [f"blocks.{l}.hook_resid_post" for l in range(N_LAYERS)]
HOOK_SET = set(HOOK_NAMES)


def extract_last_token_states(sentences):
    """Return an array [n_sentences, 13, d_model] and the list of last-token strings."""
    states = np.zeros((len(sentences), len(HOOK_NAMES), D_MODEL), dtype=np.float32)
    last_tokens = []
    for i, sentence in enumerate(tqdm(sentences)):
        # Keep only the 13 activations we need.
        _, cache = model.run_with_cache(sentence, names_filter=lambda name: name in HOOK_SET)
        for point, hook in enumerate(HOOK_NAMES):
            # [0, -1] = first (only) sentence in the batch, last token position
            states[i, point] = cache[hook][0, -1].cpu().numpy()
        last_tokens.append(model.to_str_tokens(sentence)[-1])
    return states, last_tokens


for split in ["train", "test"]:
    df = pd.read_csv(f"data/probe_{split}.csv")
    print(f"\n=== {split}: {len(df)} sentences ===")
    X, last_tokens = extract_last_token_states(df["sentence"].tolist())
    y = df["label"].to_numpy()

    np.save(f"data/hidden_{split}.npy", X)
    np.save(f"data/labels_{split}.npy", y)
    print("saved hidden states:", X.shape, "labels:", y.shape)

    # Which tokens end up in the "last" position?
    print("most common last tokens:")
    print(pd.Series(last_tokens).value_counts().head(8).to_string())

    # Average vector length per point. Compare with the Step 2 table.
    mean_norms = np.linalg.norm(X, axis=-1).mean(axis=0)
    print("mean norm per point:", np.round(mean_norms, 1).tolist())

# ---- Sanity check: the saved vector must equal a fresh run without the filter ----
df_train = pd.read_csv("data/probe_train.csv")
X_train = np.load("data/hidden_train.npy")
_, full_cache = model.run_with_cache(df_train["sentence"][0])
fresh = full_cache["blocks.5.hook_resid_post"][0, -1].cpu().numpy()  # point 6
print("\nsanity check, max difference at point 6:", np.abs(fresh - X_train[0, 6]).max())