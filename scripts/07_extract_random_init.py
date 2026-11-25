"""
Step 8a: build GPT-2 small with RANDOM weights (same architecture, never trained)
and save last-token hidden states at the same 13 points as before.
Output: data/hidden_random_train.npy, data/hidden_random_test.npy
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

SEED = 0
torch.manual_seed(SEED)          # makes the random weights identical on every run
torch.set_grad_enabled(False)

# ---- 1. Same tokenizer as before, random-weight model ----
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")   # only the tokenizer is pretrained
config = GPT2Config()                                   # default config = GPT-2 small shape
model = GPT2Model(config)                               # built from config -> random weights
model.ln_f = torch.nn.Identity()  # drop the final LayerNorm so point 12 matches our pretrained point 12
model.eval()
print("layers:", config.n_layer, "d_model:", config.n_embd, "heads:", config.n_head)


def tokenize(sentence):
    """Add <|endoftext|> at the start, like TransformerLens did. Returns shape [1, seq_len]."""
    ids = [tokenizer.bos_token_id] + tokenizer(sentence)["input_ids"]
    return torch.tensor([ids])


# ---- 2. Check the token format matches Step 2 ----
example = tokenize("This film was a complete waste of time.")
print("tokens:", tokenizer.convert_ids_to_tokens(example[0]))


# ---- 3. Extract last-token hidden states at all 13 points ----
def extract(sentences):
    states = np.zeros((len(sentences), config.n_layer + 1, config.n_embd), dtype=np.float32)
    for i, sentence in enumerate(tqdm(sentences)):
        out = model(tokenize(sentence), output_hidden_states=True)
        # out.hidden_states is a tuple of 13 tensors, each [1, seq_len, 768]:
        # index 0 = embeddings, index L+1 = after layer L
        for point, h in enumerate(out.hidden_states):
            states[i, point] = h[0, -1].numpy()
    return states


for split in ["train", "test"]:
    df = pd.read_csv(f"data/probe_{split}.csv")
    print(f"\n=== {split}: {len(df)} sentences ===")
    X = extract(df["sentence"].tolist())
    np.save(f"data/hidden_random_{split}.npy", X)
    print("saved:", X.shape)
    print("mean norm per point:", np.round(np.linalg.norm(X, axis=-1).mean(axis=0), 1).tolist())