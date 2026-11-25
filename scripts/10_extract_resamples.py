"""
Step 9b: hidden states for training samples 1-4.
  - pretrained GPT-2 (TransformerBridge, compatibility mode) on each new training sample
  - a NEW random-init GPT-2 per sample k (seed k), on training sample k AND the test set
Skips files that already exist, so the script can be re-run after an interruption.
"""
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

torch.set_grad_enabled(False)
NEW_SAMPLES = [1, 2, 3, 4]

# ---- Pretrained model, same as Step 4 ----
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
bridge.enable_compatibility_mode()
bridge.eval()
HOOK_NAMES = ["blocks.0.hook_resid_pre"] + [f"blocks.{l}.hook_resid_post" for l in range(12)]
HOOK_SET = set(HOOK_NAMES)


def extract_pretrained(sentences):
    states = np.zeros((len(sentences), 13, 768), dtype=np.float32)
    for i, s in enumerate(tqdm(sentences, desc="pretrained")):
        _, cache = bridge.run_with_cache(s, names_filter=lambda name: name in HOOK_SET)
        for point, hook in enumerate(HOOK_NAMES):
            states[i, point] = cache[hook][0, -1].cpu().numpy()
    return states


# ---- Random-init model, same as Step 8 but with a chosen seed ----
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def make_random_model(seed):
    torch.manual_seed(seed)                 # seed decides the random weights
    model = GPT2Model(GPT2Config())
    model.ln_f = torch.nn.Identity()
    return model.eval()


def tokenize(sentence):
    return torch.tensor([[tokenizer.bos_token_id] + tokenizer(sentence)["input_ids"]])


def extract_random(model, sentences):
    states = np.zeros((len(sentences), 13, 768), dtype=np.float32)
    for i, s in enumerate(tqdm(sentences, desc="random")):
        out = model(tokenize(s), output_hidden_states=True)
        for point, h in enumerate(out.hidden_states):
            states[i, point] = h[0, -1].numpy()
    return states


def save_if_missing(path, make_array):
    """Only compute and save if the file does not exist yet."""
    if os.path.exists(path):
        print("skip (exists):", path)
        return
    np.save(path, make_array())
    print("saved:", path)


test_sentences = pd.read_csv("data/probe_test.csv")["sentence"].tolist()

# ---- Reproducibility check: seed 0 must rebuild the Step 8 random model exactly ----
check_model = make_random_model(seed=0)
fresh = extract_random(check_model, test_sentences[:1])
saved = np.load("data/hidden_random_test.npy")[:1]
print("seed-0 reproducibility, max difference:", np.abs(fresh - saved).max())

# ---- Main loop over the new samples ----
for k in NEW_SAMPLES:
    print(f"\n=== sample {k} ===")
    df = pd.read_csv(f"data/probe_train_s{k}.csv")
    sentences = df["sentence"].tolist()
    np.save(f"data/labels_train_s{k}.npy", df["label"].to_numpy())

    save_if_missing(f"data/hidden_train_s{k}.npy", lambda: extract_pretrained(sentences))

    random_model = make_random_model(seed=k)
    save_if_missing(f"data/hidden_random_train_s{k}.npy",
                    lambda: extract_random(random_model, sentences))
    save_if_missing(f"data/hidden_random_test_s{k}.npy",
                    lambda: extract_random(random_model, test_sentences))