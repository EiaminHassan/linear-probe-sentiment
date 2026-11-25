"""
Shared hidden-state extraction code.
Pretrained GPT-2 small via TransformerLens 4 (TransformerBridge, compatibility mode);
random-init GPT-2 small via Hugging Face transformers.
Every function returns last-token states of shape [n_sentences, 13, 768]:
point 0 = embeddings, point L+1 = after layer L.
"""
from functools import lru_cache

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

N_POINTS = 13
D_MODEL = 768
HOOK_NAMES = ["blocks.0.hook_resid_pre"] + [f"blocks.{l}.hook_resid_post" for l in range(12)]


def load_pretrained():
    """GPT-2 small with HookedTransformer-equivalent weight processing, on CPU."""
    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    model.enable_compatibility_mode()
    return model.eval()


def extract_pretrained(model, sentences, desc="pretrained"):
    hook_set = set(HOOK_NAMES)
    states = np.zeros((len(sentences), N_POINTS, D_MODEL), dtype=np.float32)
    with torch.no_grad():
        for i, s in enumerate(tqdm(sentences, desc=desc, disable=len(sentences) < 100)):
            _, cache = model.run_with_cache(s, names_filter=lambda name: name in hook_set)
            for point, hook in enumerate(HOOK_NAMES):
                states[i, point] = cache[hook][0, -1].cpu().numpy()
    return states


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the GPT-2 tokenizer once and reuse it."""
    return GPT2TokenizerFast.from_pretrained("gpt2")


def make_random_model(seed):
    """GPT-2 small shape, random weights fixed by the seed, final LayerNorm removed."""
    torch.manual_seed(seed)
    model = GPT2Model(GPT2Config())
    model.ln_f = torch.nn.Identity()
    return model.eval()


def extract_random(model, sentences, desc="random"):
    tokenizer = get_tokenizer()
    states = np.zeros((len(sentences), N_POINTS, D_MODEL), dtype=np.float32)
    with torch.no_grad():
        for i, s in enumerate(tqdm(sentences, desc=desc, disable=len(sentences) < 100)):
            ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer(s)["input_ids"]])
            out = model(ids, output_hidden_states=True)
            for point, h in enumerate(out.hidden_states):
                states[i, point] = h[0, -1].numpy()
    return states