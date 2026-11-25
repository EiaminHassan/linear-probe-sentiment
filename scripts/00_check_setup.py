"""
Step 1: sanity check.
Confirms that the libraries are installed, GPT-2 small loads on CPU,
and the model makes sensible next-token predictions.
"""
import random
from importlib.metadata import version

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

# 1. Fix random seeds so every run gives the same result.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# We never train GPT-2 in this project, so we switch off gradient tracking.
# This saves memory and time.
torch.set_grad_enabled(False)

# 2. Print library versions. Paste these into your README later.
for pkg in ["torch", "transformer-lens", "scikit-learn", "datasets"]:
    print(f"{pkg}: {version(pkg)}")

# 3. Load GPT-2 small on the CPU through TransformerLens 4's TransformerBridge.
#    The first run downloads about 500 MB.
model = TransformerBridge.boot_transformers("gpt2", device="cpu")

# Use the same weight layout as the classic HookedTransformer,
# so our hidden states match most tutorials and papers.
model.enable_compatibility_mode()
model.eval()  # evaluation mode: turns off dropout

# 4. Print the model's size settings.
print("number of layers:", model.cfg.n_layers)
print("hidden size (d_model):", model.cfg.d_model)
print("attention heads per layer:", model.cfg.n_heads)

# 5. See how a sentence is split into tokens.
prompt = "The movie was absolutely"
print("tokens:", model.to_str_tokens(prompt))

# 6. Run the model and look at its top 5 guesses for the next token.
logits = model(prompt)            # shape: [batch, sequence_length, vocab_size]
print("logits shape:", tuple(logits.shape))
last_token_logits = logits[0, -1]  # scores for what comes after the last token
top5 = torch.topk(last_token_logits, k=5)
print("top 5 next tokens:")
for token_id in top5.indices:
    print("   ", repr(model.tokenizer.decode(token_id)))