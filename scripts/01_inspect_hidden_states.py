"""
Step 2: look at GPT-2's hidden states (the residual stream) for single sentences.
No training here. We only check shapes and test two predictions about causal attention.
"""
import torch
from transformer_lens.model_bridge import TransformerBridge

torch.manual_seed(0)
torch.set_grad_enabled(False)

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
model.enable_compatibility_mode()
model.eval()

N_LAYERS = model.cfg.n_layers  # 12

# ---- Part A: run one sentence and keep every intermediate result ----
sentence_a = "This film was a complete waste of time."
logits_a, cache_a = model.run_with_cache(sentence_a)

print("tokens:", model.to_str_tokens(sentence_a))
print("number of cached activations:", len(list(cache_a.keys())))
print("first 8 hook names:", list(cache_a.keys())[:8])


def get_hidden_states(cache):
    """Return a list of 13 tensors: point 0 = embeddings, point L+1 = after layer L."""
    states = [cache["blocks.0.hook_resid_pre"]]
    for layer in range(N_LAYERS):
        states.append(cache[f"blocks.{layer}.hook_resid_post"])
    return states


hidden_a = get_hidden_states(cache_a)

# ---- Part B: shapes and vector sizes at each point ----
# The norm is the length of a vector. It tells us how "big" the numbers are.
print("\npoint | shape            | BOS norm | mean norm of other tokens")
for point, h in enumerate(hidden_a):
    norms = h[0].norm(dim=-1)  # one length per token
    print(f"{point:5d} | {str(tuple(h.shape)):16s} | {norms[0].item():8.1f} | {norms[1:].mean().item():8.1f}")

# ---- Part C: test the causal-attention predictions ----
sentence_b = "I loved every minute of this wonderful film."
_, cache_b = model.run_with_cache(sentence_b)
hidden_b = get_hidden_states(cache_b)

point = 6  # any middle point works
bos_diff = (hidden_a[point][0, 0] - hidden_b[point][0, 0]).abs().max().item()
last_diff = (hidden_a[point][0, -1] - hidden_b[point][0, -1]).abs().max().item()
print(f"\nAt point {point}:")
print("  biggest difference between the two BOS vectors:  ", bos_diff)
print("  biggest difference between the two last-token vectors:", last_diff)