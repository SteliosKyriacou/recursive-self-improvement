# Strategy: weight-space programming (direct weight edits, no gradients)

Read the weights and activations, form a hypothesis about *which mechanism* causes the bad
behaviour, and edit the responsible weights directly. Reference: "Sparks of Weight-Space
Programming: LLMs can natively read & write neural network weights" (Takehi, Teoh, Zhang,
Khattab — MIT CSAIL, Nov 2026, https://rikiyat.github.io/blog/weight-space-programming/).
Their Exp 8 improved Qwen3.5-0.8B on MATH-500 from 7.9% to 48.1% by changing 0.016% of the
weights with a 50-forward-pass budget — the base model looped until the token limit, and the edit
removed a looping-related direction from a few MLP output projections.

## Recipe
1. **Diagnose with probes.** `python tools/probe.py --n 16` and read the completions in
   `probes/probe_*.json`. What goes wrong — looping, wrong format, arithmetic slips, early stop?
   Look at `cut`/`hit_limit` style fields and the tails of failing completions.
2. **Contrast sets.** Build two groups of texts that differ in the failure (e.g. tails of looping
   completions vs. tails of clean ones; or the same prompt continued in a good vs bad way).
3. **Find the direction.** `capture()` residual / MLP outputs at several layers, `mean_diff()`;
   check separation with `projection_stats()`; inspect with `logit_lens()`.
4. **Locate writers.** `top_writers(down_proj.weight, d, k)` finds MLP neurons that write it.
5. **Edit surgically.** `ablate_direction_in_columns`, `ablate_direction_in_rows`, `add_bias_direction`,
   or your own. Small, targeted edits generalise better and pass guards easily.
6. **Verify, then save.** Probe with `--delta`; compare to the unedited model on the *same* dev
   slice (same `--offset`/`--n`). Spend probes wisely — dev slices are small and noisy.

```python
import sys; sys.path.insert(0, ".")
from toolbox.common.delta import save_delta, parent_dir
from toolbox.weight_programming.wsp import *
MD = parent_dir()
model, tok = load_model(MD)                     # CPU float32; pass device="cuda" if you have a GPU
L = decoder_layers(model)
acts_bad  = capture(model, tok, bad_texts,  layers=range(10, 22), site="mlp", last_n_tokens=64)
acts_good = capture(model, tok, good_texts, layers=range(10, 22), site="mlp", last_n_tokens=64)
d = mean_diff(acts_bad[15], acts_good[15])
W = L[15].mlp.down_proj.weight.data
cols = top_writers(W, d, k=64)
W_new = ablate_direction_in_columns(W, d, cols)
save_delta({param_name(model, L[15].mlp.down_proj): W_new}, "candidates/<name>/delta.safetensors", model_dir=MD)
```
Parameter names from `param_name()` match checkpoint names for standard HF models; if not,
look them up in `reference/tensors.json`.
