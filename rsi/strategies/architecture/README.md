# Strategy: architecture search (with weight inheritance)

Change the network's structure, not just its weights. A candidate is a full model directory
(`candidates/<name>/model/` with `config.json` + `*.safetensors`); the harness takes tokenizer /
chat template from the parent, checks the parameter budget (`guards.max_params_ratio`), forbids
custom model code, and the model must load in the frozen evaluator (vLLM for the math example).

## Operators (`toolbox/architecture/arch.py`)
| function | effect |
|---|---|
| `describe(P)` | layers, layer-stack prefix, per-layer config lists (e.g. `layer_types`), MLP width |
| `remap_layers(P, out, order)` | new stack from source layer indices: **drop** (omit), **duplicate / depth up-scale** (repeat), **reorder** (permute). Per-layer config lists follow. |
| `prune_mlp(P, out, new_intermediate)` | width pruning of every gated MLP by neuron importance (or explicit indices) |
| `copy_parent(P, out)` | start from an exact copy, then edit config / tensors yourself |

Weights are inherited (Lamarckian, like RWNN-LMM): surviving layers/neurons keep their trained
values, duplicated layers start as copies. Optional **healing**: a short finetune of the new model
(`python toolbox/finetuning/sft.py --model-dir candidates/x/model --save-model candidates/x/model ...`).

## Ideas
- Layers whose removal barely changes dev behaviour are redundant → drop them (cheaper model, sometimes
  better: less "overthinking"/looping) — measure per-layer importance with a few probes or with
  activation similarity (input vs output of a layer, see `toolbox/weight_programming/wsp.py:capture`).
- Duplicate the most useful middle layers (depth up-scaling) — more compute per token, same width.
- Hybrid models (e.g. Qwen3.5 linear vs full attention, `layer_types`): reorder or duplicate layers to
  change the attention pattern; a full-attention layer needs its own weights (only copies are possible).
- Combine with the other strategies: an architecture candidate can later be a parent for weight
  programming or finetuning deltas; tunable surgery (which layers, how much pruning) makes a good
  agentic-optimization recipe.

## Graph-level NAS (RWNN-LMM)
Free-form DAG search over primitive ops (`~/repos/RWNN-LMM`) needs custom model code and training
from scratch, which `hf_weights` targets do not allow. It belongs in a separate target kind
(roadmap). Lesson carried over from that project: always check causality — no future-token leakage.
