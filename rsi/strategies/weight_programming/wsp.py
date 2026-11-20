"""Weight-space programming helpers: look inside a transformer, find directions, edit weights.

Typical loop (cf. "Sparks of Weight-Space Programming", Takehi et al. 2026, Exp 8):
  1. Collect two groups of texts that differ in the behaviour you want to change
     (e.g. completions that loop / hit the token limit vs. ones that finish cleanly — get them
     from tools/probe.py outputs).
  2. capture() activations for both groups; mean_diff() gives a candidate behaviour direction.
  3. Find which components write that direction (top_writers on mlp.down_proj / o_proj).
  4. Edit: remove / damp / add the direction in those components (ablate_direction_in_columns ...).
  5. save_delta() and verify with tools/probe.py; keep notes in LEARNINGS.md.

All functions work on CPU (slow but fine for short texts) or GPU.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch


# ----------------------------------------------------------------------------- loading
def load_model(model_dir: str, device: str = "cpu", dtype=torch.float32):
    """Load the text model + tokenizer. Works for plain causal LMs and VLM wrappers (e.g. Qwen3.5)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    except Exception:  # noqa: BLE001  - VLM checkpoints
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype)
    return model.to(device).eval(), tok


def decoder_layers(model) -> torch.nn.ModuleList:
    """The longest ModuleList named '*layers' that is not part of a vision tower."""
    best, best_name = None, ""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and "visual" not in name:
            if best is None or len(mod) > len(best):
                best, best_name = mod, name
    if best is None:
        raise RuntimeError("could not find decoder layers")
    return best


def param_name(model, module: torch.nn.Module, suffix: str = "weight") -> str:
    """Checkpoint-style name of a module's parameter (for save_delta)."""
    for n, m in model.named_modules():
        if m is module:
            return f"{n}.{suffix}"
    raise KeyError("module not in model")


# ----------------------------------------------------------------------------- capture
@contextmanager
def _hooks(hooks):
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def capture(model, tok, texts: list[str], layers: Iterable[int], site: str = "resid",
            max_len: int = 1024, last_n_tokens: Optional[int] = None, device: Optional[str] = None
            ) -> dict[int, torch.Tensor]:
    """Return {layer: [total_tokens, d_model]} activations (float32, CPU).

    site: "resid" (layer output), "mlp" (MLP output), "attn" (attention output).
    last_n_tokens: keep only the last n positions of each text (useful for "what happens right
    before the model loops / stops").
    """
    device = device or next(model.parameters()).device
    L = decoder_layers(model)
    store: dict[int, list] = {i: [] for i in layers}
    hooks = []
    for i in layers:
        mod = L[i] if site == "resid" else getattr(L[i], "mlp" if site == "mlp" else
                                                  ("self_attn" if hasattr(L[i], "self_attn") else "linear_attn"))
        def fn(_m, _inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            store[i].append(h.detach().float().cpu())
        hooks.append(mod.register_forward_hook(fn))
    with _hooks(hooks):
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
            model(input_ids=ids)
    out = {}
    for i, chunks in store.items():
        rows = [c[0] if last_n_tokens is None else c[0, -last_n_tokens:] for c in chunks]
        out[i] = torch.cat(rows, 0)
    return out


def mean_diff(a: torch.Tensor, b: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Direction separating group a from group b (rows = samples)."""
    d = a.float().mean(0) - b.float().mean(0)
    return d / d.norm() if normalize else d


def projection_stats(acts: torch.Tensor, d: torch.Tensor) -> dict:
    p = acts.float() @ d.float()
    return {"mean": p.mean().item(), "std": p.std().item(), "min": p.min().item(), "max": p.max().item()}


# ----------------------------------------------------------------------------- components
def top_writers(W_out: torch.Tensor, d: torch.Tensor, k: int = 64) -> torch.Tensor:
    """Columns of an output projection (shape [d_model, n_in], e.g. mlp.down_proj.weight) whose
    write vectors align most with direction d. Returns column indices sorted by |d·W[:, j]|."""
    s = (d.float() @ W_out.float()).abs()
    return torch.topk(s, k).indices


@torch.no_grad()
def logit_lens(model, vec: torch.Tensor, tok, k: int = 20) -> list[str]:
    """Tokens a residual-space vector promotes when read out through the unembedding."""
    W_U = model.get_output_embeddings().weight.float().cpu()
    scores = W_U @ vec.float().cpu()
    return [tok.decode([i]) for i in torch.topk(scores, k).indices.tolist()]


# ----------------------------------------------------------------------------- edits
def ablate_direction_in_columns(W_out: torch.Tensor, d: torch.Tensor, cols=None, alpha: float = 1.0) -> torch.Tensor:
    """W[:, j] -= alpha * (d·W[:, j]) d  for j in cols (all columns if None). Returns a new tensor."""
    W = W_out.float().clone()
    d = (d / d.norm()).float()
    idx = slice(None) if cols is None else torch.as_tensor(cols)
    W[:, idx] -= alpha * torch.outer(d, d @ W[:, idx])
    return W.to(W_out.dtype)


def ablate_direction_in_rows(W_in: torch.Tensor, d: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """For input projections (shape [n_out, d_model]): stop the component *reading* direction d."""
    W = W_in.float().clone()
    d = (d / d.norm()).float()
    W -= alpha * torch.outer(W @ d, d)
    return W.to(W_in.dtype)


def add_bias_direction(W_out: torch.Tensor, d: torch.Tensor, cols, scale: float) -> torch.Tensor:
    """Steer: make selected columns also write +scale*d (a weight-space 'steering vector')."""
    W = W_out.float().clone()
    W[:, torch.as_tensor(cols)] += scale * (d / d.norm()).float()[:, None]
    return W.to(W_out.dtype)
