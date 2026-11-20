"""Helpers shared by all strategies: read tensors from the (read-only) model dir, write deltas.

    from toolbox.common.delta import load_tensor, save_delta, diff_state, parent_dir
    MODEL_DIR = parent_dir()
    W = load_tensor(MODEL_DIR, "model.language_model.layers.15.mlp.down_proj.weight")
    W2 = W.clone(); ...edit...
    save_delta({name: W2}, "candidates/my_edit/delta.safetensors")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _files(model_dir) -> list[Path]:
    return sorted(Path(model_dir).glob("*.safetensors"))


def tensor_names(model_dir) -> list[str]:
    out = []
    for f in _files(model_dir):
        with safe_open(str(f), framework="pt") as fh:
            out += list(fh.keys())
    return out


def load_tensor(model_dir, name: str) -> torch.Tensor:
    for f in _files(model_dir):
        with safe_open(str(f), framework="pt") as fh:
            if name in fh.keys():
                return fh.get_tensor(name)
    raise KeyError(name)


def load_tensors(model_dir, names: Iterable[str]) -> dict[str, torch.Tensor]:
    want, out = set(names), {}
    for f in _files(model_dir):
        with safe_open(str(f), framework="pt") as fh:
            for k in want & set(fh.keys()):
                out[k] = fh.get_tensor(k)
    missing = want - set(out)
    if missing:
        raise KeyError(f"not in model: {sorted(missing)[:5]}")
    return out


def save_delta(changed: dict[str, torch.Tensor], path, model_dir=None) -> Path:
    """Save full replacement values for changed tensors. If model_dir is given, dtype is matched
    to the original and unchanged tensors are dropped automatically."""
    out = {}
    for k, v in changed.items():
        v = v.detach().cpu()
        if model_dir is not None:
            orig = load_tensor(model_dir, k)
            v = v.to(orig.dtype)
            if v.shape != orig.shape:
                raise ValueError(f"{k}: shape {tuple(v.shape)} != {tuple(orig.shape)}")
            if torch.equal(v, orig):
                continue
        out[k] = v.contiguous()
    if not out:
        raise ValueError("no tensor differs from the model — nothing to save")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(p))
    n = sum(t.numel() for t in out.values())
    print(f"saved delta: {len(out)} tensors, {n:,} elements -> {p}")
    return p


def diff_state(model, model_dir, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    """Given an in-memory (edited / finetuned) HF model, return tensors that differ from model_dir.
    Parameter names are mapped to checkpoint names by suffix matching."""
    ckpt = set(tensor_names(model_dir))
    sd = model.state_dict()
    out = {}
    for pname, t in sd.items():
        if names is not None and pname not in names:
            continue
        cname = pname if pname in ckpt else next((c for c in ckpt if c.endswith(pname) or pname.endswith(c)), None)
        if cname is None:
            continue
        orig = load_tensor(model_dir, cname)
        if orig.shape == t.shape and not torch.equal(orig, t.detach().cpu().to(orig.dtype)):
            out[cname] = t.detach().cpu().to(orig.dtype)
    return out


def parent_dir() -> str:
    """The (read-only) parent model directory of the current experiment workspace."""
    import os
    for base in (Path.cwd(), *Path.cwd().parents):
        f = base / ".parent_dir"
        if f.exists():
            return f.read_text().strip()
    if os.environ.get("RSI_PARENT_DIR"):
        return os.environ["RSI_PARENT_DIR"]
    raise RuntimeError("parent model dir not found (.parent_dir / RSI_PARENT_DIR)")


model_dir_from_agents_md = parent_dir  # backwards-compatible alias


if __name__ == "__main__":
    d = parent_dir()
    print(json.dumps({"model_dir": d, "n_tensors": len(tensor_names(d))}))
