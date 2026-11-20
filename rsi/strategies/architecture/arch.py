"""Architecture surgery on HF checkpoints with weight inheritance (no transformers needed).

    from toolbox.architecture.arch import describe, remap_layers, prune_mlp
    from toolbox.common.delta import parent_dir
    P = parent_dir()
    describe(P)                                                    # layers, per-layer config lists, MLP width
    remap_layers(P, "candidates/drop7/model", order=[i for i in range(24) if i != 7])   # drop layer 7
    remap_layers(P, "candidates/dup/model", order=list(range(16)) + [12, 13, 14, 15] + list(range(16, 24)))  # depth up-scale
    prune_mlp(P, "candidates/slim/model", new_intermediate=3072)  # keep the most important MLP neurons

The output directory gets config.json + model.safetensors; the harness adds tokenizer etc. from the
parent. Layer `order` is a list of *source* layer indices for the new stack: omit = drop, repeat =
duplicate, permute = reorder. Every per-layer list in the config (e.g. Qwen3.5 `layer_types`) is
remapped the same way, so hybrid models stay consistent. Optionally heal with
`toolbox/finetuning/sft.py --model-dir <out> --save-model <out>`.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _load_all(model_dir) -> tuple[dict[str, torch.Tensor], dict]:
    tensors, meta = {}, {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as fh:
            meta.update(fh.metadata() or {})
            for k in fh.keys():
                tensors[k] = fh.get_tensor(k)
    return tensors, meta


def _text_cfg(cfg: dict) -> dict:
    """The (sub)config that holds num_hidden_layers (text_config for VLM wrappers)."""
    if "num_hidden_layers" in cfg:
        return cfg
    for v in cfg.values():
        if isinstance(v, dict) and "num_hidden_layers" in v:
            return v
    raise KeyError("num_hidden_layers not found in config")


def _layer_prefix(keys, n_layers: int) -> str:
    """Prefix 'xxx.layers.' of the decoder stack whose indices are exactly 0..n_layers-1."""
    cands: dict[str, set] = {}
    for k in keys:
        m = re.match(r"(.*?\.layers\.)(\d+)\.", k)
        if m:
            cands.setdefault(m.group(1), set()).add(int(m.group(2)))
    for pre, idx in sorted(cands.items(), key=lambda t: "visual" in t[0] or "mtp" in t[0]):
        if idx == set(range(n_layers)):
            return pre
    raise RuntimeError(f"no layer stack with {n_layers} layers among {list(cands)}")


def describe(model_dir) -> dict:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    t = _text_cfg(cfg)
    n = t["num_hidden_layers"]
    with safe_open(str(sorted(Path(model_dir).glob("*.safetensors"))[0]), framework="pt") as fh:
        keys = list(fh.keys())
    info = {"num_hidden_layers": n, "layer_prefix": _layer_prefix(keys, n),
            "intermediate_size": t.get("intermediate_size"), "hidden_size": t.get("hidden_size"),
            "per_layer_config_lists": {k: v for k, v in t.items() if isinstance(v, list) and len(v) == n}}
    print(json.dumps(info, indent=1))
    return info


def _write(out_dir, tensors, meta, cfg, src_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.safetensors"):
        f.unlink()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"), metadata=meta)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    n = sum(v.numel() for v in tensors.values())
    print(f"wrote {out} ({len(tensors)} tensors, {n:,} params)")
    return out


def remap_layers(model_dir, out_dir, order: list[int]):
    """New decoder stack = [layer order[0], layer order[1], ...] of the source model."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    t = _text_cfg(cfg)
    n = t["num_hidden_layers"]
    if not order or any(i < 0 or i >= n for i in order):
        raise ValueError(f"order must index 0..{n - 1}")
    tensors, meta = _load_all(model_dir)
    pre = _layer_prefix(tensors.keys(), n)
    out = {k: v for k, v in tensors.items() if not k.startswith(pre)}
    for new_i, src in enumerate(order):
        sp = f"{pre}{src}."
        for k, v in tensors.items():
            if k.startswith(sp):
                out[f"{pre}{new_i}.{k[len(sp):]}"] = v.clone()
    for key, val in list(t.items()):
        if isinstance(val, list) and len(val) == n:
            t[key] = [val[i] for i in order]
    t["num_hidden_layers"] = len(order)
    if "layer_types" in t and "full_attention_interval" in t:
        # keep the legacy interval consistent only if the pattern still matches it
        lt = t["layer_types"]
        iv = t["full_attention_interval"]
        if any((lt[i] == "full_attention") != ((i + 1) % iv == 0) for i in range(len(lt))):
            t.pop("full_attention_interval")
    return _write(out_dir, out, meta, cfg, model_dir)


def prune_mlp(model_dir, out_dir, new_intermediate: int, keep: Optional[dict[int, list[int]]] = None):
    """Shrink every gated MLP (gate_proj/up_proj/down_proj) to new_intermediate neurons.
    Importance per neuron = ||down[:, j]|| * ||up[j, :]|| unless `keep` gives explicit indices per layer."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    t = _text_cfg(cfg)
    n = t["num_hidden_layers"]
    tensors, meta = _load_all(model_dir)
    pre = _layer_prefix(tensors.keys(), n)
    for i in range(n):
        g, u, d = (f"{pre}{i}.mlp.{x}_proj.weight" for x in ("gate", "up", "down"))
        if not all(k in tensors for k in (g, u, d)):
            raise RuntimeError(f"layer {i}: not a gated MLP")
        if keep and i in keep:
            idx = torch.as_tensor(sorted(keep[i]))
        else:
            imp = tensors[d].float().norm(dim=0) * tensors[u].float().norm(dim=1)
            idx = torch.topk(imp, new_intermediate).indices.sort().values
        tensors[g], tensors[u] = tensors[g][idx].clone(), tensors[u][idx].clone()
        tensors[d] = tensors[d][:, idx].clone()
    t["intermediate_size"] = int(new_intermediate)
    return _write(out_dir, tensors, meta, cfg, model_dir)


def copy_parent(model_dir, out_dir):
    """Start a full-model candidate as an exact copy of the parent (then edit tensors/config)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(model_dir).iterdir():
        if f.suffix == ".safetensors" or f.name == "config.json":
            shutil.copy2(f.resolve(), out / f.name)
    return out
