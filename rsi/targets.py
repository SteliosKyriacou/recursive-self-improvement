"""Targets: how a model is stored, how a candidate edit is represented and applied.

``HFWeightsTarget`` covers any safetensors checkpoint (Hugging Face layout). A candidate is a
*delta file*: ``delta.safetensors`` holding full replacement values for the tensors the agent
changed. Deltas are pure data — the agent cannot smuggle code into the evaluator — and they are
small when edits are surgical (the paper's Exp 8 edit is 131k weights).

Finetuning, LoRA-merging, model merging and direct weight programming all reduce to a delta.
Architecture changes need a different target kind (see strategies/architecture_search).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .spec import GuardSpec


def resolve_model_dir(model: str, revision: Optional[str] = None) -> Path:
    p = Path(model)
    if p.exists():
        return p.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, revision=revision))


def weight_files(model_dir: Path) -> list[Path]:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    return files


def key_index(model_dir: Path) -> dict[str, tuple[Path, tuple[int, ...], str]]:
    """name -> (file, shape, dtype) without loading tensor data."""
    out = {}
    for f in weight_files(model_dir):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                sl = fh.get_slice(k)
                out[k] = (f, tuple(sl.get_shape()), sl.get_dtype())
    return out


def total_params(model_dir: Path) -> int:
    n = 0
    for _, shape, _ in key_index(model_dir).values():
        c = 1
        for s in shape:
            c *= s
        n += c
    return n


@dataclass
class GuardReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    changed_tensors: int = 0
    changed_elements: int = 0
    changed_fraction: float = 0.0

    def as_dict(self):
        return self.__dict__.copy()


_DTYPE_ALIASES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def check_delta(delta_path: Path, model_dir: Path, guards: GuardSpec) -> GuardReport:
    """Integrity checks run by the harness (never by the agent) before any evaluation."""
    rep = GuardReport(ok=True)
    if not delta_path.exists():
        return GuardReport(ok=False, reasons=[f"missing {delta_path.name}"])
    try:
        delta = load_file(str(delta_path))
    except Exception as e:  # noqa: BLE001
        return GuardReport(ok=False, reasons=[f"unreadable delta: {e}"])
    if not delta:
        return GuardReport(ok=False, reasons=["delta is empty"])
    idx = key_index(model_dir)
    allowed = re.compile(guards.allowed_keys) if guards.allowed_keys else None
    by_file: dict[Path, list[str]] = {}
    for k, t in delta.items():
        if k not in idx:
            rep.reasons.append(f"unknown tensor {k}")
            continue
        f, shape, dt = idx[k]
        if tuple(t.shape) != shape:
            rep.reasons.append(f"{k}: shape {tuple(t.shape)} != {shape}")
        if dt in _DTYPE_ALIASES and t.dtype != _DTYPE_ALIASES[dt]:
            rep.reasons.append(f"{k}: dtype {t.dtype} != {dt}")
        if not torch.isfinite(t.float()).all():
            rep.reasons.append(f"{k}: non-finite values")
        elif t.float().abs().max().item() > guards.max_abs_value:
            rep.reasons.append(f"{k}: |value| > {guards.max_abs_value}")
        by_file.setdefault(f, []).append(k)
    # Count elements that actually differ from the current model.
    for f, keys in by_file.items():
        with safe_open(str(f), framework="pt") as fh:
            for k in keys:
                old = fh.get_tensor(k)
                new = delta[k]
                if old.shape != new.shape:
                    continue
                n = int((old != new.to(old.dtype)).sum().item())
                if n:
                    rep.changed_tensors += 1
                    rep.changed_elements += n
                    if allowed and not allowed.search(k):
                        rep.reasons.append(f"{k}: outside allowed_keys")
    rep.changed_fraction = rep.changed_elements / max(1, total_params(model_dir))
    if rep.changed_elements == 0:
        rep.reasons.append("delta changes no weights")
    if rep.changed_fraction > guards.max_changed_fraction:
        rep.reasons.append(f"changed fraction {rep.changed_fraction:.4%} > {guards.max_changed_fraction:.4%}")
    rep.ok = not rep.reasons
    return rep


def materialize(base_dir: Path, delta_path: Optional[Path], out_dir: Path) -> Path:
    """Write a full model directory = base_dir with delta applied. Non-weight files are copied."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    delta = load_file(str(delta_path)) if delta_path else {}
    for p in Path(base_dir).iterdir():
        if p.suffix == ".safetensors" or p.is_dir():
            continue
        shutil.copy2(os.path.realpath(p), out_dir / p.name)
    for f in weight_files(base_dir):
        real = Path(os.path.realpath(f))
        with safe_open(str(real), framework="pt") as fh:
            keys = list(fh.keys())
            meta = fh.metadata() or {}
        touched = [k for k in keys if k in delta]
        if not touched:
            # Hard-link if possible (same filesystem) to avoid copying 1-10s of GB.
            try:
                os.link(real, out_dir / f.name)
            except OSError:
                shutil.copy2(real, out_dir / f.name)
            continue
        tensors = load_file(str(real))
        for k in touched:
            tensors[k] = delta[k].to(tensors[k].dtype).contiguous()
        save_file(tensors, str(out_dir / f.name), metadata=meta)
    return out_dir


def merge_deltas(older: Optional[Path], newer: Path, out: Path) -> Path:
    """Cumulative delta vs. the original base (newer wins). Used for the lineage record."""
    merged = load_file(str(older)) if older and older.exists() else {}
    merged.update(load_file(str(newer)))
    save_file({k: v.contiguous() for k, v in merged.items()}, str(out))
    return out


def write_index_summary(model_dir: Path, out: Path) -> None:
    """A compact, agent-readable listing of every tensor (name, shape, dtype)."""
    idx = key_index(model_dir)
    rows = [{"name": k, "shape": list(s), "dtype": d} for k, (_, s, d) in sorted(idx.items())]
    out.write_text(json.dumps({"n_tensors": len(rows), "n_params": total_params(model_dir), "tensors": rows}, indent=0))


# ============================================================================ full-model candidates
# Architecture changes cannot be expressed as a delta. Such a candidate is a directory with
# config.json + *.safetensors. Only weights and config are taken from the candidate; tokenizer,
# chat template, generation config etc. are always copied from the parent, so a candidate can not
# change how it is prompted or decoded by the frozen evaluator.
_CANDIDATE_FILES = {"config.json"}


def candidate_kind(cdir: Path) -> Optional[str]:
    if (cdir / "delta.safetensors").exists():
        return "delta"
    if (cdir / "model" / "config.json").exists() and list((cdir / "model").glob("*.safetensors")):
        return "model"
    return None


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def check_model_candidate(model_dir: Path, parent_dir: Path, base_params: int, guards: GuardSpec) -> GuardReport:
    rep = GuardReport(ok=True)
    cfg_p = model_dir / "config.json"
    try:
        cfg = json.loads(cfg_p.read_text())
        pcfg = json.loads((parent_dir / "config.json").read_text())
    except Exception as e:  # noqa: BLE001
        return GuardReport(ok=False, reasons=[f"bad config.json: {e}"])
    if not guards.allow_custom_code:
        if list(model_dir.glob("*.py")) or "auto_map" in json.dumps(cfg):
            rep.reasons.append("custom model code (*.py / auto_map) not allowed")
    if cfg.get("model_type") != pcfg.get("model_type"):
        rep.reasons.append(f"model_type changed {pcfg.get('model_type')} -> {cfg.get('model_type')}")
    if guards.allowed_config_keys is not None:
        a, b = _flatten(cfg), _flatten(pcfg)
        changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        bad = sorted(k for k in changed if not any(k == x or k.endswith("." + x) for x in guards.allowed_config_keys))
        if bad:
            rep.reasons.append(f"config keys not allowed to change: {bad[:8]}")
    try:
        n = 0
        for f in weight_files(model_dir):
            with safe_open(str(f), framework="pt") as fh:
                for k in fh.keys():
                    t = fh.get_tensor(k)
                    n += t.numel()
                    if not torch.isfinite(t.float()).all():
                        rep.reasons.append(f"{k}: non-finite values")
                    elif t.numel() and t.float().abs().max().item() > guards.max_abs_value:
                        rep.reasons.append(f"{k}: |value| > {guards.max_abs_value}")
    except Exception as e:  # noqa: BLE001
        return GuardReport(ok=False, reasons=[f"unreadable weights: {e}"])
    rep.changed_elements = n
    rep.changed_fraction = n / max(1, base_params)  # here: size relative to the original model
    if n > guards.max_params_ratio * base_params:
        rep.reasons.append(f"params {n:,} > {guards.max_params_ratio} x original ({base_params:,})")
    rep.ok = not rep.reasons
    return rep


def finalize_model_candidate(cand_model_dir: Path, parent_dir: Path, out_dir: Path) -> Path:
    """Full model dir = candidate weights + config, everything else from the parent."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for p in Path(parent_dir).iterdir():
        if p.is_dir() or p.suffix in (".safetensors", ".py") or p.name in _CANDIDATE_FILES or \
                p.name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(os.path.realpath(p), out_dir / p.name)
    for p in Path(cand_model_dir).iterdir():
        if p.suffix == ".safetensors" or p.name in _CANDIDATE_FILES or p.name.endswith(".safetensors.index.json"):
            shutil.copy2(p, out_dir / p.name)
    return out_dir
