"""Candidate archive + model store.

Every candidate ever produced gets a record (id "k.j" = iteration k, candidate j; the original
model is "base"). A candidate is either
  * kind "delta": replacement tensors relative to its parent (small, kept forever), or
  * kind "model": a full model directory (architecture changes; kept only while useful).
``ModelStore.resolve(id)`` returns a loadable model directory for any candidate, materializing
delta chains into a small on-disk cache when needed. Candidates can therefore branch from any
archived parent, not just the current best (needed for population-based / Pareto search).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .targets import materialize


@dataclass
class CandidateRecord:
    id: str
    iteration: int
    experiment: str = ""
    strategy: str = ""
    parent: str = "base"
    kind: str = "delta"               # base | delta | model
    artifact: str = ""                # delta file, or full model dir
    summary: str = ""
    x: dict = field(default_factory=dict)        # agentic-optimization parameters, if any
    guard: dict = field(default_factory=dict)
    screen: dict = field(default_factory=dict)   # cheap dev metrics
    val: dict = field(default_factory=dict)      # hidden val metrics
    n_params: Optional[int] = None
    status: str = "pending"  # invalid | screened | promoted | accepted | rejected | eval_failed | not_promoted
    error: str = ""
    deleted: bool = False             # artifact removed from disk (not usable as parent)


def pareto_front(recs: list[CandidateRecord], objectives: list[tuple[str, bool]], source: str = "val") -> list[str]:
    """ids on the Pareto front. objectives: [(key, maximize)]; metrics read from rec.val/rec.screen
    (n_params from the record)."""
    pts = []
    for r in recs:
        m = dict(getattr(r, source) or {})
        m["n_params"] = r.n_params
        v = [m.get(k) for k, _ in objectives]
        if any(x is None for x in v):
            continue
        pts.append((r.id, [x if mx else -x for x, (_, mx) in zip(v, objectives)]))
    front = []
    for i, (a, va) in enumerate(pts):
        dominated = any(all(y >= x for x, y in zip(va, vb)) and any(y > x for x, y in zip(va, vb))
                        for j, (_, vb) in enumerate(pts) if j != i)
        if not dominated:
            front.append(a)
    return front


class ModelStore:
    def __init__(self, run_dir: Path, records: dict[str, CandidateRecord], cache_size: int = 3):
        self.run = run_dir
        self.records = records
        self.cache = run_dir / "cache"
        self.cache_size = cache_size
        self._lru: list[str] = []

    def usable(self, cid: str) -> bool:
        r = self.records.get(cid)
        if r is None or r.deleted or r.status == "invalid":
            return False
        return cid == "base" or self.usable(r.parent)

    def resolve(self, cid: str) -> Path:
        r = self.records[cid]
        if r.kind in ("base", "model"):
            if r.deleted or not Path(r.artifact).exists():
                raise FileNotFoundError(f"model for {cid} was deleted")
            return Path(r.artifact)
        out = self.cache / cid
        if not (out / "config.json").exists():
            parent = self.resolve(r.parent)
            materialize(parent, Path(r.artifact), out)
        self._touch(cid)
        return out

    def _touch(self, cid: str) -> None:
        if cid in self._lru:
            self._lru.remove(cid)
        self._lru.append(cid)

    def evict(self, protect: set[str]) -> None:
        """Drop materialized delta caches beyond cache_size (never the protected ids)."""
        if not self.cache.exists():
            return
        cached = [p.name for p in self.cache.iterdir() if p.is_dir()]
        order = [c for c in self._lru if c in cached] + [c for c in cached if c not in self._lru]
        excess = len(cached) - self.cache_size
        for cid in order:
            if excess <= 0:
                break
            if cid in protect:
                continue
            shutil.rmtree(self.cache / cid, ignore_errors=True)
            if cid in self._lru:
                self._lru.remove(cid)
            excess -= 1

    def ancestors(self, cid: str) -> list[str]:
        out = []
        while cid in self.records and cid != "base":
            out.append(cid)
            cid = self.records[cid].parent
        return out + ["base"]

    def prune_models(self, keep: int, protect: set[str]) -> None:
        """Delete full-model artifacts beyond `keep` (oldest first), except protected ids and any
        ancestor of a protected id."""
        prot = set()
        for p in protect:
            prot.update(self.ancestors(p))
        models = [r for r in self.records.values() if r.kind == "model" and not r.deleted]
        removable = [r for r in models if r.id not in prot]
        excess = len(models) - keep
        for r in sorted(removable, key=lambda r: (r.status == "accepted", r.iteration)):
            if excess <= 0:
                break
            shutil.rmtree(r.artifact, ignore_errors=True)
            r.deleted = True
            excess -= 1
