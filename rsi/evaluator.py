"""Frozen evaluator. Every evaluation runs in a fresh subprocess so that

* a crashing / OOM-ing candidate cannot take the harness down,
* GPU memory is fully released between candidates,
* the metric code is loaded from the run directory snapshot, never from the agent workspace.

A metric file must define::

    def evaluate(model_dir: str, examples: list[dict], config: dict) -> dict:
        return {"score": ..., "per_example": [...], ...}
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def load_examples(path: Path, limit: Optional[int] = None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def run_eval(
    metric: Path,
    model_dir: Path,
    examples: Path,
    config: dict[str, Any],
    out_path: Path,
    timeout_s: int,
    python: Optional[str] = None,
    limit: Optional[int] = None,
    env: Optional[dict] = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python or sys.executable, "-m", "rsi.evaluator",
        "--metric", str(metric), "--model-dir", str(model_dir), "--examples", str(examples),
        "--config", json.dumps(config), "--out", str(out_path),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    log = out_path.with_suffix(".log")
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, **(env or {}), "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return {"ok": False, "error": f"timeout after {timeout_s}s", "log": str(log)}
    if rc != 0 or not out_path.exists():
        tail = log.read_text()[-3000:]
        return {"ok": False, "error": f"exit code {rc}", "log": str(log), "log_tail": tail}
    res = json.loads(out_path.read_text())
    res["ok"] = True
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _load_metric(path: Path):
    spec = importlib.util.spec_from_file_location("rsi_metric", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    mod = _load_metric(Path(a.metric))
    ex = load_examples(Path(a.examples), a.limit)
    res = mod.evaluate(a.model_dir, ex, json.loads(a.config))
    if not isinstance(res, dict):
        raise TypeError("evaluate() must return a dict")
    res.setdefault("n", len(ex))
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    _main()
