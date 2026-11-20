"""Pre-flight checks for a goal YAML (used by `rsi validate`, `rsi new` and before `rsi run`)."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

from .spec import GoalSpec, load_spec

_TEXT_KEYS = ("problem", "prompt", "question", "input", "text")


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _key(r: dict) -> str:
    for k in _TEXT_KEYS:
        if isinstance(r.get(k), str):
            return " ".join(r[k].split())
    return json.dumps({k: v for k, v in r.items() if k != "id"}, sort_keys=True)


def validate(spec_path: str | Path, check_model: bool = True) -> tuple[GoalSpec | None, list[str], list[str]]:
    errors, warnings = [], []
    try:
        spec = load_spec(spec_path)
    except Exception as e:  # noqa: BLE001
        return None, [f"spec does not parse: {e}"], []
    ev = spec.evaluation
    if not ev.metric.exists():
        errors.append(f"metric file missing: {ev.metric}")
    else:
        try:
            tree = ast.parse(ev.metric.read_text())
            fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            if "evaluate" not in fns:
                errors.append("metric file has no top-level evaluate(model_dir, examples, config)")
            elif len(fns["evaluate"].args.args) != 3:
                errors.append("evaluate() must take exactly (model_dir, examples, config)")
        except SyntaxError as e:
            errors.append(f"metric file has a syntax error: {e}")
    rows = {}
    for name in ("dev", "val", "test"):
        p = getattr(ev.splits, name)
        if p is None:
            if name == "test":
                warnings.append("no test split: the final report will have no held-out number")
            continue
        if not p.exists():
            errors.append(f"{name} split missing: {p}")
            continue
        try:
            rows[name] = _rows(p)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name} split is not valid JSONL: {e}")
            continue
        if not rows[name]:
            errors.append(f"{name} split is empty")
    if "val" in rows and len(rows["val"]) < 50:
        warnings.append(f"val has only {len(rows['val'])} examples: acceptance decisions will be noisy "
                        f"(consider min_delta > 0 or more examples)")
    if "dev" in rows:
        dk = {_key(r) for r in rows["dev"]}
        for h in ("val", "test"):
            if h in rows:
                n = sum(_key(r) in dk for r in rows[h])
                if n:
                    errors.append(f"{n} {h} examples also appear in dev (hidden split leaks to the agent)")
    if "val" in rows and "test" in rows:
        vk = {_key(r) for r in rows["val"]}
        n = sum(_key(r) in vk for r in rows["test"])
        if n:
            warnings.append(f"{n} test examples also appear in val")
    if check_model:
        m = spec.target.model
        if not Path(m).exists():
            try:
                from huggingface_hub import HfApi
                HfApi().model_info(m, revision=spec.target.revision)
            except Exception as e:  # noqa: BLE001
                errors.append(f"target model '{m}' not found locally or on the HF Hub: {str(e)[:150]}")
        elif not list(Path(m).glob("*.safetensors")):
            errors.append(f"target model dir {m} has no *.safetensors")
    from .agents import make_backend
    b = make_backend(spec.agent)
    if not b.available():
        errors.append(f"agent backend '{spec.agent.backend}' not available on PATH")
    if spec.population.controller == "llm" or ("agentic_optimization" in spec.search.strategies
                                              and spec.agentic_optimization.llm):
        try:
            b.complete_command("x")
        except NotImplementedError:
            warnings.append(f"backend '{spec.agent.backend}' cannot do one-shot completions: planner falls back "
                            f"to the bandit and agentic optimization to its built-in sampler")
    if (spec.agent.backend == "claude_code" and spec.agent.permission_mode == "bypassPermissions") or \
            spec.agent.backend in ("antigravity", "codex"):
        warnings.append("the agent runs with permission prompts bypassed (it can run any command as you)")
    if not shutil.which("nvidia-smi"):
        warnings.append("no nvidia-smi found: GPU evaluations will not work")
    return spec, errors, warnings


def print_report(spec_path, errors, warnings) -> None:
    print(f"validate {spec_path}:")
    for e in errors:
        print(f"  ✗ {e}")
    if any("missing" in e for e in errors):
        print("  → paths in a goal YAML are relative to the YAML's own folder; metric.py and data/ must be there.\n"
              "    For a built-in example run `rsi init <example>` in that folder (lists: `rsi init --list`);\n"
              "    for data produced by a script, run it first (e.g. `python prepare_data.py`).")
    for w in warnings:
        print(f"  ! {w}")
    if not errors:
        print("  ✓ ready to run")
