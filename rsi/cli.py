"""rsi — recursive self-improvement harness.

  rsi new [NAME] [--dir DIR] [--backend B]     chat with an agent that designs the goal YAML for you
  rsi run GOAL.yaml [--run-dir D] [--backend B] [--max-iterations N] [--controller llm|bandit|fixed]
  rsi resume RUN_DIR [--max-iterations N] [--patience N]
  rsi status RUN_DIR
  rsi validate GOAL.yaml
  rsi eval GOAL.yaml MODEL_DIR [--split dev|val|test] [--limit N]      one-off evaluation
  rsi init [EXAMPLE] [--dir D]                 copy a ready-made example goal (+ metric, data) here
  rsi doctor                                   check the installation (GPU, vLLM, agent CLIs)

Runs are written to ./runs/ of the current directory (override with --run-dir).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from .orchestrator import Orchestrator
from .spec import GoalSpec, load_spec
from .validate import print_report, validate

BACKENDS = ["claude_code", "antigravity", "codex", "command"]
REPO = Path(__file__).resolve().parent.parent


def _run_dir(spec: GoalSpec, given) -> Path:
    return Path(given) if given else Path("runs") / f"{spec.name}-{time.strftime('%Y%m%d-%H%M%S')}"


def cmd_new(a) -> None:
    from .agents import make_backend
    from .spec import AgentSpec
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", a.name or "my-goal").strip("-") or "my-goal"
    gdir = Path(a.dir or Path("goals") / name).resolve()
    gdir.mkdir(parents=True, exist_ok=True)
    ref = gdir / "reference"
    ref.mkdir(exist_ok=True)
    (ref / "goal_schema.json").write_text(json.dumps(GoalSpec.model_json_schema(), indent=1))
    tdir = Path(__file__).parent / "templates"
    for f in ("goal_template.yaml", "metric_template.py"):  # generic, problem-agnostic
        shutil.copy2(tdir / f, ref / f)
    text = (Path(__file__).parent / "templates" / "DESIGNER.md").read_text().format(
        goal_dir=gdir, python=sys.executable)
    instr = ["CLAUDE.md", "AGENTS.md", "GEMINI.md"]
    for f in instr:
        (gdir / f).write_text(text)
    backend = make_backend(AgentSpec(backend=a.backend))
    if not backend.available():
        sys.exit(f"agent backend '{a.backend}' is not installed")
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so the cleanup below still runs
    kickoff = (f"Hi! I want to set up a recursive self-improvement run (goal name '{name}'). "
               "Please interview me and then write goal.yaml, metric.py and the data splits here.")
    if a.prompt:
        kickoff += " Here is what I have in mind: " + a.prompt
    try:
        if a.headless:  # no conversation: the designer works from --prompt alone
            if not a.prompt:
                sys.exit("--headless needs --prompt describing the problem")
            kickoff += (" I am not available for questions in this session: choose sensible defaults, "
                        "record every assumption in DESIGN_NOTES.md, and do everything in 'Before you finish' "
                        "except asking me.")
            print(f"[rsi] goal designer ({a.backend}, headless) working in {gdir} ...")
            res = backend.run(kickoff, gdir, [], {"PATH": f"{Path(sys.executable).parent}:{__import__('os').environ.get('PATH', '')}"},
                              gdir / ".designer", timeout_s=a.timeout)
            print(res.final_text[-3000:] or res.error)
        else:
            print(f"[rsi] goal designer ({a.backend}) in {gdir} — exit the session when you are done.\n")
            backend.interactive(kickoff, text, gdir)
    finally:
        for f in instr:
            (gdir / f).unlink(missing_ok=True)
    goal = gdir / "goal.yaml"
    if not goal.exists():
        sys.exit(f"[rsi] no goal.yaml was written in {gdir}. Run `rsi new {name}` again to continue.")
    spec, errors, warnings = validate(goal)
    print_report(goal, errors, warnings)
    if errors:
        sys.exit(f"[rsi] fix the errors above (edit {goal} or rerun `rsi new {name}`), then `rsi run {goal}`.")
    if a.headless:
        print(f"Start it with:  rsi run {goal}")
    elif input("\nStart the self-improvement run now? [y/N] ").strip().lower() in ("y", "yes"):
        Orchestrator(spec, _run_dir(spec, None)).run_loop()
    else:
        print(f"Start it later with:  rsi run {goal}")


def cmd_status(a) -> None:
    rd = Path(a.run_dir)
    st = json.loads((rd / "state.json").read_text())
    primary = load_spec(rd / "spec.yaml").evaluation.primary
    print(f"{st['name']}: original={st['baseline']} best={st['best_score']} (candidate {st['best_id']}, "
          f"iter {st['best_iteration']}) finished={st['finished']} {st['stop_reason']}")
    for it in st["iterations"]:
        ex = ", ".join(e["strategy"] for e in it["experiments"])
        print(f"  iter {it['iteration']:>3} [{it['controller']}] {ex} | {len(it['candidates'])} cands | "
              f"promoted {it['promoted']} | accepted {it['accepted']} | best {it['best_after']}")
    cands = st.get("candidates", {})
    if len(cands) > 1:
        print(f"  {len(cands) - 1} candidates; latest:")
        for c in list(cands.values())[-8:]:
            sc = (c.get("screen") or {}).get(primary)
            print(f"    {c['id']:<7} {c['strategy']:<22} {c['kind']:<6} {c['status']:<13} screen={sc} "
                  f"{(c['error'] or c['summary'])[:70]}")


def cmd_init(a) -> None:
    import subprocess
    ex_root = REPO / "examples"
    names = sorted(p.name for p in ex_root.iterdir() if (p / "goal.yaml").exists()) if ex_root.exists() else []
    if a.list or not a.example:
        print("examples:", ", ".join(names) or "(none: non-editable install)")
        print("usage: rsi init <example> [--dir DIR]")
        return
    if a.example not in names:
        sys.exit(f"unknown example '{a.example}'. Available: {', '.join(names)}")
    src, dst = ex_root / a.example, Path(a.dir or ".").resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_dir() or f.suffix not in (".yaml", ".py", ".md"):
            continue  # data / models are generated below, never copied
        t = dst / f.name
        if t.exists() and not a.force:
            print(f"  keep   {t.name} (exists; --force to overwrite)")
            continue
        shutil.copy2(f, t)
        print(f"  copied {t.name}")
    for script in ("prepare_data.py", "make_toy.py"):
        if (dst / script).exists() and not a.no_data:
            print(f"  running {script} (builds data/) ...")
            subprocess.run([sys.executable, script], cwd=dst, check=True)
    goals = sorted(p.name for p in dst.glob("goal*.yaml"))
    print(f"\nready in {dst}. Goals: {', '.join(goals)}\n  rsi validate {goals[0] if goals else 'goal.yaml'}"
          f"\n  rsi run {goals[0] if goals else 'goal.yaml'}")


def cmd_doctor() -> None:
    import shutil as sh
    from . import __version__
    print(f"rsi {__version__}")
    print(f"  package : {Path(__file__).resolve().parent}")
    print(f"  python  : {sys.executable}")
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "-"
        print(f"  torch   : {torch.__version__}  cuda={cuda}  {gpu}")
    except Exception as e:  # noqa: BLE001
        print(f"  torch   : not usable ({e})")
    try:
        import vllm
        print(f"  vllm    : {vllm.__version__}")
    except Exception:  # noqa: BLE001
        print("  vllm    : not installed (needed for the math example; ./install.sh installs it)")
    for name, b in (("claude_code", "claude"), ("antigravity", "agy"), ("codex", "codex")):
        print(f"  agent   : {name:<12} {sh.which(b) or 'not found'}")
    ex = REPO / "examples"
    print(f"  examples: {ex if ex.exists() else 'not available (non-editable install)'}")


def main(argv=None) -> None:
    from . import __version__
    ap = argparse.ArgumentParser(prog="rsi", description="Recursive self-improvement harness")
    ap.add_argument("--version", action="version", version=f"rsi {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the installation")
    i = sub.add_parser("init", help="copy a ready-made example goal into a folder")
    i.add_argument("example", nargs="?")
    i.add_argument("--dir", help="target folder (default: current)")
    i.add_argument("--list", action="store_true")
    i.add_argument("--force", action="store_true", help="overwrite existing files")
    i.add_argument("--no-data", action="store_true", help="do not run the data preparation script")
    n = sub.add_parser("new", help="design a goal interactively with an agent")
    n.add_argument("name", nargs="?")
    n.add_argument("--dir")
    n.add_argument("--backend", default="claude_code", choices=BACKENDS[:3])
    n.add_argument("--prompt", help="what you want, to start the conversation")
    n.add_argument("--headless", action="store_true", help="no conversation: design from --prompt alone")
    n.add_argument("--timeout", type=int, default=3600, help="headless: max seconds")
    r = sub.add_parser("run", help="start a run from a goal YAML")
    r.add_argument("spec")
    r.add_argument("--run-dir")
    r.add_argument("--backend", choices=BACKENDS)
    r.add_argument("--model", help="model for the agent backend")
    r.add_argument("--max-iterations", type=int)
    r.add_argument("--controller", choices=["llm", "bandit", "fixed"])
    r.add_argument("--skip-validate", action="store_true")
    rs = sub.add_parser("resume", help="continue (or extend) a run")
    rs.add_argument("run_dir")
    rs.add_argument("--max-iterations", type=int, help="raise the cap to extend a finished run")
    rs.add_argument("--patience", type=int)
    s = sub.add_parser("status", help="summary of a run")
    s.add_argument("run_dir")
    v = sub.add_parser("validate", help="pre-flight checks for a goal YAML")
    v.add_argument("spec")
    e = sub.add_parser("eval", help="one-off evaluation of a model with a goal's metric")
    e.add_argument("spec")
    e.add_argument("model_dir", help="local dir or HF repo id")
    e.add_argument("--split", default="dev", choices=["dev", "val", "test"])
    e.add_argument("--limit", type=int)
    e.add_argument("--config", default="{}", help="JSON overrides of evaluation.config")
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        cmd_doctor()
    elif a.cmd == "init":
        cmd_init(a)
    elif a.cmd == "new":
        cmd_new(a)
    elif a.cmd == "run":
        if not a.skip_validate:
            _, errors, warnings = validate(a.spec)
            print_report(a.spec, errors, warnings)
            if errors:
                sys.exit(1)
        spec = load_spec(a.spec)
        if a.backend:
            spec.agent.backend = a.backend
        if a.model:
            spec.agent.model = a.model
        if a.max_iterations:
            spec.search.max_iterations = a.max_iterations
        if a.controller:
            spec.population.controller = a.controller
        Orchestrator(spec, _run_dir(spec, a.run_dir)).run_loop()
    elif a.cmd == "resume":
        rd = Path(a.run_dir)
        spec = load_spec(rd / "spec.yaml")
        if a.max_iterations:
            spec.search.max_iterations = a.max_iterations
        if a.patience:
            spec.search.patience = a.patience
        Orchestrator(spec, rd).run_loop()
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "validate":
        _, errors, warnings = validate(a.spec)
        print_report(a.spec, errors, warnings)
        sys.exit(1 if errors else 0)
    elif a.cmd == "eval":
        from .evaluator import run_eval
        from .targets import resolve_model_dir
        spec = load_spec(a.spec)
        split = getattr(spec.evaluation.splits, a.split)
        cfg = {**spec.evaluation.config, **json.loads(a.config)}
        out = Path("runs") / "adhoc" / f"eval_{a.split}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        md = Path(a.model_dir) if Path(a.model_dir).exists() else resolve_model_dir(a.model_dir)
        res = run_eval(spec.evaluation.metric, md, split, cfg, out,
                       timeout_s=spec.evaluation.timeout_s, python=spec.evaluation.python, limit=a.limit)
        print(json.dumps({k: v for k, v in res.items() if k != "per_example"}, indent=1))
        print("full result:", out)


if __name__ == "__main__":
    main()
