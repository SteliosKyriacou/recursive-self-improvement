"""The self-improvement loop.

    baseline ─► for k in 1..max_iterations:
                   plan       controller (LLM planner | bandit | fixed) picks a population of
                              experiments: strategy, parent, brief, budget
                   run        agent experiments (weight programming / finetuning / architecture / free)
                              each deliver up to n candidates; agentic-optimization experiments
                              run G generations x P recipe candidates (LLM-written proposals)
                   guards     delta / model integrity + audit of the agent's actions
                   screen     cheap dev evaluation of every valid candidate
                   promote    top-m (that beat the best on screen) -> hidden val evaluation
                   select     accept the best promoted candidate iff it beats the best by min_delta
                   memory     LEARNINGS.md + toolbox (incl. recipes) persist -> the improver improves
                   converge?  patience / wall clock / max iterations
             ─► export best model, final hidden-test evaluation vs. baseline ─► report.md
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .agents import make_backend
from .archive import CandidateRecord, ModelStore, pareto_front
from .controller import STRATEGY_DOC, Controller, strategy_stats
from .evaluator import load_examples, run_eval
from .optim import AgenticOptimizer, build_candidate, list_recipes, load_recipe
from .probe import PROBE_CLIENT, ProbeServer
from .spec import GoalSpec, load_spec
from .targets import (candidate_kind, check_delta, check_model_candidate, finalize_model_candidate,
                      materialize, merge_deltas, resolve_model_dir, total_params, write_index_summary)

PKG = Path(__file__).resolve().parent


@dataclass
class RunState:
    name: str
    started: float
    base_model: str
    base_params: int = 0
    baseline: Optional[float] = None
    best_id: str = "base"
    best_score: Optional[float] = None
    best_iteration: int = 0
    no_improve: int = 0
    iterations: list = field(default_factory=list)       # list[dict]
    candidates: dict = field(default_factory=dict)       # id -> CandidateRecord as dict
    finished: bool = False
    stop_reason: str = ""
    test: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, spec: GoalSpec, run_dir: Path):
        self.spec = spec
        self.run = run_dir.resolve()
        self.backend = make_backend(spec.agent)
        self.state_path = self.run / "state.json"
        self.records: dict[str, CandidateRecord] = {}

    # ================================================================ helpers
    def log(self, msg: str, **kw: Any) -> None:
        line = {"t": round(time.time(), 1), "msg": msg, **kw}
        with open(self.run / "events.jsonl", "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"[rsi {time.strftime('%H:%M:%S')}] {msg}" + (f" {json.dumps(kw, default=str)}" if kw else ""),
              flush=True)

    def save(self) -> None:
        self.state.candidates = {k: asdict(v) for k, v in self.records.items()}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=1, default=str))
        tmp.replace(self.state_path)

    @property
    def primary(self) -> str:
        return self.spec.evaluation.primary

    def score(self, metrics: dict) -> Optional[float]:
        v = metrics.get(self.primary) if metrics else None
        return float(v) if v is not None else None

    @property
    def hidden_dir(self) -> Path:
        return next(self.run.glob("hidden_*"))

    def _eval(self, model_dir: Path, split: Path, config: dict, out: Path) -> dict:
        ev = self.spec.evaluation
        res = run_eval(self.run / "frozen" / "metric.py", model_dir, split, config, out,
                       timeout_s=ev.timeout_s, python=ev.python)
        return res

    def _summary(self, res: dict) -> dict:
        return {k: v for k, v in res.items()
                if k not in ("per_example", "log_tail") and not isinstance(v, (list, dict))}

    # ================================================================ setup / resume
    def setup(self) -> None:
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text())
            cands = raw.pop("candidates", {})
            self.state = RunState(**raw, candidates={})
            done = len(self.state.iterations)
            # Drop candidates of an interrupted iteration (it will be redone).
            self.records = {k: CandidateRecord(**v) for k, v in cands.items() if v["iteration"] <= done}
            self.store = ModelStore(self.run, self.records)
            self.log("resuming", iterations_done=done, best=self.state.best_score, best_id=self.state.best_id)
            return
        self.run.mkdir(parents=True, exist_ok=True)
        ev = self.spec.evaluation
        (self.run / "frozen").mkdir(exist_ok=True)
        shutil.copy2(ev.metric, self.run / "frozen" / "metric.py")
        (self.run / "spec.yaml").write_text(yaml.safe_dump(json.loads(self.spec.model_dump_json()), sort_keys=False))
        hidden = self.run / f"hidden_{secrets.token_hex(6)}"  # unguessable; also audited for
        hidden.mkdir()
        shutil.copy2(ev.splits.val, hidden / "val.jsonl")
        if ev.splits.test:
            shutil.copy2(ev.splits.test, hidden / "test.jsonl")
        dev = load_examples(ev.splits.dev)
        sc = self.spec.population.screen
        (self.run / "frozen" / "screen.jsonl").write_text(
            "\n".join(json.dumps(e) for e in dev[sc.offset: sc.offset + sc.n]) + "\n")
        shutil.copytree(PKG / "strategies", self.run / "toolbox")
        (self.run / "toolbox" / "recipes").mkdir(exist_ok=True)
        (self.run / "LEARNINGS.md").write_text("# Learnings\n\n(none yet)\n")
        base = resolve_model_dir(self.spec.target.model, self.spec.target.revision)
        self.state = RunState(name=self.spec.name, started=time.time(), base_model=str(base),
                              base_params=total_params(base))
        self.records = {"base": CandidateRecord(id="base", iteration=0, strategy="original", parent="",
                                                kind="base", artifact=str(base), summary="original model",
                                                status="accepted", n_params=self.state.base_params)}
        self.store = ModelStore(self.run, self.records)
        self.save()

    # ================================================================ main loop
    def run_loop(self) -> RunState:
        self.setup()
        st = self.state
        if st.finished and not self._should_stop(len(st.iterations) + 1):
            st.finished, st.stop_reason, st.test = False, "", {}
            self.log("extending finished run")
        base = self.records["base"]
        if st.baseline is None:
            self.log("evaluating baseline on hidden val")
            res = self._eval(Path(base.artifact), self.hidden_dir / "val.jsonl", self.spec.evaluation.config,
                             self.run / "baseline" / "val.json")
            if not res.get("ok"):
                raise RuntimeError(f"baseline evaluation failed: {res.get('error')}\n{res.get('log_tail', '')}")
            base.val = self._summary(res)
            st.baseline = st.best_score = self.score(base.val)
            self.save()
            self.log("baseline", score=st.baseline, metrics=base.val)
        self._ensure_screen("base")

        while not st.finished:
            k = len(st.iterations) + 1
            reason = self._should_stop(k)
            if reason:
                st.finished, st.stop_reason = True, reason
                break
            it = self.iteration(k)
            st.iterations.append(it)
            if it["accepted"]:
                st.best_id, st.best_score, st.best_iteration, st.no_improve = \
                    it["accepted"], it["best_after"], k, 0
            else:
                st.no_improve += 1
            self.store.prune_models(self.spec.population.keep_models, protect={st.best_id})
            self.store.evict(protect=set(self.store.ancestors(st.best_id)))
            self.save()
            self.log(f"iteration {k} done", accepted=it["accepted"], best=st.best_score, best_id=st.best_id,
                     n_candidates=len(it["candidates"]))
        self.save()
        self.log("stopping", reason=st.stop_reason)
        self.final_report()
        return st

    def _should_stop(self, k: int) -> str:
        st, sp = self.state, self.spec.search
        if k > sp.max_iterations:
            return f"reached max_iterations={sp.max_iterations}"
        if st.no_improve >= sp.patience:
            return f"converged: no improvement for {sp.patience} iterations"
        if sp.wall_clock_hours and (time.time() - st.started) / 3600 > sp.wall_clock_hours:
            return f"wall clock budget {sp.wall_clock_hours}h exhausted"
        return ""

    # ================================================================ one iteration
    def iteration(self, k: int) -> dict:
        sp, st = self.spec, self.state
        it_dir = self.run / "iterations" / f"iter_{k:03d}"
        if it_dir.exists():
            shutil.rmtree(it_dir)
        it_dir.mkdir(parents=True)
        self._cand_counter = 0
        plan, ctrl, why = Controller(sp, self.run, self.backend).plan(
            k, st, self.records, self.store, self._history_md(), self.log)
        exps = plan["experiments"]
        self.log(f"iteration {k}: plan", controller=ctrl, screen=plan["screen"], promote_top=plan["promote_top"],
                 experiments=[{x: e.get(x) for x in ("strategy", "parent", "n_candidates", "recipe")} for e in exps])
        (it_dir / "plan.json").write_text(json.dumps({"controller": ctrl, "rationale": why, **plan}, indent=1))
        rec = {"iteration": k, "controller": ctrl, "rationale": why, "plan": plan, "experiments": [],
               "candidates": [], "promoted": [], "accepted": None, "best_before": st.best_score,
               "best_after": st.best_score}

        for i, e in enumerate(exps, 1):
            if e["strategy"] == "agentic_optimization":
                er = self._run_ao_experiment(k, i, e, it_dir)
            else:
                er = self._run_agent_experiment(k, i, e, it_dir)
            rec["experiments"].append(er)
            rec["candidates"] += er["candidates"]
            self.save()

        # ---- screen if the plan asked for it (agentic-optimization candidates are already screened)
        valid = [self.records[c] for c in rec["candidates"] if self.records[c].status != "invalid"]
        need_screen = plan["screen"]
        if need_screen:
            self._ensure_screen(st.best_id)
            for r in valid:
                if not r.screen:
                    self._screen(r.id)
        best_screen = self.score(self.records[st.best_id].screen)
        pool = [r for r in valid if r.status != "eval_failed"]
        if need_screen:
            pool = [r for r in pool if self.score(r.screen) is not None]
            pool.sort(key=lambda r: sp.sort_key(self.score(r.screen)), reverse=True)
            if sp.population.promote_only_if_screen_improves and best_screen is not None:
                pool = [r for r in pool if sp.better(self.score(r.screen), best_screen, margin=0.0)]
        promoted = pool[: plan["promote_top"]]
        for r in valid:
            if r not in promoted and r.status in ("pending", "screened"):
                r.status = "not_promoted"

        # ---- hidden val for the promoted
        for r in promoted:
            self.log(f"iteration {k}: val eval", candidate=r.id, screen=self.score(r.screen))
            res = self._eval(self.store.resolve(r.id), self.hidden_dir / "val.jsonl", sp.evaluation.config,
                             it_dir / "val" / f"{r.id}.json")
            if not res.get("ok"):
                r.status, r.error = "eval_failed", str(res.get("error"))
                continue
            r.val, r.status = self._summary(res), "promoted"
            rec["promoted"].append(r.id)
        evaluated = [r for r in promoted if r.status == "promoted"]
        evaluated.sort(key=lambda r: sp.sort_key(self.score(r.val)), reverse=True)
        if evaluated and sp.better(self.score(evaluated[0].val), st.best_score):
            win = evaluated[0]
            win.status = "accepted"
            rec["accepted"], rec["best_after"] = win.id, self.score(win.val)
            for r in evaluated[1:]:
                r.status = "rejected"
        else:
            for r in evaluated:
                r.status = "rejected"
        (it_dir / "record.json").write_text(json.dumps(rec, indent=1, default=str))
        return rec

    # ================================================================ candidates
    def _new_id(self, k: int) -> str:
        self._cand_counter += 1
        return f"{k}.{self._cand_counter}"

    def _register(self, k: int, exp_id: str, strategy: str, parent: str, cdir: Path, meta: dict,
                  x: Optional[dict] = None, error: str = "") -> CandidateRecord:
        """Validate a candidate dir and store its artifact in the run."""
        cid = self._new_id(k)
        r = CandidateRecord(id=cid, iteration=k, experiment=exp_id, strategy=strategy, parent=parent,
                            summary=str(meta.get("summary", ""))[:300], x=x or {})
        self.records[cid] = r
        kind = candidate_kind(cdir) if cdir.exists() else None
        if not error and kind == "model" and not self.spec.guards.allow_model_candidates:
            error = "architecture (full-model) candidates are not allowed in this run"
        if error or kind is None:
            r.status, r.error = "invalid", error or "no delta.safetensors or model/ in candidate dir"
            return r
        r.kind = kind
        parent_dir = self.store.resolve(parent)
        store_dir = self.run / "candidates" / cid
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "meta.json").write_text(json.dumps({**meta, "x": x or {}, "parent": parent}, indent=1))
        if kind == "delta":
            g = check_delta(cdir / "delta.safetensors", parent_dir, self.spec.guards)
            r.guard, r.n_params = g.as_dict(), self.records[parent].n_params
            if g.ok:
                shutil.copy2(cdir / "delta.safetensors", store_dir / "delta.safetensors")
                r.artifact = str(store_dir / "delta.safetensors")
        else:
            g = check_model_candidate(cdir / "model", parent_dir, self.state.base_params, self.spec.guards)
            r.guard, r.n_params = g.as_dict(), g.changed_elements
            if g.ok:
                r.artifact = str(finalize_model_candidate(cdir / "model", parent_dir, store_dir / "model"))
        if not g.ok:
            r.status, r.error = "invalid", "guards: " + "; ".join(g.reasons[:8])
        else:
            r.status = "pending"
        return r

    def _ensure_screen(self, cid: str) -> None:
        if not self.records[cid].screen:
            self._screen(cid)

    def _screen(self, cid: str) -> None:
        r = self.records[cid]
        cfg = {**self.spec.evaluation.config, **self.spec.population.screen.config}
        try:
            md = self.store.resolve(cid)
        except Exception as e:  # noqa: BLE001
            r.status, r.error = "eval_failed", f"materialize: {e}"
            return
        res = self._eval(md, self.run / "frozen" / "screen.jsonl", cfg,
                         self.run / "screens" / f"{cid}.json")
        if res.get("ok"):
            r.screen = self._summary(res)
            if r.status == "pending":
                r.status = "screened"
        else:
            r.status, r.error = "eval_failed", f"screen: {res.get('error')}"
        self.log("screened", candidate=cid, **{self.primary: self.score(r.screen)}, error=r.error or None)
        self.store.evict(protect=set(self.store.ancestors(self.state.best_id)))

    # ================================================================ experiments
    def _run_agent_experiment(self, k: int, i: int, e: dict, it_dir: Path) -> dict:
        sp = self.spec
        exp_id = f"{k}.e{i}"
        exp_dir = it_dir / f"exp{i}_{e['strategy']}"
        ws = exp_dir / "workspace"
        parent_dir = self.store.resolve(e["parent"])
        self._build_workspace(ws, k, exp_id, e, parent_dir)
        probe = ProbeServer(sp, parent_dir, exp_dir / "probes", sp.probe.forward_pass_budget,
                            log=lambda r: self.log("probe", experiment=exp_id, **{x: r.get(x) for x in
                                                   ("probe_id", self.primary, "n", "remaining", "error")}))
        url = probe.start()
        (ws / ".probe.json").write_text(json.dumps({"url": url, "token": probe.token}))
        env = {"PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}", "RSI_EXPERIMENT": exp_id,
               "RSI_PARENT_DIR": str(parent_dir)}
        if not sp.agent.gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        self.log(f"experiment {exp_id}: agent working", strategy=e["strategy"], parent=e["parent"],
                 backend=self.backend.name, workspace=str(ws))
        try:
            ar = self.backend.run(_KICKOFF, ws, [parent_dir], env, exp_dir, timeout_s=e["minutes"] * 60)
        finally:
            probe.stop()
        (exp_dir / "agent_final.txt").write_text(ar.final_text or "")
        self._persist_agent_memory(ws, exp_dir)
        audit = self._audit(ar.transcript)
        out = {"id": exp_id, "strategy": e["strategy"], "parent": e["parent"], "seconds": ar.seconds,
               "cost_usd": ar.cost_usd, "probes_used": probe.used, "error": ar.error, "audit": audit,
               "candidates": []}
        cdirs = sorted(p for p in (ws / "candidates").iterdir() if p.is_dir()) if (ws / "candidates").exists() else []
        for cdir in cdirs[: e["n_candidates"]]:
            meta = _read_json(cdir / "meta.json")
            err = ("transcript audit: " + "; ".join(audit)) if audit else ""
            r = self._register(k, exp_id, e["strategy"], e["parent"], cdir, meta, error=err)
            out["candidates"].append(r.id)
            self.log("candidate", id=r.id, kind=r.kind, status=r.status, summary=r.summary, error=r.error or None)
        if not cdirs:
            out["error"] = out["error"] or "no candidates delivered"
        return out

    def _run_ao_experiment(self, k: int, i: int, e: dict, it_dir: Path) -> dict:
        sp = self.spec
        exp_id = f"{k}.e{i}"
        exp_dir = it_dir / f"exp{i}_agentic_optimization"
        exp_dir.mkdir(parents=True, exist_ok=True)
        recipe = load_recipe(self.run / "toolbox" / "recipes" / f"{e['recipe']}.py")
        shutil.copy2(recipe.path, exp_dir / "recipe.py")  # the exact version used
        parent_dir = self.store.resolve(e["parent"])
        llm = None
        if sp.agentic_optimization.llm:
            llm = lambda prompt: self.backend.complete(prompt, exp_dir / "llm")  # noqa: E731
        opt = AgenticOptimizer(recipe, e["population"], e["generations"], llm=llm, goal=sp.goal,
                               primary=self.primary, brief=e.get("brief", ""), seed=k)
        env = {"PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"}
        out = {"id": exp_id, "strategy": "agentic_optimization", "parent": e["parent"], "recipe": e["recipe"],
               "candidates": [], "error": ""}
        t0 = time.time()
        self.log(f"experiment {exp_id}: agentic optimization", recipe=e["recipe"], parent=e["parent"],
                 generations=e["generations"], population=e["population"])
        for g in range(e["generations"]):
            xs = opt.ask(g, exp_dir / f"gen{g}")
            for j, x in enumerate(xs):
                cdir = exp_dir / f"gen{g}" / f"c{j}"
                ok, tail = build_candidate(recipe.path, x, parent_dir, cdir, self.run,
                                           sp.agentic_optimization.build_timeout_s, env)
                meta = {"strategy": "agentic_optimization", "summary": f"{e['recipe']} {json.dumps(x)}"[:300]}
                r = self._register(k, exp_id, "agentic_optimization", e["parent"], cdir, meta, x=x,
                                   error="" if ok else f"build failed: {tail}")
                if r.status != "invalid":
                    self._screen(r.id)
                opt.tell(x, sp.sort_key(self.score(r.screen)) if r.screen else None)
                out["candidates"].append(r.id)
                shutil.rmtree(cdir / "model", ignore_errors=True)  # stored copy lives in run/candidates
            bx, bf = opt.best()
            self.log(f"experiment {exp_id}: generation {g + 1} done", best_x=bx, best_screen=bf,
                     source=opt.log[-1]["source"] if opt.log else "fallback")
            self.save()
        out["seconds"] = round(time.time() - t0, 1)
        (exp_dir / "optimizer.json").write_text(json.dumps({"log": opt.log, "learnings": opt.learnings,
                                                            "X": [recipe.to_dict(x) for x in opt.X], "F": opt.F},
                                                           indent=1, default=str))
        if opt.learnings:
            with open(self.run / "LEARNINGS.md", "a") as f:
                f.write(f"\n## Agentic optimization {exp_id} on recipe `{e['recipe']}`\n" +
                        "\n".join(f"- {l}" for l in opt.learnings) + f"\n- best: {opt.best()}\n")
        return out

    # ================================================================ workspace
    def _build_workspace(self, ws: Path, k: int, exp_id: str, e: dict, parent_dir: Path) -> None:
        sp, st = self.spec, self.state
        for d in ("data", "reference", "tools", "candidates", "probes"):
            (ws / d).mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp.evaluation.splits.dev, ws / "data" / "dev.jsonl")
        shutil.copy2(self.run / "frozen" / "metric.py", ws / "reference" / "metric.py")
        write_index_summary(parent_dir, ws / "reference" / "tensors.json")
        (ws / "tools" / "probe.py").write_text(PROBE_CLIENT)
        (ws / ".parent_dir").write_text(str(parent_dir))
        shutil.copytree(self.run / "toolbox", ws / "toolbox")
        shutil.copy2(self.run / "LEARNINGS.md", ws / "LEARNINGS.md")
        (ws / "HISTORY.md").write_text(self._history_md())
        ev, g, sc = sp.evaluation, sp.guards, sp.population.screen
        thr = (st.best_score + sp.search.min_delta) if sp.maximize else (st.best_score - sp.search.min_delta)
        pr = self.records[e["parent"]]
        text = (PKG / "templates" / "AGENTS.md").read_text().format(
            experiment_id=exp_id, iteration=k, max_iterations=sp.search.max_iterations, name=sp.name,
            goal=sp.goal.strip(), strategy=e["strategy"], strategy_doc=STRATEGY_DOC.get(e["strategy"], ""),
            brief=e.get("brief") or "(none — use your judgement)", parent_id=e["parent"],
            parent_score=_fmt(self.score(pr.val)), model_dir=parent_dir, n_candidates=e["n_candidates"],
            primary=ev.primary, direction=ev.direction, best_score=_fmt(st.best_score),
            baseline_score=_fmt(st.baseline), cmp=">" if sp.maximize else "<", threshold=_fmt(thr),
            screen_lo=sc.offset, screen_hi=sc.offset + sc.n,
            screen_note="the harness uses them to rank candidates; tune on other dev examples to avoid overfitting the screen.",
            notes=("\n## Notes from the user\n" + sp.notes.strip() + "\n") if sp.notes.strip() else "",
            probe_budget=sp.probe.forward_pass_budget, probe_max=sp.probe.max_examples, python=sys.executable,
            gpu_note=("A GPU is available to you (the probe server shares it)." if sp.agent.gpu else
                      "**No GPU for you** (CPU only): analyse weights on CPU; run the model only through tools/probe.py."),
            forbidden=", ".join(["hidden val/test files"] + list(g.forbidden_patterns)),
            max_frac=f"{g.max_changed_fraction:.4%}",
            allowed_keys=f", only tensors matching /{g.allowed_keys}/" if g.allowed_keys else "",
            max_params_ratio=g.max_params_ratio, timeout_min=e["minutes"],
        )
        for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
            (ws / name).write_text(text)

    def _history_md(self) -> str:
        st, sp, p = self.state, self.spec, self.primary
        L = [f"Original model val {p}: {_fmt(st.baseline)} · current best: {_fmt(st.best_score)} "
             f"(candidate `{st.best_id}`, iteration {st.best_iteration or '-'})", ""]
        L += ["## Iterations", "| iter | controller | experiments | candidates | promoted | accepted | best after |",
              "|---|---|---|---|---|---|---|"]
        for it in st.iterations:
            ex = ", ".join(f"{e['strategy']}" + (f"({e.get('recipe')})" if e.get("recipe") else "") for e in it["experiments"])
            L.append(f"| {it['iteration']} | {it['controller']} | {ex} | {len(it['candidates'])} | "
                     f"{', '.join(it['promoted']) or '-'} | {it['accepted'] or '-'} | {_fmt(it['best_after'])} |")
        if not st.iterations:
            L.append("| – | – | – | – | – | – | – |")
        L += ["", "## Candidates (latest 40; screen = cheap dev score, val = hidden)",
              "| id | strategy | parent | kind | screen | val | status | summary / error |", "|---|---|---|---|---|---|---|---|"]
        for r in list(self.records.values())[-40:]:
            L.append(f"| {r.id} | {r.strategy} | {r.parent or '-'} | {r.kind} | {_fmt(self.score(r.screen))} | "
                     f"{_fmt(self.score(r.val))} | {r.status} | {(r.error or r.summary)[:140]} |")
        stats = strategy_stats(self.records, st.iterations)
        if stats:
            L += ["", "## Strategy credit", "| strategy | experiments | candidates | valid | promoted | accepted |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {s} | {v['experiments']} | {v['candidates']} | {v['valid']} | {v['promoted']} | {v['accepted']} |"
                  for s, v in stats.items()]
        rerr: dict = {}
        recipes = list_recipes(self.run / "toolbox", rerr)
        if recipes or rerr:
            L += ["", "## Agentic-optimization recipes"]
            L += [f"- `{r.path.stem}`: {r.description} ({len(r.names)} vars)" for r in recipes]
            L += [f"- `{n}`: **INVALID** — {m}" for n, m in rerr.items()]
        if sp.evaluation.secondary:
            objs = [(p, sp.maximize)] + [(o.key, o.direction == "maximize") for o in sp.evaluation.secondary]
            front = pareto_front([r for r in self.records.values() if r.val], objs)
            L += ["", f"## Pareto front on val over {[o for o, _ in objs]}: {', '.join(front) or '-'}"]
        L += ["", f"Candidate artifacts: {self.run}/candidates/<id>/ (delta.safetensors or model/, meta.json)"]
        return "# History\n\n" + "\n".join(L) + "\n"

    def _persist_agent_memory(self, ws: Path, exp_dir: Path) -> None:
        if (ws / "LEARNINGS.md").exists():
            shutil.copy2(ws / "LEARNINGS.md", self.run / "LEARNINGS.md")
            shutil.copy2(ws / "LEARNINGS.md", exp_dir / "LEARNINGS.md")
        if (ws / "toolbox").exists():
            tb = self.run / "toolbox"
            snap = self.run / "toolbox_history" / f"{exp_dir.parent.name}_{exp_dir.name}"
            snap.parent.mkdir(exist_ok=True)
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(tb, snap)  # audit trail of how the toolbox evolved
            shutil.rmtree(tb)
            shutil.copytree(ws / "toolbox", tb, ignore=shutil.ignore_patterns("__pycache__", "*.safetensors"))
            (tb / "recipes").mkdir(exist_ok=True)

    def _audit(self, transcript: Path) -> list[str]:
        if not self.spec.guards.audit_transcript or not transcript.exists():
            return []
        text = _agent_actions(transcript)
        ev = self.spec.evaluation.splits
        pats = [self.hidden_dir.name, str(ev.val)] + ([str(ev.test)] if ev.test else [])
        pats += list(self.spec.guards.forbidden_patterns)
        return [f"agent action mentions forbidden '{p}'" for p in pats if p and p in text]

    # ================================================================ final
    def final_report(self) -> None:
        st, p = self.state, self.primary
        best_dir = self.run / "best" / "model"
        if not best_dir.exists():
            self.log("exporting best model", best_id=st.best_id)
            src = self.store.resolve(st.best_id)
            shutil.copytree(src, best_dir, symlinks=False)
            chain = self.store.ancestors(st.best_id)[:-1]
            if chain and all(self.records[c].kind == "delta" for c in chain):
                cum = self.run / "best" / "delta_vs_base.safetensors"
                for c in reversed(chain):
                    merge_deltas(cum if cum.exists() else None, Path(self.records[c].artifact), cum)
        test_path = self.hidden_dir / "test.jsonl"
        if test_path.exists() and not st.test:
            self.log("final test evaluation (best and original)")
            best = self._eval(best_dir, test_path, self.spec.evaluation.config, self.run / "final" / "test_best.json")
            st.test["best"] = self._summary(best)
            if st.best_id != "base":
                base = self._eval(Path(st.base_model), test_path, self.spec.evaluation.config,
                                  self.run / "final" / "test_base.json")
                st.test["baseline"] = self._summary(base)
            else:
                st.test["baseline"] = st.test["best"]
            self.save()
        lines = [f"# RSI run `{st.name}`", "", f"- Stop reason: {st.stop_reason}",
                 f"- Val {p}: original {_fmt(st.baseline)} → best {_fmt(st.best_score)} "
                 f"(candidate `{st.best_id}`, lineage: {' ← '.join(self.store.ancestors(st.best_id))})"]
        if st.test:
            lines.append(f"- **Test {p}: original {_fmt(st.test.get('baseline', {}).get(p))} → "
                         f"best {_fmt(st.test.get('best', {}).get(p))}**")
        n_c = len(self.records) - 1
        lines.append(f"- {len(st.iterations)} iterations, {n_c} candidates, "
                     f"{sum(r.status == 'accepted' for r in self.records.values()) - 1} accepted")
        lines += ["", self._history_md().split("\n", 2)[2], "", f"Best model: `{best_dir}`"]
        if (self.run / "best" / "delta_vs_base.safetensors").exists():
            lines.append(f"Cumulative weight delta vs original: `{self.run / 'best' / 'delta_vs_base.safetensors'}`")
        (self.run / "report.md").write_text("\n".join(lines) + "\n")
        self.log("report written", path=str(self.run / "report.md"))


_KICKOFF = ("Read AGENTS.md in the current directory and carry out this experiment end to end. "
            "Deliver your candidate(s) in candidates/<name>/ (delta.safetensors or model/, plus meta.json), "
            "append to LEARNINGS.md, then finish with a short summary.")


def _agent_actions(transcript: Path) -> str:
    """Only what the agent *did* (tool-call inputs / shell commands), not what it read or thought —
    so reading AGENTS.md (which names the forbidden patterns) does not trigger the audit."""
    acts = []
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        if t == "assistant":  # Claude Code stream-json
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    acts.append(json.dumps(c.get("input")))
        elif t in ("item.started", "item.completed") and isinstance(ev.get("item"), dict):  # Codex --json
            it = ev["item"]
            if it.get("type") in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
                acts.append(json.dumps({k: v for k, v in it.items() if k not in ("aggregated_output", "output")}))
        elif t not in ("user", "system", "result"):  # unknown format (e.g. Antigravity): be conservative
            if "tool" in json.dumps(ev)[:200]:
                acts.append(line)
    return "\n".join(acts)


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def run(spec_path: str, run_dir: Optional[str] = None) -> RunState:
    spec = load_spec(spec_path)
    rd = Path(run_dir) if run_dir else Path("runs") / f"{spec.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    return Orchestrator(spec, rd).run_loop()
