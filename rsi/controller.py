"""Per-iteration planning: which experiments to run, from which parent, with what budget.

Controllers:
  * llm    — the agent backend reads the run's history / strategy credit / archive / recipes and
             returns a JSON plan (falls back to bandit on any failure);
  * bandit — UCB1 over strategies, rewarded by promotions and acceptances;
  * fixed  — population.fixed_plan from the spec, every iteration.

A plan decides the shape of the iteration — nothing is forced: it can be one focused experiment
with a single candidate, or a population of experiments. It is a dict
  {"experiments": [...], "screen": bool, "promote_top": int}
where "screen" = cheap dev screening of all candidates before promotion (pointless for one
candidate) and "promote_top" = how many candidates get the hidden-val evaluation. Experiments:
  {"strategy": "weight_programming" | "finetuning" | "architecture" | "free",
   "parent": "best" | "<candidate id>", "brief": "...", "n_candidates": 2, "minutes": 60}
  {"strategy": "agentic_optimization", "recipe": "<name>", "parent": "best",
   "generations": 3, "population": 6, "brief": "..."}
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

from .archive import CandidateRecord, ModelStore
from .optim import list_recipes
from .spec import GoalSpec

AGENT_STRATEGIES = ("weight_programming", "finetuning", "architecture", "free")

STRATEGY_DOC = {
    "weight_programming": "agent reads weights/activations and edits weights directly, no gradients (Weight-Space Programming)",
    "finetuning": "agent trains selected tensors by gradient descent (SFT, self-distillation, ...) -> weight delta",
    "architecture": "agent changes the architecture with weight inheritance (drop/duplicate/reorder layers, prune MLP width) -> full model dir",
    "agentic_optimization": "LLM-guided population search over a parametric recipe (toolbox/recipes/*.py), many cheap screened candidates",
    "free": "agent chooses any approach (or a mix)",
}


def strategy_stats(records: dict[str, CandidateRecord], iterations: list[dict]) -> dict[str, dict]:
    st: dict[str, dict] = {}
    for it in iterations:
        for e in it.get("experiments", []):
            s = st.setdefault(e["strategy"], {"experiments": 0, "candidates": 0, "valid": 0, "promoted": 0,
                                              "accepted": 0, "best_gain": 0.0})
            s["experiments"] += 1
    for r in records.values():
        if r.id == "base" or r.strategy not in st:
            continue
        s = st[r.strategy]
        s["candidates"] += 1
        s["valid"] += r.status != "invalid"
        s["promoted"] += r.status in ("promoted", "accepted", "rejected")
        s["accepted"] += r.status == "accepted"
    return st


class Controller:
    def __init__(self, spec: GoalSpec, run_dir: Path, backend):
        self.spec = spec
        self.run = run_dir
        self.backend = backend

    # ------------------------------------------------------------------ public
    def plan(self, k: int, state, records: dict[str, CandidateRecord], store: ModelStore,
             history_md: str, log) -> tuple[dict, str, str]:
        """Returns (plan, controller_used, rationale); plan = {experiments, screen, promote_top}."""
        mode = self.spec.population.controller
        if mode == "fixed" and self.spec.population.fixed_plan:
            return self._finish(self.spec.population.fixed_plan, state, store), "fixed", "fixed plan"
        if mode == "llm":
            try:
                raw = self._llm_plan(k, state, records, store, history_md)
                plan = self._finish(raw, state, store)
                if plan["experiments"]:
                    return plan, "llm", str(raw.get("rationale", ""))[:1000]
                log("planner returned no usable experiments; using bandit")
            except Exception as e:  # noqa: BLE001
                log("planner failed; using bandit", error=str(e)[:300])
        return self._bandit(state, records, store), "bandit", "UCB1 over strategy credit"

    def _finish(self, raw: dict, state, store: ModelStore) -> dict:
        """Clip a raw plan to the spec's caps and fill sensible selection defaults."""
        pop = self.spec.population
        exps = self._normalize(raw.get("experiments", []), state, store)
        n_cands = sum(e.get("n_candidates", 0) for e in exps) + \
            sum(e.get("generations", 0) * e.get("population", 0) for e in exps)
        promote = max(1, min(int(raw.get("promote_top", 1)), pop.promote_top))
        screen = raw.get("screen")
        if screen is None:  # default: screen only when there is something to choose between
            screen = n_cands > promote
        has_ao = any(e["strategy"] == "agentic_optimization" for e in exps)  # AO always screens its members
        return {"experiments": exps, "screen": bool(screen) or has_ao, "promote_top": promote}

    # ------------------------------------------------------------------ normalize
    def _normalize(self, exps: list[dict], state, store: ModelStore) -> list[dict]:
        sp, pop = self.spec, self.spec.population
        recipes = {r.path.stem: r for r in list_recipes(self.run / "toolbox")}
        out, n_cands = [], 0
        for e in exps:
            if not isinstance(e, dict):
                continue
            s = e.get("strategy")
            if s not in sp.search.strategies:
                continue
            parent = str(e.get("parent", "best"))
            if parent == "best" or not store.usable(parent):
                parent = state.best_id
            x = {"strategy": s, "parent": parent, "brief": str(e.get("brief", ""))[:2000]}
            if s == "agentic_optimization":
                name = Path(str(e.get("recipe", ""))).stem
                if name not in recipes:
                    if not recipes:
                        continue
                    name = max(recipes, key=lambda n: recipes[n].path.stat().st_mtime)
                ao = sp.agentic_optimization
                x.update(recipe=name,
                         generations=max(1, min(int(e.get("generations", ao.generations)), ao.generations)),
                         population=max(1, min(int(e.get("population", ao.population)), ao.population)))
            else:
                n = max(1, int(e.get("n_candidates", 1)))
                n = min(n, pop.size - n_cands)
                if n <= 0:
                    continue
                n_cands += n
                minutes = int(e.get("minutes", sp.agent.timeout_s // 60))
                x.update(n_candidates=n, minutes=max(5, min(minutes, sp.agent.timeout_s // 60)))
            out.append(x)
            if len(out) >= pop.max_experiments:
                break
        return out

    # ------------------------------------------------------------------ bandit
    def _bandit(self, state, records, store) -> list[dict]:
        sp = self.spec
        stats = strategy_stats(records, [i if isinstance(i, dict) else i.__dict__ for i in state.iterations])
        recipes = list_recipes(self.run / "toolbox")
        arms = [s for s in sp.search.strategies if s in AGENT_STRATEGIES]
        if "agentic_optimization" in sp.search.strategies and recipes:
            arms.append("agentic_optimization")
        if not arms:
            arms = [s for s in sp.search.strategies if s != "agentic_optimization"] or ["free"]
        total = sum(max(1, stats.get(a, {}).get("experiments", 0)) for a in arms)

        def ucb(a):
            s = stats.get(a)
            if not s or s["experiments"] == 0:
                return float("inf")
            reward = (s["accepted"] + 0.3 * s["promoted"]) / s["experiments"]
            return reward + math.sqrt(2 * math.log(total) / s["experiments"])

        a = max(arms, key=ucb)
        raw = {"experiments": [{"strategy": a, "parent": "best", "n_candidates": 1,
                                "brief": f"Bandit choice. {STRATEGY_DOC.get(a, '')}"}]}
        return self._finish(raw, state, store)

    # ------------------------------------------------------------------ llm
    def _llm_plan(self, k, state, records, store, history_md) -> dict:
        sp = self.spec
        recipes = list_recipes(self.run / "toolbox")
        rtxt = "\n".join(f"- `{r.path.stem}`: {r.description} | vars: {', '.join(r.names)}" for r in recipes) \
            or "(none yet — an agent experiment can write one to toolbox/recipes/)"
        usable = [cid for cid, r in records.items() if store.usable(cid) and r.status in ("accepted", "rejected", "promoted")
                  or cid == "base"]
        learn = (self.run / "LEARNINGS.md").read_text()[-4000:] if (self.run / "LEARNINGS.md").exists() else ""
        tmpl = (Path(__file__).parent / "templates" / "PLANNER.md").read_text()
        prompt = tmpl.format(
            goal=sp.goal.strip(), iteration=k, max_iterations=sp.search.max_iterations,
            best_id=state.best_id, best_score=state.best_score, baseline=state.baseline,
            primary=sp.evaluation.primary, direction=sp.evaluation.direction,
            strategies="\n".join(f"- `{s}`: {STRATEGY_DOC[s]}" for s in sp.search.strategies),
            stats=json.dumps(strategy_stats(records, [i if isinstance(i, dict) else i.__dict__ for i in state.iterations]), indent=0),
            history=history_md[-8000:], recipes=rtxt, parents=", ".join(usable[-12:]),
            learnings=learn or "(none)", max_experiments=sp.population.max_experiments,
            size=sp.population.size, minutes=sp.agent.timeout_s // 60, promote_cap=sp.population.promote_top,
            ao_gen=sp.agentic_optimization.generations, ao_pop=sp.agentic_optimization.population,
            no_improve=state.no_improve, patience=sp.search.patience,
        )
        text = self.backend.complete(prompt, self.run / "iterations" / f"iter_{k:03d}" / "planner")
        (self.run / "iterations" / f"iter_{k:03d}" / "planner" / "plan_raw.txt").write_text(text)
        return _parse_json(text)


def _parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    cand = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    cand = re.sub(r"(?m)\s//[^\n\"]*$", "", cand)  # tolerate // comments copied from the template
    return json.loads(cand)
