"""Agentic optimization over a parametric *recipe* (port of the Metis-Agent idea from
~/repos/agentic-optimizer): every generation an LLM sees the archive of (parameters, score) and
writes NumPy code that proposes the next population; the harness builds and screens each member.

A recipe is a Python file (usually written by an agent into toolbox/recipes/):

    DESCRIPTION = "Scale the loop-direction ablation per layer"
    SPACE = {"alpha_15": (0.0, 2.0), "alpha_19": (0.0, 2.0), "k": (16, 512, "int")}
    DEFAULT = {"alpha_15": 1.0, "alpha_19": 1.0, "k": 64}          # optional
    def build(x: dict, model_dir: str, out_dir: str) -> None:
        # write out_dir/delta.safetensors   (same architecture)  or
        #       out_dir/model/              (config.json + *.safetensors; architecture change)
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class Recipe:
    path: Path
    description: str
    names: list[str]
    lb: np.ndarray
    ub: np.ndarray
    is_int: np.ndarray
    default: Optional[dict] = None

    def to_dict(self, v: np.ndarray) -> dict:
        out = {}
        for n, x, i in zip(self.names, v, self.is_int):
            out[n] = int(round(float(x))) if i else float(x)
        return out

    def to_vec(self, d: dict) -> np.ndarray:
        return np.array([float(d.get(n, (l + u) / 2)) for n, l, u in zip(self.names, self.lb, self.ub)])


_SAFE_BUILTINS = {"range": range, "dict": dict, "float": float, "int": int, "str": str, "list": list,
                  "tuple": tuple, "zip": zip, "enumerate": enumerate, "len": len, "min": min, "max": max}


def load_recipe(path: Path) -> Recipe:
    """Read DESCRIPTION / SPACE / DEFAULT without importing the recipe (no build code runs here).
    Values may be literals or simple expressions (comprehensions, range, ...)."""
    tree = ast.parse(Path(path).read_text())
    vals: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("DESCRIPTION", "SPACE", "DEFAULT"):
                try:
                    vals[name] = ast.literal_eval(node.value)
                except ValueError:
                    expr = ast.Expression(node.value)
                    vals[name] = eval(compile(expr, str(path), "eval"),  # noqa: S307
                                      {"__builtins__": _SAFE_BUILTINS}, dict(vals))
    if "SPACE" not in vals or not isinstance(vals["SPACE"], dict) or not vals["SPACE"]:
        raise ValueError(f"{path}: recipe needs a non-empty literal SPACE dict")
    if "def build" not in Path(path).read_text():
        raise ValueError(f"{path}: recipe needs build(x, model_dir, out_dir)")
    names, lb, ub, it = [], [], [], []
    for k, v in vals["SPACE"].items():
        names.append(k)
        lb.append(float(v[0]))
        ub.append(float(v[1]))
        it.append(len(v) > 2 and v[2] == "int")
    return Recipe(Path(path), str(vals.get("DESCRIPTION", "")), names, np.array(lb), np.array(ub),
                  np.array(it), vals.get("DEFAULT"))


def list_recipes(toolbox: Path, errors: dict | None = None) -> list[Recipe]:
    """Valid recipes in toolbox/recipes (files starting with '_' are ignored). Invalid ones are
    reported in `errors` (name -> message) so agents can fix them."""
    out = []
    for p in sorted((toolbox / "recipes").glob("*.py")) if (toolbox / "recipes").exists() else []:
        if p.name.startswith("_"):
            continue
        try:
            out.append(load_recipe(p))
        except Exception as e:  # noqa: BLE001
            if errors is not None:
                errors[p.stem] = f"{type(e).__name__}: {e}"[:300]
    return out


_PROPOSER_PROMPT = """You are the proposal engine of a population-based optimizer (generation {gen} of {gens}).
Goal of the overall run: {goal}
Recipe being tuned: {description}
Research director's brief for this search: {brief}

Variables (name: [low, high], type):
{variables}

Evaluated so far (score = {primary}, higher is better; best first, max 25 rows):
{table}
{learnings}
Write Python code that proposes exactly P={pop} NEW parameter vectors. Think about what the
data says (sensitivity of each variable, promising regions, unexplored regions); balance
exploitation of the best region with exploration. Do not resubmit evaluated points.

Available names: np, rng (np.random.Generator), X (n x d array of evaluated points, column order
as listed above), F (n scores, higher is better), lb, ub (arrays), names (list), P (int).
Your code must set RESULT to an array of shape (P, d) within [lb, ub].

Answer with one line `LEARNINGS: <what you infer so far>` followed by one ```python block.
"""


@dataclass
class AgenticOptimizer:
    recipe: Recipe
    population: int
    generations: int
    llm: Optional[Callable[[str], str]] = None      # prompt -> text
    goal: str = ""
    primary: str = "score"
    brief: str = ""
    seed: int = 0
    X: list = field(default_factory=list)
    F: list = field(default_factory=list)
    learnings: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ ask / tell
    def ask(self, gen: int, work: Path) -> list[dict]:
        P, R = self.population, self.recipe
        X = None
        if self.llm is not None:
            try:
                X = self._llm_propose(gen, work)
                self.log.append({"gen": gen, "source": "llm"})
            except Exception as e:  # noqa: BLE001
                self.log.append({"gen": gen, "source": "fallback", "error": str(e)[:300]})
        if X is None:
            X = self._fallback(gen)
        X = np.clip(np.asarray(X, dtype=float).reshape(-1, len(R.names))[:P], R.lb, R.ub)
        if gen == 0 and R.default is not None and not self.X:
            X[0] = np.clip(R.to_vec(R.default), R.lb, R.ub)
        while len(X) < P:  # pad if the proposal was short
            X = np.vstack([X, self._fallback(gen)[:1]])
        out, seen = [], {json.dumps(R.to_dict(x), sort_keys=True) for x in self.X}
        for x in X:
            d = R.to_dict(x)
            key = json.dumps(d, sort_keys=True)
            if key in seen:  # jitter duplicates
                d = R.to_dict(np.clip(x + self.rng.normal(0, 0.05, len(x)) * (R.ub - R.lb), R.lb, R.ub))
                key = json.dumps(d, sort_keys=True)
            seen.add(key)
            out.append(d)
        return out

    def tell(self, x: dict, score: Optional[float]) -> None:
        self.X.append(self.recipe.to_vec(x))
        self.F.append(float("-inf") if score is None else float(score))

    def best(self) -> tuple[Optional[dict], Optional[float]]:
        if not self.F:
            return None, None
        i = int(np.argmax(self.F))
        return self.recipe.to_dict(self.X[i]), self.F[i]

    # ------------------------------------------------------------------ proposals
    def _fallback(self, gen: int) -> np.ndarray:
        R, P, d = self.recipe, self.population, len(self.recipe.names)
        span = R.ub - R.lb
        if not self.X:  # Latin hypercube
            u = (np.argsort(self.rng.random((P, d)), axis=0) + self.rng.random((P, d))) / P
            return R.lb + u * span
        F = np.array(self.F)
        X = np.array(self.X)
        elite = X[np.argsort(-F)[: max(1, min(3, len(F)))]]
        sigma = 0.25 * (0.6 ** gen) * span
        n_rand = max(1, P // 5)
        kids = elite[self.rng.integers(len(elite), size=P - n_rand)] + self.rng.normal(0, 1, (P - n_rand, d)) * sigma
        rand = R.lb + self.rng.random((n_rand, d)) * span
        return np.vstack([kids, rand])

    def _llm_propose(self, gen: int, work: Path) -> np.ndarray:
        R = self.recipe
        var = "\n".join(f"- {n}: [{l:g}, {u:g}], {'int' if i else 'float'}"
                        for n, l, u, i in zip(R.names, R.lb, R.ub, R.is_int))
        rows = sorted(zip(self.F, [R.to_dict(x) for x in self.X]), key=lambda t: -t[0])[:25]
        table = "\n".join(f"{f:.4f}  {json.dumps(d)}" for f, d in rows) or "(nothing yet)"
        learn = ("\nYour earlier notes:\n" + "\n".join(f"- {l}" for l in self.learnings[-5:]) + "\n") \
            if self.learnings else ""
        prompt = _PROPOSER_PROMPT.format(gen=gen + 1, gens=self.generations, goal=self.goal[:600],
                                         description=R.description, brief=self.brief or "(none)", variables=var, primary=self.primary,
                                         table=table, learnings=learn, pop=self.population)
        text = self.llm(prompt)
        m = re.search(r"LEARNINGS:\s*(.+)", text)
        if m:
            self.learnings.append(m.group(1).strip()[:400])
        code = _extract_code(text)
        return _exec_proposal(code, np.array(self.X) if self.X else np.zeros((0, len(R.names))),
                              np.array(self.F), R, self.population, self.seed + gen, work)


def _extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if not m:
        raise ValueError("no python block in proposal")
    return max(m, key=len)


_RUNNER = """
import json, sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=False)
ns = dict(np=np, rng=np.random.default_rng(int(d['seed'])), X=d['X'], F=d['F'], lb=d['lb'], ub=d['ub'],
          names=list(d['names']), P=int(d['P']))
exec(open(sys.argv[2]).read(), ns)
np.save(sys.argv[3], np.asarray(ns['RESULT'], dtype=float))
"""


def _exec_proposal(code: str, X, F, R: Recipe, P: int, seed: int, work: Path) -> np.ndarray:
    """Run LLM-written proposal code in a subprocess with a timeout (never in the harness process)."""
    work.mkdir(parents=True, exist_ok=True)
    F = np.where(np.isfinite(F), F, np.nan) if len(F) else F
    np.savez(work / "ctx.npz", X=X, F=np.nan_to_num(F, nan=-1e9) if len(F) else np.zeros(0), lb=R.lb, ub=R.ub,
             names=np.array(R.names), P=P, seed=seed)
    (work / "proposal.py").write_text(code)
    (work / "_runner.py").write_text(textwrap.dedent(_RUNNER))
    r = subprocess.run([sys.executable, str(work / "_runner.py"), str(work / "ctx.npz"), str(work / "proposal.py"),
                        str(work / "result.npy")], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"proposal code failed: {r.stderr[-600:]}")
    return np.load(work / "result.npy")


# ---------------------------------------------------------------------- recipe build
_BUILD = """
import importlib.util, json, sys
sys.path.insert(0, sys.argv[4])
spec = importlib.util.spec_from_file_location("recipe", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.build(json.loads(sys.argv[2]), sys.argv[3], sys.argv[5])
"""


def build_candidate(recipe: Path, x: dict, parent_dir: Path, out_dir: Path, toolbox_root: Path,
                    timeout_s: int, env: Optional[dict] = None) -> tuple[bool, str]:
    """Run recipe.build(x, parent_dir, out_dir) in a subprocess. Returns (ok, log tail)."""
    import os
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run([sys.executable, "-c", _BUILD, str(recipe), json.dumps(x), str(parent_dir),
                            str(toolbox_root), str(out_dir)], capture_output=True, text=True, timeout=timeout_s,
                           cwd=out_dir, env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired:
        return False, f"build timeout after {timeout_s}s"
    (out_dir / "build.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    return r.returncode == 0, (r.stderr or r.stdout)[-800:]
