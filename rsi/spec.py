"""GoalSpec: the single YAML file that defines a self-improvement run.

It answers:
  * what model to improve                          -> ``target``
  * what "better" means and how to measure it      -> ``goal`` + ``evaluation``
  * how long to keep going                         -> ``search`` (iterations, patience, wall clock)
  * how each iteration explores                    -> ``population`` (+ ``agentic_optimization``)
  * who drives the loop and what rules apply       -> ``agent``, ``probe``, ``guards``
Write one by hand, or let ``rsi new`` interview you and write it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

STRATEGIES = ("weight_programming", "finetuning", "architecture", "agentic_optimization", "free")


class TargetSpec(BaseModel):
    # hf_weights: a Hugging Face / safetensors model. Candidates are weight deltas (same
    # architecture) or full model directories (architecture changes).
    kind: Literal["hf_weights"] = "hf_weights"
    model: str = Field(..., description="HF repo id or local directory")
    revision: Optional[str] = None


class SplitSpec(BaseModel):
    dev: Path = Field(..., description="Visible to the agent (practice / probing / screening)")
    val: Path = Field(..., description="Hidden. Decides accept/reject each iteration")
    test: Optional[Path] = Field(None, description="Hidden. Used only for the final report")


class Objective(BaseModel):
    key: str
    direction: Literal["maximize", "minimize"] = "maximize"


class EvaluationSpec(BaseModel):
    metric: Path = Field(..., description="Python file defining evaluate(model_dir, examples, config) -> dict")
    primary: str = Field("score", description="Key of the dict returned by evaluate() that is optimised")
    direction: Literal["maximize", "minimize"] = "maximize"
    secondary: list[Objective] = Field(
        default_factory=list,
        description="Extra objectives for the Pareto front shown to the agents. 'n_params' is always available.")
    splits: SplitSpec
    config: dict[str, Any] = Field(default_factory=dict, description="Passed to evaluate() for val/test")
    timeout_s: int = 3 * 3600
    python: Optional[str] = Field(None, description="Interpreter for eval workers (default: current)")


class ProbeSpec(BaseModel):
    """Agent-facing, harness-metered access to running the model on the dev split."""
    forward_pass_budget: int = Field(50, description="Probe calls allowed per agent experiment (0 disables)")
    max_examples: int = Field(32, description="Max dev examples per probe call")
    config: dict[str, Any] = Field(default_factory=dict, description="Overrides of evaluation.config for probes")


class ScreenSpec(BaseModel):
    """Cheap first-stage evaluation of every candidate in a population (harness-run, on dev)."""
    n: int = 16
    offset: int = Field(0, description="Dev slice start. Use a slice agents are told not to tune on")
    config: dict[str, Any] = Field(default_factory=dict, description="Overrides of evaluation.config")


class PopulationSpec(BaseModel):
    """CAPS for what the controller may plan per iteration. The controller decides each iteration
    whether to run one focused experiment with a single candidate or a population of experiments,
    and whether/how to screen and how many candidates to promote to hidden val."""
    controller: Literal["llm", "bandit", "fixed"] = Field(
        "llm", description="Who plans each iteration: an LLM planner, a UCB bandit over strategies, or fixed_plan")
    max_experiments: int = Field(3, description="Cap: experiments (agent turns / optimizer runs) per iteration")
    size: int = Field(6, description="Cap: candidates produced by agent experiments per iteration")
    promote_top: int = Field(2, description="Cap: candidates that get the full hidden-val evaluation")
    promote_only_if_screen_improves: bool = Field(
        True, description="When screening, only promote candidates that beat the parent's screen score")
    screen: ScreenSpec = ScreenSpec()
    keep_models: int = Field(4, description="Full-model (architecture) candidates kept on disk as parents")
    fixed_plan: dict[str, Any] = Field(
        default_factory=dict, description='controller=fixed: {"experiments": [...], "screen": bool, "promote_top": n}')


class AgenticOptimizationSpec(BaseModel):
    """Population-based search over a parametric recipe (cf. agentic-optimizer / Metis-Agent):
    each generation an LLM writes NumPy code proposing `population` parameter vectors."""
    generations: int = 3
    population: int = 6
    llm: bool = Field(True, description="LLM-written proposals; False = built-in evolutionary sampler")
    build_timeout_s: int = 1800


class SearchSpec(BaseModel):
    max_iterations: int = 20
    patience: int = Field(5, description="Stop after this many consecutive non-improving iterations")
    min_delta: float = Field(0.0, description="Required improvement on val for acceptance")
    wall_clock_hours: Optional[float] = None
    strategies: list[str] = Field(default_factory=lambda: list(STRATEGIES))

    @model_validator(mode="after")
    def _known(self) -> "SearchSpec":
        bad = [s for s in self.strategies if s not in STRATEGIES]
        if bad:
            raise ValueError(f"unknown strategies {bad}; choose from {STRATEGIES}")
        return self


class AgentSpec(BaseModel):
    backend: Literal["claude_code", "antigravity", "codex", "command"] = "claude_code"
    command: list[str] = Field(default_factory=list, description="backend=command: argv, may use {prompt} {workspace}")
    complete_command: list[str] = Field(
        default_factory=list, description="backend=command: argv for one-shot text completions ({prompt})")
    model: Optional[str] = None
    effort: Optional[str] = None
    timeout_s: int = Field(2 * 3600, description="Max minutes per agent experiment (planner may ask for less)")
    gpu: bool = Field(True, description="If false the agent process sees no GPU (CPU-only analysis); "
                                         "forward passes then only happen via the metered probe")
    permission_mode: str = Field("bypassPermissions", description="claude_code only")
    max_budget_usd: Optional[float] = Field(None, description="claude_code only, per agent turn")
    extra_args: list[str] = Field(default_factory=list)


class GuardSpec(BaseModel):
    max_changed_fraction: float = Field(1.0, description="Max fraction of parameters a delta may touch")
    allowed_keys: Optional[str] = Field(None, description="Regex; tensors outside it may not change (deltas)")
    max_abs_value: float = 1e4
    allow_model_candidates: bool = Field(True, description="Allow full-model (architecture) candidates")
    max_params_ratio: float = Field(1.5, description="Architecture candidates: max params vs. original model")
    allow_custom_code: bool = Field(False, description="Allow model dirs with *.py / auto_map (trust_remote_code)")
    allowed_config_keys: Optional[list[str]] = Field(
        None, description="Architecture candidates: config keys that may change (None = any)")
    audit_transcript: bool = Field(True, description="Reject candidates whose agent touched hidden splits")
    forbidden_patterns: list[str] = Field(
        default_factory=list,
        description="Extra strings that must not appear in the agent's actions (e.g. benchmark test-set ids)")


class GoalSpec(BaseModel):
    name: str
    goal: str = Field(..., description="Natural-language objective given to the agent")
    target: TargetSpec
    evaluation: EvaluationSpec
    probe: ProbeSpec = ProbeSpec()
    search: SearchSpec = SearchSpec()
    population: PopulationSpec = PopulationSpec()
    agentic_optimization: AgenticOptimizationSpec = AgenticOptimizationSpec()
    agent: AgentSpec = AgentSpec()
    guards: GuardSpec = GuardSpec()
    notes: str = Field("", description="Extra domain knowledge for the agent")

    @model_validator(mode="after")
    def _check(self) -> "GoalSpec":
        if self.search.max_iterations < 1:
            raise ValueError("search.max_iterations must be >= 1")
        if self.population.promote_top < 1:
            raise ValueError("population.promote_top must be >= 1")
        return self

    @property
    def maximize(self) -> bool:
        return self.evaluation.direction == "maximize"

    def better(self, a: Optional[float], b: Optional[float], margin: Optional[float] = None) -> bool:
        """True if score a beats b by more than margin (default: search.min_delta)."""
        if a is None:
            return False
        if b is None:
            return True
        d = self.search.min_delta if margin is None else margin
        return a > b + d if self.maximize else a < b - d

    def sort_key(self, score: Optional[float]) -> float:
        """Larger = better, for sorting candidates."""
        if score is None:
            return float("-inf")
        return score if self.maximize else -score


def load_spec(path: str | Path) -> GoalSpec:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    # Backwards compatibility.
    strats = (raw.get("search") or {}).get("strategies")
    if strats:
        raw["search"]["strategies"] = ["architecture" if s == "architecture_search" else s for s in strats]
    spec = GoalSpec.model_validate(raw)
    base = path.parent  # relative paths resolve against the spec file's directory
    ev = spec.evaluation
    ev.metric = (base / ev.metric).resolve()
    for k in ("dev", "val", "test"):
        p = getattr(ev.splits, k)
        if p is not None:
            setattr(ev.splits, k, (base / p).resolve())
    for attr in ("command", "complete_command"):
        setattr(spec.agent, attr, [str((base / c).resolve()) if (base / c).is_file() else c
                                   for c in getattr(spec.agent, attr)])
    if (base / spec.target.model).exists() or spec.target.model.startswith((".", "/")):
        spec.target.model = str((base / spec.target.model).resolve())
    return spec
