# rsi — a recursive self-improvement harness

Tell the harness **which model to improve**, **what "better" means and how to measure it**, and
**how long it may try**. Either write that as one YAML file, or let `rsi new` interview you and write it.
A research-director LLM then plans every iteration. It decides which strategy to use (weight-space
programming, gradient finetuning, architecture search, agentic optimization or a free-form
experiment) and whether to run one focused experiment or a population of them. Coding agents
(Claude Code by default; Antigravity or Codex optionally) carry the experiments out. The loop runs
until the score converges or the budget runs out.

```
            rsi new  ──(chat: model, goal, metric, data, budget)──►  goal.yaml + metric.py + data/
                                                                         │
 rsi run goal.yaml ─► baseline ─► iteration k ───────────────────────────┴──────────────────────────
   1 PLAN      controller (LLM planner | UCB bandit | fixed) reads history, strategy credit, learnings,
               recipes, and decides the iteration's shape: 1 experiment / 1 candidate … up to a population
   2 RUN       agent experiments ─ weight_programming · finetuning · architecture · free
                 (agent works in a workspace, probes the model on dev through a metered server,
                  delivers ≤ n candidates: weight deltas or full model dirs; may write recipes)
               agentic_optimization ─ G generations × P members of a recipe; an LLM writes the
                 NumPy proposal code each generation (Metis-Agent style), every member is screened
   3 GUARD     delta / model integrity, parameter budget, no custom code, audit of agent actions
   4 SCREEN    cheap dev evaluation of the population (when the plan asks for it)
   5 PROMOTE   top-m → evaluation on the HIDDEN val split (frozen metric, isolated subprocess)
   6 SELECT    accept the best iff it beats the current best by min_delta → new parent for later work
   7 REMEMBER  LEARNINGS.md + toolbox (helpers, recipes) persist → the improver improves itself
   8 STOP?     max_iterations · patience (convergence) · wall clock
 ─► export best model ─► HIDDEN test evaluation, best vs. original ─► report.md
```

## Contents
- [Design](#design)
- [Install](#install)
- [Quick start](#quick-start)
- [Creating a goal: `rsi new` or by hand](#creating-a-goal)
- [Strategies](#strategies)
- [How an iteration is planned](#how-an-iteration-is-planned)
- [What an agent sees](#what-an-agent-sees)
- [Run directory](#run-directory)
- [Integrity and limits](#integrity-and-limits)

---

## Design
- **Candidates are artifacts, not code paths.** A candidate is either a *weight delta*
  (replacement tensors, same architecture) or a *full model directory* (architecture changes).
  Tokenizer and chat template always come from the parent, and custom model code is refused.
  The agent therefore cannot change how the frozen evaluator prompts, decodes or scores the model.
- **Three splits.** `dev` is visible to the agent and used for screening. A hidden `val` split
  decides acceptance, and a hidden `test` split is used only in the final report. (Lesson from
  RWNN-LMM: the search found a future-token leak that "improved" loss from 3.76 to 3.03.)
- **Lineage.** Every candidate records its parent. Experiments can branch from any archived
  candidate, not just the current best. Delta chains are materialized on demand.
- **Recursive.** The agents' shared `LEARNINGS.md` and the toolbox they extend (helpers and
  recipes) carry into every later iteration and every later planner decision. The toolbox version
  after each experiment is snapshotted for audit.

Background: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights", Takehi, Teoh, Zhang & Khattab (MIT CSAIL), blog post, Nov 2026 —
https://rikiyat.github.io/blog/weight-space-programming/ (no arXiv paper; logs at
https://github.com/RikiyaT/lm_understand-logs, edited model at
https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming). The ideas also come from
`~/repos/agentic-optimizer` (Metis-Agent: LLM-written population proposals, strategy log) and
`~/repos/RWNN-LMM` (LLM-driven architecture search, weight inheritance, leak-proof evaluation).

---

## Install

Requires Linux, Python ≥ 3.10 and an NVIDIA GPU for real models. The toy example runs on CPU.

```bash
git clone <this repo> && cd recursive-self-improvement
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core harness
```

**Agent CLI.** Install at least one, and make sure it is logged in:

| backend | binary | used for |
|---|---|---|
| `claude_code` (default) | `claude` | experiments (`claude -p … --permission-mode bypassPermissions`), planner/optimizer completions (`--permission-mode dontAsk`), `rsi new` (interactive) |
| `antigravity` | `agy` | same roles (`--dangerously-skip-permissions` / `--mode plan` / `--prompt-interactive`) |
| `codex` | `codex` | same roles (`exec --dangerously-bypass-approvals-and-sandbox` / `--sandbox read-only` / interactive) |
| `command` | anything | your own script, e.g. the scripted toy agent |

> ⚠️ For unattended runs, agents use their permission-bypass modes. They can run any command as
> your user, so run them on a machine or account you are comfortable with.

**Extras for the Qwen-math example** (vLLM generation, math-verify grading):
```bash
nvidia-smi | grep "CUDA Version"           # check the driver's CUDA version

# Driver CUDA ≥ 13.0:
pip install -e ".[math]"

# Driver CUDA 12.x (e.g. driver 560 / CUDA 12.6) — use the cu129 builds:
pip install "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps torchcodec torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install math-verify datasets pandas pyarrow transformers

python -c "import torch, vllm; print(torch.cuda.is_available(), vllm.__version__)"
```

Check the install (CPU, about 1 min): `pip install pytest && python -m pytest -q tests`

Keep the venv active. The agents, eval workers and recipe builders all use the same interpreter.

---

## Quick start

```bash
# 0. Toy run: CPU, ~1 min, scripted agent + agentic optimization, no LLM needed
python examples/toy/make_toy.py
rsi run examples/toy/goal.yaml --run-dir runs/toy && cat runs/toy/report.md

# 1. Design a new goal by chatting with an agent
rsi new my-goal                                    # or: rsi new my-goal --backend codex --prompt "make X better at Y"

# 2. Or start from an existing YAML: "Make Qwen3.5-0.8B better at math" (the paper's Exp 8)
python examples/qwen_math/prepare_data.py          # dev = 128 MATH-train · val = 100 / test = 400 MATH-500
rsi run examples/qwen_math/goal.yaml               # all strategies, LLM planner, agent has GPU
rsi run examples/qwen_math/goal_paper.yaml         # paper-faithful: weight edits (+ tuning), CPU-only agent

# 3. Watch, resume, extend
rsi status runs/<run>                              # iterations, plans, candidates
tail -f runs/<run>/events.jsonl                    # live log
rsi resume runs/<run>                              # after a crash / Ctrl-C (redoes the interrupted iteration)
rsi resume runs/<run> --max-iterations 30          # extend a finished run

# Utilities
rsi validate goal.yaml                             # pre-flight checks (rsi run does this too)
rsi eval goal.yaml Qwen/Qwen3.5-0.8B --split dev --limit 8 --config '{"max_tokens": 4096}'
rsi run goal.yaml --backend antigravity --model <name> --controller bandit
```

---

## Creating a goal

### With `rsi new` (interactive)
`rsi new NAME` creates `goals/NAME/` and opens an interactive session with your agent CLI, running
a "goal designer" brief. The designer asks about:
- the model and your hardware;
- the goal, and what must not get worse;
- how to measure it (benchmark, protocol, metric);
- where dev, val and test data come from;
- which strategies and limits are allowed;
- the budget.

It then writes `goal.yaml`, `metric.py` and (if needed) `prepare_data.py`, builds the data splits,
runs `rsi validate` and a smoke evaluation, and summarises the plan for you. When you exit the
session, `rsi new` validates again and asks whether to start the run.

### By hand
One YAML file plus one metric file. Relative paths resolve against the YAML's directory. The full
schema is `GoalSpec` in `rsi/spec.py`.

```yaml
name: my-goal
goal: |                                  # what the agents are told to achieve
  Make <model> better at <task>. Constraints: ...
notes: |                                 # optional domain knowledge / cost hints for the agents
  ...

target:   {kind: hf_weights, model: org/model-id}          # or ./local/dir

evaluation:
  metric: metric.py                      # evaluate(model_dir, examples, config) -> dict
  primary: accuracy
  direction: maximize
  secondary: [{key: n_params, direction: minimize}]        # optional; Pareto front shown to agents
  splits: {dev: data/dev.jsonl, val: data/val.jsonl, test: data/test.jsonl}
  config: {max_tokens: 4096}             # passed to evaluate() for val/test
  timeout_s: 7200

probe:                                   # agents' metered access to run the model on dev
  forward_pass_budget: 50                # per agent experiment
  max_examples: 32
  config: {max_tokens: 2048}

search:
  max_iterations: 20
  patience: 5                            # converged = this many non-improving iterations in a row
  min_delta: 0.01                        # required val gain to accept
  wall_clock_hours: 24
  strategies: [weight_programming, finetuning, architecture, agentic_optimization, free]

population:                              # CAPS — the planner decides the actual shape
  controller: llm                        # llm | bandit | fixed (fixed_plan: {...})
  max_experiments: 2
  size: 4                                # candidates from agent experiments per iteration
  promote_top: 2                         # hidden-val evaluations per iteration
  promote_only_if_screen_improves: true
  screen: {n: 16, offset: 0, config: {max_tokens: 2048}}   # cheap first-stage eval on dev[offset:offset+n]
  keep_models: 4                         # full-model candidates kept on disk as possible parents

agentic_optimization: {generations: 3, population: 6, llm: true}

agent:
  backend: claude_code                   # claude_code | antigravity | codex | command
  model: null
  timeout_s: 5400                        # cap per agent experiment
  gpu: true                              # false = agent sees no GPU (strictly metered forward passes)
  max_budget_usd: null                   # claude_code only, per agent turn

guards:
  max_changed_fraction: 0.25             # deltas: fraction of all parameters
  allowed_keys: null                     # deltas: regex of tensors that may change
  allow_model_candidates: true           # architecture candidates allowed?
  max_params_ratio: 1.5                  # architecture candidates: size vs. original
  allowed_config_keys: null              # architecture candidates: config keys that may change
  forbidden_patterns: []                 # strings the agents' actions must not contain
```

**Metric contract.** The metric runs in a fresh subprocess for every evaluation, and is copied to
`runs/<run>/frozen/` when the run starts:
```python
def evaluate(model_dir: str, examples: list[dict], config: dict) -> dict:
    return {"accuracy": 0.42, "per_example": [...], "other_scalar": 1.0}
```

---

## Strategies

| strategy | who does it | candidate | toolbox |
|---|---|---|---|
| `weight_programming` | agent reads weights and activations, finds the mechanism, edits weights directly (no gradients) | delta | `weight_programming/wsp.py`: capture, mean-diff directions, top writers, logit lens, ablate/steer edits |
| `finetuning` | agent trains chosen tensors by gradient descent (SFT, self-distillation, …) | delta (or model) | `finetuning/sft.py` |
| `architecture` | agent changes the structure, inheriting weights: drop / duplicate / reorder layers, prune MLP width (hybrid `layer_types` stay consistent), optional healing finetune | full model dir | `architecture/arch.py` |
| `agentic_optimization` | harness: an LLM writes NumPy proposal code for G generations × P members of a **recipe**; each member is built and screened | deltas or models | `agentic_optimization/README.md`, `recipes/` |
| `free` | agent picks any approach or a mix | either | all |

**Recipes** (`toolbox/recipes/<name>.py`) connect the strategies. An agent that finds a mechanism
with tunable knobs (which layers, how strong, how many neurons) writes it as a parametric family:
`SPACE` (bounds), optional `DEFAULT`, and `build(x, model_dir, out_dir)`. The planner can then
schedule agentic optimization to tune it. See `rsi/strategies/agentic_optimization/README.md`.

Graph-level NAS in the style of RWNN-LMM (free-form DAGs, custom code, training from scratch)
needs its own target kind and is on the roadmap. `hf_weights` targets refuse custom model code.

---

## How an iteration is planned
The planner (`population.controller: llm`) is a one-shot call to your agent backend. It sees:
- the goal;
- the scores so far;
- every iteration and candidate (with screen and val scores, status and errors);
- a per-strategy credit table (experiments → candidates → valid → promoted → accepted);
- the recipes available (and any invalid ones);
- the candidates usable as parents;
- the agents' accumulated learnings.

It returns a JSON plan: the experiments to run (strategy, parent, specific brief, number of
candidates, minutes, or recipe/generations/population), whether to screen, and how many candidates
to promote. Nothing is forced. One promising idea gets a single experiment with one candidate,
which goes straight to val. Uncertainty gets a population. The plan is clipped to the `population`
caps. If the planner fails, the UCB bandit takes over (one experiment, one candidate). Plans are
saved in `iterations/iter_NNN/plan.json`.

## What an agent sees
`runs/<run>/iterations/iter_NNN/exp<i>_<strategy>/workspace/`:

| file | purpose |
|---|---|
| `AGENTS.md` (= `CLAUDE.md` = `GEMINI.md`) | goal, assignment (strategy, brief, parent), scores and acceptance threshold, deliverable, rules |
| `HISTORY.md` | iterations, candidates, strategy credit, recipes, Pareto front |
| `LEARNINGS.md` | shared notes of all agents and optimizer runs; persists |
| `toolbox/` | strategy packs + recipes; agent-editable; persists |
| `data/dev.jsonl`, `tools/probe.py` | dev data; metered model runs (`--candidate candidates/<name>` or `--delta`) |
| `reference/metric.py`, `reference/tensors.json` | read-only metric copy and the parent's tensor list |
| `candidates/<name>/` | **deliverable**: `delta.safetensors` or `model/`, plus `meta.json` |

## Run directory
```
runs/<run>/
  state.json  events.jsonl  report.md  spec.yaml  frozen/ (metric, screen slice)
  candidates/<id>/             every candidate: delta.safetensors or model/, meta.json
  cache/                       materialized delta chains (LRU)
  screens/<id>.json            screen results
  iterations/iter_NNN/         plan.json, record.json, planner/, val/<id>.json,
                               exp<i>_<strategy>/{workspace, agent_transcript.jsonl, probes/} | {gen*/, optimizer.json}
  LEARNINGS.md  toolbox/  toolbox_history/
  best/model/ (+ delta_vs_base.safetensors when the lineage is all deltas)
  hidden_<random>/             val/test copies (never exposed to agents)
  final/                       test results, best and original
```

## Integrity and limits
- The hidden splits live on the same disk the agents can access. The harness hides them under a
  random directory name, and it audits the agents' **actions** (tool calls and shell commands,
  not what they read or think) for hidden paths and `forbidden_patterns`. Candidates of a
  violating experiment are rejected. This is a deterrent, not a sandbox. Container isolation is on
  the roadmap.
- With `agent.gpu: true` agents can run the model themselves, so the probe budget is advisory.
  `gpu: false` makes it strict.
- Screening uses a fixed dev slice that agents are told about. Selection on a small val split is
  noisy: use `min_delta`, more samples, or a bigger split, and trust the held-out test number.
- Cost is dominated by evaluations. Budget roughly
  `(screened candidates × screen cost) + (promoted × val cost)` per iteration, plus agent time.
  On an RTX 4070 Ti with Qwen3.5-0.8B: about 3–4 min per 16-problem screen at 16k tokens, and
  about 30 min per 100-problem val at 32k tokens.
- Recipe metadata (`SPACE`, `DEFAULT`, `DESCRIPTION`) is evaluated with restricted builtins.
  `build()`, proposal code and agent code run as your user, in subprocesses with timeouts.

## Repository layout
```
rsi/spec.py          GoalSpec (YAML schema)          rsi/orchestrator.py   the loop
rsi/controller.py    planner / bandit / fixed        rsi/optim.py          agentic optimization + recipes
rsi/archive.py       candidate records, lineage      rsi/targets.py        deltas, model candidates, guards
rsi/evaluator.py     frozen subprocess evaluation    rsi/probe.py          metered dev probes
rsi/agents/          claude_code / antigravity / codex / command backends
rsi/validate.py      pre-flight checks               rsi/cli.py            new / run / resume / status / validate / eval
rsi/templates/       AGENTS.md (experiment), PLANNER.md, DESIGNER.md (rsi new)
rsi/strategies/      toolbox: common/, weight_programming/, finetuning/, architecture/, agentic_optimization/, recipes/
examples/toy/        CPU smoke test        examples/qwen_math/   Exp 8: goal.yaml, goal_paper.yaml, metric.py, prepare_data.py
tests/               CPU tests: guards, audit, store, architecture surgery, recipes/optimizer, planner caps, loop + resume
```
