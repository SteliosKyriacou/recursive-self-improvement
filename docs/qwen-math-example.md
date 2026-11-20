# Walkthrough: set up "Make Qwen3.5-0.8B better at math" by talking to `rsi new`

You won't copy any example files. You describe the problem to `rsi new`, and a goal-designer agent
interviews you. It then researches the evaluation protocol, writes the goal, the metric and the
data pipeline, builds the data, and checks everything. After that the harness improves the model
until it converges.

The problem is Experiment 8 of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang & Khattab,
MIT CSAIL, Nov 2026, https://rikiyat.github.io/blog/weight-space-programming/). In the paper, an
LLM edited Qwen3.5-0.8B's weights directly and raised MATH-500 accuracy from 7.9% to 48.1%.

## 1. Prerequisites (once per machine)

```bash
git clone git@github.com:SteliosKyriacou/recursive-self-improvement.git ~/repos/recursive-self-improvement
cd ~/repos/recursive-self-improvement && ./install.sh
rsi doctor
```
`rsi doctor` should show `cuda=True`, a `vllm` version, and at least one agent CLI that is logged in
(`claude`, `agy` or `codex`). You need an NVIDIA GPU with about 12 GB or more.

## 2. Start the conversation

```bash
mkdir -p ~/rsi-qwen && cd ~/rsi-qwen
rsi new qwen-math                       # --backend antigravity | codex to use another agent
```
This opens an interactive agent session in `~/rsi-qwen/goals/qwen-math/`. The designer has the
goal schema, an annotated YAML skeleton and a metric template. It does not have a ready-made Qwen
example: it builds the goal from what you tell it and from primary sources.

## 3. What to tell it

The designer asks about the model, the goal, the measurement, the data, what may change, and the
budget. You can answer one question at a time, or give everything up front. For example:

> I want to make **Qwen/Qwen3.5-0.8B** better at competition math, MATH-500 style, in **thinking
> mode**. Follow the evaluation protocol of the "Sparks of Weight-Space Programming" blog post
> (Exp 8): vLLM, thinking on, temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, final
> answer = last `\boxed{}` after `</think>`, graded with math-verify. The base model mostly loops
> until the token limit.

> Data: use ~128 **MATH train** problems as dev, and split **MATH-500** into **100 hidden val** and
> **400 hidden test**. Agents must never touch MATH-500 itself.

> Hardware: one RTX 4070 Ti (12 GB). Allow **all strategies**. At most **10 iterations**, patience
> **3**. Use **claude_code** for the agents.

Variations worth considering:
- *Paper-faithful:* "Only direct weight edits: no gradient training, no architecture changes,
  change at most 1% of the weights, and the agent gets no GPU, only 50 metered forward passes."
- *Cheaper first run:* "Use a 16k token limit for val and 2 iterations."
- *Smaller model:* "Also report the parameter count as a secondary objective to minimize."

The designer will then do the following. A test run took about 6 minutes, plus model and dataset
downloads.
1. Look up the protocol (blog post, the authors' code and logs, the model card) and note any
   discrepancies.
2. Write `prepare_data.py` and run it. It builds `data/dev.jsonl` (MATH train), `data/val.jsonl`
   and `data/test.jsonl` (MATH-500), checks for overlap, and writes a manifest with dataset
   fingerprints.
3. Write `metric.py`. It uses vLLM with the paper's prompt and sampling, extracts the last
   `\boxed{}` after `</think>`, grades with math-verify, and reports `accuracy` plus `cut_share`
   (answers that hit the token limit).
4. Write `goal.yaml` with budgets that fit your GPU. It names MATH-500 as forbidden to the agents,
   and adds cost notes for the planner.
5. Run `rsi validate`, run a smoke evaluation of the original model, estimate the cost per
   evaluation and per iteration, and record every assumption in `DESIGN_NOTES.md`.
6. Summarise the result and ask whether you are happy with it.

Read `DESIGN_NOTES.md`. If you disagree with a choice, say so in the conversation and the designer
changes it. For example, the paper averages over 4 samples per problem, and the designer may
choose 1 to save cost.

## 4. Start the run

When you exit the session (`/exit` or Ctrl-D), `rsi new` validates the goal and asks:
```
Start the self-improvement run now? [y/N]
```
A run lasts hours, often days: with a 32k token limit, one 100-problem val evaluation takes about
30 minutes on an RTX 4070 Ti. You can answer `N` and start it later inside `tmux`/`screen`:
```bash
cd ~/rsi-qwen
rsi run goals/qwen-math/goal.yaml       # runs go to ./runs/<name>-<timestamp>/
```

What happens in each iteration:
1. The planner LLM decides the experiments. It might run one weight-programming agent with one
   candidate, or a population: for example, an agent writes an edit recipe and agentic
   optimization then tunes it.
2. Agents work in `runs/<run>/iterations/iter_NNN/exp*/workspace/`. They probe the model on dev
   through the harness and deliver `candidates/<name>/delta.safetensors` (or `model/`).
3. The harness checks each candidate (guards), screens it on dev, promotes the best to the hidden
   val split, and accepts one only if its val accuracy beats the current best by `min_delta`.

## 5. Monitor, resume, extend

```bash
rsi status runs/<run>                              # plans, promotions, acceptances, latest candidates
tail -f runs/<run>/events.jsonl                    # live event log
cat runs/<run>/LEARNINGS.md                        # what the agents have learned
cat runs/<run>/iterations/iter_001/plan.json       # the planner's decision and rationale
rsi resume runs/<run>                              # after Ctrl-C / crash
rsi resume runs/<run> --max-iterations 20          # keep going after it finished
```

## 6. Results

- `runs/<run>/report.md`: val accuracy of the original vs the best model, the held-out **test**
  accuracy of both, the lineage of the best candidate, and every iteration and candidate.
- `runs/<run>/best/model/`: the improved model, a normal HF directory. When the lineage is all
  weight edits, `best/delta_vs_base.safetensors` holds exactly the changed tensors, comparable to
  the paper's 131,072-weight edit.

Reference points measured on the RTX 4070 Ti with the paper's protocol, on dev (MATH train):

| model | problems | max tokens | accuracy | hit token limit |
|---|---|---|---|---|
| Qwen3.5-0.8B (original) | 32 | 8,192 | 0% | 100% |
| paper's edited model (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`) | 32 | 8,192 | 3% | 97% |
| paper's edited model | 64 | 32,768 | 42% | 33% |

## Non-interactive variant

To script the setup, or if you have nothing to add beyond one description:
```bash
rsi new qwen-math --headless --prompt "<the description from step 3>"
```
The designer then makes every remaining decision itself and lists it in `DESIGN_NOTES.md`.

## Troubleshooting

| symptom | fix |
|---|---|
| `rsi run goal.yaml` says metric/splits missing | a goal is a folder (goal.yaml + metric.py + data/); run the goal created by `rsi new` (`goals/<name>/goal.yaml`) or its `prepare_data.py` |
| `NVIDIA driver on your system is too old` | your torch/vLLM build is for a newer CUDA; re-run `./install.sh` |
| vLLM out of memory / "not enough free memory" | another job holds the GPU (`nvidia-smi`), e.g. a run you started earlier; or lower `gpu_memory_utilization` in the goal |
| a stray `VLLM::EngineCore` keeps GPU memory after a crash | `nvidia-smi` to find its PID, then `kill <pid>` |
| `planner failed; using bandit` in the log | the agent CLI could not answer (not logged in, rate limit); the run continues with the bandit controller |
| you want to redo the design | `rsi new qwen-math` again; the designer sees the existing files and continues from them |
