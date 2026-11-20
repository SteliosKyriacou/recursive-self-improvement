# Walkthrough: "Make Qwen3.5-0.8B better at math" with `rsi init`

This reproduces the setting of Experiment 8 of "Sparks of Weight-Space Programming" (Takehi et al.,
MIT CSAIL, Nov 2026). The paper's LLM edited Qwen3.5-0.8B's weights directly and raised MATH-500 from
7.9% to 48.1%. Here, `rsi` lets Claude Code (or Antigravity or Codex) try the same task, plus any of
the other strategies you allow.

## 1. Prerequisites (once per machine)

```bash
git clone git@github.com:SteliosKyriacou/recursive-self-improvement.git ~/repos/recursive-self-improvement
cd ~/repos/recursive-self-improvement && ./install.sh
rsi doctor
```
`rsi doctor` should show `cuda=True`, a `vllm` version, and at least one agent CLI (`claude`, `agy`
or `codex`) that is logged in. You need an NVIDIA GPU with about 12 GB or more. The model is 0.8B
parameters (about 1.7 GB in bf16), and vLLM uses the rest of the memory for long generations.

## 2. Create a working folder and initialise the example

```bash
mkdir -p ~/rsi-qwen && cd ~/rsi-qwen
rsi init qwen_math
```
This copies four files into the folder and builds the data. It downloads MATH-500 and the MATH
training split from the Hugging Face Hub.

| file | what it is |
|---|---|
| `goal.yaml` | full harness: the planner may use weight programming, finetuning, architecture, agentic optimization or free-form experiments; the agent has the GPU |
| `goal_paper.yaml` | paper-faithful: weight edits only (plus tuning of the agent's edit recipes); CPU-only agent; strictly metered 50 forward passes; ≤1% of weights changed; no architecture changes |
| `metric.py` | the evaluation: vLLM, thinking on, T=0.6, top_p=0.95, top_k=20, presence_penalty=1.5; grading = last `\boxed{}` after `</think>`, checked with math-verify. Reports `accuracy`, `cut_share` (answers that hit the token limit) and `mean_tokens` |
| `prepare_data.py` | builds `data/`: **dev** = 128 MATH *train* problems (the agents see these), **val** = 100 MATH-500 problems (hidden, decides acceptance), **test** = the other 400 MATH-500 problems (hidden, final report only) |

`rsi init` never overwrites existing files unless you pass `--force`. Re-running it only refreshes
missing files and rebuilds `data/`. Use `--no-data` to skip the data download.

Check everything:
```bash
rsi validate goal.yaml          # expect "✓ ready to run" (one warning about permission bypass is normal)
```

## 3. (Optional) Sanity-check the metric

Evaluate the original model on a few dev problems:
```bash
rsi eval goal.yaml Qwen/Qwen3.5-0.8B --split dev --limit 32 --config '{"max_tokens": 8192}'
```
Measured on an RTX 4070 Ti:

| model | dev problems | max tokens | accuracy | hit token limit | time |
|---|---|---|---|---|---|
| Qwen3.5-0.8B (original) | 32 | 8,192 | 0% | 100% | ~2 min |
| paper's edited model (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`) | 32 | 8,192 | 3% | 97% | ~3 min |
| paper's edited model | 64 | 32,768 | 42% | 33% | ~18 min |

The original model's main failure is looping in its thinking until it hits the token limit. At
8k tokens even the fixed model is cut off almost every time, which is why the goals evaluate
hidden val at the paper's 32k and screen/probe at 16k. You can run the positive control yourself
by passing the second model id.

## 4. Choose and tune a goal

Start with **`goal_paper.yaml`** to reproduce the paper's setting, or **`goal.yaml`** to let the
planner pick strategies. The knobs that matter most for cost (rough RTX 4070 Ti numbers):

| knob | default | effect |
|---|---|---|
| `evaluation.config.max_tokens` | 32768 | hidden val/test generation limit. One 100-problem val ≈ 30 min |
| `population.screen` | n=16, 16k tokens | cheap dev screening of candidates, ≈ 3–4 min each |
| `population.promote_top` | 1–2 | hidden-val evaluations per iteration (the most expensive step) |
| `agentic_optimization` | 3 generations × 4 | 12 screened candidates per optimization experiment |
| `probe.forward_pass_budget` | 50 | agent-side runs on dev (32 problems at 16k ≈ 5–8 min each) |
| `agent.timeout_s` | 10800 | cap per agent experiment (3 h) |
| `search.max_iterations` / `patience` / `wall_clock_hours` | 8–12 / 3–4 / 48 | when to stop |
| `agent.backend` / `agent.model` | claude_code / default | which agent CLI drives the experiments |

Plan for roughly **2–4 hours per iteration** with the defaults. The final test evaluation at the
end (400 problems, best and original) adds about 2–4 hours. For a quicker first look, set
`search.max_iterations: 2`, lower `evaluation.config.max_tokens` to 16384, and use
`population.promote_top: 1`. Expect absolute accuracies to drop at that limit.

## 5. Run

```bash
cd ~/rsi-qwen
rsi run goal_paper.yaml                 # or goal.yaml; add --backend antigravity|codex to switch agents
```
The run folder is `./runs/<name>-<timestamp>/`. The process keeps running until it converges or
reaches the limits, so start it inside `tmux`/`screen` or with
`nohup rsi run goal_paper.yaml > rsi.log 2>&1 &`.

What happens in each iteration (see the main README for details):
1. The planner LLM decides the experiments. It might run one weight-programming agent with one
   candidate, or a population: for example, an agent writes an edit recipe and agentic
   optimization then tunes it.
2. Agents work in `runs/<run>/iterations/iter_NNN/exp*/workspace/`. They probe the model on dev
   through the harness and deliver `candidates/<name>/delta.safetensors` (or `model/`).
3. The harness checks each candidate (guards), screens it on dev, promotes the best to the hidden
   val split, and accepts one only if its val accuracy beats the current best by `min_delta`
   (0.03).

## 6. Monitor, resume, extend

```bash
rsi status runs/<run>                              # per-iteration plans, promotions, acceptances, latest candidates
tail -f runs/<run>/events.jsonl                    # live event log
cat runs/<run>/LEARNINGS.md                        # what the agents have learned so far
cat runs/<run>/iterations/iter_001/plan.json       # the planner's decision and rationale
rsi resume runs/<run>                              # after Ctrl-C / crash (the interrupted iteration is redone)
rsi resume runs/<run> --max-iterations 20          # keep going after it finished
```

## 7. Results

- `runs/<run>/report.md`: val accuracy of the original vs the best model, the held-out **test**
  accuracy of both, the lineage of the best candidate, and every iteration and candidate.
- `runs/<run>/best/model/`: the improved model, a normal HF directory you can load with
  transformers or vLLM. When the lineage is all weight edits, `best/delta_vs_base.safetensors`
  holds exactly the changed tensors, comparable to the paper's 131,072-weight edit.

## Troubleshooting

| symptom | fix |
|---|---|
| `✗ metric file missing` / `split missing` | goal paths are relative to the YAML's folder: run `rsi init qwen_math` in that folder (or `python prepare_data.py` if only `data/` is missing) |
| `NVIDIA driver on your system is too old` | your torch/vLLM build is for a newer CUDA; re-run `./install.sh` (it picks CUDA 12.9 builds for CUDA-12 drivers) |
| vLLM out of memory / "not enough free memory" | another process holds the GPU (`nvidia-smi`); or lower `gpu_memory_utilization` in `evaluation.config` / `probe.config` / `population.screen.config` |
| a stray `VLLM::EngineCore` process keeps GPU memory after a crash | `nvidia-smi` to find its PID, then `kill <pid>` |
| `planner failed; using bandit` in the log | the agent CLI could not answer a one-shot prompt (not logged in, rate limit); the run continues with the bandit controller |
| agent backend not available | `rsi doctor`; install/login to the CLI or set `agent.backend` to one you have |
