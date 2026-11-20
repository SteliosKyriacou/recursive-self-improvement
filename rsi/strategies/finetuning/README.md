# Strategy: gradient-based finetuning (emitted as a weight delta)

Classic weight updates by gradient descent. Output is still a delta of the tensors you trained.

## Ideas that tend to work for self-improvement
- **Rejection-sampling self-distillation (STaR / RFT):** sample the current model on *training*
  problems (public train splits, `data/dev.jsonl`), keep only verified-correct, well-formed,
  non-truncated solutions, SFT on them. Grade with the same logic as `reference/metric.py`.
- **Targeted SFT on a failure mode:** e.g. teach the model to stop and box the answer instead of
  looping; short, clean target completions.
- **Few tensors, low LR:** restrict `--params` to a band of layers / the MLPs; lr 1e-6…2e-5;
  check you do not damage formatting with a probe before submitting.
- **Combine with weight programming:** e.g. finetune, then ablate a residual failure direction.

## Tool
`python toolbox/finetuning/sft.py --data X.jsonl --params REGEX --lr 1e-5 --out candidates/<name>/delta.safetensors`
(see the docstring). Write your own trainer if you need RL / DPO / LoRA — just finish with
`toolbox.common.delta.save_delta` or `diff_state`.

## Data hygiene
Allowed: `data/dev.jsonl`, public **train** splits (e.g. `DigitalLearningGmbH/MATH-lighteval` train,
`openai/gsm8k` train). Forbidden: any benchmark test split the run is evaluated on (see AGENTS.md).

## GPU etiquette
The probe server also needs the GPU. Free your memory (`del model; torch.cuda.empty_cache()`, or
exit the process) before calling `tools/probe.py`, and keep training below ~5 GB when probing in
parallel.
