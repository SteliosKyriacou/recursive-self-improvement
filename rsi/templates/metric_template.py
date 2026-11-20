"""Metric template for rsi. Adapt it to the task; keep the evaluate() signature.

Contract
--------
evaluate(model_dir, examples, config) -> dict
  * model_dir: a complete HF model directory (the candidate being scored)
  * examples:  the JSONL rows of one split (dev slice, val or test)
  * config:    evaluation.config from goal.yaml, overridden by probe/screen configs
  * returns:   {<primary key>: float, <other scalars>..., "per_example": [...]}  (per_example
               is shown to agents for dev probes only — include short text heads/tails there)

It runs in a fresh subprocess for every evaluation, so it may load the model freely.

Hard-won tips
-------------
* Use vLLM for generation-heavy tasks (orders of magnitude faster than HF generate); pass
  `limit_mm_per_prompt={"image": 0, "video": 0}` for multimodal checkpoints.
* Follow the model card's recommended prompt format and sampling; set them explicitly in
  SamplingParams so a candidate cannot change them.
* Extract the answer cheaply (e.g. last \\boxed{...}) before any expensive parser/verifier:
  running symbolic parsers over thousands of tokens of looping text can hang for minutes.
* Report a "hit the token limit" share when the model can ramble; it is a great secondary signal.
* Measure the cost: run once with --limit 4 and with the full split before choosing budgets.
"""
from __future__ import annotations

DEFAULTS = dict(max_tokens=4096, samples=1, temperature=0.0, top_p=1.0, seed=0,
                gpu_memory_utilization=0.85, head_chars=1000, tail_chars=2000)


def build_prompt(tok, example: dict, cfg: dict) -> str:
    # TODO: task-specific prompt; use the chat template for chat models.
    return tok.apply_chat_template([{"role": "user", "content": example["prompt"]}],
                                   tokenize=False, add_generation_prompt=True)


def score(completion: str, example: dict) -> float:
    # TODO: task-specific scoring (exact match, verifier, unit tests, ...). Return 0..1.
    return float(completion.strip() == str(example["answer"]).strip())


def evaluate(model_dir: str, examples: list[dict], config: dict) -> dict:
    cfg = {**DEFAULTS, **config}
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [build_prompt(tok, e, cfg) for e in examples]
    max_prompt = max(len(tok(p).input_ids) for p in prompts)
    llm = LLM(model=model_dir, dtype="bfloat16", seed=int(cfg["seed"]),
              gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
              max_model_len=int(cfg["max_tokens"]) + max_prompt + 16)
    sp = SamplingParams(n=int(cfg["samples"]), temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]),
                        max_tokens=int(cfg["max_tokens"]), seed=int(cfg["seed"]))
    outs = llm.generate(prompts, sp)
    per, tot, cut, n = [], 0.0, 0, 0
    for e, o in zip(examples, outs):
        s = [score(c.text, e) for c in o.outputs]
        c = [int(x.finish_reason == "length") for x in o.outputs]
        tot, cut, n = tot + sum(s), cut + sum(c), n + len(s)
        t = o.outputs[0].text
        per.append({"id": e.get("id"), "score": s, "cut": c, "head": t[:cfg["head_chars"]],
                    "tail": t[-cfg["tail_chars"]:] if len(t) > cfg["head_chars"] else ""})
    return {"accuracy": tot / max(1, n), "cut_share": cut / max(1, n), "n": len(examples), "per_example": per}
