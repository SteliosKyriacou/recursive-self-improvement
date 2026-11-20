"""MATH accuracy for a chat model, following the protocol of "Sparks of Weight-Space Programming"
Exp 8: vLLM, thinking on, temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, last
\\boxed{} graded with math-verify. Token limit and #samples are configurable (the paper used
32,768 tokens; smaller limits make each evaluation much cheaper on a single consumer GPU).

evaluate() returns:
  score / accuracy  mean correctness over problems × samples
  cut_share         fraction of samples that hit the token limit (the base model's main failure)
  mean_tokens       mean generated tokens per sample
  per_example       [{id, correct, tokens, cut, head, tail}]  (first sample's text, truncated)
"""
from __future__ import annotations

import os

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DEFAULTS = dict(max_tokens=16384, samples=1, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                thinking=True, seed=0, gpu_memory_utilization=0.85, max_num_seqs=128, enforce_eager=False,
                head_chars=1200, tail_chars=2500)


def last_boxed(s: str):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def grade(completion: str, gold: str) -> bool:
    """Last \\boxed{} of the answer part (after </think> if present), checked with math-verify.
    Only the short boxed string is parsed: running math-verify over thousands of tokens of looping
    reasoning text can take minutes per sample."""
    from math_verify import parse, verify
    text = completion.split("</think>")[-1] if "</think>" in completion else completion
    pred = last_boxed(text)
    if pred is None or len(pred) > 400:
        return False
    if pred.strip() == gold.strip():
        return True
    try:
        return bool(verify(parse("$" + gold + "$"), parse("$\\boxed{" + pred + "}$")))
    except Exception:  # noqa: BLE001
        return False


def evaluate(model_dir: str, examples: list[dict], config: dict) -> dict:
    cfg = {**DEFAULTS, **config}
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [tok.apply_chat_template([{"role": "user", "content": e["problem"] + PROMPT_SUFFIX}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=bool(cfg["thinking"])) for e in examples]
    max_prompt = max(len(tok(p).input_ids) for p in prompts)
    llm = LLM(model=model_dir, dtype="bfloat16", seed=int(cfg["seed"]),
              gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
              max_model_len=int(cfg["max_tokens"]) + max_prompt + 16,
              max_num_seqs=int(cfg["max_num_seqs"]), enforce_eager=bool(cfg["enforce_eager"]),
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(n=int(cfg["samples"]), temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]),
                        top_k=int(cfg["top_k"]), presence_penalty=float(cfg["presence_penalty"]),
                        max_tokens=int(cfg["max_tokens"]), seed=int(cfg["seed"]))
    outs = llm.generate(prompts, sp)

    per, n_ok, n_cut, n_tok, n_all = [], 0, 0, 0, 0
    for e, o in zip(examples, outs):
        corr, toks, cuts = [], [], []
        for c in o.outputs:
            ok = grade(c.text, e["answer"])
            corr.append(int(ok)); toks.append(len(c.token_ids)); cuts.append(int(c.finish_reason == "length"))
        n_ok += sum(corr); n_cut += sum(cuts); n_tok += sum(toks); n_all += len(corr)
        t = o.outputs[0].text
        per.append({"id": e.get("id"), "answer": e["answer"], "correct": corr, "tokens": toks, "cut": cuts,
                    "head": t[: cfg["head_chars"]], "tail": t[-cfg["tail_chars"]:] if len(t) > cfg["head_chars"] else ""})
    acc = n_ok / max(1, n_all)
    return {"score": acc, "accuracy": acc, "cut_share": n_cut / max(1, n_all), "mean_tokens": n_tok / max(1, n_all),
            "n": len(examples), "samples": int(cfg["samples"]), "max_tokens": int(cfg["max_tokens"]),
            "per_example": per}
