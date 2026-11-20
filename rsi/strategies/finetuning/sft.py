"""Minimal gradient-based finetuning that emits a delta (only trainable tensors are saved).

    python toolbox/finetuning/sft.py --data my_sft.jsonl --params "layers\\.(1[2-9]|2[0-3])\\.mlp\\." \
        --lr 1e-5 --epochs 1 --max-len 2048 --out candidates/<name>/delta.safetensors

Data: JSONL with {"prompt": ..., "response": ...}. The prompt is wrapped in the model's chat
template; loss is on response tokens only. Restricting --params keeps the delta small and the
edit conservative (and lets it fit in GPU memory next to the probe server).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolbox.common.delta import parent_dir, save_delta  # noqa: E402
from toolbox.weight_programming.wsp import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--params", default=r"mlp\.(down|up|gate)_proj", help="regex of trainable parameter names")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4, help="sequences per optimizer step (grad accumulation)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--chat", action="store_true", default=True)
    ap.add_argument("--thinking", action="store_true", help="enable_thinking in chat template")
    ap.add_argument("--out", help="e.g. candidates/sft1/delta.safetensors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", help="write a full model dir instead of a delta (architecture candidates)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    md = a.model_dir or parent_dir()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(md, device=dev, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    pat = re.compile(a.params)
    train_names = []
    for n, p in model.named_parameters():
        p.requires_grad = bool(pat.search(n)) and "visual" not in n
        if p.requires_grad:
            train_names.append(n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {len(train_names)} tensors, {n_train:,} params")
    if not train_names:
        raise SystemExit("no parameter matches --params")
    if dev == "cuda":
        model.gradient_checkpointing_enable()
    model.train()

    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    def encode(r):
        if a.chat:
            kw = {"enable_thinking": a.thinking}
            prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, **kw)
        else:
            prompt = r["prompt"]
        p = tok(prompt, add_special_tokens=False).input_ids
        c = tok(r["response"] + (tok.eos_token or ""), add_special_tokens=False).input_ids
        ids = (p + c)[: a.max_len]
        labels = ([-100] * len(p) + c)[: a.max_len]
        return torch.tensor(ids), torch.tensor(labels)
    data = [encode(r) for r in rows]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = max(1, math.ceil(len(data) * a.epochs / a.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)))
    order = []
    while len(order) < steps * a.batch:
        order += random.sample(range(len(data)), len(data))
    for s in range(steps):
        tot = 0.0
        for i in order[s * a.batch:(s + 1) * a.batch]:
            ids, lab = data[i]
            out = model(input_ids=ids[None].to(dev), labels=lab[None].to(dev))
            (out.loss / a.batch).backward()
            tot += out.loss.item() / a.batch
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or s == steps - 1:
            print(f"step {s + 1}/{steps} loss {tot:.4f}", flush=True)

    model.eval()
    named = dict(model.named_parameters())
    changed = {n: named[n].detach() for n in train_names}
    if a.save_model:  # full model dir (e.g. healing an architecture candidate)
        model.save_pretrained(a.save_model, safe_serialization=True)
        print("saved full model ->", a.save_model)
    else:
        save_delta(changed, a.out, model_dir=md)


if __name__ == "__main__":
    main()
