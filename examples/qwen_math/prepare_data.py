"""Build dev / val / test splits for "Make Qwen3.5-0.8B better at math".

* dev  (agent-visible): 128 problems from the MATH *train* split (no overlap with MATH-500).
* val  (hidden, accept/reject): 100 problems from MATH-500.
* test (hidden, final report): the other 400 MATH-500 problems.

Usage:  python examples/qwen_math/prepare_data.py [--n-dev 128] [--n-val 100]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dev", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(exist_ok=True)
    rng = random.Random(a.seed)

    m500 = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    rng.shuffle(m500)
    to_row = lambda r: {"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
                        "level": r["level"], "subject": r["subject"]}
    val, test = [to_row(r) for r in m500[:a.n_val]], [to_row(r) for r in m500[a.n_val:]]
    seen = {norm(r["problem"]) for r in m500}

    train = list(load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train"))
    rng.shuffle(train)
    dev = []
    for i, r in enumerate(train):
        ans = last_boxed(r["solution"])
        if ans is None or norm(r["problem"]) in seen:
            continue
        dev.append({"id": f"train/{i}", "problem": r["problem"], "answer": ans, "level": r["level"],
                    "subject": r["type"], "solution": r["solution"]})
        if len(dev) >= a.n_dev:
            break

    for name, rows in (("dev", dev), ("val", val), ("test", test)):
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
