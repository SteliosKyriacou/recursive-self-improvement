"""Scripted stand-in for an LLM agent (CPU, seconds). Delivers up to 2 candidates found by random
search through the metered probe, and writes a recipe so agentic optimization can tune `w` later."""
import json, re, subprocess, sys, torch
from pathlib import Path
sys.path.insert(0, ".")
from toolbox.common.delta import load_tensor, save_delta, parent_dir

n_cand = int(re.search(r"Deliver up to (\d+) candidate", Path("AGENTS.md").read_text()).group(1))
w = load_tensor(parent_dir(), "w")

def probe(cdir):
    out = subprocess.run([sys.executable, "tools/probe.py", "--candidate", cdir, "--n", "8"],
                         capture_output=True, text=True).stdout
    return json.loads(out.split("full results:")[0]).get("score")

tries = []
for i in range(4):
    d = f"candidates/try{i}"
    save_delta({"w": w + 0.5 * torch.randn(16)}, f"{d}/delta.safetensors")
    tries.append((probe(d) or -1e9, d))
tries.sort(reverse=True)
for _, d in tries[n_cand:]:          # keep only the best n_cand
    import shutil; shutil.rmtree(d)
for s, d in tries[:n_cand]:
    Path(d, "meta.json").write_text(json.dumps({"strategy": "free", "summary": f"random step, dev={s:.3f}"}))

recipe = Path("toolbox/recipes/toy_w.py")
if not recipe.exists():
    space = {f"d{i}": (-1.0, 1.0) for i in range(16)}
    recipe.write_text(f"""DESCRIPTION = "Move w of the parent model by an offset vector d"
SPACE = {space!r}
DEFAULT = {{k: 0.01 for k in SPACE}}

def build(x, model_dir, out_dir):
    import torch
    from toolbox.common.delta import load_tensor, save_delta
    w = load_tensor(model_dir, "w")
    save_delta({{"w": w + torch.tensor([x[f"d{{i}}"] for i in range(16)])}}, f"{{out_dir}}/delta.safetensors")
""")
Path("LEARNINGS.md").write_text(Path("LEARNINGS.md").read_text() + f"\n- random step, best dev {tries[0][0]:.3f}\n")
print("done")
