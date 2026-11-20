"""Toy target: a 16-dim weight vector `w`; quality = -mean squared distance to hidden targets.
Used to test the harness loop end to end on CPU in seconds (no GPU, no LLM)."""
import json, torch
from pathlib import Path
from safetensors.torch import save_file
here = Path(__file__).resolve().parent
g = torch.Generator().manual_seed(0)
center = torch.randn(16, generator=g)
(here / "model").mkdir(exist_ok=True)
save_file({"w": torch.zeros(16)}, str(here / "model" / "model.safetensors"))
(here / "model" / "config.json").write_text("{}")
(here / "data").mkdir(exist_ok=True)
for split, n in (("dev", 8), ("val", 8), ("test", 8)):
    rows = [{"id": f"{split}{i}", "target": (center + 0.1 * torch.randn(16, generator=g)).tolist()} for i in range(n)]
    (here / "data" / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("toy ready")
