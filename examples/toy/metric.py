import torch
from pathlib import Path
from safetensors.torch import load_file

def evaluate(model_dir, examples, config):
    w = load_file(str(next(Path(model_dir).glob("*.safetensors"))))["w"]
    errs = [float(((w - torch.tensor(e["target"])) ** 2).mean()) for e in examples]
    return {"score": -sum(errs) / len(errs), "per_example": [{"id": e["id"], "err": x} for e, x in zip(examples, errs)]}
