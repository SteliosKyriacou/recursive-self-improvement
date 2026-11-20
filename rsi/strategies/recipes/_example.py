"""Example recipe (ignored by the planner because the file name starts with '_').

Scales one MLP down_proj by a per-layer factor — replace with a real mechanism you discovered.
"""
DESCRIPTION = "Scale the down_proj of one decoder layer by a factor"
SPACE = {"layer": (0, 23, "int"), "scale": (0.5, 1.5)}
DEFAULT = {"layer": 12, "scale": 1.0}


def build(x: dict, model_dir: str, out_dir: str) -> None:
    from toolbox.common.delta import load_tensor, save_delta, tensor_names
    name = next(n for n in tensor_names(model_dir)
                if f".layers.{x['layer']}.mlp.down_proj.weight" in n and "mtp" not in n)
    W = load_tensor(model_dir, name)
    scale = x["scale"] if abs(x["scale"] - 1.0) > 1e-6 else 1.0 + 1e-3  # a delta must change something
    save_delta({name: (W.float() * scale).to(W.dtype)}, f"{out_dir}/delta.safetensors", model_dir=model_dir)
