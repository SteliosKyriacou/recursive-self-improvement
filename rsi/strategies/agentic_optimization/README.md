# Strategy: agentic optimization (population search over a recipe)

Port of `~/repos/agentic-optimizer` (Metis-Agent): the harness tunes the knobs of a **recipe** — a
parametric family of candidates — with a population-based search. Every generation an LLM sees all
evaluated (parameters → screen score) pairs and writes NumPy code proposing the next population; each
member is built by the recipe and screened on dev. The best members compete for hidden-val promotion.
Without an LLM it falls back to an evolutionary sampler (Latin hypercube, then elite perturbation).

This strategy is *scheduled by the research director* (planner) once a recipe exists. Agents create
recipes: when you find a mechanism whose strength/location you are unsure about, encode it as a
recipe instead of guessing one setting.

## Recipe contract — `toolbox/recipes/<name>.py`
```python
DESCRIPTION = "Ablate the loop direction from the top-k down_proj columns of selected layers"
SPACE = {                               # literal dict: name -> (low, high[, "int"])
    "alpha": (0.0, 2.0),
    "k": (16, 512, "int"),
    "layer_lo": (8, 20, "int"),
}
DEFAULT = {"alpha": 1.0, "k": 64, "layer_lo": 15}   # optional, evaluated first

def build(x: dict, model_dir: str, out_dir: str) -> None:
    import sys; from pathlib import Path
    from toolbox.common.delta import load_tensor, save_delta   # toolbox is importable
    ...                                                        # deterministic given x
    save_delta(changed, f"{out_dir}/delta.safetensors", model_dir=model_dir)
    # or, for architecture changes, write f"{out_dir}/model/" (config.json + safetensors)
```
Rules: `SPACE`/`DESCRIPTION`/`DEFAULT` must be literals (they are read without running your code);
`build` must not read hidden data; save any data it needs (e.g. a direction vector) next to the
recipe, e.g. `toolbox/recipes/<name>_dir.pt`, and load it relative to `__file__`. Keep `build`
fast (seconds–minutes): it runs once per population member.

See `toolbox/recipes/_example.py` (files starting with `_` are ignored by the planner).
