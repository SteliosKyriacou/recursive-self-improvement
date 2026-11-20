"""CPU-only tests: guards, audit, architecture surgery, recipes, planner, full loop + resume."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from rsi.archive import CandidateRecord, ModelStore
from rsi.controller import Controller, _parse_json
from rsi.optim import AgenticOptimizer, load_recipe
from rsi.orchestrator import Orchestrator, _agent_actions
from rsi.spec import GuardSpec, load_spec
from rsi.targets import check_delta, check_model_candidate, finalize_model_candidate, materialize

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"


def _model(tmp: Path) -> Path:
    d = tmp / "m"
    d.mkdir()
    save_file({"a": torch.zeros(4, 4), "b": torch.ones(10)}, str(d / "model.safetensors"))
    (d / "config.json").write_text('{"model_type": "toy"}')
    return d


def test_delta_guards(tmp_path):
    md = _model(tmp_path)
    g = GuardSpec(max_changed_fraction=0.5)
    ok = tmp_path / "ok.safetensors"
    save_file({"a": torch.eye(4)}, str(ok))
    r = check_delta(ok, md, g)
    assert r.ok and r.changed_elements == 4
    for name, delta, reason in [
        ("shape", {"a": torch.zeros(3)}, "shape"),
        ("unknown", {"zzz": torch.zeros(1)}, "unknown"),
        ("nan", {"a": torch.full((4, 4), float("nan"))}, "non-finite"),
        ("noop", {"b": torch.ones(10)}, "changes no weights"),
        ("toomuch", {"a": torch.ones(4, 4), "b": torch.zeros(10)}, "changed fraction"),
    ]:
        p = tmp_path / f"{name}.safetensors"
        save_file(delta, str(p))
        r = check_delta(p, md, g)
        assert not r.ok and any(reason in x for x in r.reasons), (name, r.reasons)
    r = check_delta(ok, md, GuardSpec(allowed_keys="^b$"))
    assert not r.ok and any("allowed_keys" in x for x in r.reasons)


def test_materialize_and_store(tmp_path):
    md = _model(tmp_path)
    d1 = tmp_path / "d1.safetensors"
    save_file({"a": torch.eye(4)}, str(d1))
    d2 = tmp_path / "d2.safetensors"
    save_file({"b": torch.zeros(10)}, str(d2))
    out = materialize(md, d1, tmp_path / "out")
    t = load_file(str(out / "model.safetensors"))
    assert torch.equal(t["a"], torch.eye(4)) and torch.equal(t["b"], torch.ones(10))
    recs = {"base": CandidateRecord(id="base", iteration=0, kind="base", artifact=str(md)),
            "1.1": CandidateRecord(id="1.1", iteration=1, parent="base", artifact=str(d1)),
            "2.1": CandidateRecord(id="2.1", iteration=2, parent="1.1", artifact=str(d2))}
    store = ModelStore(tmp_path / "run", recs)
    t = load_file(str(store.resolve("2.1") / "model.safetensors"))  # delta chain
    assert torch.equal(t["a"], torch.eye(4)) and torch.equal(t["b"], torch.zeros(10))
    assert store.ancestors("2.1") == ["2.1", "1.1", "base"]


def test_audit_only_counts_actions(tmp_path):
    tr = tmp_path / "t.jsonl"
    lines = [
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "never touch SECRET"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "I must not read SECRET"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "input": {"command": "ls"}}]}},
    ]
    tr.write_text("\n".join(json.dumps(x) for x in lines))
    assert "SECRET" not in _agent_actions(tr)
    lines.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "input": {"command": "cat SECRET"}}]}})
    tr.write_text("\n".join(json.dumps(x) for x in lines))
    assert "SECRET" in _agent_actions(tr)


@pytest.fixture
def tiny_llama(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    d = tmp_path / "llama"
    LlamaForCausalLM(cfg).save_pretrained(d)
    return d


def test_architecture_surgery(tmp_path, tiny_llama):
    sys.path.insert(0, str(ROOT / "rsi" / "strategies"))
    from architecture.arch import prune_mlp, remap_layers
    from transformers import LlamaForCausalLM
    ids = torch.tensor([[1, 2, 3, 4]])
    drop = remap_layers(tiny_llama, tmp_path / "c1" / "model", order=[0, 1, 3])
    dup = remap_layers(tiny_llama, tmp_path / "c2" / "model", order=[0, 1, 2, 2, 3])
    slim = prune_mlp(tiny_llama, tmp_path / "c3" / "model", new_intermediate=48)
    for d, n_layers in ((drop, 3), (dup, 5), (slim, 4)):
        m = LlamaForCausalLM.from_pretrained(d)
        assert m.config.num_hidden_layers == n_layers
        assert torch.isfinite(m(ids).logits).all()
    base_params = sum(p.numel() for p in LlamaForCausalLM.from_pretrained(tiny_llama).parameters())
    assert check_model_candidate(drop, tiny_llama, base_params, GuardSpec()).ok
    assert not check_model_candidate(dup, tiny_llama, base_params, GuardSpec(max_params_ratio=1.0)).ok
    (dup / "modeling_evil.py").write_text("print('x')")
    assert not check_model_candidate(dup, tiny_llama, base_params, GuardSpec()).ok
    fin = finalize_model_candidate(drop, tiny_llama, tmp_path / "fin")
    assert (fin / "config.json").exists() and (fin / "generation_config.json").exists()


def test_recipe_and_optimizer(tmp_path):
    rp = tmp_path / "r.py"
    rp.write_text('DESCRIPTION = "x"\nSPACE = {f"d{i}": (-1.0, 1.0) for i in range(3)}\n'
                  'SPACE["k"] = (1, 5, "int")\nDEFAULT = {k: 0.0 for k in SPACE}\n'
                  'def build(x, model_dir, out_dir):\n    pass\n')
    r = load_recipe(rp)
    assert r.names[:3] == ["d0", "d1", "d2"] and r.default["d0"] == 0.0
    opt = AgenticOptimizer(r, population=5, generations=3, llm=None)
    for g in range(3):
        xs = opt.ask(g, tmp_path / f"g{g}")
        assert len(xs) == 5 and all(-1 <= x["d0"] <= 1 and isinstance(x["d2"], float) for x in xs)
        for x in xs:
            opt.tell(x, -sum(v * v for k, v in x.items() if k.startswith("d")))
    # LLM proposal path, with a fake LLM that writes valid NumPy code
    fake = lambda p: "LEARNINGS: center is good\n```python\nRESULT = np.zeros((P, len(names)))\n```"  # noqa
    opt2 = AgenticOptimizer(r, population=4, generations=1, llm=fake)
    xs = opt2.ask(0, tmp_path / "llm")
    assert opt2.log[-1]["source"] == "llm" and opt2.learnings == ["center is good"]


def test_planner_parsing_and_caps(tmp_path):
    assert _parse_json('```json\n{"screen": true, // c\n "experiments": []}\n```')["screen"] is True
    spec = load_spec(TOY / "goal.yaml")
    spec.search.strategies = ["weight_programming", "finetuning", "free"]
    spec.population.size, spec.population.max_experiments, spec.population.promote_top = 3, 2, 1
    (tmp_path / "toolbox" / "recipes").mkdir(parents=True)

    class S:
        best_id = "base"
    store = ModelStore(tmp_path, {"base": CandidateRecord(id="base", iteration=0, kind="base", artifact=".")})
    c = Controller(spec, tmp_path, None)
    plan = c._finish({"promote_top": 5, "experiments": [
        {"strategy": "weight_programming", "n_candidates": 2},
        {"strategy": "finetuning", "n_candidates": 5, "parent": "nope"},
        {"strategy": "free"}, {"strategy": "unknown"}]}, S, store)
    assert len(plan["experiments"]) == 2 and plan["promote_top"] == 1
    assert sum(e["n_candidates"] for e in plan["experiments"]) == 3
    assert plan["experiments"][1]["parent"] == "base" and plan["screen"] is True
    single = c._finish({"experiments": [{"strategy": "free", "n_candidates": 1}]}, S, store)
    assert single["screen"] is False  # one candidate -> straight to val


def test_toy_loop_and_resume(tmp_path):
    subprocess.run([sys.executable, str(TOY / "make_toy.py")], check=True)
    spec = load_spec(TOY / "goal.yaml")
    spec.search.max_iterations = 2
    rd = tmp_path / "run"
    st = Orchestrator(spec, rd).run_loop()
    assert st.finished and len(st.iterations) == 2 and st.baseline is not None
    assert st.best_score >= st.baseline
    it2 = st.iterations[1]
    strategies = {e["strategy"] for e in it2["experiments"]}
    assert strategies == {"free", "agentic_optimization"}   # recipe written in iter 1 is used in iter 2
    assert len(it2["candidates"]) == 2 + 2 * 4
    assert (rd / "report.md").exists() and (rd / "best" / "model").exists()
    assert st.test.get("best") is not None
    spec2 = load_spec(rd / "spec.yaml")
    spec2.search.max_iterations = 3
    st2 = Orchestrator(spec2, rd).run_loop()
    assert len(st2.iterations) == 3
