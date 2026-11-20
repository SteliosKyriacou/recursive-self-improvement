"""Metered probe service: the agent's only sanctioned way to *run* the model.

The orchestrator starts a localhost HTTP server for the duration of each agent turn. The agent
calls it (via ``tools/probe.py`` in its workspace) with an optional candidate delta; the harness
applies the delta, runs the frozen metric on a slice of the **dev** split, and returns scores and
sample completions. Each call costs one unit of ``probe.forward_pass_budget`` (cf. the 10/25/50
forward-pass budgets in "Sparks of Weight-Space Programming", Exp 8). Hidden splits are never served.
"""
from __future__ import annotations

import json
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

from .evaluator import load_examples, run_eval
from .spec import GoalSpec
from .targets import candidate_kind, check_delta, check_model_candidate, finalize_model_candidate, materialize, total_params

# Keys the agent may override per probe; numeric ones are capped at the configured value.
_OVERRIDABLE = {"max_tokens", "temperature", "top_p", "top_k", "samples", "thinking", "presence_penalty"}
_CAPPED = {"max_tokens", "samples"}


class ProbeServer:
    def __init__(self, spec: GoalSpec, model_dir: Path, work_dir: Path, budget: int,
                 log: Optional[Callable[[dict], None]] = None):
        self.spec = spec
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.remaining = budget
        self.used = 0
        self.token = secrets.token_hex(16)
        self.lock = threading.Lock()
        self.log = log or (lambda r: None)
        self.base_cfg = {**spec.evaluation.config, **spec.probe.config}
        self.dev = load_examples(spec.evaluation.splits.dev)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._base_params: Optional[int] = None

    # ------------------------------------------------------------------ core
    def probe(self, req: dict) -> dict:
        with self.lock:  # one GPU job at a time
            if self.remaining <= 0:
                return {"ok": False, "error": "forward-pass budget exhausted", "remaining": 0}
            n = max(1, min(int(req.get("n", self.spec.probe.max_examples)), self.spec.probe.max_examples))
            off = max(0, int(req.get("offset", 0))) % max(1, len(self.dev))
            subset = (self.dev + self.dev)[off:off + n]
            cfg = dict(self.base_cfg)
            for k, v in (req.get("config") or {}).items():
                if k not in _OVERRIDABLE:
                    continue
                if k in _CAPPED and k in self.base_cfg:
                    v = min(v, self.base_cfg[k])
                cfg[k] = v
            delta = req.get("delta")
            cand = req.get("candidate")
            if cand and not delta:
                kind = candidate_kind(Path(cand))
                if kind == "delta":
                    delta = str(Path(cand) / "delta.safetensors")
                elif kind is None:
                    self.remaining += 1
                    self.used -= 1
                    return {"ok": False, "error": "candidate dir has neither delta.safetensors nor model/",
                            "remaining": self.remaining}
            self.remaining -= 1
            self.used += 1
            pid = self.used
            pdir = self.work_dir / f"probe_{pid:03d}"
            pdir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir
            guard = None
            if delta or cand:
                if delta:
                    guard = check_delta(Path(delta), self.model_dir, self.spec.guards)
                else:
                    if self._base_params is None:
                        self._base_params = total_params(self.model_dir)
                    guard = check_model_candidate(Path(cand) / "model", self.model_dir,
                                                  self._base_params, self.spec.guards)
                if not guard.ok:
                    # Refund: a malformed candidate never reached the model.
                    self.remaining += 1
                    self.used -= 1
                    return {"ok": False, "error": "candidate rejected by guards", "guards": guard.as_dict(),
                            "remaining": self.remaining}
                model_dir = (materialize(self.model_dir, Path(delta), pdir / "model") if delta else
                             finalize_model_candidate(Path(cand) / "model", self.model_dir, pdir / "model"))
            ex_path = pdir / "examples.jsonl"
            ex_path.write_text("\n".join(json.dumps(e) for e in subset))
            res = run_eval(self.spec.evaluation.metric, model_dir, ex_path, cfg, pdir / "result.json",
                           timeout_s=self.spec.evaluation.timeout_s, python=self.spec.evaluation.python)
            if (pdir / "model").exists():
                import shutil
                shutil.rmtree(pdir / "model", ignore_errors=True)
            res.update({"probe_id": pid, "remaining": self.remaining, "n": len(subset), "offset": off,
                        "delta": delta, "candidate": cand, "config": cfg})
            if guard:
                res["guards"] = guard.as_dict()
            self.log({k: v for k, v in res.items() if k != "per_example"})
            return res

    # ------------------------------------------------------------------ http
    def start(self) -> str:
        server = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, obj: dict):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.headers.get("X-Token") != server.token:
                    return self._send(403, {"error": "bad token"})
                self._send(200, {"remaining": server.remaining, "used": server.used,
                                 "max_examples": server.spec.probe.max_examples, "config": server.base_cfg})

            def do_POST(self):
                if self.headers.get("X-Token") != server.token:
                    return self._send(403, {"error": "bad token"})
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    self._send(200, server.probe(req))
                except Exception as e:  # noqa: BLE001
                    self._send(500, {"ok": False, "error": repr(e)})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()


PROBE_CLIENT = r'''#!/usr/bin/env python3
"""Run the model (optionally with your candidate delta) on a slice of the DEV split.

Each call consumes one unit of your forward-pass budget. Usage:
    python tools/probe.py --status
    python tools/probe.py [--candidate candidates/<name> | --delta file.safetensors] [--n 16] [--offset 0] [--max-tokens 4096]
Full results (incl. completions) are saved to probes/probe_<id>.json; a summary is printed.
"""
import argparse, json, os, urllib.request
from pathlib import Path

cfg = json.loads((Path(__file__).resolve().parent.parent / ".probe.json").read_text())
ap = argparse.ArgumentParser()
ap.add_argument("--status", action="store_true")
ap.add_argument("--delta")
ap.add_argument("--candidate", help="candidate dir with delta.safetensors or model/")
ap.add_argument("--n", type=int)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--max-tokens", type=int)
ap.add_argument("--config", default="{}", help="JSON overrides, e.g. '{\"thinking\": false}'")
a = ap.parse_args()
hdr = {"X-Token": cfg["token"], "Content-Type": "application/json"}
if a.status:
    print(urllib.request.urlopen(urllib.request.Request(cfg["url"], headers=hdr)).read().decode()); raise SystemExit
req = {"offset": a.offset, "config": json.loads(a.config)}
if a.n: req["n"] = a.n
if a.max_tokens: req["config"]["max_tokens"] = a.max_tokens
if a.delta: req["delta"] = str(Path(a.delta).resolve())
if a.candidate: req["candidate"] = str(Path(a.candidate).resolve())
r = urllib.request.Request(cfg["url"], data=json.dumps(req).encode(), headers=hdr, method="POST")
res = json.loads(urllib.request.urlopen(r, timeout=6 * 3600).read())
out = Path("probes"); out.mkdir(exist_ok=True)
fn = out / f"probe_{res.get('probe_id', 'err')}.json"
fn.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("per_example", "log_tail")}, indent=1))
if res.get("log_tail"): print(res["log_tail"][-1500:])
print("full results:", fn)
'''
