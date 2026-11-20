"""Meta-agent backends. Each wraps a headless coding-agent CLI that works inside a workspace dir.

All backends receive the same short kickoff prompt; the real instructions live in the workspace
(AGENTS.md / CLAUDE.md / GEMINI.md), so switching backend changes nothing else in the loop.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..spec import AgentSpec


@dataclass
class AgentResult:
    ok: bool
    returncode: Optional[int]
    seconds: float
    transcript: Path
    final_text: str = ""
    cost_usd: Optional[float] = None
    error: str = ""


class AgentBackend:
    name = "base"
    binary = ""

    def __init__(self, spec: AgentSpec):
        self.spec = spec

    def command(self, prompt: str, workspace: Path, extra_dirs: list[Path]) -> list[str]:
        raise NotImplementedError

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def parse_final(self, transcript: Path) -> tuple[str, Optional[float]]:
        """Best-effort: last assistant/result text and cost from a JSONL transcript."""
        text, cost = "", None
        try:
            for line in transcript.read_text(errors="replace").splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") == "result":
                    text = ev.get("result") or text
                    cost = ev.get("total_cost_usd", cost)
                elif ev.get("type") in ("item.completed",) and isinstance(ev.get("item"), dict):
                    if ev["item"].get("type") in ("agent_message", "assistant_message"):
                        text = ev["item"].get("text", text)
        except FileNotFoundError:
            pass
        return text, cost

    # ---- one-shot text completion (planner, optimizer proposals). No tools needed.
    def complete_command(self, prompt: str) -> list[str]:
        raise NotImplementedError

    def complete(self, prompt: str, cwd: Path, timeout_s: int = 900) -> str:
        """Return the model's text answer to a single prompt (raises on failure)."""
        cmd = self.complete_command(prompt)
        cwd.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s,
                           stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"{self.name} completion failed ({r.returncode}): {r.stderr[-500:]}")
        out = r.stdout.strip()
        try:  # single JSON object (claude/agy --output-format json)
            obj = json.loads(out)
            if isinstance(obj, dict) and obj.get("result") is not None:
                return str(obj["result"])
        except json.JSONDecodeError:
            pass
        tmp = cwd / "_completion.jsonl"
        tmp.write_text(out)
        text, _ = self.parse_final(tmp)
        return text or out

    # ---- interactive session in the user's terminal (rsi new)
    def interactive_command(self, prompt: str, system: str) -> list[str]:
        raise NotImplementedError

    def interactive(self, prompt: str, system: str, cwd: Path) -> int:
        return subprocess.call(self.interactive_command(prompt, system), cwd=cwd)

    def run(self, prompt: str, workspace: Path, extra_dirs: list[Path], env: dict, log_dir: Path,
            timeout_s: Optional[int] = None) -> AgentResult:
        if not self.available():
            raise RuntimeError(f"agent backend '{self.name}': '{self.binary}' not found on PATH")
        log_dir.mkdir(parents=True, exist_ok=True)
        transcript = log_dir / "agent_transcript.jsonl"
        stderr = log_dir / "agent_stderr.log"
        cmd = self.command(prompt, workspace, extra_dirs) + list(self.spec.extra_args)
        (log_dir / "agent_cmd.json").write_text(json.dumps(cmd[:-1] + ["<prompt>"] if cmd else cmd))
        t0 = time.time()
        with open(transcript, "w") as out, open(stderr, "w") as err:
            proc = subprocess.Popen(cmd, cwd=workspace, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                    env={**os.environ, **env}, start_new_session=True)
            try:
                timeout_s = timeout_s or self.spec.timeout_s
                rc = proc.wait(timeout=timeout_s)
                error = "" if rc == 0 else f"exit code {rc}"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc, error = None, f"timeout after {timeout_s}s"
        text, cost = self.parse_final(transcript)
        return AgentResult(ok=rc == 0, returncode=rc, seconds=round(time.time() - t0, 1),
                           transcript=transcript, final_text=text, cost_usd=cost, error=error)


class ClaudeCode(AgentBackend):
    name, binary = "claude_code", "claude"

    def command(self, prompt, workspace, extra_dirs):
        cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
               "--permission-mode", self.spec.permission_mode, "--no-session-persistence"]
        for d in extra_dirs:
            cmd += ["--add-dir", str(d)]
        if self.spec.model:
            cmd += ["--model", self.spec.model]
        if self.spec.effort:
            cmd += ["--effort", self.spec.effort]
        if self.spec.max_budget_usd:
            cmd += ["--max-budget-usd", str(self.spec.max_budget_usd)]
        return cmd + [prompt]

    def complete_command(self, prompt):
        cmd = ["claude", "-p", "--output-format", "json", "--permission-mode", "dontAsk", "--no-session-persistence"]
        if self.spec.model:
            cmd += ["--model", self.spec.model]
        return cmd + [prompt]

    def interactive_command(self, prompt, system):
        cmd = ["claude", "--append-system-prompt", system, "--permission-mode", "acceptEdits"]
        if self.spec.model:
            cmd += ["--model", self.spec.model]
        return cmd + [prompt]


class Antigravity(AgentBackend):
    name, binary = "antigravity", "agy"

    def command(self, prompt, workspace, extra_dirs):
        cmd = ["agy", "--output-format", "stream-json", "--dangerously-skip-permissions"]
        for d in extra_dirs:
            cmd += ["--add-dir", str(d)]
        if self.spec.model:
            cmd += ["--model", self.spec.model]
        if self.spec.effort:
            cmd += ["--effort", self.spec.effort]
        return cmd + ["--print", prompt]

    def complete_command(self, prompt):
        cmd = ["agy", "--output-format", "json", "--mode", "plan"]
        if self.spec.model:
            cmd += ["--model", self.spec.model]
        return cmd + ["--print", prompt]

    def interactive_command(self, prompt, system):
        # agy has no system-prompt flag: the instructions travel in GEMINI.md/AGENTS.md in cwd.
        cmd = ["agy", "--mode", "accept-edits"]
        if self.spec.model:
            cmd += ["--model", self.spec.model]
        return cmd + ["--prompt-interactive", prompt]


class Codex(AgentBackend):
    name, binary = "codex", "codex"

    def command(self, prompt, workspace, extra_dirs):
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox", "-C", str(workspace)]
        for d in extra_dirs:
            cmd += ["--add-dir", str(d)]
        if self.spec.model:
            cmd += ["-m", self.spec.model]
        if self.spec.effort:
            cmd += ["-c", f"model_reasoning_effort={self.spec.effort}"]
        return cmd + [prompt]

    def complete_command(self, prompt):
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only"]
        if self.spec.model:
            cmd += ["-m", self.spec.model]
        return cmd + [prompt]

    def interactive_command(self, prompt, system):
        cmd = ["codex"]  # instructions travel in AGENTS.md in cwd
        if self.spec.model:
            cmd += ["-m", self.spec.model]
        return cmd + [prompt]


def _argv(cmd: list[str]) -> list[str]:
    """`python`/`python3` means rsi's own interpreter, so commands work from any folder/venv."""
    import sys
    return [sys.executable] + cmd[1:] if cmd and cmd[0] in ("python", "python3") else list(cmd)


class Command(AgentBackend):
    """Any executable: spec.command is an argv list; '{prompt}' and '{workspace}' are substituted."""
    name = "command"

    def available(self) -> bool:
        argv = _argv(self.spec.command)
        return bool(argv) and shutil.which(argv[0]) is not None

    def command(self, prompt, workspace, extra_dirs):
        return [a.replace("{prompt}", prompt).replace("{workspace}", str(workspace)) for a in _argv(self.spec.command)]

    def complete_command(self, prompt):
        if not self.spec.complete_command:
            raise NotImplementedError("agent.complete_command not set")
        return [a.replace("{prompt}", prompt) for a in _argv(self.spec.complete_command)]


BACKENDS = {b.name: b for b in (ClaudeCode, Antigravity, Codex, Command)}


def make_backend(spec: AgentSpec) -> AgentBackend:
    return BACKENDS[spec.backend](spec)
