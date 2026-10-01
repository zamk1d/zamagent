"""
zamagent core: agent loop with native tool calling, memory, safety gates.

Providers live in providers.py (Groq / Ollama / any OpenAI-compatible).
Tools live in tools.py and are executed in a separate process (mcp_server.py).
"""
import atexit
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from typing import Callable

from providers import Provider, ProviderError, ToolUseFailed
from tools_inspector import get_tool_schemas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.getcwd()

MAX_STEPS = 25

# Tools that need the user's approval (unless auto_approve / "always" was chosen)
DANGEROUS = {"run_command", "run_python", "delete_file", "move_file"}
# Tools that change the workspace (reset the repeated-call detector)
MUTATING = {"write_file", "append_file", "edit_file", "delete_file", "move_file",
            "create_directory", "run_command", "run_python"}


# --------------------------------------------------------------------------- #
# Tool process (MCP-style, JSON lines over stdio)                              #
# --------------------------------------------------------------------------- #

class MCPClient:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        atexit.register(self.close)

    def _ensure(self) -> subprocess.Popen:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(
                [sys.executable, os.path.join(BASE_DIR, "mcp_server.py")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None if os.getenv("ZAMAGENT_DEBUG") else subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                cwd=WORK_DIR,
            )
        return self.proc

    def call(self, tool: str, arguments: dict) -> dict:
        proc = self._ensure()
        try:
            proc.stdin.write(json.dumps({"tool": tool, "arguments": arguments}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("tool server closed unexpectedly")
            return json.loads(line)
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError) as exc:
            self.close()
            return {"status": "error", "result": f"tool server failed: {exc}"}

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        self.proc = None


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #

class Agent:
    def __init__(self, provider: Provider, auto_approve: bool = False, max_steps: int = MAX_STEPS):
        self.provider = provider
        self.auto_approve = auto_approve
        self.max_steps = max_steps
        self.messages: list[dict] = []          # persistent conversation memory
        self.approved: set[str] = set()         # tools approved with "always"
        self.usage = {"prompt": 0, "completion": 0, "calls": 0}
        self.mcp = MCPClient()

        # UI callbacks
        self.on_step: Callable[[int], None] | None = None
        self.on_token: Callable[[str], None] | None = None
        self.on_tool_call: Callable[[str, dict], None] | None = None
        self.on_tool_result: Callable[[str, dict], None] | None = None
        # returns "y" | "n" | "a"
        self.confirm: Callable[[str, dict], str] | None = None

    # ------------------------------------------------------------------ #
    # Prompt                                                              #
    # ------------------------------------------------------------------ #
    def _project_notes(self) -> str:
        for name in ("ZAMAGENT.md", "AGENTS.md"):
            path = os.path.join(WORK_DIR, name)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        return f"\n## Project instructions ({name})\n{f.read()[:4000]}\n"
                except OSError:
                    pass
        return ""

    def _system(self) -> dict:
        try:
            files = sorted(os.listdir(WORK_DIR))
            listing = ", ".join(files[:40]) + (" ..." if len(files) > 40 else "") if files else "(empty)"
        except OSError as exc:
            listing = f"(could not list: {exc})"

        prompt = f"""\
You are zamagent, an autonomous coding and file-system agent running in the user's terminal.

Workspace: {WORK_DIR}  (all paths are relative to it; you cannot leave it)
System: {platform.system()} {platform.release()}, Python {platform.python_version()}
Date: {datetime.date.today().isoformat()}
Top-level contents: {listing}

## How to work
- Understand the request first, then act with the provided tools. Read-only exploration needs no permission.
- Explore before changing things: tree / find_files / search_in_files / read_file.
  Big files: read a line range instead of the whole file.
- Read a file before editing it. edit_file needs an exact, unique old_str; use write_file for new files or full rewrites.
- Do the minimum needed for the request. No extra files, folders or tests unless asked.
- Independent tool calls can go in the same turn. If a call needs another call's result, wait for it.
- After changing code, verify it when that is cheap (run it / run the tests) and fix what breaks.
- If a tool returns an error, read it and adapt. Never repeat an identical failing call.
- Shell commands, running code, deleting and moving need the user's approval. If the user denies, do not retry - propose an alternative.
- Thinking, planning and analysing happen in your head. Only call tools for real workspace actions.

## Answer style
Reply in the language the user writes in. Be concise: say what you did / found and anything the user must know. Use markdown only when it helps.
{self._project_notes()}"""
        return {"role": "system", "content": prompt}

    # ------------------------------------------------------------------ #
    # Memory                                                              #
    # ------------------------------------------------------------------ #
    def reset(self):
        self.messages = []

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.messages = json.load(f)

    @staticmethod
    def _msg_size(m: dict) -> int:
        n = len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(tc["function"]["arguments"])
        return n

    def _size(self) -> int:
        return sum(self._msg_size(m) for m in self.messages)

    @staticmethod
    def _shrink(m: dict, keep: int = 300):
        """Shorten an old message in place (keeps JSON arguments valid)."""
        if m["role"] == "tool" and len(m.get("content") or "") > keep:
            m["content"] = m["content"][:keep] + " ...[old result trimmed]"
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                continue
            changed = False
            for k, v in list(args.items()):
                if isinstance(v, str) and len(v) > keep:
                    args[k] = v[:keep] + " ...[trimmed]"
                    changed = True
            if changed:
                tc["function"]["arguments"] = json.dumps(args, ensure_ascii=False)

    def _compact(self, force: bool = False):
        """Keep the history inside the provider's budget."""
        limit = self.provider.context_chars * (0.35 if force else 1)
        if self._size() <= limit:
            return

        users = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        last_turn = users[-1] if users else 0
        boundary = len(self.messages) - 3 if force else last_turn

        for m in self.messages[:max(0, boundary)]:
            self._shrink(m)

        while self._size() > limit:
            users = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
            if len(users) < 2:
                break
            del self.messages[:users[1]]     # drop the oldest turn as a whole

    def _trim_tool_result(self, result: dict) -> str:
        limit = self.provider.tool_result_limit
        text = json.dumps(result, ensure_ascii=False)
        if len(text) <= limit:
            return text
        payload = result.get("result")
        note = f"\n... [trimmed, {len(text)} chars total; use start_line/end_line or search_in_files for the rest]"
        if isinstance(payload, str):
            return json.dumps({**result, "result": payload[:limit] + note}, ensure_ascii=False)
        return text[:limit] + note

    # ------------------------------------------------------------------ #
    # Tool execution                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(props: dict, args: dict) -> dict:
        """Models often send '5' for ints, null for optionals, numbers for strings."""
        out = {}
        for key, val in args.items():
            if val is None:
                continue
            kind = (props.get(key) or {}).get("type")
            if kind == "integer" and isinstance(val, str) and val.strip().lstrip("-").isdigit():
                val = int(val)
            elif kind == "integer" and isinstance(val, float) and val.is_integer():
                val = int(val)
            elif kind == "boolean" and isinstance(val, str):
                val = val.strip().lower() in ("true", "1", "yes")
            elif kind == "string" and not isinstance(val, str):
                val = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
            elif kind == "array" and isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
            out[key] = val
        return out

    def _execute(self, call: dict, schemas: dict, seen: dict) -> dict:
        name, args = call["name"], call["arguments"]

        if name not in schemas:
            result = {"status": "error",
                      "result": f"unknown tool {name!r}. Available: {', '.join(schemas)}"}
            if self.on_tool_result:
                self.on_tool_result(name, result)
            return result

        if args is None:
            result = {"status": "error",
                      "result": "arguments were not valid JSON. Call the tool again with a valid JSON object."}
            if self.on_tool_call:
                self.on_tool_call(name, {})
            if self.on_tool_result:
                self.on_tool_result(name, result)
            return result

        args = self._coerce(schemas[name]["properties"], args)
        if self.on_tool_call:
            self.on_tool_call(name, args)

        sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        seen[sig] = seen.get(sig, 0) + 1
        if seen[sig] >= 3:
            result = {"status": "error",
                      "result": "loop detected: this exact call was already made 3 times with no change in between. "
                                "Use the earlier results, try a different approach, or give your final answer."}
        elif name in DANGEROUS and not self.auto_approve and name not in self.approved:
            decision = self.confirm(name, args) if self.confirm else "n"
            if decision == "a":
                self.approved.add(name)
            if decision in ("y", "a"):
                result = self.mcp.call(name, args)
            else:
                result = {"status": "error", "result": "the user denied this action"}
        else:
            result = self.mcp.call(name, args)

        if name in MUTATING:
            seen.clear()
        if self.on_tool_result:
            self.on_tool_result(name, result)
        return result

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, user_input: str) -> str:
        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._loop()
        except BaseException:
            del self.messages[start:]        # never leave a half-finished turn in memory
            raise

    def _loop(self) -> str:
        tool_defs = get_tool_schemas()
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tool_defs}
        seen: dict = {}
        malformed = 0
        shrunk_retry = False

        for step in range(self.max_steps):
            if self.on_step:
                self.on_step(step)
            self._compact()

            try:
                resp = self.provider.chat([self._system()] + self.messages, tool_defs, self.on_token)
            except ToolUseFailed:
                malformed += 1
                if malformed > 3:
                    raise
                self.messages.append({
                    "role": "user",
                    "content": "Your last tool call was malformed and was rejected. "
                               "Call the tool again with valid JSON arguments that match its schema.",
                })
                continue
            except ProviderError as exc:
                if exc.status == 413 and not shrunk_retry:      # request too large for the model/tier
                    shrunk_retry = True
                    self._compact(force=True)
                    continue
                raise

            self.usage["calls"] += 1
            self.usage["prompt"] += int(resp.usage.get("prompt_tokens") or 0)
            self.usage["completion"] += int(resp.usage.get("completion_tokens") or 0)

            if not resp.tool_calls:
                if not resp.content.strip():
                    return "(the model returned an empty response)"
                self.messages.append({"role": "assistant", "content": resp.content})
                return resp.content

            self.messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [{
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c["arguments"] or {}, ensure_ascii=False)},
                    **({"extra_content": c["extra_content"]} if c.get("extra_content") else {}),
                } for c in resp.tool_calls],
            })

            for call in resp.tool_calls:
                result = self._execute(call, schemas, seen)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": self._trim_tool_result(result),
                    "_name": call["name"],
                })

        note = f"Agent stopped: reached maximum steps ({self.max_steps}). Say 'continue' to keep going."
        self.messages.append({"role": "assistant", "content": note})
        return note