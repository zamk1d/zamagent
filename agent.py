import os
import subprocess
import json
import atexit
from typing import Callable

import requests

from tools_inspector import get_tools_list

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.getcwd()

MAX_STEPS = 20
TOOL_RESULT_TRIM = 2000

# --------------------------------------------------------------------------- #
# Callbacks                                                                    #
# --------------------------------------------------------------------------- #

on_tool_call:   Callable[[str, dict], None] | None = None
on_tool_result: Callable[[str, dict], None] | None = None
on_step:        Callable[[int], None]        | None = None
on_token:       Callable[[str], None]        | None = None


# --------------------------------------------------------------------------- #
# MCP subprocess                                                               #
# --------------------------------------------------------------------------- #

_proc: subprocess.Popen | None = None


def _get_proc() -> subprocess.Popen:
    global _proc
    if _proc is None or _proc.poll() is not None:
        _proc = subprocess.Popen(
            ["python", os.path.join(BASE_DIR, "mcp_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=WORK_DIR,
        )
        atexit.register(_cleanup_proc)
    return _proc


def _cleanup_proc():
    global _proc
    if _proc and _proc.poll() is None:
        try:
            _proc.stdin.close()
            _proc.wait(timeout=2)
        except Exception:
            _proc.kill()


def call_mcp(tool: str, arguments: dict) -> dict:
    proc = _get_proc()

    if on_tool_call:
        on_tool_call(tool, arguments)

    try:
        proc.stdin.write(json.dumps({"tool": tool, "arguments": arguments}) + "\n")
        proc.stdin.flush()
        response_line = proc.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed unexpectedly")
        result = json.loads(response_line)
    except BrokenPipeError:
        raise RuntimeError("MCP server died, restarting on next call")

    if on_tool_result:
        on_tool_result(tool, result)

    return result


# --------------------------------------------------------------------------- #
# Tool rendering                                                                #
# --------------------------------------------------------------------------- #

def render_tools() -> str:
    tools = get_tools_list()
    parts = []

    for tool_name, info in tools.items():
        args = info.get("args", {})
        args_str = ", ".join(f"{n}: {t}" for n, t in args.items())
        line = f"- {tool_name}({args_str})"

        if desc := info.get("description"):
            first_line = desc.strip().splitlines()[0]
            line += f"\n  Description: {first_line}"

        if ret := info.get("returns"):
            line += f"\n  Returns: {ret}"

        parts.append(line)

    return "\n\n".join(parts)


def _tool_names() -> str:
    return ", ".join(get_tools_list().keys())


# --------------------------------------------------------------------------- #
# LLM                                                                          #
# --------------------------------------------------------------------------- #

def ask_llm(messages: list) -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:8b",
            "messages": messages,
            "think": False,
            "stream": True,
        },
        stream=True,
        timeout=(5, None),
    )
    response.raise_for_status()

    chunks: list[str] = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        try:
            chunk = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        token = chunk.get("message", {}).get("content", "")
        if token:
            chunks.append(token)
            if on_token:
                on_token(token)

        if chunk.get("done"):
            break

    return "".join(chunks)


# --------------------------------------------------------------------------- #
# Prompts                                                                       #
# --------------------------------------------------------------------------- #

def _system_prompt() -> dict:
    prompt = f"""\
You are a file-system agent. You complete tasks by calling tools.

Current workspace: {WORK_DIR}

## Available tools (ONLY these exist — do NOT invent others):

{render_tools()}

## Strict rules

1. To call tools respond with a raw JSON array and NOTHING else — no text, no markdown:
   [{{"tool": "name", "arguments": {{"key": "value"}}}}]

2. NEVER call a tool that is not listed above. If you call an unknown tool you will
   get an error. Read the error, then retry using only the listed tools.

You may call multiple tools in one response ONLY if none of them depend on the result of another tool.

If a later action requires information from a previous tool result,
call only the first tool and wait for its result before continuing.

3. You receive each tool result before deciding the next step. Use the result.

4. All file paths must be relative. Never go outside the workspace.

5. Before editing a file you have not read yet — read it first with read_file,
   so you know the exact content to put in old_str.

6. When you are done — respond in plain text (not JSON). Be concise.
"""
    return {"role": "system", "content": prompt}


def _initial_context() -> dict:
    """
    Первое сообщение от 'assistant' с содержимым рабочей директории.
    Даём агенту контекст сразу, без лишнего tool call на старте.
    """
    import os as _os
    try:
        files = _os.listdir(WORK_DIR)
        listing = ", ".join(files) if files else "(empty)"
    except Exception as e:
        listing = f"(could not list: {e})"

    return {
        "role": "assistant",
        "content": f"Workspace contents: {listing}"
    }


# --------------------------------------------------------------------------- #
# Parsing                                                                       #
# --------------------------------------------------------------------------- #

def _parse_response(response: str) -> list[dict] | str:
    stripped = response.strip()

    if not stripped.startswith("["):
        return stripped

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if not isinstance(data, list):
        return stripped

    calls = [
        item for item in data
        if isinstance(item, dict) and "tool" in item and "arguments" in item
    ]
    return calls if calls else stripped


# --------------------------------------------------------------------------- #
# Context trimming                                                              #
# --------------------------------------------------------------------------- #

def _trim_tool_result(result: dict) -> str:
    serialized = json.dumps(result, ensure_ascii=False)
    if len(serialized) <= TOOL_RESULT_TRIM:
        return serialized

    if isinstance(result.get("result"), str):
        trimmed = result.copy()
        trimmed["result"] = (
            result["result"][:TOOL_RESULT_TRIM]
            + f"\n... [trimmed, {len(result['result'])} chars total]"
        )
        return json.dumps(trimmed, ensure_ascii=False)

    return serialized[:TOOL_RESULT_TRIM] + " ... [trimmed]"


# --------------------------------------------------------------------------- #
# Main run loop                                                                 #
# --------------------------------------------------------------------------- #

def run(user_input: str) -> str:
    messages: list[dict] = [
        _system_prompt(),
        _initial_context(),
        {"role": "user", "content": user_input},
    ]

    executed: set[tuple] = set()

    for step in range(MAX_STEPS):
        if on_step:
            on_step(step)

        response = ask_llm(messages)
        parsed = _parse_response(response)

        if isinstance(parsed, str):
            return parsed

        tool_calls: list[dict] = parsed

        signature = tuple(
            (c["tool"], json.dumps(c["arguments"], sort_keys=True))
            for c in tool_calls
        )
        if signature in executed:
            return "Agent stopped: repeated identical tool calls with no progress."
        executed.add(signature)

        messages.append({"role": "assistant", "content": response})

        for call in tool_calls:
            result = call_mcp(call["tool"], call.get("arguments", {}))
            messages.append({
                "role": "tool",
                "name": call["tool"],
                "content": _trim_tool_result(result),
            })

    return f"Agent stopped: reached maximum steps ({MAX_STEPS})."