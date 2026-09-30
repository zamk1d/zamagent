import inspect
import json
import sys

# Protocol goes through the real stdout; anything a tool prints ends up on stderr.
_out = sys.stdout
sys.stdout = sys.stderr

import tools

TOOLS = {
    name: obj
    for name, obj in vars(tools).items()
    if inspect.isfunction(obj) and obj.__module__ == tools.__name__ and not name.startswith("_")
}


def _send(payload: dict) -> None:
    _out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    _out.flush()


try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            tool_name = request.get("tool")
            arguments = request.get("arguments") or {}

            if tool_name not in TOOLS:
                response = {
                    "status": "error",
                    "result": f"unknown tool: {tool_name!r}. Available: {list(TOOLS)}",
                }
            else:
                response = TOOLS[tool_name](**arguments)

        except TypeError as e:
            response = {"status": "error", "result": f"bad arguments: {e}"}
        except Exception as e:
            response = {"status": "error", "result": str(e)}

        _send(response)

except (BrokenPipeError, KeyboardInterrupt):
    pass
