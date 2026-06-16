import sys
import json
import tools

TOOLS = {
    name: obj
    for name, obj in vars(tools).items()
    if callable(obj) and not name.startswith("_")
}

try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            tool_name = request.get("tool")
            arguments = request.get("arguments", {})

            if tool_name not in TOOLS:
                response = {
                    "status": "error",
                    "result": f"unknown tool: {tool_name!r}. Available: {list(TOOLS)}"
                }
            else:
                response = TOOLS[tool_name](**arguments)

        except TypeError as e:
            response = {"status": "error", "result": f"bad arguments: {e}"}
        except Exception as e:
            response = {"status": "error", "result": str(e)}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

except BrokenPipeError:
    pass
except KeyboardInterrupt:
    pass