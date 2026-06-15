import sys
import json
import tools

TOOLS = {
    name: obj
    for name, obj in vars(tools).items()
    if callable(obj)
}

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
                "result": f"unknown tool: {tool_name}"
            }
        else:
            tool = TOOLS[tool_name]
            response = tool(**arguments)

    except Exception as e:
        response = {
            "status": "error",
            "result": str(e)
        }

    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()