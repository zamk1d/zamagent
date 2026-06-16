from pathlib import Path
import ast

BASE_DIR = Path(__file__).parent

TOOLS_FILE = BASE_DIR / "tools.py"

def get_tools_list():
    tools = {}
    with open(TOOLS_FILE, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            tool_name = node.name
            tools[tool_name] = {}
            tool_args = {}
            for arg in node.args.args:
                param_name = arg.arg

                if arg.annotation:
                    param_type = ast.unparse(arg.annotation)
                else:
                    param_type = None
                tool_args[param_name] = param_type
            if tool_args:
                tools[tool_name]["args"] = tool_args
            if node.returns:
                tools[tool_name]["returns"] = ast.unparse(node.returns)
            description = ast.get_docstring(node)
            if description and description != "":
                tools[tool_name]["description"] = description
    return tools