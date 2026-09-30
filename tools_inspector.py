"""
Reads tools.py and turns every public function into an LLM tool schema
(OpenAI / Groq / Ollama "function calling" format).
"""
import ast
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
TOOLS_FILE = BASE_DIR / "tools.py"

_TYPES = {"str": "string", "int": "integer", "float": "number",
          "bool": "boolean", "list": "array", "dict": "object"}
_cache: dict = {"mtime": None, "tools": None}


def _split_doc(doc: str | None) -> tuple[str, dict]:
    """docstring -> (description, {param: description})"""
    if not doc:
        return "", {}
    desc, params, cur = [], {}, None
    for line in doc.strip().splitlines():
        s = line.strip()
        m = re.match(r":param\s+(\w+):\s*(.*)", s)
        if m:
            cur = m.group(1)
            params[cur] = m.group(2)
        elif cur and s and not s.startswith(":"):
            params[cur] += " " + s
        elif not params and s:
            desc.append(s)
    return " ".join(desc), params


def _schema(annotation: str | None) -> dict:
    if not annotation:
        return {"type": "string"}
    parts = [p.strip() for p in annotation.split("|") if p.strip() != "None"]
    base = parts[0] if parts else "str"
    if base.startswith("Optional["):
        base = base[9:-1]
    head = base.split("[")[0].lower()
    schema = {"type": _TYPES.get(head, "string")}
    if schema["type"] == "array":
        inner = base[base.find("[") + 1:-1] if "[" in base else "str"
        schema["items"] = {"type": _TYPES.get(inner.lower(), "string")}
    return schema


def _load() -> dict:
    mtime = TOOLS_FILE.stat().st_mtime
    if _cache["mtime"] == mtime:
        return _cache["tools"]

    tools = {}
    tree = ast.parse(TOOLS_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        args = node.args.args
        n_required = len(args) - len(node.args.defaults)
        description, param_docs = _split_doc(ast.get_docstring(node))
        properties, required, legacy_args = {}, [], {}
        for i, arg in enumerate(args):
            ann = ast.unparse(arg.annotation) if arg.annotation else None
            legacy_args[arg.arg] = ann
            prop = _schema(ann)
            if arg.arg in param_docs:
                prop["description"] = param_docs[arg.arg]
            properties[arg.arg] = prop
            if i < n_required:
                required.append(arg.arg)
        tools[node.name] = {
            "description": description,
            "args": legacy_args,
            "returns": ast.unparse(node.returns) if node.returns else None,
            "parameters": {"type": "object", "properties": properties,
                           **({"required": required} if required else {})},
        }
    _cache.update(mtime=mtime, tools=tools)
    return tools


def get_tools_list() -> dict:
    """Legacy shape: {name: {"args": {...}, "description": str, "returns": str}}"""
    out = {}
    for name, t in _load().items():
        entry = {}
        if t["args"]:
            entry["args"] = t["args"]
        if t["returns"]:
            entry["returns"] = t["returns"]
        if t["description"]:
            entry["description"] = t["description"]
        out[name] = entry
    return out


def get_tool_schemas() -> list[dict]:
    """Tool definitions in OpenAI 'tools' format (also accepted by Groq and Ollama)."""
    return [
        {"type": "function", "function": {
            "name": name,
            "description": t["description"] or name,
            "parameters": t["parameters"],
        }}
        for name, t in _load().items()
    ]
