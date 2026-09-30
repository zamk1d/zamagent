import os
import re
import ast
import subprocess
import shutil

from src.zamagent.tool_helpers import _resolve_path


def read_file(filepath: str):
    """
    Reads and returns the full content of a file with line numbers.
    """
    try:
        target = _resolve_path(filepath)
        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Возвращаем с номерами строк — модель видит точные координаты
        numbered = "".join(f"{i+1:4d}  {line}" for i, line in enumerate(lines))
        return {"status": "ok", "result": numbered}
    except FileNotFoundError:
        return {"status": "error", "result": "file not found"}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def read_file_lines(filepath: str, start: int, end: int):
    """
    Reads a specific range of lines from a file (1-indexed, inclusive).

    Use this instead of read_file when you only need part of a large file.
    Much cheaper on context than reading the whole file.

    :param filepath: relative path to the file
    :param start: first line to read (1-indexed)
    :param end: last line to read (inclusive)
    """
    try:
        target = _resolve_path(filepath)
        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        s = max(0, start - 1)
        e = min(total, end)
        chunk = lines[s:e]
        numbered = "".join(f"{s+i+1:4d}  {line}" for i, line in enumerate(chunk))
        return {
            "status": "ok",
            "result": numbered,
            "meta": {"total_lines": total, "shown": f"{s+1}-{s+len(chunk)}"}
        }
    except FileNotFoundError:
        return {"status": "error", "result": "file not found"}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def read_dir(path: str | None = None):
    """
    Returns files list in directory.
    Leave path empty to list current directory.
    """
    try:
        target = _resolve_path(path)
        return {"status": "ok", "result": os.listdir(target)}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def get_current_directory():
    return {"status": "ok", "result": os.getcwd()}


def create_directory(path: str | None = None):
    """
    Creates directory.

    IMPORTANT: If user did not specify a path, leave path empty.

    Example:
        create_directory("src")
    NOT:
        create_directory("/absolute/path/src")
    """
    try:
        target = _resolve_path(path)
        os.makedirs(target, exist_ok=True)
        return {
            "status": "ok",
            "result": f"directory created: {target}, contents: {os.listdir(target)}"
        }
    except Exception as e:
        return {"status": "error", "result": str(e)}


def write_file(filepath: str, content: str):
    """
    Creates or fully overwrites a file with the given content.

    Use this to write new files or replace an existing file entirely.
    After writing Python files, verify with check_syntax.

    :param filepath: relative path, e.g. "main.py" or "src/utils.py"
    :param content: full file content to write
    """
    try:
        target = _resolve_path(filepath)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        lines = content.count("\n") + 1
        return {"status": "ok", "result": f"wrote {lines} lines to '{filepath}'"}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def edit_file(filepath: str, old_str: str, new_str: str):
    """
    Replaces an exact substring in a file with new text.

    Use this to surgically edit part of an existing file.
    old_str must match the file exactly (indentation and newlines included)
    and must appear exactly once. After editing Python files, call check_syntax.

    :param filepath: relative path to the file
    :param old_str: the exact text to find and replace
    :param new_str: the text to put in its place (can be empty to delete)
    """
    try:
        target = _resolve_path(filepath)
        with open(target, "r", encoding="utf-8") as f:
            original = f.read()
        count = original.count(old_str)
        if count == 0:
            return {"status": "error", "result": "old_str not found in file"}
        if count > 1:
            return {
                "status": "error",
                "result": f"old_str appears {count} times — make it more specific"
            }
        updated = original.replace(old_str, new_str, 1)
        with open(target, "w", encoding="utf-8") as f:
            f.write(updated)
        return {"status": "ok", "result": f"edit applied to '{filepath}'"}
    except FileNotFoundError:
        return {"status": "error", "result": "file not found"}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def check_syntax(filepath: str):
    """
    Checks Python file for syntax errors without executing it.

    Works even if the file has unresolved imports or missing dependencies.
    Use this after every write_file or edit_file on a .py file.

    :param filepath: relative path to the .py file
    """
    try:
        target = _resolve_path(filepath)
        with open(target, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        return {"status": "ok", "result": "syntax OK"}
    except SyntaxError as e:
        return {
            "status": "error",
            "result": f"SyntaxError at line {e.lineno}: {e.msg}\n  {e.text}"
        }
    except FileNotFoundError:
        return {"status": "error", "result": "file not found"}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def search_in_file(filepath: str, pattern: str):
    """
    Searches for a substring or regex pattern in a file.
    Returns all matching lines with their line numbers.

    Use this to locate the exact insertion point before calling edit_file,
    instead of reading the whole file into context.

    :param filepath: relative path to the file
    :param pattern: plain string or Python regex to search for
    """
    try:
        target = _resolve_path(filepath)
        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()

        results = []
        try:
            rx = re.compile(pattern)
            use_regex = True
        except re.error:
            use_regex = False

        for i, line in enumerate(lines, 1):
            matched = rx.search(line) if use_regex else pattern in line
            if matched:
                results.append(f"{i:4d}  {line.rstrip()}")

        if not results:
            return {"status": "ok", "result": "no matches found"}

        return {
            "status": "ok",
            "result": "\n".join(results),
            "meta": {"matches": len(results), "total_lines": len(lines)}
        }
    except FileNotFoundError:
        return {"status": "error", "result": "file not found"}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def run_python(filepath: str, args: str = ""):
    """
    Runs a Python file and returns stdout, stderr and exit code.

    Use only for self-contained scripts that can run standalone.
    Do NOT use on library files that require imports from other modules.
    For syntax checking use check_syntax instead.

    :param filepath: relative path to the .py file
    :param args: optional command-line arguments as a single string
    """
    try:
        target = _resolve_path(filepath)
        cmd = ["python", target] + (args.split() if args else [])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        output = {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[:2000] if proc.stdout else "",
            "stderr": proc.stderr[:500] if proc.stderr else "",
        }
        status = "ok" if proc.returncode == 0 else "error"
        return {"status": status, "result": output}
    except subprocess.TimeoutExpired:
        return {"status": "error", "result": "timed out after 30s"}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def run_command(command: str):
    """
    Runs a shell command and returns stdout, stderr and exit code.

    Use for: pip install, git, grep, find, etc.
    Forbidden: destructive system commands.

    :param command: shell command string, e.g. "pip install requests"
    """
    BLOCKED = ("rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=")
    for blocked in BLOCKED:
        if blocked in command:
            return {"status": "error", "result": f"command blocked: '{blocked}'"}
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        output = {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[:2000] if proc.stdout else "",
            "stderr": proc.stderr[:500] if proc.stderr else "",
        }
        status = "ok" if proc.returncode == 0 else "error"
        return {"status": status, "result": output}
    except subprocess.TimeoutExpired:
        return {"status": "error", "result": "timed out after 30s"}
    except Exception as e:
        return {"status": "error", "result": str(e)}