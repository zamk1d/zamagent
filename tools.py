import os
import subprocess

from tool_helpers import _resolve_path


def read_file(filepath: str):
    try:
        target = _resolve_path(filepath)

        with open(target, "r", encoding="utf-8") as f:
            return {"status": "ok", "result": f.read()}

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

    Use this to surgically edit part of an existing file without
    rewriting the whole thing. old_str must match the file exactly
    (including indentation and newlines) and must appear exactly once.

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


def run_python(filepath: str, args: str = ""):
    """
    Runs a Python file and returns its stdout, stderr and exit code.

    Use this to verify that written code actually works.

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

    Use for: pip install, git, ls, cat, etc.
    Forbidden: anything outside the current workspace or destructive system commands.

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