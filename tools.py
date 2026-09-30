import fnmatch
import html
import os
import re
import shutil
import subprocess
import sys

import requests

from tool_helpers import _resolve_path, _rel, _is_text, IGNORE_DIRS


def _ok(result):
    return {"status": "ok", "result": result}


def _err(result):
    return {"status": "error", "result": result}


def _clip(text: str | None, limit: int, tail: bool = False) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return ("..." + text[-limit:]) if tail else (text[:limit] + "...")


def _run(cmd, shell: bool, timeout: int):
    try:
        proc = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return _err(f"timed out after {timeout}s")
    except Exception as e:
        return _err(str(e))

    output = {
        "exit_code": proc.returncode,
        "stdout": _clip(proc.stdout, 4000),
        "stderr": _clip(proc.stderr, 1500, tail=True),
    }
    return {"status": "ok" if proc.returncode == 0 else "error", "result": output}


# --------------------------------------------------------------------------- #
# Reading / exploring                                                          #
# --------------------------------------------------------------------------- #

def read_file(filepath: str, start_line: int = 1, end_line: int | None = None):
    """
    Reads a UTF-8 text file. Optionally returns only a line range (1-based, inclusive),
    which is useful for big files.

    :param filepath: relative path to the file
    :param start_line: first line to return (default 1)
    :param end_line: last line to return (default: end of file)
    """
    try:
        target = _resolve_path(filepath)
        with open(target, "r", encoding="utf-8") as f:
            text = f.read()

        if start_line in (None, 1) and end_line is None:
            return _ok(text)

        lines = text.splitlines(keepends=True)
        first = max(1, start_line or 1)
        last = min(len(lines), end_line or len(lines))
        return _ok(f"[lines {first}-{last} of {len(lines)}]\n" + "".join(lines[first - 1:last]))

    except FileNotFoundError:
        return _err("file not found")
    except UnicodeDecodeError:
        return _err("not a UTF-8 text file")
    except Exception as e:
        return _err(str(e))


def read_dir(path: str | None = None):
    """
    Returns files list in directory.
    Leave path empty to list current directory.
    """
    try:
        target = _resolve_path(path)
        return _ok(sorted(os.listdir(target)))
    except Exception as e:
        return _err(str(e))


def get_current_directory():
    return _ok(os.getcwd())


def tree(path: str | None = None, max_depth: int = 3):
    """
    Shows a directory tree (skips .git, .venv, node_modules, __pycache__).
    Best first step to understand a project's structure.

    :param path: directory to show (default: workspace root)
    :param max_depth: how many levels to descend (default 3)
    """
    try:
        root = _resolve_path(path)
        lines, state = [f"{_rel(root) if root != _resolve_path() else '.'}/"], {"n": 0, "cut": False}

        def walk(directory, prefix, depth):
            try:
                entries = sorted(os.scandir(directory), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return
            entries = [e for e in entries if e.name not in IGNORE_DIRS]
            for i, entry in enumerate(entries):
                if state["n"] >= 300:
                    state["cut"] = True
                    return
                last = i == len(entries) - 1
                lines.append(f"{prefix}{'`-- ' if last else '|-- '}{entry.name}{'/' if entry.is_dir() else ''}")
                state["n"] += 1
                if entry.is_dir() and depth < max_depth:
                    walk(entry.path, prefix + ("    " if last else "|   "), depth + 1)

        walk(root, "", 1)
        if state["cut"]:
            lines.append("... (truncated)")
        return _ok("\n".join(lines))
    except Exception as e:
        return _err(str(e))


def find_files(pattern: str, path: str | None = None):
    """
    Finds files by name using a glob pattern, recursively.
    Example patterns: "*.py", "test_*.py", "src/*.js"

    :param pattern: glob pattern matched against the file name (or the relative path if it contains '/')
    :param path: directory to search in (default: workspace root)
    """
    try:
        root = _resolve_path(path)
        found = []
        for folder, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
            for name in sorted(files):
                full = os.path.join(folder, name)
                rel = _rel(full)
                if fnmatch.fnmatch(rel if "/" in pattern else name, pattern):
                    found.append(rel)
                    if len(found) >= 200:
                        return _ok(found + ["... (truncated at 200)"])
        return _ok(found)
    except Exception as e:
        return _err(str(e))


def search_in_files(pattern: str, path: str | None = None, file_glob: str = "*", ignore_case: bool = True):
    """
    Searches file contents with a regular expression (like grep -rn).
    Returns matching lines as 'file:line: text'. Skips .git, .venv, node_modules and binary files.

    :param pattern: regular expression to search for
    :param path: directory or single file to search in (default: workspace root)
    :param file_glob: only search files whose name matches this glob, e.g. "*.py"
    :param ignore_case: case-insensitive search (default true)
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return _err(f"invalid regex: {e}")

    try:
        root = _resolve_path(path)
        if os.path.isfile(root):
            candidates = [root]
        else:
            candidates = []
            for folder, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
                candidates += [os.path.join(folder, n) for n in sorted(files)
                               if fnmatch.fnmatch(n, file_glob or "*")]

        matches = []
        for full in candidates:
            if os.path.getsize(full) > 1_000_000 or not _is_text(full):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    for no, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{_rel(full)}:{no}: {line.strip()[:200]}")
                            if len(matches) >= 80:
                                return _ok(matches + ["... (truncated at 80 matches)"])
            except OSError:
                continue
        return _ok(matches if matches else "no matches")
    except Exception as e:
        return _err(str(e))


def fetch_url(url: str):
    """
    Downloads a web page or text resource and returns its readable text (max ~8000 chars).
    Use for documentation lookups. Only http/https.

    :param url: full URL starting with http:// or https://
    """
    if not url.startswith(("http://", "https://")):
        return _err("only http(s) URLs are allowed")
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "zamagent/1.0"})
        text = r.text
        if "html" in r.headers.get("content-type", ""):
            text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return {"status": "ok" if r.ok else "error", "result": f"[HTTP {r.status_code}]\n" + _clip(text, 8000)}
    except Exception as e:
        return _err(str(e))


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #

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
        return _ok(f"directory created: {target}, contents: {os.listdir(target)}")
    except Exception as e:
        return _err(str(e))


def write_file(filepath: str, content: str):
    """
    Creates or fully overwrites a file with the given content.

    Use this to write new files or replace an existing file entirely.

    :param filepath: relative path, e.g. "main.py" or "src/utils.py"
    :param content: full file content to write
    """
    try:
        target = _resolve_path(filepath)
        existed = os.path.exists(target)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

        lines = content.count("\n") + 1
        verb = "overwrote" if existed else "created"
        return _ok(f"{verb} '{filepath}' ({lines} lines)")
    except Exception as e:
        return _err(str(e))


def append_file(filepath: str, content: str):
    """
    Appends text to the end of a file (creates the file if missing).

    :param filepath: relative path to the file
    :param content: text to append
    """
    try:
        target = _resolve_path(filepath)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(content)
        return _ok(f"appended {content.count(chr(10)) + 1} lines to '{filepath}'")
    except Exception as e:
        return _err(str(e))


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
            return _err("old_str not found in file (re-read the file and copy the text exactly)")
        if count > 1:
            return _err(f"old_str appears {count} times - make it more specific")

        with open(target, "w", encoding="utf-8") as f:
            f.write(original.replace(old_str, new_str, 1))

        return _ok(f"edit applied to '{filepath}'")

    except FileNotFoundError:
        return _err("file not found")
    except Exception as e:
        return _err(str(e))


def delete_file(filepath: str):
    """
    Deletes a single file. Directories are never deleted.

    :param filepath: relative path to the file
    """
    try:
        target = _resolve_path(filepath)
        if not os.path.isfile(target):
            return _err("not a file (or does not exist)")
        os.remove(target)
        return _ok(f"deleted '{filepath}'")
    except Exception as e:
        return _err(str(e))


def move_file(src: str, dst: str):
    """
    Moves or renames a file or directory inside the workspace.

    :param src: existing relative path
    :param dst: new relative path (must not exist)
    """
    try:
        source, dest = _resolve_path(src), _resolve_path(dst)
        if not os.path.exists(source):
            return _err("source not found")
        if os.path.exists(dest):
            return _err("destination already exists")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.move(source, dest)
        return _ok(f"moved '{src}' -> '{dst}'")
    except Exception as e:
        return _err(str(e))


# --------------------------------------------------------------------------- #
# Execution                                                                    #
# --------------------------------------------------------------------------- #

def run_python(filepath: str, args: str = ""):
    """
    Runs a Python file and returns its stdout, stderr and exit code.

    Use this to verify that written code actually works.

    :param filepath: relative path to the .py file
    :param args: optional command-line arguments as a single string
    """
    try:
        target = _resolve_path(filepath)
    except Exception as e:
        return _err(str(e))
    return _run([sys.executable, target] + (args.split() if args else []), shell=False, timeout=30)


def run_command(command: str, timeout: int = 30):
    """
    Runs a shell command and returns stdout, stderr and exit code.

    Use for: pip install, git, ls, cat, running tests, etc.
    Forbidden: anything outside the current workspace or destructive system commands.

    :param command: shell command string, e.g. "pip install requests"
    :param timeout: seconds before the command is killed (default 30, max 120)
    """
    BLOCKED = ("rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "format c:", ":(){")
    lowered = command.lower()
    for blocked in BLOCKED:
        if blocked in lowered:
            return _err(f"command blocked: '{blocked}'")
    return _run(command, shell=True, timeout=max(1, min(int(timeout or 30), 120)))
