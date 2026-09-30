import os

IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".zamagent",
}


def _resolve_path(path: str | None = None) -> str:
    """
    Internal helper.

    Resolves path (following symlinks) and ensures it stays inside workspace.
    """
    base = os.path.realpath(os.getcwd())

    if path is None or path == "":
        return base

    joined = path if os.path.isabs(path) else os.path.join(base, path)
    target = os.path.realpath(joined)

    try:
        inside = os.path.commonpath([base, target]) == base
    except ValueError:  # e.g. different drives on Windows
        inside = False

    if not inside:
        raise Exception(f"outside workspace. Current dir is '{base}'")

    return target


def _rel(path: str) -> str:
    return os.path.relpath(path, os.path.realpath(os.getcwd())).replace(os.sep, "/")


def _is_text(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" not in f.read(2048)
    except OSError:
        return False
