import os


def _resolve_path(path: str | None = None) -> str:
    """
    Internal helper.

    Resolves path and ensures it stays inside workspace.
    """

    base = os.path.abspath(os.getcwd())

    if path is None:
        return base

    target = os.path.abspath(path)

    if os.path.commonpath([base, target]) != base:
        raise Exception(
            f"outside workspace. Current dir is '{base}'"
        )

    return target