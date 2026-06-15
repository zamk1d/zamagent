import inspect
import os


def read_file(filepath:str):
    try:
        with open(filepath, "r") as f:
            return {
                "status": "ok",
                "result": f.read()
            }
    except FileNotFoundError:
        return {
            "status": "error",
            "result": "file not found"
        }

def read_dir(path: str | None=None):
    """
    Returns files list in directory. "None" for current dir
    """
    if not path:
        files = os.listdir()
    else:
        files = os.listdir(path)

    return {
        "status": "ok",
        "result": files
    }