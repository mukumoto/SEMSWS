"""Bundled shell scripts shipped with the driver."""

from importlib.resources import files
from pathlib import Path


def script_path(name: str) -> Path:
    """Resolve a bundled script name to its absolute path."""
    p = files(__name__) / name
    # `files()` returns a Traversable; for filesystem-backed packages this
    # has an `as_file()` API but we just need a plain Path here.
    return Path(str(p))
