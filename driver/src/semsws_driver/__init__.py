"""semsws-driver: thin run-only layer for SEMSWS."""

__version__ = "0.2.0"

from .core.run import RunResult, run

__all__ = ["run", "RunResult", "__version__"]
