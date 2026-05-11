"""Runner Protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from .binding import BindingPolicy


@dataclass
class ShotTask:
    """Description of a single shot to run."""
    shot_id: str            # unique identifier (e.g. "shot_001")
    config_path: Path       # path to the per-shot SEMSWS YAML
    workdir: Path           # working directory (cwd for the binary)
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class ShotResult:
    """Outcome of running one shot."""
    shot_id: str
    workdir: Path
    return_code: int
    elapsed_seconds: float
    stdout_log: Path
    stderr_log: Path
    config_path: Path
    start_utc: Optional[str] = None      # ISO-8601, e.g. "2026-05-08T19:25:13.421Z"
    finish_utc: Optional[str] = None
    extra: dict = field(default_factory=dict)


class Runner(Protocol):
    """Submits a list of ShotTask and returns a list of ShotResult."""

    def submit(
        self,
        tasks: Sequence[ShotTask],
        *,
        binding: Optional["BindingPolicy"] = None,
    ) -> list[ShotResult]: ...
