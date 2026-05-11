"""Run record written to <workdir>/manifest.json (argv, timings, binary,
driver version, per-shot status).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .. import __version__ as DRIVER_VERSION

SCHEMA_VERSION = "2.1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_rev_for(path: Path) -> Optional[str]:
    """Resolve the git rev (short) for the working tree containing path.
    Returns None if path is not in a git repo or git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


@dataclass
class ShotEntry:
    shot_id: str                              # "0001"
    started_at_utc: Optional[str] = None      # ISO-8601, set from ShotResult.start_utc
    elapsed_seconds: float = 0.0
    return_code: int = 0
    launch_argv: list[str] = field(default_factory=list)   # actual mpirun/srun cmd


@dataclass
class Manifest:
    schema_version: str = SCHEMA_VERSION
    invocation: list[str] = field(default_factory=list)   # original sys.argv
    cwd: str = ""                                         # working dir at invocation
    started_at_utc: str = ""
    finished_at_utc: str = ""
    semsws_binary: str = ""
    semsws_git_rev: Optional[str] = None
    driver_version: str = DRIVER_VERSION
    shots: list[ShotEntry] = field(default_factory=list)

    @classmethod
    def begin(cls, *, semsws_binary: Path,
              invocation: Optional[list[str]] = None,
              cwd: Optional[Path] = None) -> "Manifest":
        return cls(
            invocation=list(invocation if invocation is not None else sys.argv),
            cwd=str(cwd if cwd is not None else Path.cwd()),
            started_at_utc=utc_now_iso(),
            semsws_binary=str(semsws_binary),
            semsws_git_rev=git_rev_for(semsws_binary),
        )

    def finalize(self) -> "Manifest":
        self.finished_at_utc = utc_now_iso()
        return self

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "invocation": list(self.invocation),
            "cwd": self.cwd,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "semsws_binary": self.semsws_binary,
            "semsws_git_rev": self.semsws_git_rev,
            "driver_version": self.driver_version,
            "shots": [
                {
                    "shot_id": s.shot_id,
                    "started_at_utc": s.started_at_utc,
                    "elapsed_seconds": s.elapsed_seconds,
                    "return_code": s.return_code,
                    "launch_argv": list(s.launch_argv),
                }
                for s in self.shots
            ],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path
