"""Strict directory layout for a semsws-driver run.

A `Layout` instance owns a single workdir and exposes named accessors for
every path the driver writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    workdir: Path

    @classmethod
    def at(cls, path: str | Path) -> "Layout":
        return cls(workdir=Path(path).resolve())

    # ---- top-level files ------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.workdir / "manifest.json"

    @property
    def config_copy_path(self) -> Path:
        return self.workdir / "config.yaml"

    # ---- subdirs --------------------------------------------------------

    @property
    def inputs_dir(self) -> Path:
        return self.workdir / "inputs"

    @property
    def inputs_h5(self) -> Path:
        return self.inputs_dir / "observations.h5"

    @property
    def shots_dir(self) -> Path:
        return self.workdir / "shots"

    @property
    def results_dir(self) -> Path:
        return self.workdir / "results"

    @property
    def merged_path(self) -> Path:
        return self.results_dir / "merged.h5"

    @property
    def shared_mesh_dir(self) -> Path:
        return self.workdir / "_shared_mesh"

    @property
    def shared_mesh_partitions_dir(self) -> Path:
        return self.shared_mesh_dir / "partitions"

    @property
    def shared_model_dir(self) -> Path:
        return self.workdir / "_shared_model"

    @property
    def shared_model_bp_dir(self) -> Path:
        return self.shared_model_dir / "bp"

    # ---- per-shot -------------------------------------------------------

    @staticmethod
    def shot_key(shot_id: int) -> str:
        if shot_id < 0:
            raise ValueError(f"shot_id must be >= 0, got {shot_id}")
        return f"shot_{int(shot_id):04d}"

    def shot_dir(self, shot_id: int) -> Path:
        return self.shots_dir / self.shot_key(shot_id)

    def shot_config(self, shot_id: int) -> Path:
        return self.shot_dir(shot_id) / "config.yaml"

    def shot_seismograms(self, shot_id: int) -> Path:
        """Resolve the per-shot output HDF5.

        SEMSWS writes `seismograms<source_id:04d>.h5` (one per source id),
        which under the F1 simultaneous-superposition rule means at most one
        file per shot. We pick the first match if the file exists; otherwise
        return the canonical `seismograms.h5` (used pre-run for path
        announcements).
        """
        d = self.shot_dir(shot_id)
        if d.is_dir():
            matches = sorted(d.glob("seismograms*.h5"))
            if matches:
                return matches[0]
        return d / "seismograms.h5"

    def shot_stdout(self, shot_id: int) -> Path:
        return self.shot_dir(shot_id) / "stdout.log"

    def shot_stderr(self, shot_id: int) -> Path:
        return self.shot_dir(shot_id) / "stderr.log"

    # ---- create skeleton ------------------------------------------------

    def make_skeleton(self) -> None:
        """Create the always-present directories. Per-shot dirs are made
        on demand by the runner."""
        for d in (self.workdir, self.inputs_dir, self.shots_dir,
                  self.results_dir):
            d.mkdir(parents=True, exist_ok=True)
