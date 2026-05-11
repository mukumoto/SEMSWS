"""Top-level `run()` entry point for the semsws-driver."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py

from ..io import hdf5_schema as S
from ..io.hdf5_v2 import merge_shot_outputs
from ..runner import binding_policy_from, make_runner
from ..runner.base import ShotResult, ShotTask
from .layout import Layout
from .manifest import Manifest, ShotEntry
from .preflight import (
    material_needs_export, mesh_needs_partition,
    run_material_preflight, run_mesh_preflight,
)
from .run_config import RunConfig
from .template import (
    extract_run_config, load_template, render_all_shots,
)

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    workdir: Path
    layout: Layout
    shot_outcomes: list[ShotResult] = field(default_factory=list)
    merged_path: Optional[Path] = None
    manifest_path: Optional[Path] = None


def _enumerate_shot_ids(inputs_h5: Path) -> list[int]:
    """Return the shot ids (as ints) present in the bundled HDF5."""
    with h5py.File(str(inputs_h5), "r") as f:
        if S.GROUP_SHOTS not in f:
            raise ValueError(
                f"{inputs_h5}: missing /{S.GROUP_SHOTS} group "
                "(not a v2.0 input file)"
            )
        keys = sorted(f[S.GROUP_SHOTS].keys())
    ids: list[int] = []
    for k in keys:
        try:
            ids.append(int(k))
        except ValueError:
            raise ValueError(
                f"{inputs_h5}: shot key {k!r} is not a 4-digit integer"
            ) from None
    return ids


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink the input HDF5 into <workdir>/inputs/, falling back to copy
    if the FS does not support symlinks (rare but possible)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def run(
    *,
    config: str | Path,
    inputs: str | Path,
    workdir: str | Path,
    merge: bool = False,
    only_shots: Optional[list[int]] = None,
    dry_run: bool = False,
    report_affinity: bool = False,
) -> RunResult:
    """End-to-end orchestration.

    Parameters
    ----------
    config
        Path to the user-authored YAML template.
    inputs
        Path to the v2.0 multi-shot HDF5 (sources + receivers + observed).
    workdir
        Output directory. Will be created (skeleton: inputs/, shots/, results/).
    merge
        When True, run `merge_shot_outputs` after all shots succeed and
        write `<workdir>/results/merged.h5`.
    only_shots
        Optional subset of shot ids to run. Defaults to all shots in the
        input HDF5.
    dry_run
        Generate per-shot YAMLs and the manifest skeleton but do not invoke
        Runner. Returns a RunResult with empty shot_outcomes.
    """
    cfg_path = Path(config).resolve()
    inputs_path = Path(inputs).resolve()
    L = Layout.at(workdir)
    L.make_skeleton()

    # 1. Load template + extract RunConfig.
    template = load_template(cfg_path)
    rc = extract_run_config(template)

    # 2. Place a copy of the template in <workdir>/config.yaml.
    shutil.copy2(cfg_path, L.config_copy_path)

    # 3. Stage the input HDF5 under inputs/observations.h5.
    _link_or_copy(inputs_path, L.inputs_h5)

    # 4. Enumerate shot ids.
    shot_ids = _enumerate_shot_ids(L.inputs_h5)
    if only_shots is not None:
        wanted = set(int(s) for s in only_shots)
        unknown = wanted - set(shot_ids)
        if unknown:
            raise ValueError(
                f"only_shots references missing shot ids: {sorted(unknown)}"
            )
        shot_ids = [s for s in shot_ids if s in wanted]
    if not shot_ids:
        raise ValueError(f"{inputs_path}: no shots to run")

    # 5. Preflight.
    mesh_override: Optional[dict] = None
    material_override: Optional[dict] = None
    launcher = rc.effective_launcher()
    if mesh_needs_partition(template.get("mesh", {}), rc.ranks_per_shot):
        log.info("Running mesh preflight (partition_mesh)")
        mesh_override = run_mesh_preflight(
            mesh=template["mesh"], rc=rc, layout=L, launcher=launcher,
        )
    if material_needs_export(template.get("material", {})):
        log.info("Running material preflight (semsws_export_model)")
        material_override = run_material_preflight(
            template=template, rc=rc, layout=L,
            inputs_h5=L.inputs_h5, launcher=launcher,
        )

    # 6. Render per-shot YAMLs.
    render_all_shots(
        template=template,
        shot_ids=shot_ids,
        layout=L,
        inputs_h5=L.inputs_h5,
        mesh_override=mesh_override,
        material_override=material_override,
    )

    if dry_run:
        return RunResult(workdir=L.workdir, layout=L)

    # 7. Build tasks + submit.
    tasks: list[ShotTask] = []
    for sid in shot_ids:
        L.shot_dir(sid).mkdir(parents=True, exist_ok=True)
        tasks.append(ShotTask(
            shot_id=L.shot_key(sid),
            config_path=L.shot_config(sid),
            workdir=L.shot_dir(sid),
            extra_env=dict(rc.env),
        ))

    runner = make_runner(rc)
    binding = binding_policy_from(rc)
    if report_affinity:
        # Append the bundled rank wrapper so each rank prints [AFF] ...
        # to its shot's stderr.log before exec'ing the next thing.
        from ..scripts import script_path
        aff_script = script_path("report_affinity.sh")
        binding.rank_wrapper = (
            list(binding.rank_wrapper or []) + [str(aff_script)]
        )
        log.info("Affinity reporting enabled via %s", aff_script)
    log.info("Submitting %d shots via %s/%s",
             len(tasks), rc.scheduler, launcher)
    outcomes = runner.submit(tasks, binding=binding)

    # 8. Optional merge.
    merged: Optional[Path] = None
    if merge:
        per_shot_files = [L.shot_seismograms(sid) for sid in shot_ids
                          if L.shot_seismograms(sid).exists()]
        if per_shot_files:
            merged = merge_shot_outputs(
                sources=per_shot_files,
                output=L.merged_path,
                output_shot_ids=shot_ids[:len(per_shot_files)],
            )

    # 9. Manifest.
    manifest = Manifest.begin(semsws_binary=rc.binary)
    for outcome in outcomes:
        manifest.shots.append(ShotEntry(
            shot_id=outcome.shot_id.split("_", 1)[-1],
            started_at_utc=outcome.start_utc,
            elapsed_seconds=outcome.elapsed_seconds,
            return_code=outcome.return_code,
            launch_argv=list(outcome.extra.get("command", [])),
        ))
    manifest.finalize()
    manifest.write(L.manifest_path)

    return RunResult(
        workdir=L.workdir, layout=L,
        shot_outcomes=list(outcomes),
        merged_path=merged,
        manifest_path=L.manifest_path,
    )
