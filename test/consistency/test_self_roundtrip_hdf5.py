"""Self-roundtrip HDF5 consistency test.

Verifies that a per-shot HDF5 output written by SEMSWS can be re-read as
input on a subsequent simulation, producing bit-exact identical receiver
waveforms. Exercises the v2.0 input path end-to-end.

Pipeline per case:
  (a) Run simulation with inline YAML source / receivers
        write ASCII traces + per-shot HDF5 (`seis_<id>.h5` containing
        `/shots/0000/sources/...` + `/shots/0000/receivers/...`)
  (b) Run simulation with the **same HDF5 as input** for both `sources:`
      and `receivers:` (format: hdf5, shot_id: 0)
        write ASCII traces
  Compare ASCII traces (a) vs (b): bit-exact.

Usage:
    pytest test/consistency/test_self_roundtrip_hdf5.py --build-dir ./build -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

TEST_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = TEST_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from compare_waveforms import read_sem_ascii  # noqa: E402

CASES = [
    ("2D/elastic", TEST_ROOT / "waveform" / "2D" / "elastic" / "config.yaml"),
    ("2D/acoustic", TEST_ROOT / "waveform" / "2D" / "acoustic" / "config.yaml"),
    ("3D/elastic", TEST_ROOT / "waveform" / "3D" / "elastic" / "config.yaml"),
]

SHORT_STEPS = 500


def _run(executable: Path, np_procs: int, config: Path,
         mpi_cmd: str, cwd: Path, timeout: int = 600) -> tuple[bool, str]:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(executable),
               "--config", str(config)]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(executable),
               "--config", str(config)]
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0:
            return False, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


def _read_all_ascii(outdir: Path, source_id: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path in sorted(outdir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in (".d", ".v", ".a", ".p", ".g"):
            continue
        if f"_{source_id}" not in path.stem:
            continue
        out[path.name] = read_sem_ascii(path).data
    return out


def _materialise_yaml_a(src: Path, dst: Path, outdir: Path,
                        device: str, base_filename: str) -> None:
    """Run-(a) config: original sources / receivers; HDF5 + ASCII output."""
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("simulation", {}).setdefault("time", {})["steps"] = SHORT_STEPS
    sim_out = cfg["simulation"].setdefault("output", {})
    sim_out["directory"] = str(outdir)
    if "wavefield" in sim_out:
        sim_out["wavefield"]["enabled"] = False
    cfg.setdefault("device", {})["type"] = device

    # Emit BOTH ascii (for trace comparison) and hdf5 (for self-roundtrip).
    cfg["receivers"]["output"]["formats"] = [
        {"type": "ascii"},
        {"type": "hdf5"},
    ]
    cfg["receivers"]["output"]["filename"] = base_filename

    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _materialise_yaml_b(src: Path, dst: Path, outdir: Path,
                        device: str, h5_input: Path) -> None:
    """Run-(b) config: read sources + receivers from the run-(a) HDF5."""
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("simulation", {}).setdefault("time", {})["steps"] = SHORT_STEPS
    sim_out = cfg["simulation"].setdefault("output", {})
    sim_out["directory"] = str(outdir)
    if "wavefield" in sim_out:
        sim_out["wavefield"]["enabled"] = False
    cfg.setdefault("device", {})["type"] = device

    # Replace sources to read from HDF5.
    cfg["sources"] = {
        "mode": cfg["sources"].get("mode", "sequential"),
        "format": "hdf5",
        "file": str(h5_input),
        "shot_id": 0,
    }
    # Replace receivers to read from HDF5 (geometry-only). Keep parent
    # `type:` from the original config.
    parent_type = cfg["receivers"]["type"]
    cfg["receivers"] = {
        "type": parent_type,
        "format": "hdf5",
        "file": str(h5_input),
        "shot_id": 0,
        "output": {
            "formats": [{"type": "ascii"}],
            "filename": "dummy_b",
        },
    }

    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


@pytest.mark.parametrize("case_id,config_path", CASES,
                         ids=[c[0] for c in CASES])
def test_self_roundtrip(
    case_id: str,
    config_path: Path,
    executable: Path,
    np_procs: int,
    device_type: str,
    mpi_cmd: str,
    keep_results: bool,
    tmp_path_factory,
):
    if not config_path.exists():
        pytest.skip(f"config not found: {config_path}")

    work = tmp_path_factory.mktemp(f"roundtrip_{case_id.replace('/', '_')}")

    # --- Run (a): inline YAML, ASCII + HDF5 output --------------------------
    out_a = work / "results_a"; out_a.mkdir()
    cfg_a = work / "config_a.yaml"
    _materialise_yaml_a(config_path, cfg_a, out_a, device_type,
                        base_filename="seis")
    ok, msg = _run(executable.resolve(), np_procs, cfg_a, mpi_cmd, cwd=work)
    assert ok, f"{case_id}: run-(a) failed:\n{msg}"

    # The per-shot HDF5 emitted by run (a). Source id is 1 in every test
    # config we're parametrising → file name `seis0001.h5`.
    h5_input = out_a / "seis0001.h5"
    assert h5_input.exists(), f"{case_id}: missing per-shot HDF5 {h5_input}"

    # --- Run (b): use the run-(a) HDF5 as both source + receiver input ------
    out_b = work / "results_b"; out_b.mkdir()
    cfg_b = work / "config_b.yaml"
    _materialise_yaml_b(config_path, cfg_b, out_b, device_type, h5_input)
    ok, msg = _run(executable.resolve(), np_procs, cfg_b, mpi_cmd, cwd=work)
    assert ok, f"{case_id}: run-(b) failed:\n{msg}"

    # --- Compare ASCII traces ------------------------------------------------
    a_traces = _read_all_ascii(out_a, "0001")
    b_traces = _read_all_ascii(out_b, "0001")
    assert a_traces, f"{case_id}: no ASCII output from run-(a)"
    assert a_traces.keys() == b_traces.keys(), (
        f"{case_id}: trace filename sets differ\n"
        f"  (a) only: {a_traces.keys() - b_traces.keys()}\n"
        f"  (b) only: {b_traces.keys() - a_traces.keys()}"
    )

    failures: list[str] = []
    for name, ax in a_traces.items():
        bx = b_traces[name]
        if ax.shape != bx.shape:
            failures.append(f"{name}: shape {ax.shape} vs {bx.shape}")
            continue
        if not np.array_equal(ax, bx):
            diff = float(np.max(np.abs(ax - bx)))
            failures.append(f"{name}: max abs diff = {diff:.3e}")

    if not keep_results:
        shutil.rmtree(work, ignore_errors=True)

    assert not failures, f"{case_id}: {len(failures)} mismatches:\n" + \
                          "\n".join(failures)
