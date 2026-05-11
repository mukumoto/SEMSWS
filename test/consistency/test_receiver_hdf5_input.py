"""HDF5 receiver-input consistency test.

Runs the same waveform config twice — once with the inline YAML receiver
list, once with `receivers.format: hdf5` pointing at an h5py-generated v2.0
file under `/shots/<shot_id>/receivers/...` — and verifies the ASCII
seismograms are bit-exact identical.

Usage:
    pytest test/consistency/test_receiver_hdf5_input.py --build-dir ./build -v
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
    ("3D/elastic", TEST_ROOT / "waveform" / "3D" / "elastic" / "config.yaml"),
]

SHORT_STEPS = 500   # keep runtime small


def write_hdf5_receivers(path: Path, recv_list: list[dict],
                         space_dim: int, dt: float, n_samples: int,
                         shot_id: int = 0) -> None:
    """Write a v2.0 SEMSWS HDF5 file with geometry-only receivers under
    /shots/<shot_id>/receivers/. No waveform datasets — pure forward input."""
    import h5py
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "2.0"
        f.attrs["dt"] = float(dt)
        f.attrs["t0"] = 0.0
        f.attrs["n_samples"] = np.int64(n_samples)
        f.attrs["space_dim"] = np.int32(space_dim)
        f.attrs["coord_system"] = "cartesian"
        f.attrs["units"] = "SI"

        gshot = f.create_group(f"shots/{shot_id:04d}")
        gshot.attrs["shot_id"] = np.int32(shot_id)
        grecv = gshot.create_group("receivers")

        for i, r in enumerate(recv_list, start=1):
            g = grecv.create_group(f"R{i:04d}")
            g.attrs["position"] = np.asarray(
                r["location"][:space_dim], dtype=np.float64)
            # @label preserves the YAML user-friendly name so ASCII output
            # filenames stay aligned across the two runs.
            g.attrs["label"] = r["name"]


def run_simulation(executable: Path, np_procs: int, config: Path,
                   mpi_cmd: str, cwd: Path, timeout: int = 600
                   ) -> tuple[bool, str]:
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
    """Map filename → time series for every ASCII trace in outdir."""
    out: dict[str, np.ndarray] = {}
    for path in sorted(outdir.iterdir()):
        if not path.is_file():
            continue
        # SEM ASCII traces have suffixes like .d/.v/.a/.p/.g; require the
        # source-id token in the stem to avoid catching unrelated files.
        if path.suffix not in (".d", ".v", ".a", ".p", ".g"):
            continue
        if f"_{source_id}" not in path.stem:
            continue
        out[path.name] = read_sem_ascii(path).data
    return out


def _materialise_yaml(src: Path, dst: Path, outdir: Path, device: str,
                      patch: dict) -> None:
    with open(src) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("simulation", {}).setdefault("time", {})["steps"] = SHORT_STEPS
    sim_out = cfg["simulation"].setdefault("output", {})
    sim_out["directory"] = str(outdir)
    if "wavefield" in sim_out:
        sim_out["wavefield"]["enabled"] = False

    cfg.setdefault("device", {})["type"] = device

    cfg["receivers"]["output"]["formats"] = [{"type": "ascii"}]
    cfg["receivers"]["output"]["filename"] = "dummy"

    # Apply test-specific patch (e.g. swap inline list → format: hdf5).
    for k, v in patch.items():
        if v is None:
            cfg["receivers"].pop(k, None)
        else:
            cfg["receivers"][k] = v

    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


@pytest.mark.parametrize("case_id,config_path", CASES,
                         ids=[c[0] for c in CASES])
def test_hdf5_receiver_input_matches_yaml(
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

    src = yaml.safe_load(config_path.read_text())
    space_dim = int(src["simulation"]["dimension"])
    inline_recv = src["receivers"]["list"]
    dt = float(src["simulation"]["time"]["dt"])

    work = tmp_path_factory.mktemp(f"hdf5in_{case_id.replace('/', '_')}")

    # --- Run 1: inline YAML receivers -----------------------------------------
    out_yaml = work / "results_yaml"
    out_yaml.mkdir()
    cfg_yaml = work / "config_yaml.yaml"
    _materialise_yaml(config_path, cfg_yaml, out_yaml, device_type, patch={})
    ok, msg = run_simulation(executable.resolve(), np_procs, cfg_yaml,
                             mpi_cmd, cwd=work)
    assert ok, f"{case_id}: YAML run failed:\n{msg}"

    # --- Run 2: HDF5 receivers ------------------------------------------------
    h5_path = work / "receivers.h5"
    write_hdf5_receivers(h5_path, inline_recv, space_dim, dt,
                         n_samples=SHORT_STEPS, shot_id=0)

    out_h5 = work / "results_h5"
    out_h5.mkdir()
    cfg_h5 = work / "config_h5.yaml"
    _materialise_yaml(config_path, cfg_h5, out_h5, device_type, patch={
        "format": "hdf5",
        "file": str(h5_path),
        "shot_id": 0,
        "list": None,   # remove inline
        "line": None,
    })
    ok, msg = run_simulation(executable.resolve(), np_procs, cfg_h5,
                             mpi_cmd, cwd=work)
    assert ok, f"{case_id}: HDF5 run failed:\n{msg}"

    # --- Compare --------------------------------------------------------------
    yaml_traces = _read_all_ascii(out_yaml, "0001")
    h5_traces   = _read_all_ascii(out_h5,   "0001")
    assert yaml_traces, f"{case_id}: no ASCII output from YAML run"
    assert yaml_traces.keys() == h5_traces.keys(), (
        f"{case_id}: trace filename sets differ\n"
        f"  YAML only: {yaml_traces.keys() - h5_traces.keys()}\n"
        f"  HDF5 only: {h5_traces.keys() - yaml_traces.keys()}"
    )

    failures: list[str] = []
    for name, a in yaml_traces.items():
        b = h5_traces[name]
        if a.shape != b.shape:
            failures.append(f"{name}: shape {a.shape} vs {b.shape}")
            continue
        if not np.array_equal(a, b):
            diff = float(np.max(np.abs(a - b)))
            failures.append(f"{name}: max abs diff = {diff:.3e}")

    if not keep_results:
        shutil.rmtree(work, ignore_errors=True)

    assert not failures, f"{case_id}: {len(failures)} mismatches:\n" + \
                          "\n".join(failures)
