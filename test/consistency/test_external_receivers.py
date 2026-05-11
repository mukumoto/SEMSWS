"""
External receiver file consistency test.

Verifies that loading receivers from an external YAML file produces
identical output to inline receiver definitions.

Usage:
    pytest test/consistency/test_external_receivers.py --build-dir ./build -v
    pytest test/consistency/test_external_receivers.py --build-dir ./build --device hip -v
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

# Base config: 2D acoustic (simple, fast)
BASE_CONFIG = TEST_ROOT / "waveform" / "2D" / "acoustic" / "config.yaml"

# 1 second of simulation time (dt=0.0001)
SHORT_STEPS = 10000


# ---------------------------------------------------------------------------
# Config generation helpers
# ---------------------------------------------------------------------------

def make_inline_config(src: Path, dst: Path, outdir: Path, device: str):
    """Write config with inline receivers, reduced steps."""
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["time"]["steps"] = SHORT_STEPS
    cfg["simulation"]["output"]["directory"] = str(outdir)
    cfg["simulation"]["output"]["wavefield"]["enabled"] = False
    cfg["device"]["type"] = device
    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def make_external_config(src: Path, dst_config: Path, dst_receivers: Path,
                         outdir: Path, device: str):
    """Write config with receivers.file pointing to external YAML file."""
    with open(src) as f:
        cfg = yaml.safe_load(f)

    # Extract receiver list/line into external file
    ext = {}
    if "list" in cfg["receivers"]:
        ext["list"] = cfg["receivers"].pop("list")
    if "line" in cfg["receivers"]:
        ext["line"] = cfg["receivers"].pop("line")

    with open(dst_receivers, "w") as f:
        yaml.dump(ext, f, default_flow_style=False)

    # Point main config to external file (relative path)
    cfg["receivers"]["file"] = dst_receivers.name
    cfg["simulation"]["time"]["steps"] = SHORT_STEPS
    cfg["simulation"]["output"]["directory"] = str(outdir)
    cfg["simulation"]["output"]["wavefield"]["enabled"] = False
    cfg["device"]["type"] = device

    with open(dst_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# ASCII output reader
# ---------------------------------------------------------------------------

def read_all_ascii(outdir: Path) -> dict[str, np.ndarray]:
    """Read all ASCII seismogram files into {filename: data}."""
    traces = {}
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.suffix in (".p", ".d", ".v", ".a", ".g"):
            wf = read_sem_ascii(p)
            traces[p.name] = wf.data
    return traces


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_external_receiver_file(
    executable, np_procs, device_type, mpi_cmd, keep_results, tmp_path
):
    """Inline vs external receiver file must produce identical output."""
    if not BASE_CONFIG.exists():
        pytest.skip(f"Base config not found: {BASE_CONFIG}")

    exe = executable.resolve()

    # --- Run with inline receivers ---
    work_inline = tmp_path / "inline"
    work_inline.mkdir()
    outdir_inline = work_inline / "results"
    outdir_inline.mkdir()
    cfg_inline = work_inline / "config.yaml"
    make_inline_config(BASE_CONFIG, cfg_inline, outdir_inline, device_type)

    flag = "-n" if mpi_cmd == "srun" else "-np"
    cmd_inline = [mpi_cmd, flag, str(np_procs),
                  str(exe), "--config", str(cfg_inline)]
    result = subprocess.run(cmd_inline, cwd=work_inline,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"Inline simulation failed (rc={result.returncode}):\n{result.stderr}"
    )

    # --- Run with external receiver file ---
    work_ext = tmp_path / "external"
    work_ext.mkdir()
    outdir_ext = work_ext / "results"
    outdir_ext.mkdir()
    cfg_ext = work_ext / "config.yaml"
    recv_file = work_ext / "receivers.yaml"
    make_external_config(BASE_CONFIG, cfg_ext, recv_file, outdir_ext, device_type)

    cmd_ext = [mpi_cmd, flag, str(np_procs),
               str(exe), "--config", str(cfg_ext)]
    result = subprocess.run(cmd_ext, cwd=work_ext,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"External-file simulation failed (rc={result.returncode}):\n{result.stderr}"
    )

    # --- Compare outputs ---
    inline_traces = read_all_ascii(outdir_inline)
    ext_traces = read_all_ascii(outdir_ext)

    assert len(inline_traces) > 0, "No ASCII output from inline simulation"
    assert inline_traces.keys() == ext_traces.keys(), (
        f"Different output files:\n"
        f"  inline only: {inline_traces.keys() - ext_traces.keys()}\n"
        f"  external only: {ext_traces.keys() - inline_traces.keys()}"
    )

    # Normalised-L2 tolerance. The two runs use semantically identical
    # receiver definitions (inline vs external YAML path), so on a
    # deterministic device they SHOULD match to round-off. On CPU the
    # difference is consistently 0 (bit-identical); on GPU small
    # non-determinism in accumulation order or receiver interpolation
    # can surface as ~1e-5 drift. 1e-3 keeps real regressions (code
    # paths that silently drop/reorder/shift receivers, the original
    # motivation for this test) visible while tolerating benign GPU
    # floating-point reordering.
    L2_TOL = 1.0e-3

    for name in sorted(inline_traces):
        ref = inline_traces[name]
        test = ext_traces[name]

        # Verify waveforms contain non-zero values
        assert np.any(ref != 0), f"{name}: inline waveform is all zeros"
        assert np.any(test != 0), f"{name}: external waveform is all zeros"

        ref_norm = np.linalg.norm(ref)
        assert ref_norm > 0, f"{name}: inline waveform norm is zero"
        rel_err = np.linalg.norm(test - ref) / ref_norm
        max_abs = float(np.max(np.abs(test - ref)))
        print(f"  {name}: L2 relative error = {rel_err:.2e}  "
              f"(max abs = {max_abs:.3e})")
        assert rel_err <= L2_TOL, (
            f"{name}: inline/external normalized L2 = {rel_err:.3e} "
            f"exceeds tolerance {L2_TOL:.1e} (max abs diff {max_abs:.3e}). "
            "If this reflects real GPU non-determinism, lower the tolerance "
            "of the offending simulation; it must not be silently absorbed."
        )

    # --- Cleanup ---
    if not keep_results:
        shutil.rmtree(tmp_path, ignore_errors=True)
