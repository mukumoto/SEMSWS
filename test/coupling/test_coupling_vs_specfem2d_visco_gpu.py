"""
GPU regression of the 2D fluid-solid viscoacoustic/viscoelastic coupling
against SPECFEM2D reference traces. Mirrors test_coupling_vs_specfem2d_visco.py
(CPU) and additionally exercises the paths that only fire on GPU + multi-rank:

  * `AtomicAdd` scatter on shared interface GLL nodes (Commit 1)
  * `MFEM_DEVICE_SYNC` between pack forall and MPI_Isend
  * Remote-interface MPI path (`have_remote=true`), with either
    GPU-aware MPI zero-copy or the `HostRead()/HostReadWrite()`
    host-staging fallback depending on the MPI build

The np=2 decomposition is REQUIRED — single-rank skips `have_remote`
and leaves both the remote MPI path and cross-face atomic contention
untested. Selected config is `coupled.{device_type}.yaml`.

Skipped on CPU runs (the whole point is GPU coverage). Tolerance is
1.5× the CPU benchmark to absorb FMA-reordering drift between CPU and
GPU accumulation order; retune after a first clean GPU run if desired.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "specfem_2d_fluid_solid_visco"
REF_DIR  = DATA_DIR / "ref"


SEMSWS_TYPE_MAP = {
    "p": "pressure",
    "d": "displacement",
    "v": "velocity",
    "a": "acceleration",
}

SPECFEM_KIND_SUFFIX = {
    "pressure":     ".semp",
    "displacement": ".semd",
    "velocity":     ".semv",
    "acceleration": ".sema",
}

SEMSWS_TO_SPECFEM_COMP = {"x": "X", "y": "Z"}


@pytest.fixture(scope="module")
def main_exe(build_dir: Path) -> Path:
    candidates = [
        build_dir / "src" / "semsws",   # CMake build dir
        build_dir / "bin" / "semsws",   # Spack install prefix
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    pytest.skip(
        f"semsws not found in any of: "
        + ", ".join(str(c) for c in candidates)
    )


@pytest.fixture(scope="module")
def run_semsws_visco_gpu(tmp_path_factory, main_exe: Path, mpi_cmd: str,
                         np_procs: int, device_type: str, request) -> Path:
    """Run the visco coupled benchmark under --device {cuda|hip} and
    --np >= 2. Returns the receiver output dir. Skips on CPU runs or
    when the caller did not request multi-rank execution."""
    if device_type == "cpu":
        pytest.skip("GPU coupling test; rerun with --device cuda|hip")
    # np >= 2 is a hard requirement for this module: single-rank skips
    # the remote-interface MPI exchange and the cross-face atomic
    # scatter on shared GLL nodes, which are the whole point of the
    # test. Callers must pass `--np 2` (or more) explicitly.
    if np_procs < 2:
        pytest.skip(
            f"GPU canary requires --np >= 2 to exercise the remote "
            f"fluid-solid interface; got --np {np_procs}"
        )

    config_name = f"coupled.{device_type}.yaml"
    config_src = DATA_DIR / config_name
    if not config_src.exists():
        pytest.skip(f"No GPU config overlay at {config_src}")

    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp(f"coupling_vs_specfem2d_visco_{device_type}")
    shutil.copy(config_src, work / "coupled.yaml")
    shutil.copy(DATA_DIR / "probe.msh", work / "probe.msh")
    (work / "results").mkdir(exist_ok=True)

    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(main_exe),
               "--config", "coupled.yaml"]
    else:
        cmd = ["mpirun", "-np", str(np_procs), str(main_exe),
               "--config", "coupled.yaml"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          cwd=work)
    assert proc.returncode == 0, (
        f"SEMSWS exited {proc.returncode} (device={device_type}, np={np_procs})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    outdir = work / "results"
    assert outdir.is_dir(), f"SEMSWS output directory missing: {outdir}"

    yield outdir
    if keep:
        print(f"\n[--keep-results] GPU coupling-vs-specfem2d-visco "
              f"workspace ({device_type}): {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def _load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    assert data.ndim == 2 and data.shape[1] >= 2, (
        f"unexpected trace shape {data.shape} in {path}"
    )
    return data[:, 0], data[:, 1]


def _find_semsws_trace(outdir: Path, station: str, kind: str,
                       comp: str | None) -> Path:
    type_char = {v: k for k, v in SEMSWS_TYPE_MAP.items()}[kind]
    if comp is None:
        pattern = f"{station}_*.{type_char}"
    else:
        pattern = f"{station}_{comp}_*.{type_char}"
    hits = sorted(outdir.glob(pattern))
    assert hits, (
        f"no SEMSWS trace matching {pattern} under {outdir} "
        f"(contents: {sorted(p.name for p in outdir.iterdir())})"
    )
    return hits[0]


def _specfem_path(station: str, kind: str, comp: str | None) -> Path:
    suffix = SPECFEM_KIND_SUFFIX[kind]
    if comp is None:
        name = f"AA.{station}.PRE{suffix}"
    else:
        name = f"AA.{station}.BX{SEMSWS_TO_SPECFEM_COMP[comp]}{suffix}"
    return REF_DIR / name


def _align(t_a: np.ndarray, a: np.ndarray,
           t_b: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same alignment as the CPU test — shift both traces to start at 0,
    trim to the common window, resample onto the shorter cadence."""
    t_a_shift = t_a - t_a[0]
    t_b_shift = t_b - t_b[0]
    t_lo = max(t_a_shift[0],  t_b_shift[0])
    t_hi = min(t_a_shift[-1], t_b_shift[-1])
    mask_a = (t_a_shift >= t_lo) & (t_a_shift <= t_hi)
    mask_b = (t_b_shift >= t_lo) & (t_b_shift <= t_hi)
    t_ref = (t_a_shift[mask_a] if mask_a.sum() <= mask_b.sum()
             else t_b_shift[mask_b])
    return np.interp(t_ref, t_a_shift, a), np.interp(t_ref, t_b_shift, b)


def _normalized_l2(sem: np.ndarray, ref: np.ndarray) -> float:
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(sem - ref) / denom) if denom > 0 else float("inf")


CHANNELS: list[tuple[str, str, str | None]] = [
    ("R001", "pressure", None),
    *[("R002", kind, comp)
      for kind in ("displacement", "velocity", "acceleration")
      for comp in ("x", "y")],
    *[("R003", kind, comp)
      for kind in ("displacement", "velocity", "acceleration")
      for comp in ("x", "y")],
]

# 1.5× the CPU tolerance (0.05 → 0.075). FMA ordering differs between
# serial host += and hardware atomicAdd, producing sub-1% drift on the
# output traces. Retune after calibration on real GPU hardware.
L2_TOL = 0.075


@pytest.mark.parametrize("station,kind,comp", CHANNELS)
def test_channel_matches_specfem2d_visco_gpu(
    run_semsws_visco_gpu, station: str, kind: str, comp: str | None,
) -> None:
    """Same channel-by-channel comparison as the CPU benchmark, under a
    slightly looser tolerance to accommodate GPU FMA reordering."""
    outdir = run_semsws_visco_gpu
    sem_path = _find_semsws_trace(outdir, station, kind, comp)
    ref_path = _specfem_path(station, kind, comp)
    assert ref_path.exists(), f"missing reference file: {ref_path}"

    t_s, v_s = _load_trace(sem_path)
    t_r, v_r = _load_trace(ref_path)
    v_sa, v_ra = _align(t_s, v_s, t_r, v_r)

    err = _normalized_l2(v_sa, v_ra)
    label = kind + (f"-{comp}" if comp else "")
    assert err <= L2_TOL, (
        f"{station} {label}: L2={err:.4f} exceeds tolerance {L2_TOL:.3f} "
        f"(sem={sem_path.name}, ref={ref_path.name})"
    )
