"""
Quantitative 2D fluid-solid coupling benchmark with the OPPOSITE layout:
fluid on TOP, solid on BOTTOM (inverts the +y interface normal compared
to the sibling top=solid / bottom=fluid cases). Reference traces under
`specfem_2d_fluid_solid_oposite/ref/` were produced by SPECFEM2D and are
compared per-channel via normalised L2 error.

SEMSWS 2D vector components are labelled "x" and "y"; the physical axes
are (x horizontal, z vertical), so SEMSWS's "y" maps to SPECFEM2D's BXZ.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "specfem_2d_fluid_solid_oposite"
REF_DIR  = DATA_DIR / "ref"


# SEMSWS 2D ASCII receiver file naming (unchanged across benchmarks):
#   <name>_<sourceid:04d>.<channel>                  (scalar, e.g. pressure)
#   <name>_<comp>_<sourceid:04d>.<channel>           (vector, comp ∈ {x,y})
SEMSWS_TYPE_MAP = {
    "p": "pressure",
    "d": "displacement",
    "v": "velocity",
    "a": "acceleration",
}

# SPECFEM2D ASCII seismogram naming with seismotype=1,2,3,4:
#   AA.<station>.PRE.semp                    # pressure
#   AA.<station>.BX{X|Z}.sem{d|v|a}          # disp/vel/acc, per-axis
SPECFEM_KIND_SUFFIX = {
    "pressure":     ".semp",
    "displacement": ".semd",
    "velocity":     ".semv",
    "acceleration": ".sema",
}

# SEMSWS 2D "y" component is the vertical axis, which SPECFEM2D labels
# "Z". Horizontal "x" matches SPECFEM's "X".
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
def run_semsws_oposite(tmp_path_factory, main_exe: Path, mpi_cmd: str,
                       np_procs: int, device_type: str, request) -> Path:
    """Copy the opposite-layout coupled inputs into a tmp workspace and
    run SEMSWS once for the module. Returns the receiver output dir.

    The YAML embeds the Ricker wavelet inline, so no external STF file
    needs to be copied (the SPECFEM2D reference was generated from the
    same (f0, delay)). The 2D mesh is the external probe.msh shipped
    under DATA_DIR.

    When `--device != cpu` the `coupled.<device>.yaml` overlay is used
    in place of `coupled.yaml` (aborts if missing), so the same test
    body exercises the chosen backend on each invocation."""
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp(f"coupling_vs_specfem2d_oposite_{device_type}")
    config_src = (DATA_DIR / f"coupled.{device_type}.yaml"
                  if device_type != "cpu" else DATA_DIR / "coupled.yaml")
    assert config_src.exists(), f"missing config overlay: {config_src}"
    shutil.copy(config_src, work / "coupled.yaml")
    shutil.copy(DATA_DIR / "probe.msh", work / "probe.msh")
    # SEMSWS YAML writes into ./results (not ./out for this benchmark).
    (work / "results").mkdir(exist_ok=True)

    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(main_exe),
               "--config", "coupled.yaml"]
    else:
        cmd = ["mpirun", "-np", str(np_procs), str(main_exe),
               "--config", "coupled.yaml"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=work)
    assert proc.returncode == 0, (
        f"SEMSWS exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    outdir = work / "results"
    assert outdir.is_dir(), f"SEMSWS output directory missing: {outdir}"

    yield outdir
    if keep:
        print(f"\n[--keep-results] coupling-vs-specfem2d-oposite workspace: {work}")
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
    """Shift both traces to start at 0 (cancels SPECFEM2D's automatic
    pre-trigger offset t0 ≈ -1.2/f0), trim to the common window and
    resample the longer trace onto the shorter's cadence. Pairing is
    per-sample after shift — each sample index corresponds to the same
    STF file index in both codes, which IS the physically matched
    instant when both apply the identical external/inline Ricker."""
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


# OPPOSITE layout: R001 sits in the SOLID bottom (disp/vel/acc), R002
# and R003 sit in the FLUID top (pressure only). Compared to the sibling
# top=solid benchmark the domain-vs-station assignment is exactly
# flipped — that's the whole point of this regression gate.
CHANNELS: list[tuple[str, str, str | None]] = [
    # Solid station R001 (z = 0.25·LZ, bottom)
    *[("R001", kind, comp)
      for kind in ("displacement", "velocity", "acceleration")
      for comp in ("x", "y")],
    # Fluid stations R002, R003 (z = 0.55·LZ, 0.85·LZ, top)
    ("R002", "pressure", None),
    ("R003", "pressure", None),
]

# Normalised L2 tolerance. With the Dirichlet re-enforce fix in
# CoupledSimulationFacade::Step all 8 channels currently come in well
# under 1% (max ≈ 0.006 on the acceleration norms); 5% matches the
# other coupling-vs-SPECFEM tests so future small numerical drift
# doesn't immediately break the gate.
L2_TOL = 0.05


@pytest.mark.parametrize("station,kind,comp", CHANNELS)
def test_channel_matches_specfem2d_oposite(
    run_semsws_oposite, station: str, kind: str, comp: str | None,
) -> None:
    """Each channel's SEMSWS trace must match the committed SPECFEM2D
    reference under the normalised L2 tolerance, on the opposite-layout
    (fluid top / solid bottom) 2D benchmark."""
    outdir = run_semsws_oposite
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
