"""
Quantitative 2D fluid-solid coupling benchmark WITH ATTENUATION on
BOTH domains: viscoacoustic in the fluid and viscoelastic in the solid
(same Q_κ/Q_μ = 40 in each, f0 = 5 Hz, 3 SLS — Liu-Anderson fit).

Rationale:
  * SPECFEM3D's acoustic solver SKIPS attenuation regardless of the
    Q value in the mesh file, which is why the 3D viscoelastic
    benchmark only enables Q on the SOLID side. SPECFEM2D does
    NOT share that limitation — it has separate
    `ATTENUATION_VISCOELASTIC` and `ATTENUATION_VISCOACOUSTIC` flags
    and a real viscoacoustic time-integrator. So 2D is the first
    benchmark in SEMSWS's coupling test suite where the fluid side
    actually carries a non-trivial Q against a reference solver.
  * The Par_file sets `READ_VELOCITIES_AT_f0 = .true.` so SPECFEM2D
    interprets the user-supplied Vp / Vs as the velocities at the
    reference frequency f0 (not as the unrelaxed "infinite frequency"
    values). SEMSWS does the same unrelaxed→f0 correction inside
    `IsotropicAcousticMaterial::ApplyAttenuationCorrection` /
    `IsotropicElasticMaterial::ApplyAttenuationCorrection`, so the
    two codes see the same effective phase velocities at f0.
  * The fluid-solid interface coupling still works: corner DOFs at
    Γ_fs ∩ free-surface are zeroed by the facade's Dirichlet re-enforce
    after `FinalizeAndApplyMass`, matching SPECFEM2D's
    `enforce_acoustic_free_surface` at the tail of
    `compute_forces_viscoacoustic_main`.

Reference traces under `specfem_2d_fluid_solid_visco/ref/` were
produced by SPECFEM2D from the same `coupled.yaml` / probe mesh via
`examples/coupling_vs_specfem2d/gen.py` + xmeshfem2D / xspecfem2D.
Each channel is compared via normalised L2 error under the same 5%
tolerance the other coupling-vs-SPECFEM tests use.
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

# SEMSWS 2D "y" is the vertical axis, SPECFEM2D labels the same axis
# "Z". Horizontal component matches ("x" ↔ "X").
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
def run_semsws_visco(tmp_path_factory, main_exe: Path, mpi_cmd: str,
                     np_procs: int, device_type: str, request) -> Path:
    """Copy the visco coupled inputs into a tmp workspace and run
    SEMSWS once for the module. Returns the receiver output dir.

    The YAML embeds the Ricker wavelet inline, so no external STF
    needs to be copied (the SPECFEM2D reference was produced from
    the same (f0, delay)). The 2D mesh is the external probe.msh
    shipped under DATA_DIR.

    When `--device != cpu` the `coupled.<device>.yaml` overlay is used
    in place of `coupled.yaml` (aborts if missing); the dedicated
    `test_coupling_vs_specfem2d_visco_gpu.py` module provides an
    additional np=2 canary that exercises the remote-interface MPI
    path. This single-rank module just swaps the backend."""
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp(f"coupling_vs_specfem2d_visco_{device_type}")
    config_src = (DATA_DIR / f"coupled.{device_type}.yaml"
                  if device_type != "cpu" else DATA_DIR / "coupled.yaml")
    assert config_src.exists(), f"missing config overlay: {config_src}"
    shutil.copy(config_src, work / "coupled.yaml")
    shutil.copy(DATA_DIR / "probe.msh", work / "probe.msh")
    # SEMSWS yaml writes into ./results for this benchmark.
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
        print(f"\n[--keep-results] coupling-vs-specfem2d-visco workspace: {work}")
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
    resample the longer trace onto the shorter's cadence."""
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


# Regular (non-opposite) layout: R001 fluid bottom (pressure), R002/R003
# solid top (disp/vel/acc). Both domains carry non-trivial Q in this
# benchmark.
CHANNELS: list[tuple[str, str, str | None]] = [
    ("R001", "pressure", None),
    *[("R002", kind, comp)
      for kind in ("displacement", "velocity", "acceleration")
      for comp in ("x", "y")],
    *[("R003", kind, comp)
      for kind in ("displacement", "velocity", "acceleration")
      for comp in ("x", "y")],
]

# Normalised L2 tolerance. On the reference commit the 13 channels land
# between 0.008 and 0.033; 5% matches the other coupling-vs-SPECFEM
# tests and leaves headroom for small time-integration / attenuation-
# fit drift.
L2_TOL = 0.05


@pytest.mark.parametrize("station,kind,comp", CHANNELS)
def test_channel_matches_specfem2d_visco(
    run_semsws_visco, station: str, kind: str, comp: str | None,
) -> None:
    """Each channel's SEMSWS trace must match the committed SPECFEM2D
    reference under the normalised L2 tolerance — both fluid (Q_κ) and
    solid (Q_κ, Q_μ) carry attenuation in this benchmark."""
    outdir = run_semsws_visco
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
