"""
Quantitative fluid-solid coupling benchmark WITH ATTENUATION:
SEMSWS coupled facade vs SPECFEM3D reference seismograms.

Mirror of `test_coupling_vs_specfem.py` but with viscoelastic
attenuation enabled on the SOLID side (Qkappa = Qmu = 40, f0 = 2 Hz,
3 SLS). Acoustic attenuation is deliberately disabled on both sides:
SPECFEM3D's acoustic solver skips attenuation at runtime regardless
of the Mesh_Par_file Q_Kappa value (src/shared/get_attenuation_model.f90:326
`if (ispec_is_elastic(ispec) .eqv. .false.) cycle`), so applying fluid
Q in SEMSWS would make the two codes physically inequivalent.

See `examples/coupling_vs_specfem_visco/README.md` for the physics
setup and how the reference was generated. The SPECFEM reference
traces in `specfem_3d_fluid_solid_visco/ref/` and the SEMSWS input
(coupled.yaml + probe.msh) are the very files `gen.py` writes into
`examples/coupling_vs_specfem_visco/runs/`; the test fixture runs
SEMSWS once on that config and asserts every one of the 19 comparable
channels matches the SPECFEM trace under a normalised L2 tolerance.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "specfem_3d_fluid_solid_visco"
REF_DIR  = DATA_DIR / "ref"

# Same naming conventions as the non-visco sibling test. Kept local so
# each test file is self-contained even if the mappings diverge later.
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
    """Copy the visco coupled inputs into a tmp workspace and run SEMSWS
    once for the module. Returns the receiver output directory.

    The visco YAML embeds the Ricker wavelet inline, so unlike the
    non-visco benchmark no source_time_function.txt needs to be copied.

    When `--device != cpu` the `coupled.<device>.yaml` overlay is used
    in place of `coupled.yaml` (aborts if missing), so the same test
    body exercises the chosen backend on each invocation.
    """
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp(f"coupling_vs_specfem_visco_{device_type}")
    config_src = (DATA_DIR / f"coupled.{device_type}.yaml"
                  if device_type != "cpu" else DATA_DIR / "coupled.yaml")
    assert config_src.exists(), f"missing config overlay: {config_src}"
    shutil.copy(config_src, work / "coupled.yaml")
    shutil.copy(DATA_DIR / "probe.msh", work / "probe.msh")
    (work / "out").mkdir(exist_ok=True)

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
    outdir = work / "out"
    assert outdir.is_dir(), f"SEMSWS output directory missing: {outdir}"

    yield outdir
    if keep:
        print(f"\n[--keep-results] coupling-vs-specfem-visco workspace: {work}")
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


def _align(t_a: np.ndarray, a: np.ndarray,
           t_b: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_lo = max(t_a[0],  t_b[0])
    t_hi = min(t_a[-1], t_b[-1])
    mask_a = (t_a >= t_lo) & (t_a <= t_hi)
    mask_b = (t_b >= t_lo) & (t_b <= t_hi)
    t_ref = t_a[mask_a] if mask_a.sum() <= mask_b.sum() else t_b[mask_b]
    return np.interp(t_ref, t_a, a), np.interp(t_ref, t_b, b)


def _normalized_l2(sem: np.ndarray, ref: np.ndarray) -> float:
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(sem - ref) / denom) if denom > 0 else float("inf")


# (station, kind, component). `component = None` for scalar (pressure).
CHANNELS: list[tuple[str, str, str | None]] = [
    ("R01", "pressure",     None),
] + [
    (st, kind, comp)
    for st in ("R02", "R03")
    for kind in ("displacement", "velocity", "acceleration")
    for comp in ("x", "y", "z")
]

# Tolerance on normalised L2 error (||SEM − SPEC|| / ||SPEC||). On the
# reference commit all 19 channels came in under ~4.2%; we pin at 5%
# to match the non-visco sibling and the compare.py threshold in
# examples/coupling_vs_specfem_visco/config.py.
L2_TOL = 0.05


@pytest.mark.parametrize("station,kind,comp", CHANNELS)
def test_channel_matches_specfem_visco(run_semsws_visco, station: str,
                                       kind: str, comp: str | None) -> None:
    """Each channel's SEMSWS trace must match the committed SPECFEM3D
    viscoelastic reference under the normalised L2 tolerance."""
    outdir = run_semsws_visco
    sem_path = _find_semsws_trace(outdir, station, kind, comp)

    suffix = SPECFEM_KIND_SUFFIX[kind]
    if comp is None:
        ref_path = REF_DIR / f"XX.{station}.FXP{suffix}"
    else:
        ref_path = REF_DIR / f"XX.{station}.FX{comp.upper()}{suffix}"
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
