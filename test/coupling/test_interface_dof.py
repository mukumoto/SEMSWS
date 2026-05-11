"""
Phase 2A tests for FluidSolidInterface (Doc §Q.2 Test 2.1/2.2/2.3):

- Test 2.1/2.3 (attribute + normal + unit length):
  Auto-detect the interface bdr attribute via set-difference, then verify
  every quadrature point's cached normal is the fluid→solid direction
  (+ẑ for this horizontally-split probe mesh) and has unit length.

- Test 2.2 (Extract u·n):
  Fill the solid displacement with a known linear field u_s = (0, 0, z),
  call ExtractSolidDisplacementNormal, and require bit-exact agreement
  with the expected value z * n_z at every quad point.

Runs under np=1 and np=2; larger rank counts on this 144-hex mesh hit
the known zero-local-elements case (Doc §L.3) so they are skipped here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


TEST_DIR        = Path(__file__).parent
GENERATE_SCRIPT = TEST_DIR / "generate_probe_mesh.py"


@pytest.fixture(scope="module")
def probe_exe(build_dir: Path) -> Path:
    candidates = [
        build_dir / "src" / "coupling_probe_interface",   # CMake build dir
        build_dir / "bin" / "coupling_probe_interface",   # Spack install prefix
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    pytest.skip(
        f"coupling_probe_interface not found in any of: "
        + ", ".join(str(c) for c in candidates)
    )


@pytest.fixture(scope="module")
def gmsh_available() -> None:
    try:
        import gmsh  # noqa: F401
    except ImportError:
        pytest.skip("gmsh python module not installed")


@pytest.fixture(scope="module")
def probe_mesh(tmp_path_factory, gmsh_available, request) -> Path:
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_iface_probe")
    mesh_file = work / "probe.msh"
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", "600", "--ly", "300", "--lz", "400",
         "--nx", "6", "--ny", "3", "--nz-half", "4"],
        check=True, capture_output=True, text=True,
    )
    yield mesh_file
    if keep:
        print(f"\n[--keep-results] interface probe workspace retained: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def _parse_kv(stdout: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _run(exe: Path, mesh: Path, np_procs: int, mpi_cmd: str,
         order: int = 4) -> dict[str, str]:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe),
               "--mesh", str(mesh), "--order", str(order), "--dim", "3"]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe),
               "--mesh", str(mesh), "--order", str(order), "--dim", "3"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\ncmd: {' '.join(cmd)}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    kv = _parse_kv(proc.stdout)
    assert kv.get("PROBE_OK") == "1", f"probe not OK:\n{proc.stdout}"
    return kv


@pytest.mark.parametrize("np_procs", [1, 2])
def test_interface_attr_and_quad_count(probe_exe: Path, probe_mesh: Path,
                                       mpi_cmd: str, np_procs: int) -> None:
    kv = _run(probe_exe, probe_mesh, np_procs, mpi_cmd)

    assert int(kv["INTERFACE_ATTR"]) > 0, (
        "interface attribute must be auto-detected as a positive integer"
    )
    # Probe mesh layout: 6x3 interface quad faces × (order+1)² GLL nodes each.
    # For order=4, that's 18 × 25 = 450 interface quadrature points.
    assert int(kv["NUM_GLOBAL_QUAD"]) == 18 * (4 + 1) ** 2


@pytest.mark.parametrize("np_procs", [1, 2])
def test_normals_are_plus_z_and_unit_length(probe_exe: Path, probe_mesh: Path,
                                            mpi_cmd: str, np_procs: int) -> None:
    """For the horizontally-split probe (interface plane z=Lz/2), the
    fluid→solid normal must be (essentially) +ẑ at every quadrature
    point. That makes Σ n_x ≈ Σ n_y ≈ 0 and Σ n_z = NUM_GLOBAL_QUAD.

    The tangential-sum tolerance accommodates ulp-level rounding that
    GMSH's transfinite mesher bakes into node positions (e.g. 500 →
    499.9999999999999, seen in ~25% of nodes on this probe). Those
    rounding errors propagate through CalcOrtho's cross product into
    tangential components of size ~3.8e-8 per quadrature point, which
    sum non-cancelling to ~1e-5 across 450 quad points. A physically-
    tilted interface would produce residues of order 1e-2 or larger,
    so 1e-4 is comfortably above noise and below any real bug."""
    kv = _run(probe_exe, probe_mesh, np_procs, mpi_cmd)
    nq = int(kv["NUM_GLOBAL_QUAD"])
    assert abs(float(kv["SUM_N_X"])) < 1e-4, (
        f"SUM_N_X = {kv['SUM_N_X']} — tangential residue exceeds mesh-"
        "rounding noise floor (1e-4).")
    assert abs(float(kv["SUM_N_Y"])) < 1e-4, (
        f"SUM_N_Y = {kv['SUM_N_Y']} — tangential residue exceeds mesh-"
        "rounding noise floor (1e-4).")
    assert abs(float(kv["SUM_N_Z"]) - nq) < 1e-9
    assert float(kv["MAX_UNIT_DEV"]) < 1e-12, (
        "normal length deviates from 1 — CalcOrtho normalization issue?"
    )


@pytest.mark.parametrize("np_procs", [1, 2])
def test_jacobianw_sum_equals_face_area(probe_exe: Path, probe_mesh: Path,
                                        mpi_cmd: str, np_procs: int) -> None:
    """Σ w_i × |J_i| over every interface quadrature point must equal the
    physical area of Γ_fs. For the probe mesh the interface lives at
    z=Lz/2 and spans (Lx=600) × (Ly=300) = 180_000 m². Phase 3 integrators
    rely on this being the correct quadrature weight, not the Phase 2A
    placeholder 1.0."""
    kv = _run(probe_exe, probe_mesh, np_procs, mpi_cmd)
    expected_area = 600.0 * 300.0
    jw_sum = float(kv["JACOBIAN_W_SUM"])
    rel_err = abs(jw_sum - expected_area) / expected_area
    # ~1e-7 relative is float32 round-off territory for 450 accumulated
    # contributions; allow a comfortable margin.
    assert rel_err < 5e-5, (
        f"Σ w*|J| = {jw_sum} (expected {expected_area}, "
        f"rel err = {rel_err:.2e})"
    )


@pytest.fixture(scope="module")
def tall_mesh(tmp_path_factory, gmsh_available, request) -> Path:
    """Mesh shaped so METIS at np=2 lands on an interface-aligned cut
    (all fluid on rank 0, all solid on rank 1). Used to force the Phase
    2B-i remote-face classification path."""
    keep = bool(request.config.getoption("--keep-results"))
    work = tmp_path_factory.mktemp("coupling_tall_probe")
    mesh_file = work / "tall.msh"
    subprocess.run(
        ["python3", str(GENERATE_SCRIPT),
         "--output", str(mesh_file),
         "--lx", "100", "--ly", "100", "--lz", "1000",
         "--nx", "2", "--ny", "2", "--nz-half", "4"],
        check=True, capture_output=True, text=True,
    )
    yield mesh_file
    if keep:
        print(f"\n[--keep-results] tall-mesh probe workspace retained: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def test_phase2b_i_remote_face_classification(probe_exe: Path,
                                              tall_mesh: Path,
                                              mpi_cmd: str) -> None:
    """Phase 2B-i: on a mesh where METIS at np=2 places all fluid elements
    on one rank and all solid elements on the other, Setup() must detect
    interface faces that span a rank boundary and count them as remote
    (fluid-owned on one rank, solid-owned on the other). The total
    number of local quadrature points should be ZERO in this case —
    every interface face is MPI-spanning.

    Expected counts for this 2×2×{4+4} hex probe at order=2:
      - interface faces = 2×2 = 4
      - quads per face = (order+1)^2 = 9
      - NUM_REMOTE_FLUID_OWNED_QUAD = 4 × 9 = 36 (one perspective)
      - NUM_REMOTE_SOLID_OWNED_QUAD = 4 × 9 = 36 (the other perspective)
      - NUM_LOCAL_QUAD = 0 (nothing is both-sides-local)
    """
    kv = _run(probe_exe, tall_mesh, np_procs=2, mpi_cmd=mpi_cmd, order=2)
    assert int(kv["NUM_LOCAL_QUAD"]) == 0, (
        "tall mesh at np=2 should trigger 100% MPI-spanning interface; "
        "got some local quads — METIS choice drifted?"
    )
    expected_remote = 2 * 2 * (2 + 1) ** 2  # 4 faces × 9 GLL nodes
    assert int(kv["NUM_REMOTE_FLUID_OWNED_QUAD"]) == expected_remote
    assert int(kv["NUM_REMOTE_SOLID_OWNED_QUAD"]) == expected_remote


@pytest.mark.parametrize("np_procs", [1, 2])
def test_extract_u_normal_bit_exact(probe_exe: Path, probe_mesh: Path,
                                    mpi_cmd: str, np_procs: int) -> None:
    """ExtractSolidDisplacementNormal must satisfy un[i] = z_quad[i] * n_z[i]
    for the test field u_s = (0, 0, z), bit-exactly. A non-zero error here
    means the cached solid DOF indices don't correspond to the cached
    quadrature positions / normals — Phase 3 coupling would be wrong."""
    kv = _run(probe_exe, probe_mesh, np_procs, mpi_cmd)
    assert float(kv["EXTRACT_UN_MAX_ERR"]) == 0.0, (
        "u·n extract disagrees with analytical value "
        f"(max err = {kv['EXTRACT_UN_MAX_ERR']})"
    )
    assert float(kv["EXTRACT_UN_MAX_ABS"]) > 0.0, (
        "extracted u·n is identically zero — the analytical field failed "
        "to project onto the solid submesh or the DOF lookup is empty"
    )
