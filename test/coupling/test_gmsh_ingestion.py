"""Verify GMSH Physical Volume tags survive the SEMSWS/MFEM mesh pipeline as
ParMesh::GetAttribute() values, and that ParSubMesh::CreateFromDomain
correctly splits the parent mesh.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


TEST_DIR = Path(__file__).parent
GENERATE_SCRIPT = TEST_DIR / "generate_probe_mesh.py"


def _parse_kv(stdout: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
    return result


@pytest.fixture(scope="module")
def probe_exe(build_dir: Path) -> Path:
    candidates = [
        build_dir / "src" / "coupling_probe_gmsh",   # CMake build dir
        build_dir / "bin" / "coupling_probe_gmsh",   # Spack install prefix
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    pytest.skip(
        f"coupling_probe_gmsh not found in any of: "
        + ", ".join(str(c) for c in candidates)
    )


@pytest.fixture(scope="module")
def gmsh_available() -> None:
    try:
        import gmsh  # noqa: F401
    except ImportError:
        pytest.skip("gmsh python module not installed; skipping GMSH ingestion probe.")


@pytest.fixture(scope="module")
def probe_mesh(tmp_path_factory, gmsh_available) -> tuple[Path, dict[str, int]]:
    work = tmp_path_factory.mktemp("coupling_probe_mesh")
    mesh_file = work / "probe.msh"
    proc = subprocess.run(
        ["python3", str(GENERATE_SCRIPT), "--output", str(mesh_file)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"mesh generation failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert mesh_file.exists(), "mesh file not produced"
    kv = _parse_kv(proc.stdout)
    expected = {
        "fluid_ne": int(kv["EXPECTED_FLUID_NE"]),
        "solid_ne": int(kv["EXPECTED_SOLID_NE"]),
        "total_ne": int(kv["EXPECTED_TOTAL_NE"]),
    }
    yield mesh_file, expected
    shutil.rmtree(work, ignore_errors=True)


def _run_probe(exe: Path, mesh: Path, np_procs: int, mpi_cmd: str) -> dict[str, str]:
    if mpi_cmd == "srun":
        cmd = ["srun", "-n", str(np_procs), str(exe), "--mesh", str(mesh)]
    else:
        cmd = [mpi_cmd, "-np", str(np_procs), str(exe), "--mesh", str(mesh)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\ncmd: {' '.join(cmd)}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    kv = _parse_kv(proc.stdout)
    assert kv.get("PROBE_OK") == "1", f"probe did not report OK:\n{proc.stdout}"
    return kv


@pytest.mark.parametrize("np_procs", [1, 2])
def test_gmsh_physical_volume_to_mfem_attribute(
    probe_exe: Path, probe_mesh, mpi_cmd: str, np_procs: int
) -> None:
    mesh_file, expected = probe_mesh
    kv = _run_probe(probe_exe, mesh_file, np_procs, mpi_cmd)

    assert int(kv["PARENT_NE_PARALLEL"]) == expected["total_ne"], (
        "parent element count mismatch — parallelization changed NE?"
    )
    assert int(kv.get("PARENT_ATTR_1", 0)) == expected["fluid_ne"], (
        "GMSH Physical Volume 1 did not land in parent attribute 1"
    )
    assert int(kv.get("PARENT_ATTR_2", 0)) == expected["solid_ne"], (
        "GMSH Physical Volume 2 did not land in parent attribute 2"
    )


@pytest.mark.parametrize("np_procs", [1, 2])
def test_parsubmesh_splits_by_attribute(
    probe_exe: Path, probe_mesh, mpi_cmd: str, np_procs: int
) -> None:
    mesh_file, expected = probe_mesh
    kv = _run_probe(probe_exe, mesh_file, np_procs, mpi_cmd)

    assert int(kv["FLUID_SUB_NE"]) == expected["fluid_ne"]
    assert int(kv["SOLID_SUB_NE"]) == expected["solid_ne"]


@pytest.mark.parametrize("np_procs", [1, 2])
def test_interface_bdr_attr_auto_detected_via_set_diff(
    probe_exe: Path, probe_mesh, mpi_cmd: str, np_procs: int
) -> None:
    """The interface bdr attr is whichever attribute appears in the submeshes
    but NOT in the parent (set difference). This must be robust to parents
    that already carry arbitrary bdr attrs (Dirichlet/ABC/free-surface tags),
    i.e. we must NOT assume the interface attr is hardcoded or equals
    `max_parent_bdr + 1` derived in the test side. See Doc Test 1.3."""
    mesh_file, _expected = probe_mesh
    kv = _run_probe(probe_exe, mesh_file, np_procs, mpi_cmd)

    parent_bdrs = {int(x) for x in kv.get("PARENT_BDR_ATTRS", "").split(",") if x}
    fluid_bdrs = {int(x) for x in kv.get("FLUID_SUB_BDR_ATTRS", "").split(",") if x}
    solid_bdrs = {int(x) for x in kv.get("SOLID_SUB_BDR_ATTRS", "").split(",") if x}

    # Sanity: the test mesh is intentionally generated with multiple parent
    # bdr attrs (bottom/top/sides) so that auto detection must work on a
    # realistic, multi-attr parent rather than a trivial {1}.
    assert len(parent_bdrs) >= 2, (
        f"probe mesh degenerated to single-bdr parent; coverage lost. "
        f"parent_bdrs={sorted(parent_bdrs)}"
    )

    fluid_new = fluid_bdrs - parent_bdrs
    solid_new = solid_bdrs - parent_bdrs

    assert len(fluid_new) == 1, (
        f"expected exactly one auto-generated bdr attr on fluid submesh, "
        f"got {sorted(fluid_new)} (parent={sorted(parent_bdrs)}, "
        f"fluid={sorted(fluid_bdrs)})"
    )
    assert len(solid_new) == 1, (
        f"expected exactly one auto-generated bdr attr on solid submesh, "
        f"got {sorted(solid_new)} (parent={sorted(parent_bdrs)}, "
        f"solid={sorted(solid_bdrs)})"
    )
    assert fluid_new == solid_new, (
        f"fluid/solid submeshes disagree on interface bdr attr: "
        f"fluid_new={sorted(fluid_new)} solid_new={sorted(solid_new)}"
    )
