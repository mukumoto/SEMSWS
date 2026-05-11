"""
Pytest configuration for SEMSWS waveform validation tests.

Provides fixtures and automatic test discovery for waveform comparison tests.
"""

import pytest
import shutil
import subprocess
from pathlib import Path
from typing import Generator

# Add scripts to path
import sys
SCRIPT_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--build-dir",
        action="store",
        default=str(Path(__file__).parent.parent / "build"),
        help="Build directory containing executable"
    )
    parser.addoption(
        "--np",
        action="store",
        default="1",
        help="Number of MPI processes"
    )
    parser.addoption(
        "--threshold",
        action="store",
        default="0.05",
        help="L2 error threshold"
    )
    parser.addoption(
        "--keep-results",
        action="store_true",
        default=False,
        help="Keep simulation results after test"
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device type for simulation (cpu, cuda, hip)"
    )
    parser.addoption(
        "--mpi-cmd",
        action="store",
        default="mpirun",
        help="MPI launcher command (default: mpirun, use 'srun' on Cray/SLURM systems)"
    )


@pytest.fixture(scope="session")
def build_dir(request) -> Path:
    """Build directory path."""
    return Path(request.config.getoption("--build-dir"))


@pytest.fixture(scope="session")
def executable(build_dir: Path) -> Path:
    """Path to the SEMSWS executable."""
    candidates = [
        build_dir / "src" / "semsws",
        build_dir / "semsws",
        build_dir / "bin" / "semsws",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pytest.skip(
        "semsws binary not found under {b}. Looked in: {c}. "
        "To enable: cmake --build {b} --target semsws (or pass "
        "--build-dir <correct-path>).".format(
            b=build_dir,
            c=", ".join(str(c) for c in candidates),
        )
    )


@pytest.fixture(scope="session")
def np_procs(request) -> int:
    """Number of MPI processes."""
    return int(request.config.getoption("--np"))


@pytest.fixture(scope="session")
def threshold(request) -> float:
    """L2 error threshold."""
    return float(request.config.getoption("--threshold"))


@pytest.fixture(scope="session")
def keep_results(request) -> bool:
    """Whether to keep simulation results."""
    return request.config.getoption("--keep-results")


@pytest.fixture(scope="session")
def device_type(request) -> str:
    """Device type for simulation."""
    return request.config.getoption("--device")


@pytest.fixture(scope="session")
def mpi_cmd(request) -> str:
    """MPI launcher command."""
    return request.config.getoption("--mpi-cmd")


def discover_waveform_tests() -> list[tuple[str, Path, Path]]:
    """
    Discover all waveform test cases.

    Returns:
        List of (test_id, config_path, ref_dir) tuples
    """
    test_root = Path(__file__).parent
    waveform_dir = test_root / "waveform"
    tests = []

    if not waveform_dir.exists():
        return tests

    for dim_dir in sorted(waveform_dir.iterdir()):
        if not dim_dir.is_dir():
            continue

        for material_dir in sorted(dim_dir.iterdir()):
            if not material_dir.is_dir():
                continue

            config_path = material_dir / "config.yaml"
            ref_dir = material_dir / "ref"

            if config_path.exists() and ref_dir.exists():
                test_id = f"{dim_dir.name}/{material_dir.name}"
                tests.append((test_id, config_path, ref_dir))

    return tests


def pytest_generate_tests(metafunc):
    """Parametrize waveform tests automatically."""
    if "waveform_test_case" in metafunc.fixturenames:
        tests = discover_waveform_tests()
        if tests:
            ids = [t[0] for t in tests]
            metafunc.parametrize("waveform_test_case", tests, ids=ids)


@pytest.fixture
def run_simulation(executable: Path, np_procs: int, device_type: str, mpi_cmd: str):
    """Factory fixture to run simulations."""
    # Resolve to an absolute path once; subprocess runs cwd=config_path.parent
    # so relative executable paths (e.g. ./build/src/...) would otherwise fail.
    exe = executable.resolve()

    def _run(config_path: Path, timeout: int = 600) -> tuple[bool, str]:
        # Get device-specific config if needed
        if device_type != "cpu":
            config_device = config_path.parent / f"config.{device_type}.yaml"
            if config_device.exists():
                config_path = config_device

        if mpi_cmd == "srun":
            cmd = ["srun", "-n", str(np_procs), str(exe), "--config", str(config_path)]
        else:
            cmd = [mpi_cmd, "-np", str(np_procs), str(exe), "--config", str(config_path)]

        try:
            result = subprocess.run(
                cmd,
                cwd=config_path.parent,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                return False, f"Simulation failed:\n{result.stderr}\n{result.stdout}"

            return True, ""

        except subprocess.TimeoutExpired:
            return False, f"Simulation timed out ({timeout}s)"
        except Exception as e:
            return False, f"Failed to run simulation: {e}"

    return _run
