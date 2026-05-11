"""Launcher command-construction snapshot tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from semsws_driver.runner.binding import BindingPolicy
from semsws_driver.runner.launchers import (
    LocalLauncher,
    MpiexecLauncher,
    MpirunLauncher,
    PjmLauncher,
    SlurmLauncher,
    launcher_for,
)
from semsws_driver.runner.resources import Slot


@pytest.fixture
def slot():
    return Slot(
        index=0, nodes=["n01"],
        cpu_ids=[0, 1, 2, 3], gpu_ids=[0, 1],
        env_overrides={"ROCR_VISIBLE_DEVICES": "0,1"},
    )


@pytest.fixture
def policy():
    return BindingPolicy(
        extra_launcher_args={
            "slurm": ["--cpu-bind=cores", "--mem-bind=local"],
            "mpirun": [],
            "mpiexec": [],
            "pjm": [],
            "local": [],
        },
        rank_wrapper=["numactl", "--localalloc"],
        extra_env={"OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "close"},
    )


def test_slurm_launcher(slot, policy):
    cmd = SlurmLauncher().build(
        slot=slot, policy=policy, binary="./semsws",
        config_path="/tmp/shot.yaml", cpus_per_rank=1,
    )
    argv = " ".join(cmd.argv)
    assert "srun --exact" in argv
    assert "--ntasks=4" in argv
    assert "--nodelist=n01" in argv
    assert "--cpu-bind=cores" in argv
    assert "--mem-bind=local" in argv
    assert "numactl --localalloc" in argv
    assert "./semsws -config /tmp/shot.yaml" in argv
    assert any(a.startswith("--export=ALL,") for a in cmd.argv)
    assert "ROCR_VISIBLE_DEVICES=0,1" in argv


def test_mpirun_launcher_writes_rankfile(slot, policy, tmp_path):
    cfg = tmp_path / "shot.yaml"
    cfg.write_text("")
    cmd = MpirunLauncher().build(
        slot=slot, policy=policy, binary="./semsws",
        config_path=str(cfg), cpus_per_rank=1,
    )
    argv = " ".join(cmd.argv)
    rankfile = tmp_path / "rankfile"
    assert rankfile.exists()
    body = rankfile.read_text().splitlines()
    assert body == [
        "rank 0=n01 slot=0",
        "rank 1=n01 slot=1",
        "rank 2=n01 slot=2",
        "rank 3=n01 slot=3",
    ]
    assert argv.startswith(
        f"mpirun -n 4 --use-hwthread-cpus --rankfile {rankfile}")
    assert "-x OMP_NUM_THREADS=1" in argv
    assert "-x ROCR_VISIBLE_DEVICES=0,1" in argv
    assert f"numactl --localalloc ./semsws -config {cfg}" in argv


def test_local_launcher_writes_rankfile(slot, policy, tmp_path):
    cfg = tmp_path / "shot.yaml"
    cfg.write_text("")
    cmd = LocalLauncher().build(
        slot=slot, policy=policy, binary="./semsws",
        config_path=str(cfg), cpus_per_rank=1,
    )
    argv = " ".join(cmd.argv)
    rankfile = tmp_path / "rankfile"
    assert rankfile.exists()
    assert argv.startswith(
        f"mpirun -n 4 --use-hwthread-cpus --rankfile {rankfile}")
    # No --cpu-set, no --bind-to in our output.
    assert "--cpu-set" not in argv
    assert "--bind-to" not in argv


def test_mpiexec_launcher_intel_pin_env(slot, policy):
    cmd = MpiexecLauncher().build(
        slot=slot, policy=policy, binary="./semsws",
        config_path="/tmp/shot.yaml", cpus_per_rank=1,
    )
    argv = " ".join(cmd.argv)
    assert argv.startswith("mpiexec -n 4 -host n01")
    assert "-genv I_MPI_PIN_PROCESSOR_LIST 0,1,2,3" in argv
    assert "-genv OMP_NUM_THREADS 1" in argv


def test_pjm_launcher_writes_vcoordfile(slot, policy, tmp_path):
    cfg = tmp_path / "shot.yaml"
    cfg.write_text("")
    cmd = PjmLauncher().build(
        slot=slot, policy=policy, binary="./semsws",
        config_path=str(cfg), cpus_per_rank=1,
    )
    argv = " ".join(cmd.argv)
    vcoord = tmp_path / "vcoord.txt"
    assert vcoord.exists()
    body = vcoord.read_text().splitlines()
    assert body == [
        "(0,0)", "(0,1)", "(0,2)", "(0,3)",
    ]
    assert argv.startswith(f"mpiexec -n 4 --vcoordfile {vcoord}")
    assert f"./semsws -config {cfg}" in argv


def test_passthrough_policy_slurm(slot):
    policy = BindingPolicy()
    cmd = SlurmLauncher().build(
        slot=slot, policy=policy, binary="./semsws",
        config_path="/tmp/shot.yaml", cpus_per_rank=1,
    )
    argv = " ".join(cmd.argv)
    assert "--cpu-bind" not in argv
    assert "numactl" not in argv
    assert "ROCR_VISIBLE_DEVICES=0,1" in argv


def test_launcher_for_dispatch():
    assert isinstance(launcher_for("slurm"), SlurmLauncher)
    assert isinstance(launcher_for("pjm"), PjmLauncher)
    assert isinstance(launcher_for("local"), LocalLauncher)
    assert isinstance(launcher_for("pbs"), MpirunLauncher)
    assert isinstance(launcher_for("lsf"), MpirunLauncher)


def test_per_launcher_args_isolation(slot, tmp_path):
    """slurm-only flag should not leak into mpirun command."""
    cfg = tmp_path / "shot.yaml"
    cfg.write_text("")
    policy = BindingPolicy(extra_launcher_args={"slurm": ["--mem-bind=local"]})
    slurm_cmd = SlurmLauncher().build(slot=slot, policy=policy,
                                        binary="./s", config_path=str(cfg),
                                        cpus_per_rank=1)
    mpi_cmd = MpirunLauncher().build(slot=slot, policy=policy,
                                       binary="./s", config_path=str(cfg),
                                       cpus_per_rank=1)
    assert "--mem-bind=local" in " ".join(slurm_cmd.argv)
    assert "--mem-bind=local" not in " ".join(mpi_cmd.argv)


def test_multi_node_rankfile(tmp_path):
    """OpenMPI rankfile for a 2-node × 4-cpu/node slot (256-rank style).

    Demonstrates the multi-node generator output."""
    slot = Slot(
        index=0, nodes=["n1", "n2"],
        cpu_ids=[0, 1, 2, 3], gpu_ids=[],
    )
    cfg = tmp_path / "shot.yaml"
    cfg.write_text("")
    cmd = MpirunLauncher().build(
        slot=slot, policy=BindingPolicy(), binary="./s",
        config_path=str(cfg), cpus_per_rank=1,
    )
    body = (tmp_path / "rankfile").read_text().splitlines()
    assert body == [
        "rank 0=n1 slot=0",
        "rank 1=n1 slot=1",
        "rank 2=n1 slot=2",
        "rank 3=n1 slot=3",
        "rank 4=n2 slot=0",
        "rank 5=n2 slot=1",
        "rank 6=n2 slot=2",
        "rank 7=n2 slot=3",
    ]
    assert " ".join(cmd.argv).startswith(
        "mpirun -n 8 --use-hwthread-cpus --rankfile ")


def test_multi_node_vcoordfile(tmp_path):
    slot = Slot(
        index=0, nodes=["n1", "n2", "n3"],
        cpu_ids=[0, 1, 2, 3], gpu_ids=[],
    )
    cfg = tmp_path / "shot.yaml"
    cfg.write_text("")
    PjmLauncher().build(
        slot=slot, policy=BindingPolicy(), binary="./s",
        config_path=str(cfg), cpus_per_rank=1,
    )
    body = (tmp_path / "vcoord.txt").read_text().splitlines()
    assert body == [
        "(0,0)", "(0,1)", "(0,2)", "(0,3)",
        "(1,0)", "(1,1)", "(1,2)", "(1,3)",
        "(2,0)", "(2,1)", "(2,2)", "(2,3)",
    ]


def test_cpus_per_rank_4_writes_range(tmp_path):
    """cpus_per_rank=4 → each rank gets a 4-cpu range in rankfile."""
    slot = Slot(
        index=0, nodes=["n1"],
        cpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], gpu_ids=[],
    )
    cfg = tmp_path / "shot.yaml"
    cfg.write_text("")
    MpirunLauncher().build(
        slot=slot, policy=BindingPolicy(), binary="./s",
        config_path=str(cfg), cpus_per_rank=4,
    )
    body = (tmp_path / "rankfile").read_text().splitlines()
    assert body == [
        "rank 0=n1 slot=0-3",
        "rank 1=n1 slot=4-7",
    ]
