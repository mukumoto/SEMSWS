"""Local launcher: Open MPI mpirun for laptop / WSL2 / no scheduler.

Identical mechanism to `MpirunLauncher`: per-shot rankfile pinning so
concurrent shots on the same node never collide on cores.

`--cpu-set` and `--bind-to <X>` are intentionally NOT used because of
OpenMPI issue #12884 (concurrent mpirun invocations with `--cpu-set`
overlap each other's CPUs). Rankfile makes each rank's CPU explicit so
no auto-binding happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..binding import BindingPolicy
from ..resources import Slot
from .base import LaunchCommand, merge_env, write_openmpi_rankfile


@dataclass
class LocalLauncher:
    name: str = "local"

    def build(
        self,
        *,
        slot: Slot,
        policy: BindingPolicy,
        binary: str,
        config_path: str,
        cpus_per_rank: int,
    ) -> LaunchCommand:
        rankfile = Path(config_path).parent / "rankfile"
        ntasks = write_openmpi_rankfile(rankfile, slot, cpus_per_rank)
        argv: list[str] = [
            "mpirun",
            "-n", str(ntasks),
            # --use-hwthread-cpus makes mpirun interpret slot numbers in the
            # rankfile as logical CPU ids (matching /proc/cpuinfo and
            # slot.cpu_ids), not physical cores.
            "--use-hwthread-cpus",
            "--rankfile", str(rankfile),
        ]
        argv += policy.args_for("local")

        env = merge_env(slot, policy)
        for k, v in sorted(env.items()):
            argv += ["-x", f"{k}={v}"]

        if policy.rank_wrapper:
            argv += list(policy.rank_wrapper)
        argv += [binary, "-config", config_path]

        return LaunchCommand(argv=argv, env=env, scheduler="local")
