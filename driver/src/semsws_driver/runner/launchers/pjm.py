"""Fujitsu MPI mpiexec launcher (PJM scheduler: Fugaku / ES4).

Inter-shot pinning is done with per-shot `vcoordfile` (Fujitsu MPI's
native per-rank (node, core) mapping file). The driver writes the file
next to the per-shot YAML and passes `--vcoordfile <path>` to mpiexec.

The exact format expected by Fugaku/ES4 Fujitsu MPI should be verified
against the target system's `mpiexec --help`. The generator here writes
the form documented in the Fujitsu MPI User's Guide:
    (<node_index>,<core_index>)
one line per rank, node_index 0-based within the allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..binding import BindingPolicy
from ..resources import Slot
from .base import LaunchCommand, merge_env, write_fujitsu_vcoordfile


@dataclass
class PjmLauncher:
    name: str = "pjm"

    def build(
        self,
        *,
        slot: Slot,
        policy: BindingPolicy,
        binary: str,
        config_path: str,
        cpus_per_rank: int,
    ) -> LaunchCommand:
        vcoord = Path(config_path).parent / "vcoord.txt"
        ntasks = write_fujitsu_vcoordfile(vcoord, slot, cpus_per_rank)
        argv: list[str] = [
            "mpiexec",
            "-n", str(ntasks),
            "--vcoordfile", str(vcoord),
        ]
        argv += policy.args_for("pjm")

        env = merge_env(slot, policy)
        for k, v in sorted(env.items()):
            argv += ["-x", f"{k}={v}"]

        if policy.rank_wrapper:
            argv += list(policy.rank_wrapper)
        argv += [binary, "-config", config_path]

        return LaunchCommand(argv=argv, env=env, scheduler="pjm")
