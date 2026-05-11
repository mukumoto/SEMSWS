"""Intel MPI `mpiexec` launcher.

Inter-shot CPU isolation: per-slot `I_MPI_PIN_PROCESSOR_LIST` environment
variable carrying the slot's logical CPU ids. Intel MPI pins each rank
to one CPU in that list (round-robin).

Notes:
  * `MpiexecLauncher` here targets Intel MPI specifically. For MPICH the
    pinning syntax differs (`-bind-to user:...`) and is NOT auto-emitted;
    users on MPICH should write the equivalent into
    `run.binding.extra_launcher_args.mpiexec`.
  * For Fujitsu MPI on Fugaku/ES4, see `PjmLauncher` which uses
    `--vcoordfile` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..binding import BindingPolicy
from ..resources import Slot
from .base import LaunchCommand, merge_env


@dataclass
class MpiexecLauncher:
    name: str = "mpiexec"

    def build(
        self,
        *,
        slot: Slot,
        policy: BindingPolicy,
        binary: str,
        config_path: str,
        cpus_per_rank: int,
    ) -> LaunchCommand:
        ranks_per_node = len(slot.cpu_ids) // max(1, cpus_per_rank)
        ntasks = ranks_per_node * len(slot.nodes)
        argv: list[str] = [
            "mpiexec",
            "-n", str(ntasks),
            "-host", ",".join(slot.nodes),
        ]
        argv += policy.args_for("mpiexec")

        env = merge_env(slot, policy)
        # Intel MPI pinning: drive each rank to one CPU on its node.
        env.setdefault(
            "I_MPI_PIN_PROCESSOR_LIST",
            ",".join(str(c) for c in slot.cpu_ids),
        )
        for k, v in sorted(env.items()):
            argv += ["-genv", k, v]

        if policy.rank_wrapper:
            argv += list(policy.rank_wrapper)
        argv += [binary, "-config", config_path]

        return LaunchCommand(argv=argv, env=env, scheduler="mpiexec")
