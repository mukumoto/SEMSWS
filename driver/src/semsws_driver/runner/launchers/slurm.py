"""SLURM launcher: srun --exact for source-parallel within an allocation."""

from __future__ import annotations

from dataclasses import dataclass

from ..binding import BindingPolicy
from ..resources import Slot
from .base import LaunchCommand, env_to_export_list, merge_env


@dataclass
class SlurmLauncher:
    name: str = "slurm"

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
            "srun",
            "--exact",
            "--unbuffered",
            f"--nodes={len(slot.nodes)}",
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpus_per_rank}",
            f"--nodelist={','.join(slot.nodes)}",
        ]
        # User-supplied launcher args
        argv += policy.args_for("slurm")

        env = merge_env(slot, policy)
        if env:
            export_tokens = env_to_export_list(env)
            argv.append("--export=ALL," + ",".join(export_tokens))

        # Rank wrapper before the binary
        if policy.rank_wrapper:
            argv += list(policy.rank_wrapper)
        argv += [binary, "-config", config_path]

        return LaunchCommand(argv=argv, env=env, scheduler="slurm")
