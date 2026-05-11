"""Launcher Protocol and shared helpers. A Launcher maps a (Slot, BindingPolicy,
binary, config) tuple to the argv and env needed to spawn one shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..binding import BindingPolicy
from ..resources import Slot


@dataclass(frozen=True)
class LaunchCommand:
    argv: list[str]
    env: dict[str, str]
    scheduler: str

    def pretty(self) -> str:
        env_str = " ".join(f"{k}={v}" for k, v in sorted(self.env.items()))
        if env_str:
            env_str = env_str + " \\\n   "
        return env_str + " ".join(self.argv)


class Launcher(Protocol):
    name: str

    def build(
        self,
        *,
        slot: Slot,
        policy: BindingPolicy,
        binary: str,
        config_path: str,
        cpus_per_rank: int,
    ) -> LaunchCommand: ...


def merge_env(slot: Slot, policy: BindingPolicy) -> dict[str, str]:
    """Merge slot env_overrides (driver-set GPU isolation) with user extra_env.

    User extra_env takes precedence on conflict, except for *_VISIBLE_DEVICES
    which the driver always wins (safety: prevent GPU collisions).
    """
    env = dict(policy.extra_env)
    for k, v in slot.env_overrides.items():
        env[k] = v  # driver wins
    return env


def env_to_export_list(env: dict[str, str]) -> list[str]:
    """Render env as 'K=V' tokens, sorted for stable test snapshots."""
    return [f"{k}={v}" for k, v in sorted(env.items())]


def write_openmpi_rankfile(
    rankfile: Path, slot: Slot, cpus_per_rank: int,
) -> int:
    """Write an Open MPI rankfile for the slot. Returns total ntasks.

    Format (per Open MPI scheduling docs):
        rank <N>=<host> slot=<P>
        rank <N>=<host> slot=<P>-<Q>    # range for cpus_per_rank > 1

    The `slot=<P>` value is the OS-assigned logical CPU id (same as in
    /proc/cpuinfo). Multi-node slots produce one rank-line per (node,
    cpu) pair; ranks are numbered consecutively node-by-node.
    """
    rankfile.parent.mkdir(parents=True, exist_ok=True)
    K = max(1, cpus_per_rank)
    r = 0
    with rankfile.open("w") as f:
        for node in slot.nodes:
            for i in range(0, len(slot.cpu_ids), K):
                cpus = slot.cpu_ids[i:i + K]
                if not cpus:
                    continue
                if len(cpus) == 1:
                    spec = str(cpus[0])
                elif cpus == list(range(cpus[0], cpus[-1] + 1)):
                    spec = f"{cpus[0]}-{cpus[-1]}"
                else:
                    spec = ",".join(str(c) for c in cpus)
                f.write(f"rank {r}={node} slot={spec}\n")
                r += 1
    return r


def write_fujitsu_vcoordfile(
    vcoordfile: Path, slot: Slot, cpus_per_rank: int,
) -> int:
    """Write a Fujitsu MPI vcoordfile for the slot. Returns total ntasks.

    Format per the Fujitsu MPI User's Guide: one line per rank,
        (<node_index>,<core_index>)
    where node_index is 0-based within the allocation and core_index is
    the rank's preferred core on that node. Note: the exact format
    expected by Fugaku/ES4 Fujitsu MPI should be verified against
    `mpiexec --help` on the target system.
    """
    vcoordfile.parent.mkdir(parents=True, exist_ok=True)
    K = max(1, cpus_per_rank)
    r = 0
    with vcoordfile.open("w") as f:
        for node_idx, _ in enumerate(slot.nodes):
            for i in range(0, len(slot.cpu_ids), K):
                cpus = slot.cpu_ids[i:i + K]
                if not cpus:
                    continue
                # We pin to the first cpu of the rank's cpu group; multi-cpu
                # ranks rely on OMP_NUM_THREADS / fork from that anchor.
                f.write(f"({node_idx},{cpus[0]})\n")
                r += 1
    return r
