"""LocalRunner: asyncio subprocess driver for laptop / single node, dispatching
concurrent shots via the LocalLauncher (mpirun by default).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .binding import BindingPolicy
from .base import Runner, ShotResult, ShotTask
from .launchers import launcher_for
from .resources import Allocation, ResourcePool, Slot, detect_allocation

log = logging.getLogger(__name__)


@dataclass
class LocalRunner:
    """Run shots locally (no scheduler)."""
    semsws_binary: str
    ranks_per_shot: int = 1
    cpus_per_rank: int = 1
    max_concurrent_shots: int = 1
    gpus_per_shot: int = 0
    device_kind: str = "cpu"
    numa_aware: bool = False
    allocation_override: Optional[Allocation] = None

    def submit(
        self,
        tasks: Sequence[ShotTask],
        *,
        binding: Optional[BindingPolicy] = None,
    ) -> list[ShotResult]:
        return asyncio.run(self._submit_async(tasks, binding or BindingPolicy()))

    async def _submit_async(
        self, tasks: Sequence[ShotTask], binding: BindingPolicy,
    ) -> list[ShotResult]:
        alloc = self.allocation_override or detect_allocation(force="local")
        # Cap concurrency by what fits in the allocation.
        pool = ResourcePool(
            allocation=alloc,
            ranks_per_shot=self.ranks_per_shot,
            gpus_per_shot=self.gpus_per_shot,
            cpus_per_rank=self.cpus_per_rank,
            device_kind=self.device_kind,  # type: ignore[arg-type]
            numa_aware=self.numa_aware,
        )
        gate = asyncio.Semaphore(min(self.max_concurrent_shots, pool.n_slots))

        async def _run_one(task: ShotTask) -> ShotResult:
            async with gate:
                slot = await pool.acquire()
                try:
                    return await _exec_shot(
                        task=task,
                        slot=slot,
                        allocation=alloc,
                        binding=binding,
                        binary=self.semsws_binary,
                        cpus_per_rank=self.cpus_per_rank,
                        scheduler="local",
                    )
                finally:
                    await pool.release(slot)

        results = await asyncio.gather(*(_run_one(t) for t in tasks))
        return list(results)


async def _exec_shot(
    *,
    task: ShotTask,
    slot: Slot,
    allocation: Allocation,
    binding: BindingPolicy,
    binary: str,
    cpus_per_rank: int,
    scheduler: str,
) -> ShotResult:
    """Build the launcher command for this slot, run it, capture logs."""
    launcher = launcher_for(scheduler)
    cmd = launcher.build(
        slot=slot,
        policy=binding,
        binary=binary,
        config_path=str(task.config_path),
        cpus_per_rank=cpus_per_rank,
    )

    task.workdir.mkdir(parents=True, exist_ok=True)
    stdout_path = task.workdir / f"{task.shot_id}.stdout.log"
    stderr_path = task.workdir / f"{task.shot_id}.stderr.log"
    env = os.environ.copy()
    env.update(cmd.env)
    env.update(task.extra_env)

    log.info("[%s] launching: %s", task.shot_id, " ".join(cmd.argv))
    start_dt = datetime.now(timezone.utc)
    t0 = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = await asyncio.create_subprocess_exec(
            *cmd.argv,
            cwd=str(task.workdir),
            stdout=out,
            stderr=err,
            env=env,
        )
        rc = await proc.wait()
    elapsed = time.monotonic() - t0
    finish_dt = datetime.now(timezone.utc)
    log.info("[%s] finished rc=%d in %.2fs (started %s)",
             task.shot_id, rc, elapsed, start_dt.isoformat(timespec="milliseconds"))

    return ShotResult(
        shot_id=task.shot_id,
        workdir=task.workdir,
        return_code=rc,
        elapsed_seconds=elapsed,
        start_utc=start_dt.isoformat(timespec="milliseconds"),
        finish_utc=finish_dt.isoformat(timespec="milliseconds"),
        stdout_log=stdout_path,
        stderr_log=stderr_path,
        config_path=task.config_path,
        extra={"slot": slot, "command": cmd.argv, "scheduler": scheduler},
    )
