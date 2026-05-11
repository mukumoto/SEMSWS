"""MpiDirectRunner: HPC-aware source-parallel runner that dispatches shots
concurrently within an existing SLURM/PJM/PBS/LSF allocation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from .binding import BindingPolicy
from .base import Runner, ShotResult, ShotTask
from .local import _exec_shot
from .resources import Allocation, ResourcePool, detect_allocation

log = logging.getLogger(__name__)


@dataclass
class MpiDirectRunner:
    semsws_binary: str
    ranks_per_shot: int
    cpus_per_rank: int = 1
    gpus_per_shot: int = 0
    max_concurrent_shots: int = 0   # 0 = use all carved slots
    device_kind: str = "cpu"        # "cpu" | "cuda" | "hip"
    numa_aware: bool = False
    allocation_override: Optional[Allocation] = None
    # Scheduler override (e.g. "slurm" if auto-detect is fragile in your env).
    force_scheduler: Optional[str] = None

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
        alloc = self.allocation_override or detect_allocation(
            force=self.force_scheduler  # type: ignore[arg-type]
        )
        pool = ResourcePool(
            allocation=alloc,
            ranks_per_shot=self.ranks_per_shot,
            gpus_per_shot=self.gpus_per_shot,
            cpus_per_rank=self.cpus_per_rank,
            device_kind=self.device_kind,  # type: ignore[arg-type]
            numa_aware=self.numa_aware,
        )
        max_concurrent = (
            pool.n_slots
            if self.max_concurrent_shots <= 0
            else min(self.max_concurrent_shots, pool.n_slots)
        )
        gate = asyncio.Semaphore(max_concurrent)

        log.info(
            "MpiDirectRunner: scheduler=%s n_slots=%d max_concurrent=%d "
            "ranks/shot=%d gpus/shot=%d",
            alloc.scheduler, pool.n_slots, max_concurrent,
            self.ranks_per_shot, self.gpus_per_shot,
        )

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
                        scheduler=alloc.scheduler,
                    )
                finally:
                    await pool.release(slot)

        results = await asyncio.gather(*(_run_one(t) for t in tasks))
        return list(results)
