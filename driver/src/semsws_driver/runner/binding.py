"""Static binding policy passed to launchers (extra launcher args, rank
wrapper, and per-rank env), translated from `RunConfig.binding`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BindingPolicy:
    extra_launcher_args: dict[str, list[str]] = field(default_factory=dict)
    rank_wrapper: Optional[list[str]] = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def args_for(self, scheduler: str) -> list[str]:
        return list(self.extra_launcher_args.get(scheduler, []))

    def is_passthrough(self) -> bool:
        return (not self.extra_launcher_args
                and not self.rank_wrapper
                and not self.extra_env)
