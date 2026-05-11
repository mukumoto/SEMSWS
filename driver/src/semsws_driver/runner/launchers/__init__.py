from .base import Launcher, LaunchCommand
from .slurm import SlurmLauncher
from .mpirun import MpirunLauncher
from .mpiexec import MpiexecLauncher
from .pjm import PjmLauncher
from .local import LocalLauncher


def launcher_for(scheduler: str) -> Launcher:
    return {
        "slurm": SlurmLauncher(),
        "pjm": PjmLauncher(),
        "pbs": MpirunLauncher(),     # PBS jobs typically launch via mpirun
        "lsf": MpirunLauncher(),     # likewise LSF
        "local": LocalLauncher(),
        "manual": MpirunLauncher(),
    }[scheduler]


__all__ = [
    "Launcher",
    "LaunchCommand",
    "SlurmLauncher",
    "MpirunLauncher",
    "MpiexecLauncher",
    "PjmLauncher",
    "LocalLauncher",
    "launcher_for",
]
