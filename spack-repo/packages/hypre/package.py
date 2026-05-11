"""Shadow recipe that disables cuSOLVER in hypre when building with CUDA.

cuSOLVER's ``cusolverDnDpotrfBatched`` is double-precision-only, so
``par_fsai_device.c`` fails to build under ``precision=single + cuda``
(hypre-space/hypre#557). cuSOLVER is not on any SEMSWS code path, so
disabling it is a no-op for us while eliminating the FSAI failure.

Both build systems are overridden:
  * AutotoolsBuilder  (hypre 2.x):  ``--disable-cusolver``
  * CMakeBuilder      (hypre 3.x):  ``-DHYPRE_ENABLE_CUSOLVER=OFF``

Drop this file once spack/spack exposes a ``cusolver`` variant on hypre.
"""

from spack.package import *  # noqa: F401, F403
from spack_repo.builtin.packages.hypre.package import (
    AutotoolsBuilder as BuiltinAutotoolsBuilder,
)
from spack_repo.builtin.packages.hypre.package import (
    CMakeBuilder as BuiltinCMakeBuilder,
)
from spack_repo.builtin.packages.hypre.package import Hypre as BuiltinHypre


class Hypre(BuiltinHypre):
    pass


class AutotoolsBuilder(BuiltinAutotoolsBuilder):
    def configure_args(self):
        args = super().configure_args()
        if self.pkg.spec.satisfies("+cuda"):
            args = [a for a in args if a != "--enable-cusolver"]
            if "--disable-cusolver" not in args:
                args.append("--disable-cusolver")
        return args


class CMakeBuilder(BuiltinCMakeBuilder):
    def cmake_args(self):
        args = super().cmake_args()
        if self.pkg.spec.satisfies("+cuda"):
            # Strip any prior HYPRE_ENABLE_CUSOLVER definition the parent
            # may have set, then force it OFF.
            args = [a for a in args if "HYPRE_ENABLE_CUSOLVER" not in a]
            args.append(self.define("HYPRE_ENABLE_CUSOLVER", False))
        return args
