"""Shadow recipe that adds MFEM 4.9 to the catalog.

Spack v1.1.1's builtin mfem package only ships up to 4.8.0. SEMSWS depends on
two MFEM features that landed in 4.9:

  * mfem.hpp pulls general/forall.hpp transitively, so mfem::Reshape is
    visible to consumers via #include <mfem.hpp>. (4.8 dropped this.)
  * mfem::future::Dual<> autodiff types in linalg/dual.hpp.

This shadow inherits the builtin Mfem package and only adds the 4.9 version
entry; all variants, dependencies, and build logic remain the upstream ones.
Drop this file once spack/spack ships mfem 4.9 in builtin.
"""

from spack.package import *  # noqa: F401, F403  (provides version, etc.)
from spack_repo.builtin.packages.mfem.package import Mfem as BuiltinMfem


class Mfem(BuiltinMfem):
    # MFEM 4.9 (release tag v4.9 on github.com/mfem/mfem).
    version(
        "4.9.0",
        sha256="ea3ac13e182c09f05b414b03a9bef7a4da99d45d67ee409112b8f11058447a7c",
        url="https://github.com/mfem/mfem/archive/refs/tags/v4.9.tar.gz",
    )
