from spack.package import *


class Semsws(CMakePackage, CudaPackage, ROCmPackage):
    """SEMSWS — Spectral Element Method Seismic Wave Solver.

    A finite-element wave-equation forward and inversion code built on
    MFEM, with adjoint-state FWI workflows. Acoustic and isotropic
    elastic media in 2D/3D, with optional viscoelastic Q via SLS.
    """

    homepage = "https://github.com/mukumoto/SEMSWS"
    git = "https://github.com/mukumoto/SEMSWS.git"

    license("BSD-3-Clause", checked_by="kota-mukumoto")

    maintainers("kota-mukumoto")

    version("main", branch="main")
    version("0.1.0", tag="v0.1.0")

    variant("precision",
            default="double",
            values=("single", "double"),
            description="Floating-point precision (must match MFEM and hypre)")
    variant("gpu_aware_mpi",
            default=False,
            description="Assert the linked MPI is GPU-aware (SEM_USE_GPU_AWARE_MPI). "
                        "Only enable on Cray MPICH with MPICH_GPU_SUPPORT_ENABLED=1, "
                        "CUDA-aware OpenMPI, or other verified GPU-aware MPI stacks. "
                        "Wrong ON crashes inside MPI_Isend at runtime.")
    variant("coupling_probes",
            default=False,
            description="Build dev-only fluid-solid coupling probes")
    variant("tests",
            default=False,
            description="Build C++ unit tests")

    # GPU variants are mutually exclusive: ROCmPackage and CudaPackage each
    # add their own +rocm / +cuda variant. Make them conflict explicitly.
    conflicts("+cuda", when="+rocm")

    # Both C and C++ compilers are needed: C++ for SEMSWS itself, C for the
    # downstream find_package(HDF5 COMPONENTS C) test compile and similar
    # transitive checks. Spack v1.x requires explicit language declarations
    # before the compiler wrappers will export SPACK_CC_* / SPACK_CXX_*.
    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("cmake@3.20:", type="build")

    # MPI is non-optional in the current code path (ParMesh / hypre).
    depends_on("mpi")

    # MFEM 4.9+ required: SEMSWS uses mfem::Reshape (scoped via
    # general/forall.hpp re-include in 4.9) and mfem::future::Dual<>
    # autodiff. Spack builtin only ships up to 4.8.0; spack-repo/packages/
    # mfem/ adds 4.9.0 as a shadow.
    depends_on("mfem@4.9:+mpi")
    # Precision must match across mfem, hypre, and SEMSWS itself.
    depends_on("mfem precision=single", when="precision=single")
    depends_on("mfem precision=double", when="precision=double")
    depends_on("hypre precision=single", when="precision=single")
    depends_on("hypre precision=double", when="precision=double")
    # NetCDF mesh I/O (Cubit/Genesis) is unconditionally required by
    # SEMSWS pre-processing and exporters.
    depends_on("mfem+netcdf")

    # GPU-time hypre tuning. Spack's mfem package does not propagate
    # GPU variants to hypre, so we force them here. +unified-memory
    # avoids OOM on large meshes; +mixedint halves the index footprint
    # on GPU; +gpu-aware-mpi must match SEMSWS's own +gpu_aware_mpi.
    # cuda_arch / amdgpu_target are propagated as for mfem.
    depends_on("hypre+cuda+unified-memory+mixedint", when="+cuda")
    depends_on("hypre+rocm+unified-memory+mixedint", when="+rocm")
    depends_on("hypre+gpu-aware-mpi",                when="+gpu_aware_mpi")
    for _arch in CudaPackage.cuda_arch_values:
        depends_on(
            "hypre+cuda cuda_arch={0}".format(_arch),
            when="+cuda cuda_arch={0}".format(_arch),
        )
    for _arch in ROCmPackage.amdgpu_targets:
        depends_on(
            "hypre+rocm amdgpu_target={0}".format(_arch),
            when="+rocm amdgpu_target={0}".format(_arch),
        )
    depends_on("hdf5+mpi")
    # SEMSWS only uses ADIOS2 BP5 binary I/O. Variants left at upstream
    # defaults: the `~xxx` set required to trim engines shifts between
    # spack stable and develop, and a single missing flag breaks concretization.
    depends_on("adios2@2.9:+mpi")

    # GPU variants must propagate to MFEM and arch flags must match.
    depends_on("mfem+cuda", when="+cuda")
    depends_on("mfem+rocm", when="+rocm")
    for _arch in CudaPackage.cuda_arch_values:
        depends_on(
            "mfem+cuda cuda_arch={0}".format(_arch),
            when="+cuda cuda_arch={0}".format(_arch),
        )
    for _arch in ROCmPackage.amdgpu_targets:
        depends_on(
            "mfem+rocm amdgpu_target={0}".format(_arch),
            when="+rocm amdgpu_target={0}".format(_arch),
        )

    # yaml-cpp 0.9.0+ required for the cstdint fix (GCC 15+ otherwise
    # fails on `uint16_t was not declared`). Spack builtin only has 0.8.0;
    # spack-repo/packages/yaml-cpp/ adds 0.9.0 as a shadow.
    depends_on("yaml-cpp@0.9.0:")

    def cmake_args(self):
        spec = self.spec
        args = [
            self.define("MFEM_DIR", spec["mfem"].prefix),
            self.define("HDF5_ROOT", spec["hdf5"].prefix),
            self.define_from_variant("SEM_USE_GPU_AWARE_MPI", "gpu_aware_mpi"),
            self.define_from_variant("SEM_BUILD_COUPLING_PROBES",
                                     "coupling_probes"),
            self.define_from_variant("SEMSWS_BUILD_TESTS", "tests"),
        ]
        if spec.satisfies("+cuda"):
            cuda_archs = spec.variants["cuda_arch"].value
            args.append(
                self.define("CMAKE_CUDA_ARCHITECTURES", ";".join(cuda_archs))
            )
        if spec.satisfies("+rocm"):
            archs = spec.variants["amdgpu_target"].value
            args.append(self.define("AMDGPU_TARGETS", ";".join(archs)))
            # MFEM_FORALL kernels only become real HIP device code when
            # compiled with hipcc; other CXX compilers silently fall back
            # to host loops and produce wrong results.
            args.append(self.define("CMAKE_CXX_COMPILER",
                                    str(spec["hip"].prefix.bin.hipcc)))
        return args

    def flag_handler(self, name, flags):
        # Cray PE: cray-mpich rejects MPI_Init under MPICH_GPU_SUPPORT_ENABLED=1
        # unless its GPU Transport Layer plugin (libmpi_gtl_hsa/cuda) is
        # linked. We pull the right -L/-l/-rpath via the gtl_lib property
        # of cray-mpich; non-Cray MPI providers fall through as a noop.
        wrapper_flags = list(flags)
        build_system_flags = []
        if (self.spec.satisfies("+rocm +gpu_aware_mpi")
                or self.spec.satisfies("+cuda +gpu_aware_mpi")):
            if self.spec.satisfies("^[virtuals=mpi] cray-mpich"):
                gtl = self.spec["cray-mpich"].package.gtl_lib
                build_system_flags.extend(gtl.get(name) or [])
        return (wrapper_flags, [], build_system_flags)

    def test_help(self):
        """Run the installed semsws --help as a smoke check."""
        semsws = which(self.prefix.bin.semsws)
        out = semsws("--help", output=str.split, error=str.split, fail_on_error=False)
        assert "config" in out.lower() or "Usage" in out
