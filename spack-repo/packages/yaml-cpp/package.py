"""Standalone yaml-cpp recipe that ships 0.9.0 with a narrowed cstdint patch.

Builtin's yaml-cpp applies the upstream ``#include <cstdint>`` fix
unconditionally for ``@0.7:``, but that fix is already merged into 0.9.0;
re-applying it fails with "Reversed (or previously applied) patch
detected". Since Spack's ``patch()`` is append-only, we re-declare the
recipe standalone with ``when="@0.7:0.8"``.

Drop this file once builtin tightens the patch range and registers 0.9.0+.
"""

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.packages.boost.package import Boost

from spack.package import *


yaml_cpp_tests_libcxx_error_msg = "yaml-cpp tests incompatible with libc++"


class YamlCpp(CMakePackage):
    """A YAML parser and emitter in C++"""

    homepage = "https://github.com/jbeder/yaml-cpp"
    url = "https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz"
    git = "https://github.com/jbeder/yaml-cpp.git"
    maintainers("eschnett")

    license("MIT")

    version("develop", branch="master")
    version(
        "0.9.0",
        sha256="25cb043240f828a8c51beb830569634bc7ac603978e0f69d6b63558dadefd49a",
    )
    version(
        "0.8.0",
        sha256="fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16",
    )
    version(
        "0.7.0",
        sha256="43e6a9fcb146ad871515f0d0873947e5d497a1c9c60c58cb102a97b47208b7c3",
    )

    variant("shared", default=True, description="Build shared instead of static libraries")
    variant("pic", default=True, description="Build with position independent code")
    variant("tests", default=False, description="Build yaml-cpp tests using internal gtest")

    depends_on("c", type="build")
    depends_on("cxx", type="build")

    depends_on("boost@:1.66", when="@0.5.0:0.5.3")
    depends_on(Boost.with_default_variants, when="@0.5.0:0.5.3")

    # cstdint patch (jbeder/yaml-cpp@7b469b4, upstream PR #1310). Bounded
    # at 0.7-0.8: 0.9.0+ already has the change merged.
    patch(
        "https://github.com/jbeder/yaml-cpp/commit/7b469b4220f96fb3d036cf68cd7bd30bd39e61d2.patch?full_index=1",
        sha256="0bb42bea4f38ac5e9b51a46938cf7ed12c23e62c8690a166101caa00f09dd639",
        when="@0.7:0.8",
    )

    conflicts("%gcc@:4.7", when="@0.6.0:", msg="versions 0.6.0: require c++11 support")
    conflicts("%clang@:3.3.0", when="@0.6.0:", msg="versions 0.6.0: require c++11 support")
    conflicts("%apple-clang@:4.0.0", when="@0.6.0:", msg="versions 0.6.0: require c++11 support")
    conflicts(
        'cxxflags="-stdlib=libc++" %clang', when="+tests", msg=yaml_cpp_tests_libcxx_error_msg
    )

    def flag_handler(self, name, flags):
        if (
            name in ("cxxflags", "cppflags")
            and self.spec.satisfies("+tests")
            and "-stdlib=libc++" in flags
        ):
            raise InstallError(yaml_cpp_tests_libcxx_error_msg)
        return (flags, None, None)

    def cmake_args(self):
        return [
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("YAML_BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("CMAKE_POSITION_INDEPENDENT_CODE", "pic"),
            self.define_from_variant("YAML_CPP_BUILD_TESTS", "tests"),
            # Suppress yaml-cpp's GCC-only `-Weffc++` / `-pedantic-errors`
            # warning flags via PR #1430. CMake silently ignores the
            # option on 0.8.0 / 0.9.0 tarballs, so it is safe to set
            # unconditionally; released versions still need
            # `cxxflags="-noswitcherror"` under NVHPC.
            self.define("YAML_CPP_USE_STRICT_FLAGS", False),
        ]

    def url_for_version(self, version):
        url = "https://github.com/jbeder/yaml-cpp/archive/{0}.tar.gz"
        if version < Version("0.5.3"):
            return url.format(f"release-{version}")
        elif version < Version("0.8.0"):
            return url.format(f"yaml-cpp-{version}")
        elif version == Version("0.8.0"):
            # 0.8.0 used a bare version tag (the convention briefly drifted)
            return url.format(f"{version}")
        else:
            # 0.9.0+ went back to the yaml-cpp-X.Y.Z tag convention
            return url.format(f"yaml-cpp-{version}")
