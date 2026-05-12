<p align="center">
  <img src="assets/logofinal.png" alt="SEMSWS Logo" width="800">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-green.svg" alt="License: BSD-3-Clause"></a>
  <a href="https://isocpp.org/"><img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17"></a>
  <a href="https://mfem.org/"><img src="https://img.shields.io/badge/built%20on-MFEM-orange.svg" alt="MFEM"></a>
</p>

---

## Overview

**SEMSWS** (Spectral Element Method — Seismic Wave Simulator) is a
high-performance seismic wave propagation solver written in C++17 and built
on [MFEM](https://mfem.org/).

- **Parallel** MPI domain decomposition for HPC clusters
- **Portable GPU** support (CUDA / HIP) through MFEM's device layer
- **Multiple physics**: isotropic acoustic, isotropic elastic,
  viscoelastic (generalized Maxwell body), and coupled fluid-solid
  wave equations in 2D/3D

---

## Repository layout

```
src/                     C++ sources (simulation, operators, materials, integrators)
include/                 C++ headers (mirrors src/)
cmake/                   CMake helpers
driver/                  Python driver (semsws_driver) for shot orchestration
test/                    pytest suite (unit, consistency, waveform, coupling)
scripts/plot/            Python helpers for wavefield / mesh visualization
examples/visualization/  Wavefield / kernel rendering examples
assets/                  Logos
spack-repo/              Spack package recipe + LUMI / Earth Simulator site-config
```

---

## Third-party components

SEMSWS depends on the following projects. See each project for its own license:

- [MFEM](https://mfem.org/) — finite element library (BSD-3)
- [hypre](https://github.com/hypre-space/hypre) — preconditioners and solvers (Apache 2.0)
- [METIS](https://github.com/KarypisLab/METIS) — graph / mesh partitioning (Apache 2.0)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) — hierarchical data format (BSD-3)
- [ADIOS2](https://github.com/ornladios/ADIOS2) — parallel I/O (Apache 2.0)
- [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) — Cubit / Exodus mesh I/O (BSD-3)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) — YAML parser (MIT)

## Documentation

Full installation and usage documentation is distributed as a separate PDF
manual. A copy will be included in this repository once released.

## License

SEMSWS is released under the [BSD 3-Clause License](LICENSE), the same
license used by [MFEM](https://github.com/mfem/mfem).
