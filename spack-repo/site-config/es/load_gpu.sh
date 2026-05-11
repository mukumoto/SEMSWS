#!/bin/bash
# Earth Simulator (JAMSTEC) GPU build environment for SEMSWS.
# Source this before any `spack` invocation; do NOT execute.
#
# Override these if your install location differs:
#   WORK         project working directory (Spack root + cache live here)
#   SPACK_ROOT   Spack checkout (defaults to $WORK/spack)

: "${WORK:=/S/data01/G3506/${USER}/program_test}"
: "${SPACK_ROOT:=${WORK}/spack}"

module purge
module load zlib/1.2.11 szip/2.1.1
module load HDF5/1.10.5
module load NetCDF4/all
module load NVIDIAHPCSDK/24.7/nvhpc-hpcx-cuda12
module load METIS/5.1.0
module load CUDA/12.0.0
module load Python/3.12.6

# Make sure /usr/bin (system git, etc.) is reachable. Spack's startup check
# subprocesses git via a sanitized PATH, so prepend it explicitly. Also point
# Python's git-using libraries at the same binary.
export PATH=/usr/bin:$PATH
export SPACK_GIT_COMMAND=/usr/bin/git
export GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

export WORK SPACK_ROOT
export SPACK_USER_CONFIG_PATH="${WORK}/.spack"
export SPACK_USER_CACHE_PATH="${WORK}/.spack-cache"
mkdir -p "$SPACK_USER_CONFIG_PATH" "$SPACK_USER_CACHE_PATH"

source "${SPACK_ROOT}/share/spack/setup-env.sh"
