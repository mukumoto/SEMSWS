#!/bin/bash
# LUMI partition/G environment for SEMSWS Spack builds.
# Source this before any `spack` invocation; do NOT execute.
#
# Override these if your project / Spack checkout lives elsewhere:
#   PROJECT     LUMI allocation, e.g. project_465002833
#   SPACK_ROOT  Spack checkout (shared with the project is fine)

: "${PROJECT:=project_465002833}"
: "${SPACK_ROOT:=/flash/${PROJECT}/program/spack}"

module load LUMI/25.09 partition/G
module load PrgEnv-cray
module load cray-mpich/9.0.1
module load craype-accel-amd-gfx90a
module load rocm/6.4.4
module load lumi-CrayPath

export MPICH_GPU_SUPPORT_ENABLED=1

source "${SPACK_ROOT}/share/spack/setup-env.sh"

export SPACK_USER_CACHE_PATH="/flash/${PROJECT}/${USER}/.spack-cache"
