# LUMI partition/G site config for SEMSWS

Spack build settings for SEMSWS on LUMI.

```text
load.sh         module load + spack environment (PROJECT/SPACK_ROOT overridable)
packages.yaml   externals: cce / cray-mpich / cray-hdf5 / metis / rocm 6.4.4
                           cray-libsci / cmake
```

Prefixes in `packages.yaml` are confined to `/opt/cray`, `/appl/lumi`, and
`/usr`, so the file is user-independent.

## First-time setup

```bash
# 0) project number
export PROJECT=project_465002833

# 1) Clone Spack to flash (project-shared, one time)
git clone --depth=1 https://github.com/spack/spack.git \
    /flash/${PROJECT}/program/spack

# 2) Clone SEMSWS
cd /flash/${PROJECT}/${USER}
git clone https://github.com/mukumoto/SEMSWS.git
cd SEMSWS

# 3) Drop site config files into place
cp spack-repo/site-config/lumi/load.sh ~/load.sh
mkdir -p ~/.spack
cp spack-repo/site-config/lumi/packages.yaml ~/.spack/packages.yaml

# 4) Load environment, register compiler
source ~/load.sh
spack compiler find        # registers cce@20.0.0

# 5) Tell Spack about the SEMSWS recipe
spack repo add /flash/${PROJECT}/${USER}/SEMSWS/spack-repo

# 6) Install (production GPU build: single + gpu_aware_mpi)
spack install semsws@main +rocm amdgpu_target=gfx90a precision=single +gpu_aware_mpi

# 7) Use it
spack load semsws
which semsws
```

## Subsequent sessions

```bash
source ~/load.sh
spack load semsws
```

## Notes

- Back up any existing `~/.spack/packages.yaml` before copying the new one
  in step 3.
- If `PROJECT` differs from the default (`project_465002833`), edit the
  `: "${PROJECT:=...}"` line at the top of `load.sh`, or call
  `PROJECT=projectXXXX source ~/load.sh` per session.
- `MPICH_GPU_SUPPORT_ENABLED=1` is required for GPU-aware MPI. `load.sh`
  exports it; remember to also export it from your job script.
