# LUMI partition/G site config for SEMSWS

LUMI で SEMSWS を Spack ビルドするための環境設定一式。

```
load.sh         module load + spack 環境ロード（PROJECT/SPACK_ROOT で上書き可）
packages.yaml   externals: cce / cray-mpich / cray-hdf5 / metis / rocm 6.4.4
                           cray-libsci / cmake
```

`packages.yaml` の prefix は `/opt/cray` `/appl/lumi` `/usr` で完結しているのでユーザ非依存。

## 初回セットアップ

```bash
# 0) project 番号
export PROJECT=project_465002833

# 1) Spack を flash に clone（プロジェクト共有可。1 回だけ）
git clone --depth=1 https://github.com/spack/spack.git \
    /flash/${PROJECT}/program/spack

# 2) SEMSWS を clone
cd /flash/${PROJECT}/${USER}
git clone https://github.com/mukumoto/SEMSWS.git
cd SEMSWS

# 3) site config を所定の場所へコピー
cp spack-repo/site-config/lumi/load.sh ~/load.sh
mkdir -p ~/.spack
cp spack-repo/site-config/lumi/packages.yaml ~/.spack/packages.yaml

# 4) 環境ロード & コンパイラ登録
source ~/load.sh
spack compiler find        # cce@20.0.0 が登録される

# 5) SEMSWS recipe を Spack に教える
spack repo add /flash/${PROJECT}/${USER}/SEMSWS/spack-repo

# 6) install（GPU 本番想定: single + gpu_aware_mpi）
spack install semsws@main +rocm amdgpu_target=gfx90a precision=single +gpu_aware_mpi

# 7) 使う
spack load semsws
which semsws
```

## 2 回目以降のセッション

```bash
source ~/load.sh
spack load semsws
```

## 注意

- 既存の `~/.spack/packages.yaml` がある場合は退避してから `cp` すること。
- `PROJECT` がデフォルト (`project_465002833`) と異なる場合は `load.sh` 先頭の
  `: "${PROJECT:=...}"` を書き換えるか、`PROJECT=projectXXXX source ~/load.sh`
  で都度指定する。
- `MPICH_GPU_SUPPORT_ENABLED=1` は GPU-aware MPI に必須。`load.sh` で export
  済み。job script でも忘れず export すること。
