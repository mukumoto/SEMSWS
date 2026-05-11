# Earth Simulator (JAMSTEC) site config for SEMSWS

GPU パーティション (NVIDIA A100, sm_80) と CPU パーティション両対応の Spack
ビルド設定一式。LUMI と同じ形式 (load.sh + packages.yaml) を提供しつつ、
ES では GPU/CPU の 2 ターゲットがあるため両方分の設定を同梱しています。

```text
load_gpu.sh           module load + Spack 環境ロード (NVHPC + CUDA)
load_cpu.sh           module load + Spack 環境ロード (gcc + system OpenMPI)
packages_gpu.yaml     externals: nvhpc / hpcx-ompi / cuda 12.0 / cmake
                      Spack-built (%gcc): hdf5 / metis / netcdf / openblas
packages_cpu.yaml     externals: system OpenMPI 4.0.5 / cmake
                      Spack-built (%gcc): hdf5 / metis / netcdf / openblas
```

## 初回セットアップ

```bash
# 0) 作業ディレクトリ
export WORK=/S/data01/G3506/${USER}/program_test   # 自分の path に
mkdir -p "$WORK"

# 1) Spack を clone (1 回だけ。site 共有なら省略可)
git clone --depth=1 https://github.com/spack/spack.git "${WORK}/spack"

# 2) SEMSWS を clone
cd "$WORK"
git clone https://github.com/mukumoto/SEMSWS.git
cd SEMSWS

# 3) load 系を所定の場所へコピー
cp spack-repo/site-config/es/load_gpu.sh ~/load_gpu.sh
cp spack-repo/site-config/es/load_cpu.sh ~/load_cpu.sh
chmod +x ~/load_gpu.sh ~/load_cpu.sh
```

ここから先は **(A) GPU/CPU の片方だけ使う** か **(B) 両方並立** で手順が変わります。

## (A) 単一ターゲット (LUMI と同じ形)

GPU だけ、または CPU だけ使う場合。LUMI 用 README と同じ流れ。

```bash
# GPU で使う例
source ~/load_gpu.sh
mkdir -p ~/.spack
[ -f ~/.spack/packages.yaml ] && mv ~/.spack/packages.yaml ~/.spack/packages.yaml.bak
cp spack-repo/site-config/es/packages_gpu.yaml ~/.spack/packages.yaml
# あるいは SPACK_USER_CONFIG_PATH 経由 (load_gpu.sh が $WORK/.spack を指す):
cp spack-repo/site-config/es/packages_gpu.yaml "$SPACK_USER_CONFIG_PATH/packages.yaml"

spack compiler find                    # nvhpc@24.7 + gcc@8.5 が登録される
spack repo add "$WORK/SEMSWS/spack-repo"
spack install semsws@main +cuda cuda_arch=80 precision=single +gpu_aware_mpi \
  cxxflags=="-noswitcherror" cflags=="-noswitcherror" ldlibs=="-lstdc++fs" \
  ^hypre@2.33 %nvhpc
```

CPU の場合は `load_cpu.sh` + `packages_cpu.yaml` に差し替え、`spack install
semsws@main precision=single %gcc` とする。

## (B) GPU と CPU を並立 (Spack environment 推奨)

両方 install して使い分けたい場合は Spack environment で分離する。
`SPACK_USER_CONFIG_PATH` を切り替える方式より、env 単位で完結する方が
事故が少ない。

```bash
# GPU env
source ~/load_gpu.sh
spack compiler find
spack env create semsws-gpu
spack env activate semsws-gpu
spack config edit                       # エディタが開く
# spack.yaml に以下を貼る (packages_gpu.yaml の内容を packages: の下にネスト)
```

`semsws-gpu` の `spack.yaml`:

```yaml
spack:
  repos:
  - /S/data01/G3506/<USER>/program_test/SEMSWS/spack-repo

  packages:
    # ↓ packages_gpu.yaml の中身をここに丸ごとインデント
    cmake:
      externals:
      - spec: cmake@3.26.5
        prefix: /usr
      buildable: false
    nvhpc: ...
    openmpi: ...
    cuda: ...
    blas:   { require: openblas }
    lapack: { require: openblas }
    openblas:
      require: '%gcc'
    hypre:
      variants: +shared
    all:
      providers:
        mpi: [openmpi]
      variants: cuda_arch=80

  specs:
  - semsws@main +cuda cuda_arch=80 precision=single +gpu_aware_mpi cxxflags=="-noswitcherror" cflags=="-noswitcherror" ldlibs=="-lstdc++fs" ^hypre@2.33 %nvhpc

  concretizer:
    unify: true
  config:
    build_jobs: 4
```

```bash
spack concretize -f
spack install --fail-fast
spack env deactivate

# CPU env
source ~/load_cpu.sh
spack compiler find
spack env create semsws-cpu
spack env activate semsws-cpu
spack config edit
# packages_cpu.yaml の内容 + spec: semsws@main precision=single %gcc
spack concretize -f
spack install --fail-fast
spack env deactivate
```

## 2 回目以降のセッション

```bash
# GPU 使うとき
source ~/load_gpu.sh
spack env activate semsws-gpu          # (B) 方式の場合
spack load semsws

# CPU 使うとき
source ~/load_cpu.sh
spack env activate semsws-cpu
spack load semsws
```

## 注意

- **module 競合**: `NVIDIAHPCSDK/24.7/...` と `OpenMPI/4.0.5` は同時 load
  できない。load_gpu.sh / load_cpu.sh は冒頭で `module purge` してから
  入れ直す設計。切替時は必ず適切な `source ~/load_xxx.sh` を実行。
- **CUDA 12.0 を使う理由**: NVHPC SDK 24.7 同梱の CUDA 12.5 は分割
  レイアウトで MFEM が `cusparse.h` を見失う。`/opt/share/CUDA/12.0.0` の
  flat レイアウト standalone モジュールを採用。
- **hypre 2.33 を強制**: hypre 3.x は ES の gcc 8.5 + nvcc で cstddef の
  extern "C" 問題と CMake FindMPI の NVHPC 互換問題を踏むため。shadow
  recipe (`spack-repo/packages/hypre/`) で cuSOLVER OFF + FSAI device
  バグ回避済み。
