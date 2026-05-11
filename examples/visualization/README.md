# Wavefield Visualization Demos

Short forward simulations that exercise every wavefield output format
(GLVis, ParaView, GMT xyz) and exercise the plot scripts in
[`scripts/plot/`](../../scripts/plot/).

Three cases:

| Dir | Dim | Material | Plots |
|---|---|---|---|
| `2D/acoustic` | 2D | acoustic  | `plot_wavefield_2d.py`, `plot_gmt.sh` (GMT CLI) |
| `3D/acoustic` | 3D | acoustic  | `plot_wavefield_3d.py`, `plot_3d_slices.py`, `plot_gmt.sh` |
| `3D/elastic`  | 3D | elastic   | same as 3D acoustic |

Each directory has:

- `config.yaml` — simulation config with all 3 wavefield formats enabled
- `run.sh` — forward run + invokes the appropriate plot scripts
- `plot_gmt.sh` — local shell wrapper for GMT CLI plots

## Prerequisites

- SEMSWS build (`build/src/semsws`)
- Python 3 with `numpy`, `matplotlib`
- Optional: `pyvista` (for `plot_3d_slices.py`), GMT 6 CLI (for `plot_gmt.sh`)

## Usage

```bash
cd examples/visualization/3D/acoustic
bash run.sh 4                 # 4 MPI ranks
```

Plots land under `./results/figures/`.
