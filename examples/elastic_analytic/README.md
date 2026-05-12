# 3D Elastic Analytic Benchmark

Homogeneous isotropic-elastic full-space with a point Mrp double-couple
source. Validates SEMSWS against the Aki & Richards closed-form solution.

- Domain: 44 km × 44 km × 34 km, internal hex mesh 120×120×100, order 4
- Material: Vp=2800, Vs=1500, rho=2300
- Source: Ricker / Gaussian at the domain center (see `config.yaml`)
- Receivers: 6 stations (Z1–Z6) at increasing offset

## Files

```
elastic_analytic/
├── config.yaml                  # SEMSWS simulation config
├── generate_analytical.py       # Aki & Richards solution → ref/*.semd
├── plot_comparison.py           # paper-style analytic/numerical overlay (3 stations, dashed synthetic)
└── run_benchmark.sh             # 3-step: generate ref → run sim → inline L2 diff
```

No mesh / ref / result artifacts are checked in — everything is regenerated
from `generate_analytical.py` and `semsws`.

## Prerequisites

- SEMSWS build: `build/src/semsws` (MPI)
- Python 3 with `numpy`
- `mpirun`

## Run

```bash
cd examples/elastic_analytic
bash run_benchmark.sh 4            # 4 = number of MPI ranks
```

That single command:

1. runs `python3 generate_analytical.py` (writes `ref/*.semd`)
2. runs `mpirun -np 4 .../semsws --config config.yaml`
   (writes `results/`)
3. prints the normalized L2 error per trace (18 traces = 6 stations × 3 comps)

## Plots

```bash
python3 plot_comparison.py         # 3-station overlay (analytic solid, numerical dashed)
```
