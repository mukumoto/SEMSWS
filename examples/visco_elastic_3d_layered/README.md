# 3D Two-Layer (Visco)Elastic Benchmark

10 km × 10 km × 10 km domain with a horizontal interface at z = –4 km, run as
either isotropic elastic or visco-elastic (3 SLS units, f0=2 Hz).

- Mesh: Cubit-generated hex grid (from `mesh_layered_3d.jou`), one mandatory
  refinement via `refine_mesh` → `mesh/` (partitioned for MPI)
- Materials: read from attribute table (`materials_elastic.txt` /
  `materials_visco.txt`); layer 1 Vp=6000, layer 2 Vp=3000
- Source: moment-tensor at (5, 5, -3) km, Ricker 2 Hz
- Receivers: 5 stations at z = 0 on the free surface

## Files

```
visco_elastic_3d_layered/
├── README.md
├── mesh_layered_3d.jou          # Cubit journal (only mesh input)
├── config_elastic.yaml          # external .exo + no attenuation
├── config_elastic_sim.yaml      # partitioned mesh + wavefield output
├── config_visco.yaml            # external .exo + attenuation on
├── config_visco_sim.yaml        # partitioned mesh + wavefield output
├── materials_elastic.txt        # Q=9999 (effectively elastic)
├── materials_visco.txt          # Q=100 / 50
├── run_elastic.sh               # refine → evaluate → forward → plot
├── run_visco.sh                 # same, visco variant
└── plot_comparison.py           # elastic vs visco trace overlay
                                # (plot_wavefield_3d.py / plot_3d_slices.py
                                #  live under scripts/plot/ and are shared)
```

No mesh binaries, no partitioned mesh directory, no results are checked in —
everything is regenerated from `.jou` + the two `run_*.sh` scripts.

## Prerequisites

- SEMSWS build: `build/src/{refine_mesh, evaluate_mesh, semsws}` with MPI
- Cubit (or Trelis) for mesh generation from `.jou`. Without Cubit, you need
  to provide a compatible `mesh_layered_3d.exo` by other means.
- Python 3 with `numpy`, `matplotlib` (for `scripts/plot/plot_wavefield_3d.py`)
- Optional: `pyvista` for `scripts/plot/plot_3d_slices.py` (the 3D slice viewer)

Note on "GMT": the plot scripts read SEMSWS's **GMT-format xyz text files**
(plain `x y value` ASCII dumps of each wavefield slice). This is just a file
format; you do **not** need the GMT CLI / PyGMT installed.

## Workflow (elastic shown; swap `elastic` → `visco` for the viscoelastic case)

```bash
cd examples/visco_elastic_3d_layered

# 1. Regenerate the mesh (~1 min) — writes mesh_layered_3d.exo
cubit -batch mesh_layered_3d.jou

# 2-4. Refine → evaluate → forward simulation (with optional plotting)
bash run_elastic.sh 4                          # 4 = MPI ranks

# Equivalent manual invocation of the three SEMSWS binaries:
#   mpirun -np 4 ../../build/src/refine_mesh        --config config_elastic.yaml --output-dir mesh
#   mpirun -np 4 ../../build/src/evaluate_mesh      --config config_elastic_sim.yaml --histogram eval
#   mpirun -np 4 ../../build/src/semsws --config config_elastic_sim.yaml
```

**Important:** step 1 (refine) is mandatory. `config_elastic_sim.yaml` and
`config_visco_sim.yaml` both declare `mesh.type: partitioned` pointing at
`mesh/`, so running `semsws` without first populating
`mesh/` will fail.

If you change the MPI rank count (`NPROCS`), you must re-run the refine step:
the partition count is baked into `mesh/` during refinement.

## Comparing elastic vs visco

After both runs finish, drop their results into sibling directories and:

```bash
python3 plot_comparison.py
```

See the script header for the exact expected layout.
