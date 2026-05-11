# semsws-driver

Thin run-only driver for SEMSWS large-scale forward / FWI runs.

The driver does **not** assemble SEMSWS YAML. The user supplies (1) a YAML
template (with a `run:` section) and (2) a v2.0 HDF5 file bundling sources,
receivers, and observed waveforms. The driver:

1. copies the template per shot, substituting `sources` and `receivers`
   with `{format: hdf5, file: ..., shot_id: NNNN}`,
2. forces `receivers.output.formats: [{type: hdf5}]` (HDF5 outputs only),
3. runs `partition_mesh` / `semsws_export_model` automatically when needed,
4. dispatches via Local / SLURM / PJM / PBS / LSF runners,
5. writes `manifest.json` (sha256 + git rev) for reproducibility.

## Install

```bash
pip install -e driver/
```

## CLI

```bash
semsws-driver run --config my_sim.yaml --inputs observations.h5 --workdir ./work
semsws-driver run ... --merge          # also produce results/merged.h5
semsws-driver dry-run --config ... --inputs ... --workdir ...
semsws-driver merge --workdir ./work -o ./work/results/merged.h5
```

## Python API

```python
from semsws_driver import run

result = run(
    config="my_sim.yaml",
    inputs="observations.h5",
    workdir="./work",
    merge=True,
)
```

## Output layout

```
<workdir>/
  manifest.json            # sha256 + git rev + timing per shot
  config.yaml              # copy of user template (lock)
  inputs/observations.h5
  shots/shot_NNNN/
    config.yaml
    seismograms*.h5
    *.stdout.log
    *.stderr.log
  results/merged.h5        # only with --merge
  _shared_mesh/, _shared_model/   # preflight cache, when applicable
```

## YAML template

Required top-level: `simulation`, `mesh`, `material`, `device`, `boundary`,
`receivers` (`type:` only — driver fills the rest), `sources` (placeholder
— driver overrides), `run:`.

The driver-only `run:` section is stripped before SEMSWS sees the YAML.
Minimum:

```yaml
run:
  binary: /path/to/build/src/semsws
  scheduler: local                    # local | slurm | pjm | pbs | lsf
  ranks_per_shot: 1
```

GPU runs assume `gpus_per_shot == ranks_per_shot` for `device_kind` ∈
{`cuda`, `hip`}. CPU thread counts go via `env.OMP_NUM_THREADS`; CPU
allocation via `#SBATCH --cpus-per-task` in the user's job script.

## What the driver does NOT do

- Assemble or validate SEMSWS YAML beyond `sources/receivers/output.formats`.
- Allocate SLURM jobs (use `#SBATCH` in your job script).
- Convert HDF5 outputs to SU / SEG-Y (future work).
- Load waveforms into memory (use `h5py` / `xarray` post-run).
