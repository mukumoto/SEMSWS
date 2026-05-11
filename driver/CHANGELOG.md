# Changelog

## 0.2.0 — 2026-05-10

Complete rewrite as a thin run-only layer.

**Removed (breaking):**
- `Simulation`, `Source`, `Receiver`, `Wavelet`, `Mesh`, `Material`,
  `Model`, `MomentTensor`, `OutputSpec`, `WavefieldOutput`,
  `MaterialOutput`, `ParaviewFormat`/`GlvisFormat`/`GmtFormat`,
  `TimeSpec`, `BoundarySpec`, `DeviceSpec`, `SimulationBlock`, `Shot`,
  `ShotOutcome` — driver no longer assembles SEMSWS YAML.
- `binding/` package, `forward/`, `inversion/`, `config/`, `tracking/`, `viz/`.
- `io/su.py`, `io/segy.py`, `io/stf.py`, `io/hdf5.py` (v1 reader),
  `io/grid.py`, `io/adios.py`.
- `[segy]`, `[adios]`, `[interactive]`, `[all]` extras.
- Tests for the deleted classes and example scripts.

**Added:**
- `core/run.py:run()` — top-level entry: read template + HDF5 inputs,
  optional preflight, render per-shot YAMLs, dispatch via Runner,
  optional merge, write `manifest.json`.
- `core/layout.py` — strict directory layout (single source of truth).
- `core/run_config.py` — typed parser for the YAML `run:` section.
- `core/template.py` — template-copy + `sources/receivers/output` injection.
- `core/preflight.py` — auto `partition_mesh` / `semsws_export_model`.
- `core/manifest.py` — sha256 + git rev reproducibility lock.
- `cli/main.py` — `semsws-driver run|dry-run|merge` console scripts.
- `runner/factory.py` — `RunConfig` → Runner + BindingPolicy bridge.
- `runner/binding.py` — minimal `BindingPolicy` (3 fields).

**Behavioural changes:**
- Receiver outputs are forced to HDF5; ASCII / SU paths are not driver-supported.
- `sources.mode` is auto-set to `simultaneous` (F1 rule).
- `simulation.output.directory` is auto-filled per shot.
- HPC binding flows through `run.binding` in YAML, not Python `BindingPolicy`.

## 0.1.0 — earlier

Initial 1:1 SEMSWS YAML wrapper (now removed).
