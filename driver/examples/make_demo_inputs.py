#!/usr/bin/env python3
"""Build a v2.0 SEMSWS HDF5 input file from simple CSV source/receiver lists.

This is the suggested workflow when a user has source / receiver tables
exported from an external tool (geometry surveys, vibroseis logs, etc.)
and wants to feed them into `semsws-driver run`.

Usage:
    python examples/make_demo_inputs.py \\
        --sources    examples/demo_inputs/sources.csv \\
        --receivers  examples/demo_inputs/receivers.csv \\
        --output     observations.h5 \\
        --dt 5e-4 --n-samples 200 --space-dim 2

    semsws-driver run \\
        --config examples/2d_acoustic_local.yaml \\
        --inputs observations.h5 \\
        --workdir ./work

CSV formats
-----------

sources.csv columns (header row required):
    shot_id, source_id, type, x, y[, z], freq, delay
    [, dx, dy[, dz]]                    # required for type=force
    [, mxx, myy, mzz, mxy, mxz, myz]    # required for type=moment_tensor

    Allowed `type` values: pressure | force | moment_tensor.
    `freq` and `delay` parameterise a Ricker STF (Hz, seconds).
    For 2D space_dim=2, omit z / dz / Mxz / Myz / Mzz columns.

receivers.csv columns:
    shot_id, receiver_id, x, y[, z]

Both shot_id and source_id / receiver_id must be positive integers and
unique per shot. The script groups rows by shot_id automatically.

For more advanced inputs (custom STFs, observed waveforms with weights),
see `semsws_driver.io.hdf5_v2.{SourceEntry, ReceiverEntry, write_v2}`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running directly from the repo without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsws_driver.io import hdf5_schema as S
from semsws_driver.io.hdf5_v2 import (
    ReceiverEntry, ShotEntry, SourceEntry, write_v2,
)


def ricker_samples(n: int, dt: float, freq: float, delay: float) -> np.ndarray:
    """Standard Ricker (Mexican hat) wavelet samples."""
    t = np.arange(n) * dt - delay
    x = (np.pi * freq * t) ** 2
    return ((1.0 - 2.0 * x) * np.exp(-x)).astype(np.float64)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_position(row: dict, space_dim: int) -> list[float]:
    pos = [float(row["x"]), float(row["y"])]
    if space_dim == 3:
        pos.append(float(row["z"]))
    return pos


def _read_force_direction(row: dict, space_dim: int) -> list[float]:
    d = [float(row["dx"]), float(row["dy"])]
    if space_dim == 3:
        d.append(float(row["dz"]))
    return d


def _read_moment_tensor(row: dict, space_dim: int) -> list[float]:
    if space_dim == 2:
        # 2D canonical order: Mxx, Myy, Mxy
        return [float(row["mxx"]), float(row["myy"]), float(row["mxy"])]
    # 3D canonical order: Mxx, Myy, Mzz, Mxy, Mxz, Myz
    return [float(row[k]) for k in ("mxx", "myy", "mzz", "mxy", "mxz", "myz")]


def build_shots(
    *,
    sources_csv: Path,
    receivers_csv: Path,
    dt: float,
    n_samples: int,
    space_dim: int,
) -> list[ShotEntry]:
    src_rows = _read_csv(sources_csv)
    rcv_rows = _read_csv(receivers_csv)

    by_shot: dict[int, dict] = defaultdict(
        lambda: {"sources": [], "receivers": []})

    for row in src_rows:
        sid = int(row["shot_id"])
        kind = row["type"].strip().lower()
        if kind not in (S.SOURCE_TYPE_PRESSURE, S.SOURCE_TYPE_FORCE,
                        S.SOURCE_TYPE_MOMENT_TENSOR):
            raise ValueError(
                f"sources.csv: unknown type {kind!r} (allowed: pressure | "
                f"force | moment_tensor)")
        stf = ricker_samples(n_samples, dt,
                              freq=float(row["freq"]),
                              delay=float(row["delay"]))
        entry = SourceEntry(
            source_id=int(row["source_id"]),
            type=kind,
            position=_read_position(row, space_dim),
            stf=stf,
            direction=(_read_force_direction(row, space_dim)
                       if kind == S.SOURCE_TYPE_FORCE else ()),
            moment_tensor=(_read_moment_tensor(row, space_dim)
                           if kind == S.SOURCE_TYPE_MOMENT_TENSOR else ()),
            label=row.get("label", "") or "",
        )
        by_shot[sid]["sources"].append(entry)

    for row in rcv_rows:
        sid = int(row["shot_id"])
        entry = ReceiverEntry(
            receiver_id=int(row["receiver_id"]),
            position=_read_position(row, space_dim),
            label=row.get("label", "") or "",
        )
        by_shot[sid]["receivers"].append(entry)

    return [ShotEntry(shot_id=sid,
                      sources=by_shot[sid]["sources"],
                      receivers=by_shot[sid]["receivers"])
            for sid in sorted(by_shot.keys())]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", required=True, type=Path)
    ap.add_argument("--receivers", required=True, type=Path)
    ap.add_argument("--output", "-o", required=True, type=Path)
    ap.add_argument("--dt", type=float, required=True,
                    help="time step (seconds); must match simulation.time.dt")
    ap.add_argument("--n-samples", type=int, required=True,
                    help="number of time samples (must match simulation.time.steps)")
    ap.add_argument("--space-dim", type=int, choices=(2, 3), default=2)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--no-overwrite", action="store_true")
    args = ap.parse_args(argv)

    shots = build_shots(
        sources_csv=args.sources, receivers_csv=args.receivers,
        dt=args.dt, n_samples=args.n_samples, space_dim=args.space_dim,
    )
    if not shots:
        print("error: no shots produced (empty CSVs?)", file=sys.stderr)
        return 1

    out = write_v2(
        path=args.output, shots=shots,
        dt=args.dt, n_samples=args.n_samples, space_dim=args.space_dim,
        t0=args.t0, overwrite=not args.no_overwrite,
        created_by="examples/make_demo_inputs.py",
    )
    n_src = sum(len(s.sources) for s in shots)
    n_rcv = sum(len(s.receivers) for s in shots)
    print(f"wrote {out}")
    print(f"  {len(shots)} shot(s), {n_src} source(s), {n_rcv} receiver(s), "
          f"dt={args.dt}, n_samples={args.n_samples}, space_dim={args.space_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
