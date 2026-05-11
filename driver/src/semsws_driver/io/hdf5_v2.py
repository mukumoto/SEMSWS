"""SEMSWS v2.0 HDF5 writer (driver-side, h5py-based).

Writes files that the C++ readers (`HDF5SourceReceiverReader`,
`HDF5ObservedReader`) can consume directly. Covers three use cases:
forward input (receiver positions only), observed input (receivers carry
waveform datasets for FWI), and source input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from . import hdf5_schema as S


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReceiverEntry:
    """One receiver. For geometry-only input, `channels` may be empty.
    For observed input, fill `channels` with name → array (shape (nt,) for
    PS, (space_dim, nt) for vector channels). Channel names must be one of
    the canonical short forms (PS / VEL / DISP / ACC)."""
    receiver_id: int
    position: Sequence[float]
    label: str = ""
    types: Sequence[str] = ()           # optional override of parent receivers.type
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    weights: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SourceEntry:
    """One source. `direction` is required for force; ignored for pressure
    / moment_tensor. `moment_tensor` is required (canonical order) for
    moment_tensor type. `stf` is the scalar (n_samples,) array."""
    source_id: int
    type: str                                   # "force" | "pressure" | "moment_tensor"
    position: Sequence[float]
    stf: np.ndarray                             # shape (n_samples,) f64
    direction: Sequence[float] = ()
    moment_tensor: Sequence[float] = ()         # canonical order, 3 (2D) or 6 (3D)
    label: str = ""


@dataclass
class ShotEntry:
    """One shot's worth of sources + receivers."""
    shot_id: int
    sources: list[SourceEntry] = field(default_factory=list)
    receivers: list[ReceiverEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_v2(
    path: str | Path,
    shots: Sequence[ShotEntry],
    *,
    dt: float,
    n_samples: int,
    space_dim: int,
    t0: float = 0.0,
    coord_system: str = S.DEFAULT_COORD_SYSTEM,
    units: str = S.DEFAULT_UNITS,
    created_by: str | None = None,
    created_at: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a v2.0 SEMSWS HDF5 file containing the given shots.

    Each entry is validated against the F-rules where it's cheap to do so
    in Python (4-digit ids, MT shape per space_dim, channel name set,
    unique source ids per shot). Most bulk validation lives in the C++
    readers; we only catch the easy-to-spot writer mistakes here.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if space_dim not in (2, 3):
        raise ValueError(f"space_dim must be 2 or 3 (got {space_dim})")

    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(str(path), "w") as f:
        # Root attrs (B section).
        f.attrs[S.ATTR_FORMAT_VERSION] = S.FORMAT_VERSION
        f.attrs[S.ATTR_DT]        = float(dt)
        f.attrs[S.ATTR_T0]        = float(t0)
        f.attrs[S.ATTR_N_SAMPLES] = np.int64(n_samples)
        f.attrs[S.ATTR_SPACE_DIM] = np.int32(space_dim)
        f.attrs[S.ATTR_COORD_SYSTEM] = coord_system
        f.attrs[S.ATTR_UNITS]        = units
        if created_by is not None:
            f.attrs[S.ATTR_CREATED_BY] = created_by
        if created_at is not None:
            f.attrs[S.ATTR_CREATED_AT] = created_at

        for shot in shots:
            _write_shot(f, shot, space_dim=space_dim, n_samples=n_samples,
                        str_dt=str_dt)

    return path


def _write_shot(parent: h5py.File, shot: ShotEntry, *,
                space_dim: int, n_samples: int, str_dt) -> None:
    g = parent.create_group(f"{S.GROUP_SHOTS}/{S.shot_key(shot.shot_id)}")
    g.attrs[S.ATTR_SHOT_ID] = np.int32(shot.shot_id)

    if shot.sources:
        gsrc = g.create_group(S.GROUP_SOURCES)
        seen_ids: set[int] = set()
        for src in shot.sources:
            if src.source_id in seen_ids:
                raise ValueError(
                    f"shot {shot.shot_id}: duplicate source @id={src.source_id}"
                )
            seen_ids.add(src.source_id)
            _write_source(gsrc, src, space_dim=space_dim,
                          n_samples=n_samples, str_dt=str_dt)

    if shot.receivers:
        grcv = g.create_group(S.GROUP_RECEIVERS)
        seen_rids: set[int] = set()
        for rec in shot.receivers:
            if rec.receiver_id in seen_rids:
                raise ValueError(
                    f"shot {shot.shot_id}: duplicate receiver @id={rec.receiver_id}"
                )
            seen_rids.add(rec.receiver_id)
            _write_receiver(grcv, rec, space_dim=space_dim,
                            n_samples=n_samples, str_dt=str_dt)


def _write_source(gsrc: h5py.Group, src: SourceEntry, *,
                  space_dim: int, n_samples: int, str_dt) -> None:
    if src.source_id <= 0:
        raise ValueError(f"source @id must be > 0 (got {src.source_id})")
    valid = {S.SOURCE_TYPE_FORCE, S.SOURCE_TYPE_PRESSURE,
             S.SOURCE_TYPE_MOMENT_TENSOR}
    if src.type not in valid:
        raise ValueError(f"source type must be one of {valid} (got {src.type!r})")
    pos = np.asarray(src.position, dtype=np.float64)
    if pos.shape != (space_dim,):
        raise ValueError(
            f"source position shape {pos.shape} != ({space_dim},)")
    stf = np.asarray(src.stf, dtype=np.float64)
    if stf.shape != (n_samples,):
        raise ValueError(
            f"source stf shape {stf.shape} != ({n_samples},)")

    g = gsrc.create_group(S.source_key(src.source_id))
    g.attrs[S.ATTR_ID]   = np.int32(src.source_id)
    if src.label:
        g.attrs[S.ATTR_LABEL] = src.label
    g.attrs[S.ATTR_TYPE] = src.type
    g.attrs[S.ATTR_POSITION] = pos

    if src.type == S.SOURCE_TYPE_FORCE:
        direction = np.asarray(src.direction, dtype=np.float64)
        if direction.shape != (space_dim,):
            raise ValueError(
                f"force direction shape {direction.shape} != ({space_dim},)"
            )
        g.attrs[S.ATTR_DIRECTION] = direction

    if src.type == S.SOURCE_TYPE_MOMENT_TENSOR:
        canonical = (S.MT_COMPONENT_ORDER_3D
                     if space_dim == 3 else S.MT_COMPONENT_ORDER_2D)
        mt = np.asarray(src.moment_tensor, dtype=np.float64)
        if mt.shape != (len(canonical),):
            raise ValueError(
                f"moment_tensor shape {mt.shape} != ({len(canonical)},)")
        ds = g.create_dataset(S.DATASET_MOMENT_TENSOR, data=mt)
        ds.attrs[S.ATTR_COMPONENT_ORDER] = np.asarray(canonical, dtype=str_dt)
        ds.attrs[S.ATTR_COORD_SYSTEM]   = "xyz"

    g.create_dataset(S.DATASET_STF, data=stf)


def _write_receiver(grcv: h5py.Group, rec: ReceiverEntry, *,
                    space_dim: int, n_samples: int, str_dt) -> None:
    if rec.receiver_id <= 0:
        raise ValueError(f"receiver @id must be > 0 (got {rec.receiver_id})")
    pos = np.asarray(rec.position, dtype=np.float64)
    if pos.shape != (space_dim,):
        raise ValueError(
            f"receiver position shape {pos.shape} != ({space_dim},)")

    g = grcv.create_group(S.receiver_key(rec.receiver_id))
    g.attrs[S.ATTR_ID] = np.int32(rec.receiver_id)
    if rec.label:
        g.attrs[S.ATTR_LABEL] = rec.label
    g.attrs[S.ATTR_POSITION] = pos
    if rec.types:
        g.attrs["types"] = np.asarray(list(rec.types), dtype=str_dt)

    valid_channels = {S.CHANNEL_PRESSURE, S.CHANNEL_VELOCITY,
                      S.CHANNEL_DISPLACEMENT, S.CHANNEL_ACCELERATION}
    for ch_name, arr in rec.channels.items():
        if ch_name not in valid_channels:
            raise ValueError(
                f"unknown channel {ch_name!r} (valid: {valid_channels})")
        a = np.asarray(arr, dtype=np.float32)
        expected = ((n_samples,) if ch_name == S.CHANNEL_PRESSURE
                    else (space_dim, n_samples))
        if a.shape != expected:
            raise ValueError(
                f"channel {ch_name} shape {a.shape} != {expected}")
        g.create_dataset(ch_name, data=a)

    for ch_name, arr in rec.weights.items():
        if ch_name not in valid_channels:
            raise ValueError(
                f"unknown weight channel {ch_name!r}")
        a = np.asarray(arr, dtype=np.float32)
        expected = ((n_samples,) if ch_name == S.CHANNEL_PRESSURE
                    else (space_dim, n_samples))
        if a.shape != expected:
            raise ValueError(
                f"weight {ch_name} shape {a.shape} != {expected}")
        g.create_dataset(f"{S.WEIGHT_PREFIX}{ch_name}", data=a)


# ---------------------------------------------------------------------------
# Merge multiple per-shot HDF5 files into one bundled v2.0 file
# ---------------------------------------------------------------------------


def merge_shot_outputs(
    sources: Sequence[str | Path],
    output: str | Path,
    *,
    output_shot_ids: Sequence[int] | None = None,
    overwrite: bool = True,
) -> Path:
    """Merge N per-shot SEMSWS HDF5 files into a single multi-shot file.

    Each input file is expected to be a v2.0 file with exactly one shot
    (typically `/shots/0000/`). The merged file places them under
    `/shots/<NNNN>/` with consecutive ids by default, or the explicit
    `output_shot_ids` provided by the caller.

    Root attrs (dt, n_samples, space_dim, ...) must agree across all
    inputs (R2: file-wide common time axis). Mismatches raise ValueError.
    """
    if not sources:
        raise ValueError("merge_shot_outputs: no inputs")
    sources = [Path(p) for p in sources]
    if output_shot_ids is None:
        output_shot_ids = list(range(len(sources)))
    if len(output_shot_ids) != len(sources):
        raise ValueError(
            f"output_shot_ids length {len(output_shot_ids)} != "
            f"sources length {len(sources)}")

    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Inspect the first file to seed root attrs; assert subsequent files agree.
    first = sources[0]
    with h5py.File(str(first), "r") as f0:
        _check_v2(f0, str(first))
        root_attrs = {
            S.ATTR_FORMAT_VERSION: S.FORMAT_VERSION,
            S.ATTR_DT:        float(f0.attrs[S.ATTR_DT]),
            S.ATTR_T0:        float(f0.attrs[S.ATTR_T0]),
            S.ATTR_N_SAMPLES: int(np.asarray(f0.attrs[S.ATTR_N_SAMPLES])),
            S.ATTR_SPACE_DIM: int(np.asarray(f0.attrs[S.ATTR_SPACE_DIM])),
            S.ATTR_COORD_SYSTEM: _decode(f0.attrs.get(
                S.ATTR_COORD_SYSTEM, S.DEFAULT_COORD_SYSTEM)),
            S.ATTR_UNITS:        _decode(f0.attrs.get(
                S.ATTR_UNITS, S.DEFAULT_UNITS)),
        }

    with h5py.File(str(output), "w") as fout:
        for k, v in root_attrs.items():
            if k == S.ATTR_N_SAMPLES:
                fout.attrs[k] = np.int64(v)
            elif k == S.ATTR_SPACE_DIM:
                fout.attrs[k] = np.int32(v)
            else:
                fout.attrs[k] = v

        for src_path, target_id in zip(sources, output_shot_ids):
            with h5py.File(str(src_path), "r") as fin:
                _check_v2(fin, str(src_path))
                _check_root_attrs_match(fin, root_attrs, str(src_path))
                # Find the shot group inside (single-shot per-file output
                # always has one).
                shot_groups = list(fin[S.GROUP_SHOTS].keys())
                if len(shot_groups) != 1:
                    raise ValueError(
                        f"{src_path}: expected exactly 1 shot group, "
                        f"got {len(shot_groups)}: {shot_groups}")
                src_shot = fin[f"{S.GROUP_SHOTS}/{shot_groups[0]}"]
                target_path = (
                    f"{S.GROUP_SHOTS}/{S.shot_key(target_id)}")
                fout.copy(src_shot, fout, name=target_path)
                # Ensure the @shot_id attr matches the new key (writers
                # may have stamped 0).
                fout[target_path].attrs[S.ATTR_SHOT_ID] = np.int32(target_id)

    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def _check_v2(f: h5py.File, ctx: str) -> None:
    fv = _decode(f.attrs.get(S.ATTR_FORMAT_VERSION, ""))
    if not fv:
        raise ValueError(f"{ctx}: missing root attr '{S.ATTR_FORMAT_VERSION}'")
    major = fv.split(".", 1)[0]
    if major != S.FORMAT_VERSION_MAJOR:
        raise ValueError(
            f"{ctx}: unsupported format_version '{fv}' (expected major "
            f"{S.FORMAT_VERSION_MAJOR})")
    if S.GROUP_SHOTS not in f:
        raise ValueError(f"{ctx}: missing /{S.GROUP_SHOTS} group")


def _check_root_attrs_match(f: h5py.File, expected: dict, ctx: str) -> None:
    for k, v in expected.items():
        if k == S.ATTR_FORMAT_VERSION:
            actual = _decode(f.attrs.get(k, ""))
            if actual.split(".", 1)[0] != v.split(".", 1)[0]:
                raise ValueError(
                    f"{ctx}: format_version major mismatch ({actual} vs {v})")
            continue
        if k in (S.ATTR_COORD_SYSTEM, S.ATTR_UNITS):
            actual = _decode(f.attrs.get(k, ""))
            if actual != v:
                raise ValueError(
                    f"{ctx}: {k} mismatch ({actual!r} vs {v!r})")
            continue
        actual = f.attrs.get(k)
        if actual is None:
            raise ValueError(f"{ctx}: missing root attr '{k}'")
        # Numeric comparison.
        if k in (S.ATTR_N_SAMPLES, S.ATTR_SPACE_DIM):
            actual_i = int(np.asarray(actual))
            if actual_i != v:
                raise ValueError(
                    f"{ctx}: {k} mismatch ({actual_i} vs {v}); "
                    "merging files with different time axes is forbidden (R2)")
            continue
        if abs(float(actual) - float(v)) > 0.0:
            raise ValueError(
                f"{ctx}: {k} mismatch ({actual} vs {v}) (R2)")
