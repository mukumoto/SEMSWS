"""Round-trip tests for the driver-side v2.0 HDF5 writer and merge tool."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from semsws_driver.io import hdf5_schema as S
from semsws_driver.io.hdf5_v2 import (
    ReceiverEntry, ShotEntry, SourceEntry, merge_shot_outputs, write_v2,
)


def _decode(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


@pytest.fixture
def base_args():
    return dict(dt=1e-3, n_samples=8, space_dim=2)


def _make_force_shot(shot_id: int, base_args, with_receiver_data: bool):
    nt = base_args["n_samples"]
    sd = base_args["space_dim"]
    src = SourceEntry(
        source_id=1,
        type=S.SOURCE_TYPE_FORCE,
        position=[10.0, 20.0],
        direction=[0.0, -1.0],
        stf=np.linspace(0.0, 1.0, nt),
        label="my_source",
    )
    rec = ReceiverEntry(
        receiver_id=1,
        position=[100.0, 0.0],
        label="STA1",
    )
    if with_receiver_data:
        rec.channels[S.CHANNEL_VELOCITY] = (
            np.arange(sd * nt, dtype=np.float32).reshape(sd, nt))
        rec.weights[S.CHANNEL_VELOCITY] = (
            np.ones((sd, nt), dtype=np.float32))
    return ShotEntry(shot_id=shot_id, sources=[src], receivers=[rec])


def test_write_force_geometry_only(tmp_path: Path, base_args):
    path = tmp_path / "geo.h5"
    shot = _make_force_shot(0, base_args, with_receiver_data=False)
    write_v2(path, [shot], **base_args)

    with h5py.File(path, "r") as f:
        assert _decode(f.attrs[S.ATTR_FORMAT_VERSION]) == "2.0"
        assert int(f.attrs[S.ATTR_N_SAMPLES]) == 8
        assert int(f.attrs[S.ATTR_SPACE_DIM]) == 2
        s = f[f"{S.GROUP_SHOTS}/0000"]
        assert int(s.attrs[S.ATTR_SHOT_ID]) == 0
        src = s[f"{S.GROUP_SOURCES}/S0001"]
        assert _decode(src.attrs[S.ATTR_TYPE]) == "force"
        assert int(src.attrs[S.ATTR_ID]) == 1
        np.testing.assert_array_equal(
            np.asarray(src.attrs[S.ATTR_DIRECTION]), [0.0, -1.0])
        np.testing.assert_array_equal(
            np.asarray(src["stf"]), np.linspace(0.0, 1.0, 8))
        rec = s[f"{S.GROUP_RECEIVERS}/R0001"]
        assert "VEL" not in rec  # geometry-only mode


def test_write_observed_with_weights(tmp_path: Path, base_args):
    path = tmp_path / "obs.h5"
    shot = _make_force_shot(0, base_args, with_receiver_data=True)
    write_v2(path, [shot], **base_args)
    with h5py.File(path, "r") as f:
        rec = f[f"{S.GROUP_SHOTS}/0000/{S.GROUP_RECEIVERS}/R0001"]
        assert "VEL" in rec
        assert "weight_VEL" in rec
        assert rec["VEL"].shape == (2, 8)


def test_write_moment_tensor_3d(tmp_path: Path):
    path = tmp_path / "mt.h5"
    nt = 4
    src = SourceEntry(
        source_id=42,
        type=S.SOURCE_TYPE_MOMENT_TENSOR,
        position=[0.0, 0.0, 0.0],
        moment_tensor=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        stf=np.zeros(nt),
    )
    write_v2(path, [ShotEntry(0, sources=[src])],
             dt=1e-3, n_samples=nt, space_dim=3)
    with h5py.File(path, "r") as f:
        ds = f[f"{S.GROUP_SHOTS}/0000/{S.GROUP_SOURCES}/S0042/moment_tensor"]
        np.testing.assert_array_equal(np.asarray(ds), [1, 2, 3, 4, 5, 6])
        order = [_decode(s) for s in
                 np.asarray(ds.attrs[S.ATTR_COMPONENT_ORDER])]
        assert order == ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]


def test_write_rejects_force_without_direction(tmp_path: Path, base_args):
    src = SourceEntry(source_id=1, type=S.SOURCE_TYPE_FORCE,
                      position=[0.0, 0.0],
                      stf=np.zeros(base_args["n_samples"]))
    with pytest.raises(ValueError, match="force direction"):
        write_v2(tmp_path / "x.h5",
                 [ShotEntry(0, sources=[src])], **base_args)


def test_write_rejects_bad_position_size(tmp_path: Path, base_args):
    src = SourceEntry(source_id=1, type=S.SOURCE_TYPE_PRESSURE,
                      position=[0.0, 0.0, 0.0],   # 3D in 2D file
                      stf=np.zeros(base_args["n_samples"]))
    with pytest.raises(ValueError, match="source position"):
        write_v2(tmp_path / "x.h5",
                 [ShotEntry(0, sources=[src])], **base_args)


def test_write_rejects_bad_stf_length(tmp_path: Path, base_args):
    src = SourceEntry(source_id=1, type=S.SOURCE_TYPE_PRESSURE,
                      position=[0.0, 0.0],
                      stf=np.zeros(base_args["n_samples"] + 1))
    with pytest.raises(ValueError, match="source stf"):
        write_v2(tmp_path / "x.h5",
                 [ShotEntry(0, sources=[src])], **base_args)


def test_write_rejects_unknown_channel(tmp_path: Path, base_args):
    nt = base_args["n_samples"]
    rec = ReceiverEntry(
        receiver_id=1, position=[0.0, 0.0],
        channels={"GRAD": np.zeros((2, nt), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="unknown channel"):
        write_v2(tmp_path / "x.h5",
                 [ShotEntry(0, receivers=[rec])], **base_args)


def test_write_rejects_duplicate_source_ids(tmp_path: Path, base_args):
    a = SourceEntry(source_id=1, type=S.SOURCE_TYPE_PRESSURE,
                    position=[0.0, 0.0],
                    stf=np.zeros(base_args["n_samples"]))
    b = SourceEntry(source_id=1, type=S.SOURCE_TYPE_PRESSURE,
                    position=[1.0, 1.0],
                    stf=np.zeros(base_args["n_samples"]))
    with pytest.raises(ValueError, match="duplicate source"):
        write_v2(tmp_path / "x.h5",
                 [ShotEntry(0, sources=[a, b])], **base_args)


def test_merge_two_files_round_trip(tmp_path: Path, base_args):
    p1 = tmp_path / "a.h5"
    p2 = tmp_path / "b.h5"
    write_v2(p1, [_make_force_shot(0, base_args, True)], **base_args)
    write_v2(p2, [_make_force_shot(0, base_args, True)], **base_args)

    out = tmp_path / "merged.h5"
    merge_shot_outputs([p1, p2], out)

    with h5py.File(out, "r") as f:
        assert int(f.attrs[S.ATTR_N_SAMPLES]) == base_args["n_samples"]
        assert "0000" in f[S.GROUP_SHOTS]
        assert "0001" in f[S.GROUP_SHOTS]
        # @shot_id should reflect the new keys (0 and 1), even though the
        # input files both stamped 0.
        assert int(f[f"{S.GROUP_SHOTS}/0000"].attrs[S.ATTR_SHOT_ID]) == 0
        assert int(f[f"{S.GROUP_SHOTS}/0001"].attrs[S.ATTR_SHOT_ID]) == 1


def test_merge_rejects_dt_mismatch(tmp_path: Path, base_args):
    p1 = tmp_path / "a.h5"
    p2 = tmp_path / "b.h5"
    write_v2(p1, [_make_force_shot(0, base_args, False)], **base_args)
    write_v2(p2, [_make_force_shot(0, base_args, False)],
             **{**base_args, "dt": 2e-3})
    out = tmp_path / "merged.h5"
    with pytest.raises(ValueError, match="dt mismatch|R2"):
        merge_shot_outputs([p1, p2], out)


def test_merge_custom_shot_ids(tmp_path: Path, base_args):
    p1 = tmp_path / "a.h5"
    p2 = tmp_path / "b.h5"
    write_v2(p1, [_make_force_shot(0, base_args, False)], **base_args)
    write_v2(p2, [_make_force_shot(0, base_args, False)], **base_args)
    out = tmp_path / "merged.h5"
    merge_shot_outputs([p1, p2], out, output_shot_ids=[5, 23])
    with h5py.File(out, "r") as f:
        assert "0005" in f[S.GROUP_SHOTS]
        assert "0023" in f[S.GROUP_SHOTS]
        assert int(f[f"{S.GROUP_SHOTS}/0023"].attrs[S.ATTR_SHOT_ID]) == 23
