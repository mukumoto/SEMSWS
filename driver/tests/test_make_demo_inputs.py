"""Smoke test for examples/make_demo_inputs.py: CSV → v2.0 HDF5."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from semsws_driver.io import hdf5_schema as S

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "make_demo_inputs.py"


@pytest.fixture(scope="module")
def make_demo():
    """Load the example script as a module so we can call its functions."""
    spec = importlib.util.spec_from_file_location("make_demo_inputs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_demo_inputs"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_csvs(tmp_path: Path, *, multi_shot: bool):
    src = tmp_path / "src.csv"
    rcv = tmp_path / "rcv.csv"
    src.write_text("shot_id,source_id,type,x,y,freq,delay\n"
                    "0,1,pressure,500.0,500.0,10.0,0.1\n" +
                    ("1,1,pressure,700.0,500.0,10.0,0.1\n"
                     if multi_shot else ""))
    rcv.write_text("shot_id,receiver_id,x,y\n"
                    "0,1,600.0,500.0\n"
                    "0,2,700.0,500.0\n" +
                    ("1,1,800.0,500.0\n" if multi_shot else ""))
    return src, rcv


def test_ricker_samples_shape(make_demo):
    samples = make_demo.ricker_samples(n=100, dt=1e-3, freq=10.0, delay=0.1)
    assert samples.shape == (100,)
    assert samples.dtype == np.float64
    # peak roughly at the delay sample
    assert samples.argmax() in range(95, 105)


def test_build_shots_single_shot(make_demo, tmp_path):
    src, rcv = _write_csvs(tmp_path, multi_shot=False)
    shots = make_demo.build_shots(
        sources_csv=src, receivers_csv=rcv,
        dt=1e-3, n_samples=100, space_dim=2,
    )
    assert len(shots) == 1
    assert shots[0].shot_id == 0
    assert len(shots[0].sources) == 1
    assert len(shots[0].receivers) == 2
    assert shots[0].sources[0].type == S.SOURCE_TYPE_PRESSURE
    assert shots[0].sources[0].stf.shape == (100,)


def test_build_shots_multi_shot(make_demo, tmp_path):
    src, rcv = _write_csvs(tmp_path, multi_shot=True)
    shots = make_demo.build_shots(
        sources_csv=src, receivers_csv=rcv,
        dt=1e-3, n_samples=100, space_dim=2,
    )
    assert [s.shot_id for s in shots] == [0, 1]
    assert len(shots[0].receivers) == 2
    assert len(shots[1].receivers) == 1


def test_main_writes_valid_v2(make_demo, tmp_path):
    src, rcv = _write_csvs(tmp_path, multi_shot=True)
    out = tmp_path / "obs.h5"
    rc = make_demo.main([
        "--sources", str(src),
        "--receivers", str(rcv),
        "--output", str(out),
        "--dt", "1e-3", "--n-samples", "100", "--space-dim", "2",
    ])
    assert rc == 0
    assert out.exists()
    with h5py.File(str(out), "r") as f:
        fv = f.attrs[S.ATTR_FORMAT_VERSION]
        if isinstance(fv, bytes):
            fv = fv.decode()
        assert str(fv) == "2.0"
        assert int(np.asarray(f.attrs[S.ATTR_N_SAMPLES])) == 100
        assert "0000" in f[S.GROUP_SHOTS]
        assert "0001" in f[S.GROUP_SHOTS]


def test_unknown_type_rejected(make_demo, tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("shot_id,source_id,type,x,y,freq,delay\n"
                    "0,1,explosion,500.0,500.0,10.0,0.1\n")
    rcv = tmp_path / "rcv.csv"
    rcv.write_text("shot_id,receiver_id,x,y\n0,1,600.0,500.0\n")
    with pytest.raises(ValueError, match="unknown type"):
        make_demo.build_shots(
            sources_csv=src, receivers_csv=rcv,
            dt=1e-3, n_samples=100, space_dim=2,
        )


def test_force_with_direction(make_demo, tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("shot_id,source_id,type,x,y,freq,delay,dx,dy\n"
                    "0,1,force,500.0,500.0,10.0,0.1,0.0,1.0\n")
    rcv = tmp_path / "rcv.csv"
    rcv.write_text("shot_id,receiver_id,x,y\n0,1,600.0,500.0\n")
    shots = make_demo.build_shots(
        sources_csv=src, receivers_csv=rcv,
        dt=1e-3, n_samples=100, space_dim=2,
    )
    assert shots[0].sources[0].type == S.SOURCE_TYPE_FORCE
    assert list(shots[0].sources[0].direction) == [0.0, 1.0]


def test_moment_tensor_2d(make_demo, tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("shot_id,source_id,type,x,y,freq,delay,mxx,myy,mxy\n"
                    "0,1,moment_tensor,500.0,500.0,10.0,0.1,1.0,2.0,3.0\n")
    rcv = tmp_path / "rcv.csv"
    rcv.write_text("shot_id,receiver_id,x,y\n0,1,600.0,500.0\n")
    shots = make_demo.build_shots(
        sources_csv=src, receivers_csv=rcv,
        dt=1e-3, n_samples=100, space_dim=2,
    )
    assert shots[0].sources[0].type == S.SOURCE_TYPE_MOMENT_TENSOR
    assert list(shots[0].sources[0].moment_tensor) == [1.0, 2.0, 3.0]
