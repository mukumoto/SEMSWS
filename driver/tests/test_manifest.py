"""Manifest write/read round trip tests."""

from __future__ import annotations

import json
from pathlib import Path

from semsws_driver.core.manifest import (
    Manifest, ShotEntry, utc_now_iso,
)


def test_begin_records_binary(tmp_path: Path):
    bin_ = tmp_path / "semsws"
    bin_.write_bytes(b"\x7fELF")
    m = Manifest.begin(semsws_binary=bin_)
    assert m.semsws_binary == str(bin_)
    assert m.started_at_utc != ""
    assert m.driver_version
    assert m.schema_version == "2.1"
    assert m.invocation, "should have captured sys.argv automatically"
    assert m.cwd, "should have captured cwd"


def test_begin_explicit_invocation(tmp_path: Path):
    bin_ = tmp_path / "semsws"
    bin_.write_bytes(b"\x7fELF")
    m = Manifest.begin(
        semsws_binary=bin_,
        invocation=["semsws-driver", "run", "--config", "x.yaml"],
        cwd=tmp_path,
    )
    assert m.invocation == ["semsws-driver", "run", "--config", "x.yaml"]
    assert m.cwd == str(tmp_path)


def test_finalize_and_write(tmp_path: Path):
    bin_ = tmp_path / "semsws"
    bin_.write_bytes(b"\x7fELF")
    m = Manifest.begin(
        semsws_binary=bin_,
        invocation=["semsws-driver", "run", "--workdir", "x"],
        cwd=tmp_path,
    )
    m.shots.append(ShotEntry(
        shot_id="0001",
        started_at_utc="2026-05-10T10:00:00+00:00",
        elapsed_seconds=12.3,
        return_code=0,
    ))
    m.finalize()
    out = m.write(tmp_path / "manifest.json")
    assert out.exists()
    blob = json.loads(out.read_text())
    assert blob["schema_version"] == "2.1"
    assert blob["finished_at_utc"] != ""
    assert blob["invocation"] == ["semsws-driver", "run", "--workdir", "x"]
    assert blob["cwd"] == str(tmp_path)
    s0 = blob["shots"][0]
    assert s0["shot_id"] == "0001"
    assert s0["started_at_utc"] == "2026-05-10T10:00:00+00:00"
    assert s0["return_code"] == 0
    assert s0["elapsed_seconds"] == 12.3


def test_no_sha256_keys(tmp_path: Path):
    """Regression: sha256 fields must not appear in the manifest schema."""
    bin_ = tmp_path / "semsws"
    bin_.write_bytes(b"\x7fELF")
    m = Manifest.begin(semsws_binary=bin_)
    m.shots.append(ShotEntry(shot_id="0001"))
    blob = m.to_dict()
    forbidden = {"config_yaml_sha256", "inputs_h5_sha256",
                 "semsws_binary_sha256", "seismograms_sha256"}
    assert not (set(blob.keys()) & forbidden)
    assert not (set(blob["shots"][0].keys()) & forbidden)


def test_utc_now_format():
    s = utc_now_iso()
    assert "T" in s
    assert s.endswith("+00:00")
