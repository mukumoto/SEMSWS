"""Layout path-builder tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from semsws_driver.core.layout import Layout


def test_at_resolves_absolute(tmp_path: Path):
    L = Layout.at(tmp_path / "x")
    assert L.workdir.is_absolute()
    assert L.workdir == (tmp_path / "x").resolve()


def test_top_level_paths(tmp_path: Path):
    L = Layout.at(tmp_path)
    assert L.manifest_path == tmp_path / "manifest.json"
    assert L.config_copy_path == tmp_path / "config.yaml"
    assert L.inputs_dir == tmp_path / "inputs"
    assert L.inputs_h5 == tmp_path / "inputs" / "observations.h5"
    assert L.shots_dir == tmp_path / "shots"
    assert L.results_dir == tmp_path / "results"
    assert L.merged_path == tmp_path / "results" / "merged.h5"
    assert L.shared_mesh_partitions_dir == tmp_path / "_shared_mesh" / "partitions"
    assert L.shared_model_bp_dir == tmp_path / "_shared_model" / "bp"


def test_shot_paths(tmp_path: Path):
    L = Layout.at(tmp_path)
    assert L.shot_key(0) == "shot_0000"
    assert L.shot_key(7) == "shot_0007"
    assert L.shot_key(9999) == "shot_9999"
    assert L.shot_dir(7) == tmp_path / "shots" / "shot_0007"
    assert L.shot_config(7) == tmp_path / "shots" / "shot_0007" / "config.yaml"
    assert L.shot_seismograms(7) == tmp_path / "shots" / "shot_0007" / "seismograms.h5"
    assert L.shot_stdout(7) == tmp_path / "shots" / "shot_0007" / "stdout.log"


def test_negative_shot_id_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        Layout.shot_key(-1)


def test_make_skeleton(tmp_path: Path):
    L = Layout.at(tmp_path)
    L.make_skeleton()
    assert L.workdir.is_dir()
    assert L.inputs_dir.is_dir()
    assert L.shots_dir.is_dir()
    assert L.results_dir.is_dir()
