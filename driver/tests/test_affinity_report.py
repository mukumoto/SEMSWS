"""Smoke tests for the --report-affinity wrapper and affinity-report CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from semsws_driver.cli.main import main
from semsws_driver.scripts import script_path


def test_bundled_script_resolves():
    p = script_path("report_affinity.sh")
    assert p.exists(), f"bundled script missing: {p}"
    assert p.stat().st_mode & 0o111, "bundled script must be executable"


def test_bundled_script_runs_standalone(tmp_path):
    """Script should run as a probe and emit an [AFF] line."""
    import subprocess
    p = script_path("report_affinity.sh")
    proc = subprocess.run([str(p)], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0
    assert "[AFF]" in (proc.stdout + proc.stderr)


def test_affinity_report_no_logs(tmp_path):
    """Running affinity-report on a workdir with no [AFF] lines fails."""
    L_workdir = tmp_path / "work"
    (L_workdir / "shots" / "shot_0000").mkdir(parents=True)
    (L_workdir / "shots" / "shot_0000" / "shot_0000.stderr.log").write_text("")
    rc = main(["affinity-report", "--workdir", str(L_workdir)])
    assert rc == 1


def test_affinity_report_detects_collision(tmp_path, capsys):
    """Two shots overlapping in time on the same cpu → COLLISION reported."""
    work = tmp_path / "work"
    for sid in ("0000", "0001"):
        d = work / "shots" / f"shot_{sid}"
        d.mkdir(parents=True)
        (d / f"shot_{sid}.stderr.log").write_text(
            f"[AFF] host=H rank=0 local=0 pid=1{sid} aff=0 psr=0\n"
        )
    # Manifest with overlapping shots (0000 and 0001 both run 10:00-10:01).
    (work / "manifest.json").write_text(json.dumps({
        "shots": [
            {"shot_id": "0000",
             "started_at_utc": "2026-05-11T10:00:00+00:00",
             "elapsed_seconds": 60.0, "return_code": 0},
            {"shot_id": "0001",
             "started_at_utc": "2026-05-11T10:00:30+00:00",
             "elapsed_seconds": 60.0, "return_code": 0},
        ],
    }))
    rc = main(["affinity-report", "--workdir", str(work)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "COLLISION" in captured.err


def test_affinity_report_slot_reuse_is_ok(tmp_path, capsys):
    """Two shots sharing a cpu but with disjoint time windows → OK."""
    work = tmp_path / "work"
    for sid in ("0000", "0001"):
        d = work / "shots" / f"shot_{sid}"
        d.mkdir(parents=True)
        (d / f"shot_{sid}.stderr.log").write_text(
            f"[AFF] host=H rank=0 local=0 pid=1{sid} aff=0 psr=0\n"
        )
    # Sequential, not overlapping.
    (work / "manifest.json").write_text(json.dumps({
        "shots": [
            {"shot_id": "0000",
             "started_at_utc": "2026-05-11T10:00:00+00:00",
             "elapsed_seconds": 10.0, "return_code": 0},
            {"shot_id": "0001",
             "started_at_utc": "2026-05-11T10:00:30+00:00",
             "elapsed_seconds": 10.0, "return_code": 0},
        ],
    }))
    rc = main(["affinity-report", "--workdir", str(work)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "no time-overlapping collisions" in captured.out
