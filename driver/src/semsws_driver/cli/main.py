"""argparse-based CLI: `semsws-driver run|merge|dry-run`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .. import __version__
from ..core.layout import Layout
from ..core.run import run as core_run
from ..io.hdf5_v2 import merge_shot_outputs


def _parse_shot_ids(s: Optional[str]) -> Optional[list[int]]:
    if s is None:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def cmd_run(args: argparse.Namespace) -> int:
    res = core_run(
        config=args.config,
        inputs=args.inputs,
        workdir=args.workdir,
        merge=args.merge,
        only_shots=_parse_shot_ids(args.shots),
        dry_run=False,
        report_affinity=args.report_affinity,
    )
    failed = sum(1 for o in res.shot_outcomes if o.return_code != 0)
    if failed:
        print(f"FAILED: {failed}/{len(res.shot_outcomes)} shots", file=sys.stderr)
        return 1
    print(f"ok: {len(res.shot_outcomes)} shots → {res.workdir}")
    if res.merged_path:
        print(f"merged: {res.merged_path}")
    return 0


def cmd_affinity_report(args: argparse.Namespace) -> int:
    """Aggregate [AFF] lines from all shot stderr logs into a table.

    Flags only collisions between shots that overlapped in time (slot
    reuse after a shot finishes is OK and not a collision).
    """
    import json
    import re
    from datetime import datetime, timedelta

    workdir = Path(args.workdir).resolve()
    L = Layout.at(workdir)
    pat = re.compile(
        r"\[AFF\]\s+host=(?P<host>\S+)\s+rank=(?P<rank>\S+)\s+"
        r"local=(?P<local>\S+)\s+pid=(?P<pid>\S+)\s+"
        r"aff=(?P<aff>\S+)\s+psr=(?P<psr>\S+)"
    )
    rows: list[dict] = []
    for stderr in sorted(L.shots_dir.glob("shot_*/*.stderr.log")):
        shot_id = stderr.parent.name.split("_", 1)[1]
        for line in stderr.read_text(errors="replace").splitlines():
            m = pat.search(line)
            if m:
                d = m.groupdict()
                d["shot"] = shot_id
                rows.append(d)
    if not rows:
        print("no [AFF] lines found; run with --report-affinity",
              file=sys.stderr)
        return 1

    # Load per-shot timing from manifest.json if available.
    shot_window: dict[str, tuple[datetime, datetime]] = {}
    if L.manifest_path.exists():
        try:
            m = json.loads(L.manifest_path.read_text())
            for s in m.get("shots", []):
                t0 = s.get("started_at_utc")
                el = s.get("elapsed_seconds")
                if t0 and el is not None:
                    start = datetime.fromisoformat(t0)
                    shot_window[str(s["shot_id"])] = (
                        start, start + timedelta(seconds=float(el)))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # Per-shot time windows + cpu set summary.
    if shot_window:
        print(f"{'shot':>6}  {'started_at_utc':<28}  {'elapsed(s)':>10}  "
              f"{'finished_at_utc':<28}")
        print("-" * 80)
        for sid in sorted(shot_window.keys()):
            start, end = shot_window[sid]
            elapsed = (end - start).total_seconds()
            print(f"{sid:>6}  {start.isoformat():<28}  {elapsed:>10.3f}  "
                  f"{end.isoformat():<28}")
        print()

    # Per-rank table.
    print(f"{'shot':>6}  {'rank':>4}  {'local':>5}  {'host':<20}  "
          f"{'aff':<16}  {'psr':>4}")
    print("-" * 70)
    for r in sorted(rows, key=lambda r: (r["shot"], int(r["rank"])
                                        if r["rank"].isdigit() else -1)):
        print(f"{r['shot']:>6}  {r['rank']:>4}  {r['local']:>5}  "
              f"{r['host']:<20}  {r['aff']:<16}  {r['psr']:>4}")

    # Per-shot CPU set per host.
    shot_cpus: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        key = (r["host"], r["shot"])
        shot_cpus.setdefault(key, set()).update(r["aff"].split(","))

    # Collision = same host, shared cpu, AND time windows overlap.
    def overlap(a: tuple[datetime, datetime],
                b: tuple[datetime, datetime]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    keys = sorted(shot_cpus.keys())
    collisions = 0
    for i, (host_a, shot_a) in enumerate(keys):
        for host_b, shot_b in keys[i + 1:]:
            if host_a != host_b or shot_a == shot_b:
                continue
            shared = shot_cpus[(host_a, shot_a)] & shot_cpus[(host_b, shot_b)]
            if not shared:
                continue
            wa = shot_window.get(shot_a)
            wb = shot_window.get(shot_b)
            if wa and wb and not overlap(wa, wb):
                continue   # slot reuse, not a collision
            print(f"COLLISION: host={host_a} shots {shot_a} and {shot_b} "
                  f"share cpus {sorted(shared)}", file=sys.stderr)
            collisions += 1

    if collisions:
        return 2
    n_shots = len({r["shot"] for r in rows})
    print(f"\n{len(rows)} rank records across {n_shots} shots; "
          f"no time-overlapping collisions.")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    res = core_run(
        config=args.config,
        inputs=args.inputs,
        workdir=args.workdir,
        dry_run=True,
        only_shots=_parse_shot_ids(args.shots),
    )
    print(f"dry-run: {res.workdir}")
    print(f"  config copy: {res.layout.config_copy_path}")
    print(f"  inputs:      {res.layout.inputs_h5}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    L = Layout.at(workdir)
    seis = sorted(L.shots_dir.glob("shot_*/seismograms.h5"))
    if not seis:
        print(f"no seismograms found under {L.shots_dir}", file=sys.stderr)
        return 1
    out = Path(args.output) if args.output else L.merged_path
    out.parent.mkdir(parents=True, exist_ok=True)
    shot_ids = [int(p.parent.name.split("_", 1)[1]) for p in seis]
    merge_shot_outputs(sources=seis, output=out, output_shot_ids=shot_ids)
    print(f"merged → {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="semsws-driver",
        description="Thin run-only layer for SEMSWS large-scale runs.",
    )
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v: INFO, -vv: DEBUG")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run all shots end-to-end")
    pr.add_argument("--config", required=True,
                    help="user YAML template (with `run:` section)")
    pr.add_argument("--inputs", required=True,
                    help="bundled v2.0 HDF5 (sources + receivers + observed)")
    pr.add_argument("--workdir", required=True,
                    help="output directory (will be created)")
    pr.add_argument("--merge", action="store_true",
                    help="emit results/merged.h5 after all shots succeed")
    pr.add_argument("--shots", default=None,
                    help="comma-separated subset of shot ids (defaults: all)")
    pr.add_argument("--report-affinity", action="store_true",
                    help="inject a rank wrapper that prints each rank's CPU "
                         "affinity to its shot's stderr log; aggregate with "
                         "`semsws-driver affinity-report --workdir <path>`")
    pr.set_defaults(func=cmd_run)

    pd = sub.add_parser("dry-run",
                        help="generate per-shot YAMLs, do not invoke runner")
    pd.add_argument("--config", required=True)
    pd.add_argument("--inputs", required=True)
    pd.add_argument("--workdir", required=True)
    pd.add_argument("--shots", default=None)
    pd.set_defaults(func=cmd_dry_run)

    pm = sub.add_parser("merge", help="merge per-shot seismograms.h5 files")
    pm.add_argument("--workdir", required=True,
                    help="run workdir containing shots/shot_NNNN/seismograms.h5")
    pm.add_argument("-o", "--output", default=None,
                    help="output path (default: <workdir>/results/merged.h5)")
    pm.set_defaults(func=cmd_merge)

    pa = sub.add_parser("affinity-report",
                        help="parse [AFF] lines from a workdir and print "
                             "a per-rank CPU affinity table (use after "
                             "`run --report-affinity`)")
    pa.add_argument("--workdir", required=True)
    pa.set_defaults(func=cmd_affinity_report)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
