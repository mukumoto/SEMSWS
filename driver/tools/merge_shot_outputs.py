#!/usr/bin/env python3
"""Merge per-shot SEMSWS HDF5 outputs into a single bundled v2.0 file.

Usage:
    python -m semsws_driver.tools.merge_shot_outputs \
        --output observations.h5 \
        seis0001.h5 seis0002.h5 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make this script runnable both as `python -m driver.tools.merge_shot_outputs`
# and as a direct script on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsws_driver.io.hdf5_v2 import merge_shot_outputs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="per-shot HDF5 files (v2.0)")
    ap.add_argument("--output", "-o", required=True,
                    help="path for the merged file")
    ap.add_argument("--shot-ids", default=None,
                    help="comma-separated shot ids matching `inputs` "
                         "(default: 0..N-1)")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="fail if --output already exists")
    args = ap.parse_args()

    shot_ids = None
    if args.shot_ids is not None:
        shot_ids = [int(s) for s in args.shot_ids.split(",")]

    out = merge_shot_outputs(
        sources=args.inputs,
        output=args.output,
        output_shot_ids=shot_ids,
        overwrite=not args.no_overwrite,
    )
    print(f"merged → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
