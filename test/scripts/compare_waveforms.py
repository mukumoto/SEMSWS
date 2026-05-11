#!/usr/bin/env python3
"""
Waveform comparison tool for SEMSWS validation tests.

Compares simulation output against reference waveforms (e.g., SPECFEM)
using normalized L2 norm: ||result - ref|| / ||ref||
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Waveform:
    """Time series waveform data."""
    time: np.ndarray
    data: np.ndarray
    name: str = ""

    @property
    def dt(self) -> float:
        """Time step."""
        return self.time[1] - self.time[0] if len(self.time) > 1 else 0.0

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return len(self.time)


def read_sem_ascii(filepath: Path) -> Waveform:
    """
    Read an ASCII seismogram (SEMSWS or SPECFEM).
    Format: two columns (time, amplitude) separated by whitespace.
    """
    data = np.loadtxt(filepath)
    return Waveform(
        time=data[:, 0],
        data=data[:, 1],
        name=filepath.stem
    )


def normalized_l2_error(result: Waveform, reference: Waveform) -> float:
    """
    Compute normalized L2 error: ||result - ref|| / ||ref||

    Returns:
        float: Normalized L2 error (0 = perfect match)
    """
    ref_norm = np.linalg.norm(reference.data)
    if ref_norm < 1e-30:
        # Reference is essentially zero
        return np.linalg.norm(result.data)

    return np.linalg.norm(result.data - reference.data) / ref_norm


def _parse_ref_specfem(ref_file: Path) -> tuple[str, str | None, str]:
    """Parse SPECFEM-style ref filename: AA.{receiver}.{component}.{ext}"""
    parts = ref_file.stem.split('.')
    if len(parts) >= 3:
        receiver_name = parts[1]
        component = parts[2]
    elif len(parts) >= 2:
        receiver_name = parts[1]
        component = None
    else:
        receiver_name = ref_file.stem
        component = None

    # Map SPECFEM component to SEMSWS component
    component_map = {
        # 2D components
        'BXX': 'x', 'BXY': 'y',
        'BXZ': 'y',  # In 2D, Z is the second component (y in SEMSWS)
        # 3D components (force)
        'FXX': 'x', 'FXY': 'y', 'FXZ': 'z',
        # Acoustic
        'PRE': None,
    }
    sem_component = component_map.get(component) if component else None

    # Map SPECFEM extension to SEMSWS extension
    ext_map = {'.semd': '.d', '.semv': '.v', '.sema': '.a', '.semp': '.p'}
    sem_ext = ext_map.get(ref_file.suffix, '.d')

    return receiver_name, sem_component, sem_ext


def _parse_ref_native(ref_file: Path) -> tuple[str, str | None, str]:
    """Parse SEMSWS native ref filename: {receiver}_{component}.{ext}
    Examples: R1_x.d, R01_y.d, R001.p"""
    stem = ref_file.stem  # e.g. "R1_x"
    ext = ref_file.suffix  # e.g. ".d"

    parts = stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in ('x', 'y', 'z'):
        return parts[0], parts[1], ext
    else:
        # Acoustic or no component: R001.p
        return stem, None, ext


def find_matching_files(
    result_dir: Path,
    ref_dir: Path,
    result_pattern: str,
    ref_pattern: str
) -> list[tuple[Path, Path]]:
    """
    Find matching result and reference files based on receiver name.

    Supports reference files in two formats:
    - SPECFEM: AA.R001.PRE.semp, AA.REC00.BXX.semd
    - SEMSWS native: R1_x.d, R01_y.d, R001.p

    Returns:
        List of (result_file, reference_file) tuples
    """
    matches = []

    # Find all reference files (SPECFEM and native formats)
    ref_files = list(ref_dir.glob("*.sem*")) + list(ref_dir.glob("*.d")) + \
                list(ref_dir.glob("*.v")) + list(ref_dir.glob("*.p"))

    for ref_file in ref_files:
        # Detect format and parse
        if '.' in ref_file.stem:
            # SPECFEM format: dots in stem (e.g. AA.R001.BXX)
            receiver_name, sem_component, sem_ext = _parse_ref_specfem(ref_file)
        else:
            # SEMSWS native format (e.g. R1_x)
            receiver_name, sem_component, sem_ext = _parse_ref_native(ref_file)

        # Build candidate result file names
        result_candidates = []

        if sem_component:
            # Elastic case: R1_x_0000.d or R1_x.d
            result_candidates.extend([
                result_dir / f"{receiver_name}_{sem_component}_0000{sem_ext}",
                result_dir / f"{receiver_name}_{sem_component}{sem_ext}",
            ])
        else:
            # Acoustic case: R001_0000.p
            result_candidates.extend([
                result_dir / f"{receiver_name}_0000{sem_ext}",
                result_dir / f"{receiver_name}{sem_ext}",
            ])

        # Try each candidate
        for candidate in result_candidates:
            if candidate.exists():
                matches.append((candidate, ref_file))
                break
        else:
            # Fallback: glob pattern search
            if sem_component:
                pattern_matches = list(result_dir.glob(f"{receiver_name}_{sem_component}_*{sem_ext}"))
            else:
                pattern_matches = list(result_dir.glob(f"{receiver_name}_*{sem_ext}"))
            if pattern_matches:
                matches.append((pattern_matches[0], ref_file))

    return matches


@dataclass
class ComparisonResult:
    """Result of comparing two waveforms."""
    receiver_name: str
    l2_error: float
    passed: bool
    result_file: Path
    reference_file: Path


def compare_single(
    result_file: Path,
    ref_file: Path,
    threshold: float,
    result_format: str = "sem",
    ref_format: str = "specfem"
) -> ComparisonResult:
    """Compare a single pair of waveforms."""

    # Read waveforms
    if result_format == "sem":
        result_wf = read_sem_ascii(result_file)
    else:
        result_wf = read_sem_ascii(result_file)

    if ref_format == "specfem":
        ref_wf = read_sem_ascii(ref_file)
    else:
        ref_wf = read_sem_ascii(ref_file)

    # Compare data directly (ignore time axis)
    # Require same number of samples
    if len(result_wf.data) != len(ref_wf.data):
        raise ValueError(
            f"Sample count mismatch: result={len(result_wf.data)}, ref={len(ref_wf.data)}"
        )

    # Compute error using only amplitude data
    error = normalized_l2_error(result_wf, ref_wf)

    # Extract receiver name with component (e.g., REC00.BXX or R001.PRE)
    parts = ref_file.stem.split('.')
    if len(parts) >= 3:
        receiver_name = f"{parts[1]}.{parts[2]}"  # REC00.BXX
    elif len(parts) >= 2:
        receiver_name = parts[1]
    else:
        receiver_name = ref_file.stem

    return ComparisonResult(
        receiver_name=receiver_name,
        l2_error=error,
        passed=error <= threshold,
        result_file=result_file,
        reference_file=ref_file
    )


def run_comparison(
    result_dir: Path,
    ref_dir: Path,
    threshold: float = 0.05,
    result_pattern: str = "*",
    ref_pattern: str = "*.sem*",  # Supports SPECFEM (*.semd) and native (*.d) formats
    verbose: bool = True
) -> tuple[bool, list[ComparisonResult]]:
    """
    Run waveform comparison for all matching files.

    Returns:
        (all_passed, results): Overall pass/fail and individual results
    """
    # Find matching files
    matches = find_matching_files(result_dir, ref_dir, result_pattern, ref_pattern)

    if not matches:
        print(f"ERROR: No matching files found")
        print(f"  Result dir: {result_dir}")
        print(f"  Reference dir: {ref_dir}")
        return False, []

    results = []
    all_passed = True

    if verbose:
        print(f"\nComparing {len(matches)} waveform pairs (threshold: {threshold*100:.1f}%)")
        print("-" * 60)

    for result_file, ref_file in matches:
        comparison = compare_single(result_file, ref_file, threshold)
        results.append(comparison)

        if not comparison.passed:
            all_passed = False

        if verbose:
            status = "PASS" if comparison.passed else "FAIL"
            print(f"  {comparison.receiver_name}: L2 error = {comparison.l2_error:.4f} [{status}]")

    if verbose:
        print("-" * 60)
        overall = "PASSED" if all_passed else "FAILED"
        print(f"Overall: {overall} ({sum(r.passed for r in results)}/{len(results)} passed)")

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(
        description="Compare SEMSWS waveforms against reference (e.g., SPECFEM)"
    )
    parser.add_argument(
        "--results", "-r",
        type=Path,
        required=True,
        help="Directory containing SEMSWS output waveforms"
    )
    parser.add_argument(
        "--reference", "-R",
        type=Path,
        required=True,
        help="Directory containing reference waveforms"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.05,
        help="Normalized L2 error threshold (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--ref-pattern",
        type=str,
        default="*.semp",
        help="Glob pattern for reference files (default: *.semp)"
    )
    parser.add_argument(
        "--result-pattern",
        type=str,
        default="*.txt",
        help="Glob pattern for result files (default: *.txt)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results directory not found: {args.results}")
        sys.exit(1)

    if not args.reference.exists():
        print(f"ERROR: Reference directory not found: {args.reference}")
        sys.exit(1)

    all_passed, results = run_comparison(
        result_dir=args.results,
        ref_dir=args.reference,
        threshold=args.threshold,
        ref_pattern=args.ref_pattern,
        result_pattern=args.result_pattern,
        verbose=not args.quiet
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
