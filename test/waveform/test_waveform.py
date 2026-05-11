"""
Waveform validation tests for SEMSWS.

Compares simulation output against reference waveforms (e.g., SPECFEM)
using normalized L2 norm.

Tests are automatically discovered from test/waveform/{2D,3D}/{acoustic,elastic,...}/
"""

import pytest
import shutil
from pathlib import Path

from scripts.compare_waveforms import run_comparison


class TestWaveformValidation:
    """Waveform validation test suite."""

    def test_waveform_match(
        self,
        waveform_test_case: tuple[str, Path, Path],
        run_simulation,
        threshold: float,
        keep_results: bool
    ):
        """
        Test that simulation output matches reference waveforms.

        Args:
            waveform_test_case: (test_id, config_path, ref_dir)
            run_simulation: Fixture to run simulations
            threshold: L2 error threshold
            keep_results: Whether to keep results after test
        """
        test_id, config_path, ref_dir = waveform_test_case
        test_dir = config_path.parent
        results_dir = test_dir / "results"

        # Clean previous results
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(exist_ok=True)

        # Run simulation
        success, error_msg = run_simulation(config_path)
        assert success, f"Simulation failed for {test_id}:\n{error_msg}"

        # Compare waveforms
        all_passed, comparison_results = run_comparison(
            result_dir=results_dir,
            ref_dir=ref_dir,
            threshold=threshold,
            verbose=True
        )

        # Clean up if not keeping results
        if not keep_results and results_dir.exists():
            shutil.rmtree(results_dir)

        # Build detailed error message if failed
        if not all_passed:
            failed = [r for r in comparison_results if not r.passed]
            error_details = "\n".join(
                f"  {r.receiver_name}: L2 error = {r.l2_error:.4f} (threshold: {threshold})"
                for r in failed
            )
            pytest.fail(f"Waveform mismatch for {test_id}:\n{error_details}")
