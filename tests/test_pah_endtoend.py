"""
Slow end-to-end smoke test: synthetic maps → real stacking → PAH deconvolution.

Validates the flux-level simulator's shortcut once against the actual
SimstackAlgorithm pipeline (PSF injection, layer-matrix solve, bootstrap
errors). Run with: uv run pytest tests/test_pah_endtoend.py -m slow
"""

import numpy as np
import pytest


@pytest.mark.slow
def test_endtoend_smoke(tmp_path):
    from simstack4.scripts.pah_dither_endtoend import run_endtoend

    out = run_endtoend(tmp_path / "e2e", quick=True, seed=42, verbose=False)
    result = out["result"]
    truth = out["truth"]

    # pipeline plumbed through: all dither bins produced finite fluxes
    df = out["df"]
    assert len(df) > 0
    assert np.isfinite(df["MIPS_24"]).all()

    # the strongest group (7.7+8.6) recovers within 50% (smoke tolerance);
    # the warm temperature lands inside its bounds near the injected value
    A_true = truth.amplitudes()
    rel = abs(result.A[0][1] / A_true[1] - 1.0)
    assert rel < 0.5, f"7.7+8.6 recovery off by {rel:.1%}"
    assert abs(result.theta_global[0] - truth.T_warm) < 15.0
