"""
Tests for the compute-cost preflight estimator (simstack4.preflight).

Reuses the synthetic dataset builder from test_integration.py (catalog,
FITS maps, TOML config with known population/map counts) so the estimator
is exercised against a real (if tiny) SimstackAlgorithm/PopulationManager/
SkyMaps trio -- no mocking of the pipeline internals it reuses.
"""

from unittest.mock import patch

import numpy as np

from simstack4.config import load_config
from simstack4.preflight import (
    ComputeEstimate,
    MapEstimate,
    confirm_or_abort,
    estimate_compute_requirements,
)
from simstack4.wrapper import SimstackWrapper
from tests.test_integration import (
    make_catalog,
    make_observed_map,
    make_wcs,
    write_fits_map,
    write_noise_map,
    write_toml_config,
)


def _build_loaded_wrapper(
    tmp_path, *, bootstrap_enabled=False, bootstrap_method="per_bin", bootstrap_iterations=3
):
    """Build a small synthetic dataset and return a loaded (unstacked) wrapper."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_size = 64
    pixel_scale_arcsec = 3.0
    rng = np.random.default_rng(7)
    wcs = make_wcs(pixel_scale_arcsec=pixel_scale_arcsec, image_size=image_size)

    catalog = make_catalog(
        n_pop_a=20, n_pop_b=20, wcs=wcs, image_size=image_size, rng=rng
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    data = make_observed_map(
        catalog=catalog,
        wcs=wcs,
        image_size=image_size,
        beam_fwhm_arcsec=18.0,
        pixel_scale_arcsec=pixel_scale_arcsec,
        flux_pop_a=0.005,
        flux_pop_b=0.010,
        noise_std=0.0,
        rng=rng,
    )
    map_path = tmp_path / "map_250_signal.fits"
    noise_path = tmp_path / "map_250_noise.fits"
    write_fits_map(map_path, data, wcs, 18.0, 250.0)
    write_noise_map(noise_path, data.shape, 1e-6, wcs)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config_path = tmp_path / "test_config.toml"
    write_toml_config(
        config_path,
        catalog_path,
        [{
            "name": "map_250", "path_map": str(map_path), "path_noise": str(noise_path),
            "wavelength": 250.0, "beam_fwhm": 18.0,
        }],
        output_dir,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_method=bootstrap_method,
        bootstrap_iterations=bootstrap_iterations,
    )

    config = load_config(config_path)
    wrapper = SimstackWrapper(
        config=config_path, read_maps=True, read_catalog=True,
        stack_automatically=False,
    )
    return config, wrapper


def test_estimate_runs_and_reports_exact_pixel_counts(tmp_path):
    """Estimator runs end-to-end and n_crop never exceeds n_full."""
    config, wrapper = _build_loaded_wrapper(tmp_path)
    estimate = estimate_compute_requirements(
        wrapper.config, wrapper.population_manager, wrapper.sky_maps
    )
    assert isinstance(estimate, ComputeEstimate)
    assert estimate.n_populations == 2  # matches write_toml_config's 2 z-bins
    assert estimate.n_maps == 1
    assert estimate.bootstrap_method is None
    assert len(estimate.per_map) == 1
    m = estimate.per_map[0]
    assert isinstance(m, MapEstimate)
    assert 0 < m.n_crop_pixels <= m.n_full_pixels
    assert estimate.estimated_seconds > 0.0
    assert estimate.peak_memory_bytes > 0.0
    assert "Populations: 2" in estimate.report()


def test_all_bins_uses_uncropped_full_pixel_layers(tmp_path):
    """all_bins bootstrap should be flagged and cost far more than per_bin."""
    _, wrapper_per_bin = _build_loaded_wrapper(
        tmp_path / "per_bin", bootstrap_enabled=True, bootstrap_method="per_bin",
        bootstrap_iterations=5,
    )
    est_per_bin = estimate_compute_requirements(
        wrapper_per_bin.config, wrapper_per_bin.population_manager, wrapper_per_bin.sky_maps
    )

    _, wrapper_all_bins = _build_loaded_wrapper(
        tmp_path / "all_bins", bootstrap_enabled=True, bootstrap_method="all_bins",
        bootstrap_iterations=5,
    )
    est_all_bins = estimate_compute_requirements(
        wrapper_all_bins.config, wrapper_all_bins.population_manager, wrapper_all_bins.sky_maps
    )

    assert est_all_bins.bootstrap_method == "all_bins"
    assert any("all_bins" in note for note in est_all_bins.notes)
    # all_bins redoes full convolutions every iteration -- must cost more time
    # than per_bin's one-time cache + cheap stamping, for the same iterations.
    assert est_all_bins.estimated_seconds > est_per_bin.estimated_seconds
    # all_bins builds an uncropped 2x-population layer matrix -- heavier memory.
    assert est_all_bins.peak_memory_bytes > est_per_bin.peak_memory_bytes


def test_exceeds_flags_memory_and_time_independently():
    base = {
        "n_populations": 2, "n_maps": 1, "bootstrap_method": None,
        "bootstrap_iterations": 0, "per_map": [MapEstimate("m", 100, 50, 2)],
    }
    tight_memory = ComputeEstimate(
        **base, peak_memory_bytes=90.0, estimated_seconds=1.0,
        available_memory_bytes=100.0,
    )
    assert tight_memory.exceeds(memory_fraction=0.8, time_seconds=900.0)

    slow_but_small = ComputeEstimate(
        **base, peak_memory_bytes=1.0, estimated_seconds=1000.0,
        available_memory_bytes=100.0,
    )
    assert slow_but_small.exceeds(memory_fraction=0.8, time_seconds=900.0)

    comfortable = ComputeEstimate(
        **base, peak_memory_bytes=1.0, estimated_seconds=1.0,
        available_memory_bytes=100.0,
    )
    assert not comfortable.exceeds(memory_fraction=0.8, time_seconds=900.0)


def test_confirm_or_abort_skips_prompt_when_within_budget(capsys):
    estimate = ComputeEstimate(
        n_populations=2, n_maps=1, bootstrap_method=None, bootstrap_iterations=0,
        per_map=[MapEstimate("m", 100, 50, 2)],
        peak_memory_bytes=1.0, estimated_seconds=1.0,
        available_memory_bytes=100.0,
    )
    with patch("builtins.input", side_effect=AssertionError("should not prompt")):
        assert confirm_or_abort(estimate) is True


def test_confirm_or_abort_assume_yes_bypasses_prompt():
    estimate = ComputeEstimate(
        n_populations=2, n_maps=1, bootstrap_method=None, bootstrap_iterations=0,
        per_map=[MapEstimate("m", 100, 50, 2)],
        peak_memory_bytes=99.0, estimated_seconds=1.0,
        available_memory_bytes=100.0,
    )
    with patch("builtins.input", side_effect=AssertionError("should not prompt")):
        assert confirm_or_abort(estimate, assume_yes=True) is True


def test_confirm_or_abort_non_interactive_aborts_without_yes():
    estimate = ComputeEstimate(
        n_populations=2, n_maps=1, bootstrap_method=None, bootstrap_iterations=0,
        per_map=[MapEstimate("m", 100, 50, 2)],
        peak_memory_bytes=99.0, estimated_seconds=1.0,
        available_memory_bytes=100.0,
    )
    with patch("sys.stdin.isatty", return_value=False):
        assert confirm_or_abort(estimate, assume_yes=False) is False


def test_confirm_or_abort_interactive_honors_user_response():
    estimate = ComputeEstimate(
        n_populations=2, n_maps=1, bootstrap_method=None, bootstrap_iterations=0,
        per_map=[MapEstimate("m", 100, 50, 2)],
        peak_memory_bytes=99.0, estimated_seconds=1.0,
        available_memory_bytes=100.0,
    )
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="n"):
        assert confirm_or_abort(estimate, assume_yes=False) is False
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="y"):
        assert confirm_or_abort(estimate, assume_yes=False) is True
