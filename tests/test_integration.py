"""
Integration test: full pipeline from synthetic FITS/catalog/TOML through
SimstackWrapper to recovered flux densities.

This tests the complete chain:
  TOML config → SkyCatalogs → PopulationManager → SkyMaps
  → SimstackAlgorithm → flux recovery

Synthetic data is created in a temp directory with known injected fluxes,
then the pipeline is run and recovered fluxes are compared to truth.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.wcs import WCS


# ---------------------------------------------------------------------------
# Synthetic data generation helpers
# ---------------------------------------------------------------------------


def make_wcs(
    crval_ra: float = 150.0,
    crval_dec: float = 2.0,
    pixel_scale_arcsec: float = 3.0,
    image_size: int = 128,
) -> WCS:
    """Create a simple TAN WCS centered on (crval_ra, crval_dec)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [image_size / 2 + 0.5, image_size / 2 + 0.5]
    w.wcs.crval = [crval_ra, crval_dec]
    w.wcs.cdelt = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    return w


def make_catalog(
    n_pop_a: int = 100,
    n_pop_b: int = 100,
    wcs: WCS = None,
    image_size: int = 128,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Create a catalog with two populations distinguished by redshift bins.

    Pop A: z in [0.5, 1.0), mass ~10.5
    Pop B: z in [1.0, 2.0), mass ~10.5

    Sources are placed randomly but away from edges (10-pixel margin).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if wcs is None:
        wcs = make_wcs(image_size=image_size)

    n_total = n_pop_a + n_pop_b
    margin = 15  # pixels from edge

    # Random pixel positions with margin
    x_pix = rng.uniform(margin, image_size - margin, n_total)
    y_pix = rng.uniform(margin, image_size - margin, n_total)

    # Convert to sky coordinates
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    sky = wcs.pixel_to_world(x_pix, y_pix)
    ra = sky.ra.deg
    dec = sky.dec.deg

    # Assign populations via redshift
    z = np.empty(n_total)
    z[:n_pop_a] = rng.uniform(0.6, 0.9, n_pop_a)   # Pop A
    z[n_pop_a:] = rng.uniform(1.1, 1.8, n_pop_b)    # Pop B

    mass = rng.normal(10.5, 0.2, n_total)
    mass = np.clip(mass, 9.0, 12.0)

    # Simple split: all star-forming (sfg=0)
    sfg = np.zeros(n_total, dtype=int)

    return pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z_peak": z,
        "lmass": mass,
        "sfg": sfg,
    })


def make_observed_map(
    catalog: pd.DataFrame,
    wcs: WCS,
    image_size: int,
    beam_fwhm_arcsec: float,
    pixel_scale_arcsec: float,
    flux_pop_a: float,
    flux_pop_b: float,
    z_boundary: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Create an observed map by injecting sources with known per-population fluxes.

    Each source gets the flux of its population. Sources are convolved with
    the same Gaussian PSF that simstack4 will use, with the same normalization
    (peak = 1).
    """
    if rng is None:
        rng = np.random.default_rng(99)

    image = np.zeros((image_size, image_size), dtype=np.float64)

    # Place sources
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords = SkyCoord(ra=catalog["ra"].values * u.deg,
                      dec=catalog["dec"].values * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords)

    for i, (x, y) in enumerate(zip(x_pix, y_pix)):
        ix, iy = int(np.round(x)), int(np.round(y))
        if 0 <= ix < image_size and 0 <= iy < image_size:
            flux = flux_pop_a if catalog["z_peak"].iloc[i] < z_boundary else flux_pop_b
            image[iy, ix] += flux

    # Convolve with PSF (peak-normalized, matching simstack4)
    beam_fwhm_pix = beam_fwhm_arcsec / pixel_scale_arcsec
    sigma_pix = beam_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
    kernel_size = int(6 * sigma_pix)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = Gaussian2DKernel(
        x_stddev=sigma_pix,
        y_stddev=sigma_pix,
        x_size=kernel_size,
        y_size=kernel_size,
    )
    kernel_array = kernel.array / np.max(kernel.array)  # Peak-normalize

    image = convolve_fft(image, kernel_array, boundary="wrap",
                         nan_treatment="interpolate", normalize_kernel=False)

    # Add noise
    if noise_std > 0:
        image += rng.normal(0, noise_std, image.shape)

    return image


def write_fits_map(
    filepath: Path,
    data: np.ndarray,
    wcs: WCS,
    beam_fwhm_arcsec: float,
    wavelength_um: float,
):
    """Write a FITS map with proper headers for simstack4."""
    header = wcs.to_header()
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = beam_fwhm_arcsec / 3600.0
    header["BMIN"] = beam_fwhm_arcsec / 3600.0
    header["BPA"] = 0.0
    header["WAVELENG"] = wavelength_um
    header["TELESCOP"] = "SIMULATED"

    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(filepath, overwrite=True)


def write_noise_map(filepath: Path, shape: tuple, noise_std: float, wcs: WCS):
    """Write a uniform noise (RMS) map."""
    noise_rms = np.full(shape, noise_std, dtype=np.float32)
    header = wcs.to_header()
    hdu = fits.PrimaryHDU(data=noise_rms, header=header)
    hdu.writeto(filepath, overwrite=True)


def write_toml_config(
    filepath: Path,
    catalog_path: Path,
    map_configs: list[dict],
    output_dir: Path,
    bootstrap_enabled: bool = False,
    bootstrap_iterations: int = 5,
    bootstrap_method: str = "all_bins",
    crop_circles: bool = True,
    add_foreground: bool = True,
):
    """Write a TOML config for simstack4."""
    maps_section = ""
    for mc in map_configs:
        maps_section += f"""
[maps.{mc['name']}]
path_map = "{mc['path_map']}"
path_noise = "{mc['path_noise']}"
wavelength = {mc['wavelength']}
color_correction = 1.0

[maps.{mc['name']}.beam]
fwhm = {mc['beam_fwhm']}
"""

    toml_content = f"""
cosmology = "Planck18"

[binning]
stack_all_z_at_once = false
add_foreground = {"true" if add_foreground else "false"}
crop_circles = {"true" if crop_circles else "false"}

[error_estimator]
write_simmaps = false
randomize = false

[error_estimator.bootstrap]
enabled = {"true" if bootstrap_enabled else "false"}
iterations = {bootstrap_iterations}
initial_seed = 42
method = "{bootstrap_method}"

[output]
folder = "{output_dir}"
shortname = "integration_test"

[catalog]
path = "{catalog_path.parent}"
file = "{catalog_path.name}"

[catalog.astrometry]
ra = "ra"
dec = "dec"

[catalog.classification]
split_type = "labels"

[catalog.classification.split_params]
id = "sfg"

[catalog.classification.binning.redshift]
id = "z_peak"
bins = [0.01, 1.0, 2.0]

[catalog.classification.binning.stellar_mass]
id = "lmass"
bins = [9.0, 12.0]

{maps_section}
"""
    filepath.write_text(toml_content)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset(tmp_path):
    """
    Create a complete synthetic dataset: catalog, FITS maps, TOML config.

    Two populations (z < 1 and z > 1), two wavelength maps (250 and 350 μm),
    with known injected fluxes.
    """
    # Parameters
    image_size = 128
    pixel_scale_arcsec = 3.0
    n_pop_a = 80
    n_pop_b = 80

    # Known truth: per-population flux at each wavelength
    truth = {
        "map_250": {"pop_a": 0.005, "pop_b": 0.010},  # Jy
        "map_350": {"pop_a": 0.008, "pop_b": 0.015},  # Jy
    }

    map_specs = [
        {"name": "map_250", "wavelength": 250.0, "beam_fwhm": 18.0},
        {"name": "map_350", "wavelength": 350.0, "beam_fwhm": 25.0},
    ]

    rng = np.random.default_rng(42)
    wcs = make_wcs(pixel_scale_arcsec=pixel_scale_arcsec, image_size=image_size)

    # Create catalog
    catalog = make_catalog(
        n_pop_a=n_pop_a, n_pop_b=n_pop_b,
        wcs=wcs, image_size=image_size, rng=rng,
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    # Create maps
    map_configs = []
    for spec in map_specs:
        map_name = spec["name"]

        data = make_observed_map(
            catalog=catalog,
            wcs=wcs,
            image_size=image_size,
            beam_fwhm_arcsec=spec["beam_fwhm"],
            pixel_scale_arcsec=pixel_scale_arcsec,
            flux_pop_a=truth[map_name]["pop_a"],
            flux_pop_b=truth[map_name]["pop_b"],
            noise_std=0.0,  # Noiseless for clean recovery
            rng=rng,
        )

        map_path = tmp_path / f"{map_name}_signal.fits"
        noise_path = tmp_path / f"{map_name}_noise.fits"

        write_fits_map(map_path, data, wcs, spec["beam_fwhm"], spec["wavelength"])
        write_noise_map(noise_path, data.shape, 1e-6, wcs)  # Tiny noise RMS

        map_configs.append({
            "name": map_name,
            "path_map": str(map_path),
            "path_noise": str(noise_path),
            "wavelength": spec["wavelength"],
            "beam_fwhm": spec["beam_fwhm"],
        })

    # Create output dir
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Write TOML config
    config_path = tmp_path / "test_config.toml"
    write_toml_config(config_path, catalog_path, map_configs, output_dir)

    return {
        "config_path": config_path,
        "catalog_path": catalog_path,
        "catalog": catalog,
        "truth": truth,
        "map_configs": map_configs,
        "tmp_path": tmp_path,
        "output_dir": output_dir,
        "n_pop_a": n_pop_a,
        "n_pop_b": n_pop_b,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipelineFluxRecovery:
    """
    Run the full simstack4 pipeline on synthetic data and verify
    that recovered fluxes match injected truth.
    """

    def test_pipeline_runs_without_error(self, synthetic_dataset):
        """The pipeline should complete without exceptions."""
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        config = load_config(synthetic_dataset["config_path"])
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=True,
        )

        assert wrapper.stacking_results is not None

    def test_correct_number_of_populations(self, synthetic_dataset):
        """Pipeline should identify the expected number of population bins."""
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        config = load_config(synthetic_dataset["config_path"])
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=False,
        )

        n_pops = len(list(wrapper.population_manager.iter_populations()))
        # 2 redshift bins × 1 mass bin × 1 split type = 2 populations
        assert n_pops == 2, f"Expected 2 populations, got {n_pops}"

    def test_flux_recovery_noiseless(self, synthetic_dataset):
        """
        With noiseless maps, recovered fluxes should match injected truth
        to high precision.
        """
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        config = load_config(synthetic_dataset["config_path"])
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=True,
        )

        results = wrapper.stacking_results
        truth = synthetic_dataset["truth"]

        for map_name, map_truth in truth.items():
            labels = results.population_labels
            fluxes = results.flux_densities[map_name]

            # Find the two science populations (not foreground)
            science_mask = [l != "foreground" for l in labels]
            science_labels = [l for l, m in zip(labels, science_mask) if m]
            science_fluxes = fluxes[science_mask]

            assert len(science_fluxes) == 2, (
                f"{map_name}: Expected 2 science populations, "
                f"got {len(science_fluxes)} with labels {science_labels}"
            )

            # Identify which population is which by label ordering
            # Labels are sorted by bin ranges, so z=[0.01,1.0] comes first
            flux_pop_a = science_fluxes[0]  # Lower-z population
            flux_pop_b = science_fluxes[1]  # Higher-z population

            # Noiseless recovery should be good to ~5%
            # (limited by pixel discretization, mean subtraction, edge effects)
            assert flux_pop_a == pytest.approx(map_truth["pop_a"], rel=0.05), (
                f"{map_name} Pop A: recovered {flux_pop_a:.4e} vs truth {map_truth['pop_a']:.4e}"
            )
            assert flux_pop_b == pytest.approx(map_truth["pop_b"], rel=0.05), (
                f"{map_name} Pop B: recovered {flux_pop_b:.4e} vs truth {map_truth['pop_b']:.4e}"
            )

    def test_foreground_layer_near_zero(self, synthetic_dataset):
        """
        The foreground (constant) layer should recover near-zero flux
        since the injected map is pure source signal (no background).
        """
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        config = load_config(synthetic_dataset["config_path"])
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=True,
        )

        results = wrapper.stacking_results
        truth = synthetic_dataset["truth"]

        for map_name in truth:
            labels = results.population_labels
            fluxes = results.flux_densities[map_name]

            fg_idx = labels.index("foreground")
            fg_flux = fluxes[fg_idx]

            # Foreground should be small relative to source fluxes.
            # On small maps with peak-normalized PSF, source signal spreads
            # significantly and the foreground absorbs the DC offset from
            # the pipeline's map-level mean subtraction.
            max_source_flux = max(truth[map_name].values())
            assert abs(fg_flux) < 0.5 * max_source_flux, (
                f"{map_name}: foreground flux {fg_flux:.4e} too large "
                f"(max source flux = {max_source_flux:.4e})"
            )

    def test_population_source_counts(self, synthetic_dataset):
        """Verify that the population manager assigns the right number of sources."""
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        config = load_config(synthetic_dataset["config_path"])
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=False,
        )

        total = 0
        for pop_bin in wrapper.population_manager.iter_populations():
            total += pop_bin.n_sources

        expected = synthetic_dataset["n_pop_a"] + synthetic_dataset["n_pop_b"]
        assert total == expected, (
            f"Population manager has {total} sources, expected {expected}"
        )


class TestPipelineWithNoise:
    """Test flux recovery with realistic noise levels."""

    def test_noisy_recovery_within_errors(self, tmp_path):
        """
        With noise, recovered fluxes should be within a few sigma
        of the injected truth.
        """
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        # Create dataset with noise
        image_size = 128
        pixel_scale_arcsec = 3.0
        noise_std = 0.0005  # Moderate noise

        truth = {
            "map_250": {"pop_a": 0.005, "pop_b": 0.010},
        }

        rng = np.random.default_rng(123)
        wcs = make_wcs(pixel_scale_arcsec=pixel_scale_arcsec, image_size=image_size)

        catalog = make_catalog(
            n_pop_a=100, n_pop_b=100,
            wcs=wcs, image_size=image_size, rng=rng,
        )
        catalog_path = tmp_path / "catalog.csv"
        catalog.to_csv(catalog_path, index=False)

        data = make_observed_map(
            catalog=catalog, wcs=wcs, image_size=image_size,
            beam_fwhm_arcsec=18.0, pixel_scale_arcsec=pixel_scale_arcsec,
            flux_pop_a=0.005, flux_pop_b=0.010,
            noise_std=noise_std, rng=rng,
        )

        map_path = tmp_path / "map_250_signal.fits"
        noise_path = tmp_path / "map_250_noise.fits"
        write_fits_map(map_path, data, wcs, 18.0, 250.0)
        write_noise_map(noise_path, data.shape, noise_std, wcs)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        map_configs = [{
            "name": "map_250",
            "path_map": str(map_path),
            "path_noise": str(noise_path),
            "wavelength": 250.0,
            "beam_fwhm": 18.0,
        }]

        config_path = tmp_path / "test_config.toml"
        write_toml_config(config_path, catalog_path, map_configs, output_dir)

        config = load_config(config_path)
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=True,
        )

        results = wrapper.stacking_results
        labels = results.population_labels
        fluxes = results.flux_densities["map_250"]
        errors = results.flux_errors["map_250"]

        science_mask = [l != "foreground" for l in labels]
        science_fluxes = fluxes[science_mask]
        science_errors = errors[science_mask]

        truth_values = [0.005, 0.010]

        for i, (recovered, error, true_val) in enumerate(
            zip(science_fluxes, science_errors, truth_values)
        ):
            # Should be within 5σ (very conservative)
            if error > 0:
                n_sigma = abs(recovered - true_val) / error
                assert n_sigma < 5, (
                    f"Pop {i}: {n_sigma:.1f}σ from truth "
                    f"(recovered={recovered:.4e}, truth={true_val:.4e}, error={error:.4e})"
                )


class TestPipelineWithBootstrap:
    """Test that bootstrap error estimation works through the full pipeline."""

    def test_all_bins_bootstrap(self, synthetic_dataset):
        """Run pipeline with all_bins bootstrap and verify errors are produced."""
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        config_path = synthetic_dataset["config_path"]

        # Rewrite config with bootstrap enabled
        write_toml_config(
            config_path,
            synthetic_dataset["catalog_path"],
            synthetic_dataset["map_configs"],
            synthetic_dataset["output_dir"],
            bootstrap_enabled=True,
            bootstrap_iterations=5,
            bootstrap_method="all_bins",
        )

        config = load_config(config_path)
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=True,
        )

        results = wrapper.stacking_results
        for map_name in synthetic_dataset["truth"]:
            # Should have flux errors from bootstrap
            errors = results.flux_errors[map_name]
            assert np.all(np.isfinite(errors)), f"{map_name}: non-finite errors"
            assert np.all(errors >= 0), f"{map_name}: negative errors"

    def test_per_bin_bootstrap(self, synthetic_dataset):
        """Run pipeline with per_bin bootstrap and verify errors are produced."""
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        config_path = synthetic_dataset["config_path"]

        write_toml_config(
            config_path,
            synthetic_dataset["catalog_path"],
            synthetic_dataset["map_configs"],
            synthetic_dataset["output_dir"],
            bootstrap_enabled=True,
            bootstrap_iterations=5,
            bootstrap_method="per_bin",
        )

        config = load_config(config_path)
        wrapper = SimstackWrapper(
            config,
            read_maps=True,
            read_catalog=True,
            stack_automatically=True,
        )

        results = wrapper.stacking_results
        for map_name in synthetic_dataset["truth"]:
            errors = results.flux_errors[map_name]
            assert np.all(np.isfinite(errors)), f"{map_name}: non-finite errors"
            assert np.all(errors >= 0), f"{map_name}: negative errors"


# ---------------------------------------------------------------------------
# Test: Partial coverage (zero-pixel boundaries)
# ---------------------------------------------------------------------------


class TestPartialCoverage:
    """
    Regression test for mean subtraction consistency.

    Real Herschel maps have unobserved pixels (zero or NaN) at boundaries.
    The pipeline mean-subtracts the observed map over non-zero pixels only,
    but layers must be mean-subtracted over the SAME pixel set.

    Previously, layers used global mean subtraction (all pixels including
    zeros), causing systematic flux underestimation of 5-25% depending on
    the fraction of unobserved pixels.
    """

    @pytest.fixture
    def partial_coverage_dataset(self, tmp_path):
        """Create synthetic data with zero-pixel boundary (~15% unobserved)."""
        image_size = 128
        pixel_scale_arcsec = 3.0
        border_pixels = 7
        rng = np.random.default_rng(42)

        wcs = make_wcs(pixel_scale_arcsec=pixel_scale_arcsec, image_size=image_size)

        # Sources placed with margin=15 (default), well inside border
        catalog = make_catalog(
            n_pop_a=40, n_pop_b=120,
            wcs=wcs, image_size=image_size, rng=rng,
        )

        truth = {"map_250": {"pop_a": 0.005, "pop_b": 0.010}}
        map_specs = [{"name": "map_250", "wavelength": 250.0, "beam_fwhm": 18.0}]

        catalog_path = tmp_path / "catalog.csv"
        catalog.to_csv(catalog_path, index=False)

        map_configs_list = []
        for ms in map_specs:
            obs = make_observed_map(
                catalog, wcs, image_size,
                ms["beam_fwhm"], pixel_scale_arcsec,
                truth[ms["name"]]["pop_a"],
                truth[ms["name"]]["pop_b"],
                z_boundary=1.0, noise_std=0.0, rng=rng,
            )

            # Zero out border (simulate unobserved edge pixels)
            obs[:border_pixels, :] = 0.0
            obs[-border_pixels:, :] = 0.0
            obs[:, :border_pixels] = 0.0
            obs[:, -border_pixels:] = 0.0

            map_path = tmp_path / f"{ms['name']}.fits"
            noise_path = tmp_path / f"{ms['name']}_noise.fits"
            write_fits_map(map_path, obs, wcs, ms["beam_fwhm"], ms["wavelength"])
            write_noise_map(noise_path, (image_size, image_size), 0.001, wcs)

            map_configs_list.append({
                "name": ms["name"], "wavelength": ms["wavelength"],
                "beam_fwhm": ms["beam_fwhm"],
                "path_map": str(map_path), "path_noise": str(noise_path),
            })

        return {
            "catalog_path": catalog_path,
            "map_configs": map_configs_list,
            "truth": truth,
        }

    def _run_pipeline(self, ds, tmp_path, crop_circles, add_foreground=True):
        """Helper to run pipeline with specific settings."""
        from simstack4.config import load_config
        from simstack4.wrapper import SimstackWrapper

        output_dir = tmp_path / f"output_crop{crop_circles}_fg{add_foreground}"
        output_dir.mkdir(exist_ok=True)
        config_path = tmp_path / f"config_crop{crop_circles}_fg{add_foreground}.toml"
        write_toml_config(
            config_path, ds["catalog_path"], ds["map_configs"],
            output_dir, crop_circles=crop_circles, add_foreground=add_foreground,
        )
        config = load_config(str(config_path))
        wrapper = SimstackWrapper(
            config, read_maps=True, read_catalog=True, stack_automatically=True,
        )
        return wrapper.stacking_results

    def _check_recovery(self, results, truth, tolerance, label_prefix=""):
        """Check flux recovery against truth values."""
        for map_name, truth_fluxes in truth.items():
            fluxes = results.flux_densities[map_name]
            labels = results.population_labels
            for i, label in enumerate(labels):
                if "foreground" in label:
                    continue
                expected = truth_fluxes["pop_a"] if "0.01_1.0" in label else truth_fluxes["pop_b"]
                assert fluxes[i] == pytest.approx(expected, rel=tolerance), (
                    f"{label_prefix}{label}: recovered {fluxes[i]:.6f}, "
                    f"expected {expected}, tol={tolerance}"
                )

    def test_crop_circles_true(self, partial_coverage_dataset, tmp_path):
        """With crop_circles=True, recovery should be exact despite zero pixels."""
        results = self._run_pipeline(partial_coverage_dataset, tmp_path, crop_circles=True)
        self._check_recovery(results, partial_coverage_dataset["truth"], tolerance=0.02,
                             label_prefix="crop=T: ")

    def test_crop_circles_false(self, partial_coverage_dataset, tmp_path):
        """
        Regression: crop_circles=False must also give unbiased recovery.

        Before the mean subtraction fix, this failed with 5-25% underestimation
        because layers were globally mean-subtracted while the map was
        mean-subtracted only over non-zero (observed) pixels.
        """
        results = self._run_pipeline(partial_coverage_dataset, tmp_path, crop_circles=False)
        self._check_recovery(results, partial_coverage_dataset["truth"], tolerance=0.05,
                             label_prefix="crop=F: ")
