#!/usr/bin/env python3
"""
Generate test data for debugging simstack4 PSF bias issues
Creates:
- Test catalog (CSV) with ra, dec, mass, redshift
- Simulated FITS images with different beam sizes
- Configuration file (TOML) for simstack4
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def create_test_catalog(
    n_sources=10000,
    field_center_ra=150.0,
    field_center_dec=2.0,
    field_size_deg=1,
    output_dir="test_data",
):
    """
    Create a test catalog with realistic galaxy properties

    Args:
        n_sources: Number of sources to generate
        field_center_ra: Field center RA in degrees
        field_center_dec: Field center Dec in degrees
        field_size_deg: Field size in degrees
        output_dir: Output directory

    Returns:
        DataFrame with catalog
    """
    print(f"📊 Creating test catalog with {n_sources} sources...")

    # Generate random positions within field
    np.random.seed(42)  # For reproducibility

    # Uniform distribution in RA/Dec
    ra_min = field_center_ra - field_size_deg / 2
    ra_max = field_center_ra + field_size_deg / 2
    dec_min = field_center_dec - field_size_deg / 2
    dec_max = field_center_dec + field_size_deg / 2

    ra = np.random.uniform(ra_min, ra_max, n_sources)
    dec = np.random.uniform(dec_min, dec_max, n_sources)

    # Generate realistic redshift distribution (peaked at z~1)
    z = np.random.exponential(0.8, n_sources)
    z = np.clip(z, 0.1, 4.0)  # Reasonable redshift range

    # Generate stellar mass distribution (log-normal, peaked at 10.5)
    log_mass = np.random.normal(10.5, 0.8, n_sources)
    log_mass = np.clip(log_mass, 8.5, 12.0)  # Reasonable mass range

    # Add some scatter in mass-redshift relation
    log_mass += np.random.normal(0, 0.2, n_sources)
    log_mass = np.clip(log_mass, 8.5, 12.0)

    # Create population types (star-forming vs quiescent)
    # Higher mass galaxies more likely to be quiescent
    quiescent_prob = 1 / (1 + np.exp(-(log_mass - 10.8)))  # Sigmoid
    population_type = np.random.random(n_sources) < quiescent_prob

    # Create population labels as integers (0=star_forming, 1=quiescent)
    population_label = population_type.astype(int)

    # Create catalog dataframe
    catalog = pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "redshift": z,
            "stellar_mass": log_mass,
            "population_type": [
                "quiescent" if q else "star_forming" for q in population_type
            ],
            "population_label": population_label,  # Integer labels for simstack4
            "source_id": range(n_sources),
        }
    )

    # Save catalog
    os.makedirs(output_dir, exist_ok=True)
    catalog_path = Path(output_dir) / "test_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    print(f"✅ Catalog saved: {catalog_path}")
    print(f"   Sources: {len(catalog)}")
    print(f"   RA range: {ra.min():.3f} to {ra.max():.3f}")
    print(f"   Dec range: {dec.min():.3f} to {dec.max():.3f}")
    print(f"   Redshift range: {z.min():.2f} to {z.max():.2f}")
    print(f"   Mass range: {log_mass.min():.1f} to {log_mass.max():.1f}")
    print(
        f"   Star-forming: {np.sum(~population_type)} ({100 * np.sum(~population_type) / len(catalog):.1f}%)"
    )
    print(
        f"   Quiescent: {np.sum(population_type)} ({100 * np.sum(population_type) / len(catalog):.1f}%)"
    )

    return catalog


def create_wcs(field_center_ra, field_center_dec, pixel_scale_arcsec, image_size):
    """
    Create a WCS for the simulated image

    Args:
        field_center_ra: Field center RA in degrees
        field_center_dec: Field center Dec in degrees
        pixel_scale_arcsec: Pixel scale in arcseconds
        image_size: Image size in pixels (assumes square)

    Returns:
        WCS object
    """
    wcs = WCS(naxis=2)

    # Reference pixel at image center
    wcs.wcs.crpix = [image_size / 2 + 1, image_size / 2 + 1]  # FITS 1-indexed
    wcs.wcs.crval = [field_center_ra, field_center_dec]

    # Pixel scale in degrees
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]  # RA increases to the left

    # Coordinate system
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return wcs


def simulate_map(
    catalog,
    field_center_ra,
    field_center_dec,
    image_size=512,
    pixel_scale_arcsec=2.0,
    beam_fwhm_arcsec=10.0,
    wavelength_um=250.0,
    noise_level=0.001,
):
    """
    Create a simulated astronomical map with sources

    Args:
        catalog: Source catalog DataFrame
        field_center_ra: Field center RA in degrees
        field_center_dec: Field center Dec in degrees
        image_size: Image size in pixels
        pixel_scale_arcsec: Pixel scale in arcseconds
        beam_fwhm_arcsec: Beam FWHM in arcseconds
        wavelength_um: Wavelength in microns
        noise_level: Noise level in Jy/beam

    Returns:
        Tuple of (image_data, noise_data, header)
    """
    print(f'🗺️  Simulating {wavelength_um}μm map (FWHM={beam_fwhm_arcsec:.1f}")...')

    # Create WCS
    wcs = create_wcs(field_center_ra, field_center_dec, pixel_scale_arcsec, image_size)

    # Initialize image
    image = np.zeros((image_size, image_size))

    # Convert source positions to pixel coordinates
    coords = SkyCoord(
        ra=catalog["ra"].values * u.deg, dec=catalog["dec"].values * u.deg
    )
    x_pix, y_pix = wcs.world_to_pixel(coords)

    # Calculate source fluxes (proportional to stellar mass)
    # Typical scaling: F ∝ M_star^α, where α ~ 0.7-1.0
    alpha = 0.8
    mass_linear = 10 ** catalog["stellar_mass"].values
    base_flux = 1e-4  # Base flux in Jy
    source_fluxes = base_flux * (mass_linear / 1e10) ** alpha

    # Add wavelength-dependent scaling (crude SED model)
    # IR flux increases with wavelength up to ~200μm, then decreases
    if wavelength_um <= 200:
        wavelength_factor = (wavelength_um / 100.0) ** 1.5
    else:
        wavelength_factor = (wavelength_um / 100.0) ** 1.5 * np.exp(
            -(wavelength_um - 200) / 200
        )

    source_fluxes *= wavelength_factor

    # Place sources in image
    sources_placed = 0
    for x, y, flux in zip(x_pix, y_pix, source_fluxes, strict=True):
        # Check if source is within image bounds
        if 0 <= x < image_size and 0 <= y < image_size:
            # Add source at pixel position (simple point source)
            ix, iy = int(np.round(x)), int(np.round(y))
            if 0 <= ix < image_size and 0 <= iy < image_size:
                image[iy, ix] += flux  # Note: y,x order for numpy arrays
                sources_placed += 1

    print(f"   Placed {sources_placed} sources in image")

    # Convolve with PSF
    if beam_fwhm_arcsec > 0:
        beam_fwhm_pix = beam_fwhm_arcsec / pixel_scale_arcsec
        sigma_pix = beam_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))

        # Create PSF kernel
        kernel_size = int(6 * sigma_pix)
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size < image_size // 4:  # Only convolve if kernel is reasonable size
            kernel = Gaussian2DKernel(
                x_stddev=sigma_pix,
                y_stddev=sigma_pix,
                x_size=kernel_size,
                y_size=kernel_size,
            )

            # Normalize kernel
            # kernel_array = kernel.array / np.sum(kernel.array)
            kernel_array = kernel.array / np.max(kernel.array)

            # Convolve
            image_convolved = convolve(image, kernel_array, boundary="extend")
            print(
                f"   Convolved with {beam_fwhm_pix:.2f} pixel FWHM PSF (kernel: {kernel_size}x{kernel_size})"
            )
        else:
            image_convolved = image
            print(f"   Warning: Kernel too large ({kernel_size}), skipping convolution")
    else:
        image_convolved = image

    # Add noise
    if noise_level > 0:
        noise_map = np.random.normal(0, noise_level, (image_size, image_size))
        image_convolved += noise_map
    else:
        noise_map = np.zeros_like(image_convolved)

    # Create FITS header
    header = wcs.to_header()

    # Add additional header information
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = beam_fwhm_arcsec / 3600.0  # Beam major axis in degrees
    header["BMIN"] = beam_fwhm_arcsec / 3600.0  # Beam minor axis in degrees
    header["BPA"] = 0.0  # Beam position angle
    header["WAVELENG"] = wavelength_um
    header["TELESCOP"] = "SIMULATED"
    header["INSTRUME"] = "SIMSTACK_TEST"
    header["OBJECT"] = "TEST_FIELD"
    header["NSOURCES"] = sources_placed
    header["NOISE"] = noise_level
    header["PIXSCALE"] = pixel_scale_arcsec
    header["COMMENT"] = "Simulated map for simstack4 testing"

    # Calculate beam area in steradians (for conversion verification)
    beam_area_arcsec2 = 1.133 * beam_fwhm_arcsec**2  # Gaussian beam
    beam_area_sr = beam_area_arcsec2 * (np.pi / (180 * 3600)) ** 2
    header["BEAMAREA"] = beam_area_sr

    print(
        f"   Image stats: min={np.min(image_convolved):.2e}, max={np.max(image_convolved):.2e}, std={np.std(image_convolved):.2e}"
    )

    return image_convolved, noise_map, header


def create_test_maps(
    catalog,
    output_dir="test_data",
    field_center_ra=150.0,
    field_center_dec=2.0,
    image_size_deg=1.0,
):
    """
    Create a set of test maps with different beam sizes

    Args:
        catalog: Source catalog
        output_dir: Output directory
        field_center_ra: Field center RA
        field_center_dec: Field center Dec
        image_size_deg: Image size in degrees (default 0.5 degrees)

    Returns:
        List of map configurations
    """
    print("🎯 Creating test maps...")
    print(f"   Image size: {image_size_deg:.3f} degrees")

    # Define different "instruments" with varying beam sizes
    map_configs = [
        {
            "name": "mips",
            "wavelength": 25.0,
            "beam_fwhm": 5.0,
            "pixel_scale": 1.0,
            "noise": 0.0000005,
        },
        {
            "name": "pacs_green",
            "wavelength": 160.0,
            "beam_fwhm": 7.0,
            "pixel_scale": 2.0,
            "noise": 0.00001,
        },
        {
            "name": "pacs_red",
            "wavelength": 160.0,
            "beam_fwhm": 10.0,
            "pixel_scale": 3.0,
            "noise": 0.00001,
        },
        {
            "name": "spire_psw",
            "wavelength": 250.0,
            "beam_fwhm": 18.0,
            "pixel_scale": 6.0,
            "noise": 0.00002,
        },
        {
            "name": "spire_plw",
            "wavelength": 500.0,
            "beam_fwhm": 36.0,
            "pixel_scale": 6.0,
            "noise": 0.00003,
        },
        {
            "name": "scuba",
            "wavelength": 850.0,
            "beam_fwhm": 12.0,
            "pixel_scale": 2.0,
            "noise": 0.00003,
        },
    ]

    for config in map_configs:
        print(f"\n📡 Creating {config['name']} map...")

        # Calculate image size in pixels for this map's pixel scale
        pixel_scale_deg = config["pixel_scale"] / 3600.0  # Convert arcsec to degrees
        image_size_pixels = int(np.ceil(image_size_deg / pixel_scale_deg))

        # Ensure image size is reasonable (not too small or too large)
        min_pixels = 64  # Minimum sensible image size
        max_pixels = 2048  # Maximum for memory/speed

        if image_size_pixels < min_pixels:
            image_size_pixels = min_pixels
            actual_size_deg = image_size_pixels * pixel_scale_deg
            print(
                f"   ⚠️  Adjusted image size to {min_pixels} pixels ({actual_size_deg:.4f}°) for adequate sampling"
            )
        elif image_size_pixels > max_pixels:
            image_size_pixels = max_pixels
            actual_size_deg = image_size_pixels * pixel_scale_deg
            print(
                f"   ⚠️  Limited image size to {max_pixels} pixels ({actual_size_deg:.4f}°) for memory constraints"
            )
        else:
            actual_size_deg = image_size_deg

        print(
            f"   Pixel scale: {config['pixel_scale']}\" = {pixel_scale_deg * 3600:.2f}\" = {pixel_scale_deg:.6f}°"
        )
        print(
            f"   Image: {image_size_pixels} × {image_size_pixels} pixels = {actual_size_deg:.4f}° × {actual_size_deg:.4f}°"
        )

        # Simulate map
        image_data, noise_data, header = simulate_map(
            catalog=catalog,
            field_center_ra=field_center_ra,
            field_center_dec=field_center_dec,
            image_size=image_size_pixels,  # Now calculated based on degrees and pixel scale
            pixel_scale_arcsec=config["pixel_scale"],
            beam_fwhm_arcsec=config["beam_fwhm"],
            wavelength_um=config["wavelength"],
            noise_level=config["noise"],
        )

        # Save signal map
        signal_path = Path(output_dir) / f"{config['name']}_signal.fits"
        fits.writeto(signal_path, image_data, header=header, overwrite=True)

        # Save noise map
        noise_path = Path(output_dir) / f"{config['name']}_noise.fits"
        noise_header = header.copy()
        noise_header["COMMENT"] = "Noise map for simstack4 testing"
        fits.writeto(noise_path, noise_data, header=noise_header, overwrite=True)

        print(f"   ✅ Saved: {signal_path}")
        print(f"   ✅ Saved: {noise_path}")

        # Update config with file paths and actual dimensions
        config["signal_path"] = str(signal_path)
        config["noise_path"] = str(noise_path)
        config["image_size_pixels"] = image_size_pixels
        config["image_size_deg"] = actual_size_deg
        config["pixel_scale_deg"] = pixel_scale_deg

    return map_configs


def create_config_file(
    map_configs, output_dir="test_data", catalog_filename="test_catalog.csv"
):
    """
    Create simstack4 configuration file

    Args:
        map_configs: List of map configurations
        output_dir: Output directory
        catalog_filename: Catalog filename
    """
    print("⚙️  Creating configuration file...")

    config_content = f"""# Simstack4 Test Configuration
# Generated for PSF bias debugging

[binning]
stack_all_z_at_once = true
add_foreground = true
crop_circles = false  # Use full maps for testing

[error_estimator]
write_simmaps = false
randomize = false

[error_estimator.bootstrap]
enabled = false  # Disable for quick testing
iterations = 1
initial_seed = 42

cosmology = "Planck18"

[output]
folder = "{output_dir}/results"
shortname = "simstack_test"

[catalog]
path = "{output_dir}"
file = "{catalog_filename}"

[catalog.astrometry]
ra = "ra"
dec = "dec"

[catalog.classification]
split_type = "labels"

[catalog.classification.redshift]
id = "redshift"
bins = [0.1, 1.0, 2.0, 4.0]

[catalog.classification.stellar_mass]
id = "stellar_mass"
bins = [8.5, 10.0, 11.0, 12.0]

# No population splitting - just use redshift and mass bins
# This simplifies the test to focus on PSF bias

"""

    # Add map configurations
    for config in map_configs:
        # Calculate beam area in steradians
        beam_fwhm_arcsec = config["beam_fwhm"]
        beam_area_arcsec2 = 1.133 * beam_fwhm_arcsec**2
        beam_area_sr = beam_area_arcsec2 * (np.pi / (180 * 3600)) ** 2

        config_content += f"""
[maps.{config['name']}]
wavelength = {config['wavelength']}
color_correction = 1.0
path_map = "{config['signal_path']}"
path_noise = "{config['noise_path']}"

[maps.{config['name']}.beam]
fwhm = {config['beam_fwhm']}
area = {beam_area_sr:.3e}
"""

    # Save config file
    config_path = Path(output_dir) / "simstack_test.toml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"✅ Configuration saved: {config_path}")
    return config_path


def create_analysis_script(output_dir="test_data"):
    """
    Create a script to analyze the test results
    """
    analysis_script = f'''#!/usr/bin/env python3
"""
Analysis script for simstack4 PSF bias test
Run after simstack4 to check for beam-dependent bias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_test_results():
    """Analyze simstack test results for beam-dependent bias"""

    # Expected: flux should be proportional to stellar mass
    # Any deviation indicates bias

    print("📊 Analyzing simstack4 test results...")

    # Load original catalog for comparison
    catalog = pd.read_csv("{output_dir}/test_catalog.csv")

    # This would load your simstack results
    # results_path = "{output_dir}/results/simstack_test_results.csv"
    # if Path(results_path).exists():
    #     results = pd.read_csv(results_path)
    #
    #     # Plot flux vs stellar mass for each beam size
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #     axes = axes.ravel()
    #
    #     beam_maps = ['narrow_beam', 'medium_beam', 'wide_beam', 'very_wide_beam']
    #
    #     for i, beam_map in enumerate(beam_maps):
    #         # Plot expected vs measured relationship
    #         # This would depend on your results format
    #         pass

    print("Run this after simstack4 completes!")

if __name__ == "__main__":
    analyze_test_results()
'''

    analysis_path = Path(output_dir) / "analyze_results.py"
    with open(analysis_path, "w") as f:
        f.write(analysis_script)

    print(f"📈 Analysis script saved: {analysis_path}")


def main():
    """Create complete test dataset"""
    print("🚀 SIMSTACK4 PSF BIAS TEST DATA GENERATOR")
    print("=" * 60)

    output_dir = "simstack_test_data"

    # Create test catalog
    catalog = create_test_catalog(n_sources=150000, output_dir=output_dir)

    # Create test maps
    map_configs = create_test_maps(catalog, output_dir=output_dir)

    # Create configuration file
    config_path = create_config_file(map_configs, output_dir=output_dir)

    # Create analysis script
    create_analysis_script(output_dir=output_dir)

    print("\n🎉 TEST DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}/")
    print(f"Configuration: {config_path}")
    print()
    print("📋 GENERATED FILES:")
    print("   • test_catalog.csv - Source catalog")
    print("   • *_signal.fits - Simulated maps")
    print("   • *_noise.fits - Noise maps")
    print("   • simstack_test.toml - Configuration")
    print("   • analyze_results.py - Analysis script")
    print()
    print("🔬 NEXT STEPS:")
    print(f"1. Run simstack4 with: {config_path}")
    print("2. Check if larger beam maps show systematically higher fluxes")
    print("3. Compare measured vs expected (mass-proportional) fluxes")
    print("4. Test different convolution methods/boundaries")

    return output_dir, config_path


if __name__ == "__main__":
    main()
