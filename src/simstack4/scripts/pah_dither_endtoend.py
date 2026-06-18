"""
End-to-end map-level spot check of the PAH dithered-stacking machinery.

The flux-level simulator in pah_dither.py skips the maps entirely; this
script validates that shortcut once, with the real stacking pipeline in
the loop:

    TruthSpectrum per-source 24/70 µm fluxes (at TRUE redshifts)
      → synthetic confused maps (PSF-convolved, simstack conventions)
      → catalog with PHOTO-z (scatter + catastrophic outliers)
      → one TOML + SimstackWrapper stacking run per dither offset
      → stacked fluxes per (run, z-bin) → PAHSpectrumModel.fit_lstsq
      → injected vs recovered feature amplitudes

Usage:
    uv run python -m simstack4.scripts.pah_dither_endtoend            # full
    uv run python -m simstack4.scripts.pah_dither_endtoend --quick    # smoke

The quick mode (also used by tests/test_pah_endtoend.py) shrinks the
field, source count, and dither scheme so it finishes in ~a minute.
"""

import argparse
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from simstack4.pah_dither import (
    DitherScheme,
    TruthSpectrum,
    _draw_redshifts,
    make_dndz,
)
from simstack4.pah_spectrum import PAHSpectrumModel

BAND_SPECS = {
    "MIPS_24": {"wavelength": 24.0, "beam_fwhm": 6.0, "pixel_scale": 2.0},
    "MIPS_70": {"wavelength": 70.0, "beam_fwhm": 18.0, "pixel_scale": 4.0},
}


def _make_wcs(ra0, dec0, pixel_scale_arcsec, image_size):
    w = WCS(naxis=2)
    w.wcs.crpix = [image_size / 2 + 0.5, image_size / 2 + 0.5]
    w.wcs.crval = [ra0, dec0]
    w.wcs.cdelt = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    return w


def _inject_and_convolve(
    fluxes,
    ra,
    dec,
    wcs,
    image_size,
    beam_fwhm_arcsec,
    pixel_scale_arcsec,
    noise_std,
    rng,
):
    """Peak-normalized PSF injection, matching simstack4 and the
    integration-test conventions."""
    image = np.zeros((image_size, image_size))
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords)
    ix = np.round(x_pix).astype(int)
    iy = np.round(y_pix).astype(int)
    ok = (ix >= 0) & (ix < image_size) & (iy >= 0) & (iy < image_size)
    np.add.at(image, (iy[ok], ix[ok]), fluxes[ok])

    sigma_pix = beam_fwhm_arcsec / pixel_scale_arcsec / (2 * np.sqrt(2 * np.log(2)))
    kernel_size = int(6 * sigma_pix)
    kernel_size += 1 - kernel_size % 2
    kernel = Gaussian2DKernel(
        x_stddev=sigma_pix,
        y_stddev=sigma_pix,
        x_size=kernel_size,
        y_size=kernel_size,
    )
    image = convolve_fft(
        image,
        kernel.array / kernel.array.max(),
        boundary="wrap",
        normalize_kernel=False,
    )
    if noise_std > 0:
        image += rng.normal(0, noise_std, image.shape)
    return image


def _write_map(path, data, wcs, beam_fwhm, wavelength):
    header = wcs.to_header()
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = header["BMIN"] = beam_fwhm / 3600.0
    header["BPA"] = 0.0
    header["WAVELENG"] = wavelength
    header["TELESCOP"] = "SIMULATED"
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(
        path, overwrite=True
    )


def _write_config(
    path, catalog_path, map_configs, output_dir, z_bins, bootstrap_iterations
):
    maps_section = ""
    for mc in map_configs:
        maps_section += f"""
[maps.{mc["name"]}]
path_map = "{mc["path_map"]}"
path_noise = "{mc["path_noise"]}"
wavelength = {mc["wavelength"]}
color_correction = 1.0

[maps.{mc["name"]}.beam]
fwhm = {mc["beam_fwhm"]}
"""
    path.write_text(
        f"""
cosmology = "Planck18"

[binning]
stack_all_z_at_once = true
add_foreground = true
crop_circles = true

[error_estimator]
write_simmaps = false
randomize = false

[error_estimator.bootstrap]
enabled = true
iterations = {bootstrap_iterations}
initial_seed = 42
method = "all_bins"

[output]
folder = "{output_dir}"
shortname = "pah_endtoend"

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
bins = {z_bins}

[catalog.classification.binning.stellar_mass]
id = "lmass"
bins = [9.0, 12.0]
{maps_section}
"""
    )


def run_endtoend(
    workdir: Path | None = None,
    *,
    quick: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run the spot check; returns {"result", "truth", "df", "comparison"}."""
    rng = np.random.default_rng(seed)
    if quick:
        scheme = DitherScheme.uniform(z_min=0.5, z_max=2.5, dz=0.25, n_stagger=1)
        n_sources, image_size_24, noise_std = 15_000, 512, 2e-6
    else:
        scheme = DitherScheme.uniform(z_min=0.5, z_max=3.5, dz=0.10, n_stagger=2)
        n_sources, image_size_24, noise_std = 60_000, 1024, 2e-6
    sigma_z0, f_cat = 0.01, 0.01
    truth = TruthSpectrum(continuum_amp=1e-3)  # Jy at the warm-MBB peak

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="pah_endtoend_"))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"workdir: {workdir}")
        print(
            f"{n_sources} sources, {scheme.n_zbins} dither bins, "
            f"{len(scheme.runs)} runs"
        )

    # --- catalog: true z drives the maps, photo-z goes in the catalog ----
    dndz = make_dndz("cosmos_like")
    z_true = _draw_redshifts(rng, n_sources, dndz, 0.3, scheme.z_max + 0.3)
    z_phot = z_true + rng.normal(0, sigma_z0 * (1 + z_true))
    is_cat = rng.random(n_sources) < f_cat
    z_phot[is_cat] = rng.uniform(scheme.z_min, scheme.z_max, is_cat.sum())

    ra0, dec0 = 150.0, 2.0
    field_deg = image_size_24 * BAND_SPECS["MIPS_24"]["pixel_scale"] / 3600.0
    margin = 0.03 * field_deg
    ra = ra0 + rng.uniform(-field_deg / 2 + margin, field_deg / 2 - margin, n_sources)
    dec = dec0 + rng.uniform(-field_deg / 2 + margin, field_deg / 2 - margin, n_sources)

    catalog = pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "z_peak": z_phot,
            "lmass": np.full(n_sources, 10.5),
            "sfg": np.zeros(n_sources, dtype=int),
        }
    )
    catalog_path = workdir / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    # --- maps: per-source band fluxes at TRUE redshift ---------------------
    z_fine = np.linspace(0.2, scheme.z_max + 0.5, 1200)
    map_configs = []
    for band, spec in BAND_SPECS.items():
        image_size = int(round(field_deg * 3600.0 / spec["pixel_scale"]))
        wcs = _make_wcs(ra0, dec0, spec["pixel_scale"], image_size)
        flux_src = np.interp(z_true, z_fine, truth.band_flux_curve(z_fine, band))
        data = _inject_and_convolve(
            flux_src,
            ra,
            dec,
            wcs,
            image_size,
            spec["beam_fwhm"],
            spec["pixel_scale"],
            noise_std,
            rng,
        )
        map_path = workdir / f"{band}_signal.fits"
        noise_path = workdir / f"{band}_noise.fits"
        _write_map(map_path, data, wcs, spec["beam_fwhm"], spec["wavelength"])
        header = wcs.to_header()
        fits.PrimaryHDU(
            data=np.full(data.shape, noise_std, dtype=np.float32), header=header
        ).writeto(noise_path, overwrite=True)
        map_configs.append(
            {
                "name": band,
                "path_map": str(map_path),
                "path_noise": str(noise_path),
                "wavelength": spec["wavelength"],
                "beam_fwhm": spec["beam_fwhm"],
            }
        )
        if verbose:
            print(
                f"  {band}: {image_size}px map, "
                f"mean source flux {flux_src.mean():.2e} Jy"
            )

    # --- one stacking run per dither offset --------------------------------
    from simstack4.config import load_config
    from simstack4.wrapper import SimstackWrapper

    rows = []
    for run_id, edges in enumerate(scheme.runs):
        output_dir = workdir / f"output_run{run_id}"
        output_dir.mkdir(exist_ok=True)
        config_path = workdir / f"config_run{run_id}.toml"
        _write_config(
            config_path,
            catalog_path,
            map_configs,
            output_dir,
            [round(float(e), 3) for e in edges],
            bootstrap_iterations=4 if quick else 8,
        )
        if verbose:
            print(f"stacking run {run_id} ({len(edges) - 1} z-bins)...")
        config = load_config(config_path)
        wrapper = SimstackWrapper(
            config, read_maps=True, read_catalog=True, stack_automatically=True
        )
        results = wrapper.stacking_results
        pops = wrapper.population_manager.populations
        for p_idx, label in enumerate(results.population_labels):
            if label == "foreground" or label not in pops:
                continue
            z_lo, z_hi = pops[label].bin_ranges["redshift"]
            row = {
                "run_id": run_id,
                "zbin_id": int(np.searchsorted(edges, z_lo + 1e-6) - 1),
                "z_lo": z_lo,
                "z_hi": z_hi,
                "z_mid": 0.5 * (z_lo + z_hi),
                "prop_bin_id": 0,
                "log_M_star": 10.5,
                "log_sigma_sfr": 0.0,
                "n_sources": pops[label].n_sources,
            }
            for band in BAND_SPECS:
                flux = float(results.flux_densities[band][p_idx])
                err = float(results.flux_errors[band][p_idx])
                if not np.isfinite(err) or err <= 0:
                    err = float(results.flux_errors_systematic[band][p_idx])
                row[band] = flux
                row[f"{band}_err"] = max(err, 1e-8)
            rows.append(row)
    df = pd.DataFrame(rows)

    # --- deconvolve ---------------------------------------------------------
    model = PAHSpectrumModel(sigma_z0=sigma_z0, f_cat=f_cat)
    result = model.fit_lstsq(df, scheme=scheme, dndz=dndz)

    A_true = truth.amplitudes()
    comparison = pd.DataFrame(
        {
            "group": result.labels,
            "A_injected": A_true,
            "A_recovered": result.A[0],
            "A_err": result.A_err[0],
            "ratio": result.A[0] / A_true,
        }
    )
    if verbose:
        print()
        print(
            f"T_w = {result.theta_global[0]:.1f} ± {result.theta_err[0]:.1f} K "
            f"(injected {truth.T_warm}); chi2_red = {result.chi2_red:.2f}"
        )
        print(comparison.round(4).to_string(index=False))
    return {
        "result": result,
        "truth": truth,
        "df": df,
        "comparison": comparison,
        "workdir": workdir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_endtoend(args.workdir, quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
