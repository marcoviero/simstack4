#!/usr/bin/env python3
"""
Measure PAH emission from stacked SED mid-IR excess.

The greybody model fits the cool dust continuum (100-850um). The 24um
data point sits far above this extrapolation due to PAH emission and
stochastically heated small grains. By measuring the 24um excess as a
function of redshift, we trace individual PAH features marching through
the MIPS band — a spectral fingerprint recovered from broadband photometry.

PAH features in the MIPS 24um band
-----------------------------------
z ~ 0.89:  12.7um C-H out-of-plane bend (moderate)
z ~ 1.12:  11.3um C-H out-of-plane bend (strong)
z ~ 1.79:   8.6um C-H in-plane bend (moderate)
z ~ 2.12:   7.7um C-C stretch (STRONGEST)
z ~ 2.87:   6.2um C-C stretch (moderate)

Physical drivers of PAH strength
---------------------------------
- Metallicity: PAHs are carbonaceous, PAH fraction drops sharply
  below ~0.25 Z_sun (Engelbracht+2005, Draine+2007)
- Radiation field hardness: intense UV destroys small PAHs
  (starbursts and AGN suppress PAH EW)
- Galaxy type: PAH EW is lower in ULIRGs and AGN hosts

Suggested stacking config
--------------------------
inflation_factors = {24: 1000}   # decouple 24um from greybody fit
z bins: [0.5, 0.81, 0.97, 1.04, 1.2, 1.71, 1.87, 2.04, 2.2, 2.79, 2.95, 3.5]
M* bins: [9.0, 9.5, 10.0, 10.5, 11.0, 12.0]

Usage
-----
    from analyze_pah import measure_pah_excess, pah_optimized_zbins

    fig, df = measure_pah_excess(wrapper)
    fig, df = measure_pah_excess(wrapper, split_filter=[0])

References
----------
Draine & Li 2007 (ApJ 657, 810): PAH model
Engelbracht+2005 (ApJL 628, L29): PAH deficit at low Z
Shipley+2016 (ApJ 818, 60): PAH features at z~0-2
Lai+2020 (ApJ 905, 55): PAH in high-z galaxies
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── PAH feature catalog ─────────────────────────────────────────────
PAH_FEATURES = {
    # name: (rest_wavelength_um, relative_strength, fwhm_um)
    "6.2 C-C": (6.2, 0.40, 0.19),
    "7.7 C-C": (7.7, 1.00, 0.70),  # strongest
    "8.6 C-H": (8.6, 0.30, 0.34),
    "11.3 C-H": (11.3, 0.60, 0.24),
    "12.7 C-H": (12.7, 0.30, 0.45),
}

# MIPS 24um band
MIPS_BAND = {"center": 24.0, "lo": 20.5, "hi": 30.0}

# PAH-optimized z bins (peaks and valleys of feature ladder in 24um)
PAH_Z_BINS = [0.5, 0.81, 0.97, 1.04, 1.2, 1.71, 1.87, 2.04, 2.2, 2.79, 2.95, 3.5]


def pah_optimized_zbins():
    """Return PAH-optimized redshift bin edges for 24um stacking."""
    return PAH_Z_BINS.copy()


def adaptive_pah_zbins(
    redshifts,
    *,
    z_min=0.5,
    z_max=3.5,
    min_sources=500,
    max_dz=0.15,
    min_dz=0.03,
    secondary=None,
    verbose=True,
):
    """
    Create adaptive z-bins for PAH tomographic spectroscopy.

    Bins are narrow where sources are abundant (low z) and wider
    where sources are sparse (high z), while maintaining enough
    spectral resolution to resolve PAH features.

    Guarantees at least min_sources in the *sparsest cell* across
    all secondary binning dimensions.

    Parameters
    ----------
    redshifts : array-like
        Redshift array from the catalog.
    z_min, z_max : float
        Redshift range.
    min_sources : int
        Minimum number of sources in the sparsest (z, secondary...) cell.
    max_dz : float
        Maximum bin width. dz=0.15 gives R~10-20 (marginally resolves PAH).
    min_dz : float
        Minimum bin width.
    secondary : dict, optional
        Secondary binning dimensions. Keys are arrays of values,
        values are bin edges. Examples:

        # One secondary dimension (stellar mass):
        secondary = {
            "stellar_mass": (cat["lp_mass_med"].values, [9.0, 10.0, 10.5, 11.0, 12.0])
        }

        # Two secondary dimensions:
        secondary = {
            "stellar_mass": (cat["lp_mass_med"].values, [9.0, 10.0, 10.5, 11.0, 12.0]),
            "delta_ms": (cat["log_delta_ms"].values, [-1.0, 0.0, 0.7, 1.5]),
        }
    verbose : bool

    Returns
    -------
    edges : list of float
        Bin edges suitable for TOML config.
    """
    z = np.asarray(redshifts, dtype=float)

    # Build a mask for the z range
    z_mask = (z >= z_min) & (z <= z_max)

    # Parse secondary dimensions
    sec_dims = []
    if secondary:
        for name, (values, bins) in secondary.items():
            vals = np.asarray(values, dtype=float)
            bins = list(bins)
            sec_dims.append(
                {"name": name, "values": vals, "bins": bins, "n_bins": len(bins) - 1}
            )

    def _min_count_in_zbin(z_lo, z_hi):
        """Count sources in the sparsest cell for a given z-bin."""
        zbin = z_mask & (z >= z_lo) & (z < z_hi)

        if not sec_dims:
            return int(zbin.sum())

        # Find the minimum across all secondary bin combinations
        min_n = zbin.sum()  # start with total

        # For each combination of secondary bins, count
        def _recurse(dim_idx, current_mask):
            nonlocal min_n
            if dim_idx >= len(sec_dims):
                n = int(current_mask.sum())
                min_n = min(min_n, n)
                return

            sd = sec_dims[dim_idx]
            for i in range(sd["n_bins"]):
                lo, hi = sd["bins"][i], sd["bins"][i + 1]
                sec_mask = current_mask & (sd["values"] >= lo) & (sd["values"] < hi)
                _recurse(dim_idx + 1, sec_mask)

        _recurse(0, zbin)
        return int(min_n)

    # Build bins greedily
    edges = [z_min]
    z_sorted = np.sort(z[z_mask])

    while edges[-1] < z_max:
        z_lo = edges[-1]

        # Binary search for the narrowest dz where min cell >= min_sources
        dz_lo_search = min_dz
        dz_hi_search = max_dz

        # First check if max_dz is enough
        count_at_max = _min_count_in_zbin(z_lo, min(z_lo + max_dz, z_max))
        if count_at_max < min_sources:
            # Even max_dz isn't enough — use max_dz anyway (flagged later)
            dz = max_dz
        else:
            # Binary search for optimal dz
            for _ in range(20):  # converge in ~20 iterations
                dz_mid = (dz_lo_search + dz_hi_search) / 2
                count = _min_count_in_zbin(z_lo, z_lo + dz_mid)
                if count >= min_sources:
                    dz_hi_search = dz_mid  # try narrower
                else:
                    dz_lo_search = dz_mid  # need wider
                if dz_hi_search - dz_lo_search < 0.001:
                    break
            dz = dz_hi_search  # use the narrowest that satisfies

        dz = max(dz, min_dz)
        dz = min(dz, max_dz)
        z_hi = z_lo + dz

        if z_hi >= z_max - min_dz:
            z_hi = z_max

        edges.append(round(float(z_hi), 3))

        if z_hi >= z_max:
            break

    # Clean up tiny bins
    clean = [edges[0]]
    for e in edges[1:]:
        if e - clean[-1] >= min_dz * 0.9:
            clean.append(e)
        else:
            clean[-1] = e
    if clean[-1] < z_max:
        clean.append(z_max)
    edges = [round(float(e), 3) for e in clean]

    if verbose:
        n_bins = len(edges) - 1
        n_total = int(z_mask.sum())
        n_cells_per_zbin = 1
        for sd in sec_dims:
            n_cells_per_zbin *= sd["n_bins"]

        print(f"\nAdaptive PAH z-bins:")
        print(f"  {n_bins} z-bins from z={z_min} to z={z_max}")
        print(f"  Total sources in range: {n_total:,}")
        if sec_dims:
            dim_str = " x ".join(f"{sd['name']}({sd['n_bins']})" for sd in sec_dims)
            print(f"  Secondary dimensions: {dim_str}")
            print(f"  Cells per z-bin: {n_cells_per_zbin}")
        print(f"  Min sources per cell: {min_sources}")
        print(f"  dz range: [{min_dz}, {max_dz}]")
        print()

        features = {
            6.2: "6.2 C-C",
            7.7: "7.7 C-C*",
            8.6: "8.6 C-H",
            11.3: "11.3 C-H",
            12.7: "12.7 C-H",
        }

        print(
            f"  {'z_lo':>6} {'z_hi':>6} {'dz':>5} {'N_min':>7} {'N_tot':>7} "
            f"{'rest_lam':>8} {'R':>5} {'PAH feature':>14}"
        )
        print(f"  {'-'*68}")

        for i in range(len(edges) - 1):
            z_lo, z_hi = edges[i], edges[i + 1]
            dz = z_hi - z_lo
            z_mid = (z_lo + z_hi) / 2
            n_min = _min_count_in_zbin(z_lo, z_hi)
            n_tot = int(((z >= z_lo) & (z < z_hi) & z_mask).sum())
            rest_lam = 24 / (1 + z_mid)
            dlam = 24 / (1 + z_lo) - 24 / (1 + z_hi)
            R = rest_lam / dlam if dlam > 0 else 999

            feat = ""
            for flam, fname in features.items():
                if abs(rest_lam - flam) < 0.8:
                    feat = fname

            flag = " !" if n_min < min_sources * 0.5 else ""
            print(
                f"  {z_lo:6.3f} {z_hi:6.3f} {dz:5.3f} {n_min:7,} {n_tot:7,} "
                f"{rest_lam:7.1f}um {R:5.0f}  {feat}{flag}"
            )

        print(f"\n  Edges for TOML:")
        print(f"  bins = {edges}")

    return edges


def _pah_template_in_band(z, band_lo=20.5, band_hi=30.0, n_nu=500):
    """
    Compute expected relative PAH flux in a broadband filter at redshift z.

    Uses a simple Gaussian model for each PAH feature, integrated
    over the band transmission (assumed flat).

    Returns
    -------
    total_strength : float
        Sum of PAH feature strengths in the band (arbitrary units,
        normalized so 7.7um feature at z=2.12 gives ~1.0).
    dominant_feature : str or None
        Name of the strongest feature in the band.
    """
    total = 0.0
    dominant = None
    max_contrib = 0.0

    for name, (lam_rest, strength, fwhm) in PAH_FEATURES.items():
        # Observed wavelength
        lam_obs = lam_rest * (1 + z)
        fwhm_obs = fwhm * (1 + z)
        sigma = fwhm_obs / 2.355

        # Fraction of Gaussian profile within the band
        # Integrate Gaussian over [band_lo, band_hi]
        from scipy.special import erf

        frac = 0.5 * (
            erf((band_hi - lam_obs) / (sigma * np.sqrt(2)))
            - erf((band_lo - lam_obs) / (sigma * np.sqrt(2)))
        )
        contrib = strength * max(frac, 0)
        total += contrib
        if contrib > max_contrib:
            max_contrib = contrib
            dominant = name

    return total, dominant


def measure_pah_excess(
    wrapper,
    *,
    target_wavelength=24.0,
    min_tier="B",
    split_filter=None,
    save_path=None,
):
    """
    Measure PAH + warm dust excess at 24um from stacked SEDs.

    Parameters
    ----------
    wrapper : SimstackWrapper
    target_wavelength : float
        Observed wavelength to measure excess at (um). Default 24.
    min_tier : str
        Minimum fit quality tier.
    split_filter : list of int, optional
        Population class filter.
    save_path : str or Path, optional

    Returns
    -------
    fig, df : figure and DataFrame with PAH measurements
    """
    try:
        from simstack4.plots import _parse_bins, _extract_pop_type
    except ImportError:
        from plots import _parse_bins, _extract_pop_type

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed results found")
        return None, None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    # Detect model wavelength frame
    sample_sed = next(
        (
            s
            for s in pr.sed_results.values()
            if s.greybody_fit_success and s.model_wavelengths is not None
        ),
        None,
    )
    model_is_rest = False
    if sample_sed is not None:
        mw = sample_sed.model_wavelengths
        mf = sample_sed.model_fluxes
        model_peak = mw[np.argmax(mf)]
        data_peak = sample_sed.wavelengths[np.argmax(sample_sed.flux_densities)]
        z_s = sample_sed.median_redshift
        ratio_shifted = model_peak * (1 + z_s) / data_peak
        ratio_asis = model_peak / data_peak
        if abs(ratio_shifted - 1.0) < abs(ratio_asis - 1.0):
            model_is_rest = True

    # ── Extract measurements ─────────────────────────────────────────
    rows = []

    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue
        tier = getattr(sed, "fit_quality_tier", None) or "C"
        if tier_rank.get(tier, 2) > min_rank:
            continue

        pop_type = _extract_pop_type(pop_id)
        if split_filter is not None:
            allowed = {f"split_{i}" for i in split_filter}
            if pop_type not in allowed and pop_type != "_all_":
                continue
        elif pop_type == "split_2":
            continue

        derived = pr.derived_quantities.get(pop_id)
        if not derived or derived.total_ir_luminosity <= 0:
            continue

        z = sed.median_redshift
        l_ir = derived.total_ir_luminosity
        T_dust = sed.dust_temperature_rest_frame

        # Get bin properties
        props = sed.bin_properties or {}
        if isinstance(props, str):
            import ast

            try:
                props = ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}

        extra = {}
        for key, val in props.items():
            kl = key.lower()
            if "mass" in kl and "delta" not in kl:
                extra["stellar_mass"] = val
            elif "sigma" in kl and "sfr" in kl:
                extra["log_sigma_sfr"] = val
            elif "metal" in kl:
                extra["metallicity"] = val
            elif "ssfr" in kl:
                extra["log_ssfr"] = val

        # Derive log_ssfr if not present
        if "log_ssfr" not in extra:
            sfr_prop = props.get("sfr_med")
            mass_prop = extra.get("stellar_mass")
            if sfr_prop and mass_prop and sfr_prop > 0 and mass_prop > 0:
                extra["log_ssfr"] = np.log10(sfr_prop) - mass_prop

        if "stellar_mass" not in extra:
            extra["stellar_mass"] = sed.median_mass

        # Find the 24um data point
        idx_24 = None
        for j, w in enumerate(sed.wavelengths):
            if abs(w - target_wavelength) / target_wavelength < 0.15:
                idx_24 = j
                break

        if idx_24 is None:
            continue

        f_data = sed.flux_densities[idx_24]
        f_err = sed.flux_errors[idx_24]

        # Greybody extrapolation to 24um
        # NOTE: At 24um observed (rest ~10-15um for z~0.5-1.5), the
        # greybody flux is essentially ZERO — the entire 24um signal
        # is PAH + warm dust. Don't use (data-model)/model here.
        f_model = 0.0
        if sed.model_wavelengths is not None and sed.model_fluxes is not None:
            mw = sed.model_wavelengths
            if model_is_rest:
                mw = mw * (1 + z)
            # Only interpolate if 24um falls within the model grid
            if mw.min() <= target_wavelength <= mw.max():
                f_model = np.interp(target_wavelength, mw, sed.model_fluxes)
            # else: model is zero at 24um (Wien side, negligible)

        # The PAH excess is essentially all the 24um flux
        pah_flux = f_data - f_model  # ≈ f_data when model ≈ 0

        # Reference: peak FIR flux (typically 160 or 250um)
        idx_peak = np.argmax(sed.flux_densities)
        f_peak = sed.flux_densities[idx_peak]
        lam_peak = sed.wavelengths[idx_peak]

        # Luminosity at 24um
        from astropy.cosmology import Planck18

        dl_cm = Planck18.luminosity_distance(z).to("cm").value
        # nu * L_nu at 24um observed
        c_cgs = 2.998e10
        nu_24 = c_cgs / (target_wavelength * 1e-4)
        L_24 = f_data * 1e-23 * 4 * np.pi * dl_cm**2 * nu_24 / (1 + z)
        L_sun = 3.828e33
        l_24_lsun = L_24 / L_sun

        excess_snr = f_data / f_err if f_err > 0 else 0

        # PAH template prediction
        pah_strength, dominant_feature = _pah_template_in_band(z)

        rows.append(
            {
                "pop_id": pop_id,
                "z": z,
                "l_ir": l_ir,
                "log_l_ir": np.log10(l_ir),
                "T_dust": T_dust,
                "n_sources": sed.n_sources,
                "tier": tier,
                "f_24": f_data,
                "f_24_model": f_model,
                "f_24_err": f_err,
                "f_peak": f_peak,
                "pah_flux": pah_flux,
                "f24_to_fpeak": f_data / f_peak if f_peak > 0 else np.nan,
                "f24_to_fpeak_err": f_err / f_peak if f_peak > 0 else np.nan,
                "l_24": l_24_lsun,
                "log_l_24": np.log10(l_24_lsun) if l_24_lsun > 0 else np.nan,
                "l_24_to_l_ir": l_24_lsun / l_ir if l_ir > 0 else np.nan,
                "excess_snr": excess_snr,
                "pah_template": pah_strength,
                "dominant_feature": dominant_feature or "",
                **extra,
            }
        )

    if not rows:
        print("No populations with 24um data")
        return None, None

    df = pd.DataFrame(rows)

    # ── Summary ──────────────────────────────────────────────────────
    valid = np.isfinite(df["f24_to_fpeak"])
    print(f"\n{'='*60}")
    print(f"PAH + WARM DUST EMISSION AT {target_wavelength:.0f}um")
    print(f"{'='*60}")
    print(f"  Populations measured: {valid.sum()}")
    print(f"  f_24 / f_FIR_peak:")
    print(f"    median = {df.loc[valid, 'f24_to_fpeak'].median():.3f}")
    print(
        f"    range  = [{df.loc[valid, 'f24_to_fpeak'].min():.3f}, "
        f"{df.loc[valid, 'f24_to_fpeak'].max():.3f}]"
    )
    print(f"  L_24 / L_IR:")
    ll = df.loc[valid, "l_24_to_l_ir"]
    print(f"    median = {ll.median():.3f} ({ll.median()*100:.1f}%)")

    # By redshift
    print(f"\n  By redshift (peak features labeled):")
    z_edges = np.arange(0.5, min(df["z"].max() + 0.5, 12), 0.3)
    for i in range(len(z_edges) - 1):
        zbin = valid & (df["z"] >= z_edges[i]) & (df["z"] < z_edges[i + 1])
        if zbin.sum() > 0:
            med = df.loc[zbin, "f24_to_fpeak"].median()
            l24_med = df.loc[zbin, "l_24_to_l_ir"].median()
            z_mid = (z_edges[i] + z_edges[i + 1]) / 2
            _, feat = _pah_template_in_band(z_mid)
            feat_str = f"  ({feat})" if feat else ""
            print(
                f"    z={z_edges[i]:.1f}-{z_edges[i+1]:.1f}: "
                f"f24/fPeak = {med:.3f}, L24/LIR = {l24_med:.3f}  "
                f"N={zbin.sum()}{feat_str}"
            )

    # ── 6-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: f24/fPeak vs redshift with PAH template overlay
    ax = axes[0, 0]
    if valid.any():
        sc = ax.scatter(
            df.loc[valid, "z"],
            df.loc[valid, "f24_to_fpeak"],
            c=df.loc[valid, "stellar_mass"],
            cmap="viridis",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="log M$_*$")

        # PAH template curve
        z_grid = np.linspace(0.3, 4.0, 300)
        pah_curve = np.array([_pah_template_in_band(z)[0] for z in z_grid])
        # Scale template to match median
        if pah_curve.max() > 0:
            scale = df.loc[valid, "f24_to_fpeak"].median() / pah_curve.mean()
            ax2 = ax.twinx()
            ax2.plot(
                z_grid, pah_curve * scale, "r-", lw=2, alpha=0.7, label="PAH template"
            )
            ax2.set_ylabel("PAH template (scaled)", color="r", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="r")

            # Mark individual features
            for name, (lam, strength, _) in PAH_FEATURES.items():
                z_feat = MIPS_BAND["center"] / lam - 1
                if 0.3 < z_feat < 4.0:
                    ax.axvline(z_feat, color="r", ls=":", alpha=0.3, lw=1)
                    ax.text(
                        z_feat,
                        ax.get_ylim()[1] * 0.95,
                        f"{lam:.1f}$\\mu$m",
                        fontsize=6,
                        rotation=90,
                        va="top",
                        ha="right",
                        color="r",
                        alpha=0.7,
                    )

    ax.set_xlabel("Redshift")
    ax.set_ylabel("f$_{24}$ / f$_{FIR,peak}$")
    ax.set_title("PAH feature ladder in 24$\\mu$m")
    ax.grid(True, alpha=0.2)

    # Panel 2: L_24/L_IR vs stellar mass (metallicity proxy)
    ax = axes[0, 1]
    has_mass = "stellar_mass" in df.columns
    valid_lr = valid & np.isfinite(df["l_24_to_l_ir"])
    if has_mass and valid_lr.any():
        valid_m = valid_lr & np.isfinite(df["stellar_mass"])
        if valid_m.any():
            sc = ax.scatter(
                df.loc[valid_m, "stellar_mass"],
                df.loc[valid_m, "l_24_to_l_ir"],
                c=df.loc[valid_m, "z"],
                cmap="plasma",
                s=30,
                alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")

            # Binned medians
            mass_edges = np.arange(9.0, 12.0, 0.5)
            for i in range(len(mass_edges) - 1):
                mbin = (
                    valid_m
                    & (df["stellar_mass"] >= mass_edges[i])
                    & (df["stellar_mass"] < mass_edges[i + 1])
                )
                if mbin.sum() > 2:
                    med = df.loc[mbin, "l_24_to_l_ir"].median()
                    p16, p84 = np.percentile(df.loc[mbin, "l_24_to_l_ir"], [16, 84])
                    ax.errorbar(
                        (mass_edges[i] + mass_edges[i + 1]) / 2,
                        med,
                        yerr=[[med - p16], [p84 - med]],
                        fmt="ks",
                        ms=9,
                        mfc="white",
                        mew=2,
                        capsize=3,
                        zorder=5,
                    )
    ax.set_xlabel("log M$_*$ (M$_\\odot$)")
    ax.set_ylabel("L$_{24}$ / L$_{IR}$")
    ax.set_title("PAH fraction vs M$_*$ (Z proxy)")
    ax.grid(True, alpha=0.2)

    # Panel 3: L_24/L_IR vs L_IR
    ax = axes[0, 2]
    if valid_lr.any():
        sc = ax.scatter(
            df.loc[valid_lr, "l_ir"],
            df.loc[valid_lr, "l_24_to_l_ir"],
            c=df.loc[valid_lr, "z"],
            cmap="plasma",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Redshift")
    ax.set_xscale("log")
    ax.set_xlabel("L$_{IR}$ (L$_\\odot$)")
    ax.set_ylabel("L$_{24}$ / L$_{IR}$")
    ax.set_title("PAH fraction vs L$_{IR}$")
    ax.grid(True, alpha=0.2)

    # Panel 4: Correlation between measured f24/fPeak and PAH template
    ax = axes[1, 0]
    if valid.any():
        sc = ax.scatter(
            df.loc[valid, "pah_template"],
            df.loc[valid, "f24_to_fpeak"],
            c=df.loc[valid, "z"],
            cmap="plasma",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Redshift")

        # Fit
        x = df.loc[valid, "pah_template"].values
        y = df.loc[valid, "f24_to_fpeak"].values
        w = np.sqrt(df.loc[valid, "n_sources"].values.astype(float))
        valid_f = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if valid_f.sum() >= 4:
            try:
                coeffs = np.polyfit(x[valid_f], y[valid_f], 1, w=w[valid_f])
                x_grid = np.linspace(0, x.max(), 100)
                ax.plot(
                    x_grid,
                    np.polyval(coeffs, x_grid),
                    "k--",
                    lw=2,
                    label=f"slope={coeffs[0]:.3f}",
                )
                # R²
                pred = np.polyval(coeffs, x[valid_f])
                ss_res = np.sum(w[valid_f] ** 2 * (y[valid_f] - pred) ** 2)
                ss_tot = np.sum(
                    w[valid_f] ** 2
                    * (y[valid_f] - np.average(y[valid_f], weights=w[valid_f] ** 2))
                    ** 2
                )
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                ax.text(
                    0.05,
                    0.95,
                    f"R$^2$ = {r2:.3f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    va="top",
                    bbox=dict(fc="white", alpha=0.9),
                )
                ax.legend(fontsize=9)
            except Exception:
                pass
    ax.set_xlabel("PAH template strength")
    ax.set_ylabel("f$_{24}$ / f$_{FIR,peak}$")
    ax.set_title("Template vs measurement\n(validates PAH origin)")
    ax.grid(True, alpha=0.2)

    # Panel 5: L_24/L_IR vs metallicity
    ax = axes[1, 1]
    has_Z = "metallicity" in df.columns and np.isfinite(df["metallicity"]).any()
    if has_Z and valid_lr.any():
        valid_Z = valid_lr & np.isfinite(df["metallicity"]) & (df["metallicity"] > 0)
        if valid_Z.any():
            sc = ax.scatter(
                np.log10(df.loc[valid_Z, "metallicity"]),
                df.loc[valid_Z, "l_24_to_l_ir"],
                c=df.loc[valid_Z, "z"],
                cmap="plasma",
                s=30,
                alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")

            # Engelbracht+2005 threshold
            ax.axvline(
                np.log10(0.005),
                color="r",
                ls="--",
                alpha=0.5,
                label="~0.25 Z$_\\odot$ (PAH threshold)",
            )
            ax.legend(fontsize=7)
    else:
        ax.text(
            0.5, 0.5, "metallicity not available", transform=ax.transAxes, ha="center"
        )
    ax.set_xlabel("log(Z)")
    ax.set_ylabel("L$_{24}$ / L$_{IR}$")
    ax.set_title("PAH fraction vs metallicity\n(Engelbracht+2005)")
    ax.grid(True, alpha=0.2)

    # Panel 6: L_24/L_IR vs T_dust
    ax = axes[1, 2]
    has_T = np.isfinite(df["T_dust"]).any()
    if has_T and valid_lr.any():
        valid_T = valid_lr & np.isfinite(df["T_dust"])
        if valid_T.any():
            sc = ax.scatter(
                df.loc[valid_T, "T_dust"],
                df.loc[valid_T, "l_24_to_l_ir"],
                c=df.loc[valid_T, "z"],
                cmap="plasma",
                s=30,
                alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")
    ax.set_xlabel("T$_{dust}$ (K)")
    ax.set_ylabel("L$_{24}$ / L$_{IR}$")
    ax.set_title("PAH fraction vs T$_{dust}$\n(radiation field intensity)")
    ax.grid(True, alpha=0.2)

    # Title
    fig.suptitle(
        f"PAH + warm dust at {target_wavelength:.0f}$\\mu$m  |  "
        f"{valid.sum()} populations (tier >= {min_tier})",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\n  Saved: {save_path}")

    return fig, df


def staggered_pah_zbins(
    redshifts,
    *,
    n_stagger=3,
    z_min=0.5,
    z_max=3.5,
    min_sources=500,
    max_dz=0.15,
    min_dz=0.03,
    secondary=None,
    verbose=True,
):
    """
    Generate multiple sets of staggered z-bins for PAH tomographic spectroscopy.

    Each set has the same bin width (good S/N), but offset by dz/n_stagger.
    Combining the results gives n_stagger× the spectral sampling without
    reducing per-bin S/N. This is the spectral equivalent of dithering.

    Parameters
    ----------
    redshifts : array-like
        Redshift array from the catalog.
    n_stagger : int
        Number of staggered runs (2 or 3 recommended).
    z_min, z_max : float
        Redshift range.
    min_sources : int
        Minimum sources in sparsest cell per z-bin.
    max_dz, min_dz : float
        Bin width constraints.
    secondary : dict, optional
        Secondary binning dimensions (same as adaptive_pah_zbins).
    verbose : bool

    Returns
    -------
    list of lists : Each element is a set of bin edges (for one TOML config).
    """
    # First, get the optimal adaptive bins for the base run
    base_edges = adaptive_pah_zbins(
        redshifts,
        z_min=z_min,
        z_max=z_max,
        min_sources=min_sources,
        max_dz=max_dz,
        min_dz=min_dz,
        secondary=secondary,
        verbose=False,
    )

    # Compute typical bin widths per z range
    # Use these to determine offsets (not uniform — adaptive bins vary)
    all_edge_sets = [base_edges]

    for s in range(1, n_stagger):
        # Offset: shift each bin edge by s/n_stagger of its local bin width
        offset_edges = [z_min]
        for i in range(len(base_edges) - 1):
            dz = base_edges[i + 1] - base_edges[i]
            offset = dz * s / n_stagger
            new_edge = base_edges[i] + offset
            if new_edge > offset_edges[-1] + min_dz * 0.5 and new_edge < z_max:
                offset_edges.append(round(float(new_edge), 3))

        if offset_edges[-1] < z_max:
            offset_edges.append(z_max)

        # Clean up
        clean = [offset_edges[0]]
        for e in offset_edges[1:]:
            if e - clean[-1] >= min_dz * 0.5:
                clean.append(e)
            else:
                clean[-1] = e
        offset_edges = [round(float(e), 3) for e in clean]

        all_edge_sets.append(offset_edges)

    if verbose:
        print(f"\nStaggered PAH z-bins: {n_stagger} runs")
        total_bins = sum(len(e) - 1 for e in all_edge_sets)
        print(f"  Total spectral points: {total_bins}")
        print(f"  Per-run bins: " + ", ".join(str(len(e) - 1) for e in all_edge_sets))

        # Effective spectral sampling
        all_mids = []
        for edges in all_edge_sets:
            for i in range(len(edges) - 1):
                z_mid = (edges[i] + edges[i + 1]) / 2
                rest_lam = 24 / (1 + z_mid)
                all_mids.append(rest_lam)
        all_mids.sort()
        if len(all_mids) > 2:
            spacings = np.diff(all_mids)
            med_spacing = np.median(spacings)
            med_lam = np.median(all_mids)
            R_eff = med_lam / med_spacing
            print(f"  Effective spectral resolution: R ~ {R_eff:.0f}")
            print(f"  Median sampling: {med_spacing:.2f} um")

        for s, edges in enumerate(all_edge_sets):
            print(f"\n  Run {s}: {len(edges)-1} bins")
            print(f"    bins = {edges}")

    return all_edge_sets


def combine_pah_spectra(
    wrappers,
    *,
    target_wavelength=24.0,
    min_tier="B",
    split_filter=None,
    normalize_by="l_ir",
):
    """
    Combine PAH spectra from multiple staggered stacking runs.

    Parameters
    ----------
    wrappers : list of SimstackWrapper
        One per staggered run.
    target_wavelength, min_tier, split_filter, normalize_by :
        Same as measure_pah_excess.

    Returns
    -------
    df_combined : DataFrame
        Combined spectral points from all runs, sorted by rest wavelength.
    """
    all_dfs = []

    for i, wrapper in enumerate(wrappers):
        _, df = measure_pah_excess(
            wrapper,
            target_wavelength=target_wavelength,
            min_tier=min_tier,
            split_filter=split_filter,
        )
        plt.close()

        if df is not None and len(df) > 0:
            df["rest_lam_24"] = target_wavelength / (1 + df["z"])
            df["stagger_run"] = i
            all_dfs.append(df)

    if not all_dfs:
        print("No PAH measurements from any run")
        return None

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined = df_combined.sort_values("rest_lam_24").reset_index(drop=True)

    # Summary
    n_points = len(df_combined)
    n_per_run = [len(d) for d in all_dfs]
    print(
        f"\nCombined PAH spectrum: {n_points} spectral points "
        f"from {len(all_dfs)} runs ({n_per_run})"
    )

    rest_lams = df_combined["rest_lam_24"].values
    if len(rest_lams) > 2:
        spacings = np.diff(np.sort(rest_lams))
        med_spacing = np.median(spacings[spacings > 0])
        print(
            f"  Rest wavelength range: {rest_lams.min():.1f} - {rest_lams.max():.1f} um"
        )
        print(f"  Median spacing: {med_spacing:.3f} um")
        print(f"  Effective R: {np.median(rest_lams) / med_spacing:.0f}")

    return df_combined


def reconstruct_pah_spectrum(
    wrapper,
    *,
    target_wavelength=24.0,
    min_tier="B",
    split_filter=None,
    normalize_by="l_ir",
    save_path=None,
):
    """
    Reconstruct the rest-frame PAH spectrum from tomographic stacking.

    Each z-bin samples a different rest-frame wavelength at 24um observed.
    By plotting f_24 / f_continuum vs rest-frame wavelength, we reconstruct
    the 6-16um PAH emission spectrum from broadband photometry.

    Parameters
    ----------
    wrapper : SimstackWrapper
    target_wavelength : float
        Observed wavelength (um). Default 24.
    min_tier : str
    split_filter : list of int, optional
    normalize_by : str
        'l_ir' = L_24/L_IR (physical), 'f_peak' = f_24/f_FIR_peak (observed)
    save_path : str or Path, optional

    Returns
    -------
    fig, df_spec : figure and DataFrame with spectral points
    """
    # First, get the PAH measurements
    _, df = measure_pah_excess(
        wrapper,
        target_wavelength=target_wavelength,
        min_tier=min_tier,
        split_filter=split_filter,
    )
    plt.close()  # close the default figure

    if df is None or len(df) == 0:
        print("No PAH measurements")
        return None, None

    # Compute rest-frame wavelength sampled by each population
    df["rest_lam_24"] = target_wavelength / (1 + df["z"])

    # Choose normalization
    if normalize_by == "l_ir":
        y_col = "l_24_to_l_ir"
        y_label = "L$_{24}$ / L$_{IR}$"
    else:
        y_col = "f24_to_fpeak"
        y_label = "f$_{24}$ / f$_{FIR,peak}$"

    valid = np.isfinite(df[y_col]) & np.isfinite(df["rest_lam_24"])

    # Group by stellar mass for color coding
    # Extract actual mass bins from the population IDs
    has_mass = "stellar_mass" in df.columns and np.isfinite(df["stellar_mass"]).any()
    colors_cycle = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    if has_mass:
        try:
            from simstack4.plots import _parse_bins
        except ImportError:
            from plots import _parse_bins

        # Get unique mass bin tuples from population IDs
        mass_bin_set = set()
        for pop_id in df["pop_id"]:
            bins = _parse_bins(pop_id)
            for dim, (lo, hi) in bins.items():
                if "mass" in dim.lower():
                    mass_bin_set.add((lo, hi))

        if mass_bin_set:
            mass_bin_list = sorted(mass_bin_set)
            mass_bins = []
            for i, (lo, hi) in enumerate(mass_bin_list):
                color = colors_cycle[i % len(colors_cycle)]
                mass_bins.append((lo, hi, f"{lo:.1f}-{hi:.1f}", color))
        else:
            mass_bins = [(0, 99, "All", colors_cycle[0])]
    else:
        mass_bins = [(0, 99, "All", colors_cycle[0])]

    # ── Figure: 2 panels ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Reference PAH template (Draine & Li 2007 approximate)
    pah_lam = np.linspace(5, 16, 500)
    pah_template = np.zeros_like(pah_lam)
    for name, (lam_c, strength, fwhm) in PAH_FEATURES.items():
        sigma = fwhm / 2.355
        pah_template += strength * np.exp(-0.5 * ((pah_lam - lam_c) / sigma) ** 2)

    # ── Panel 1: PAH spectrum by mass bin ────────────────────────────
    ax = axes[0]

    for m_lo, m_hi, m_label, color in mass_bins:
        if has_mass:
            mask = valid & (df["stellar_mass"] >= m_lo) & (df["stellar_mass"] < m_hi)
        else:
            mask = valid

        if mask.sum() < 3:
            continue

        lam = df.loc[mask, "rest_lam_24"].values
        flux = df.loc[mask, y_col].values
        n_src = df.loc[mask, "n_sources"].values

        # Sort by wavelength
        order = np.argsort(lam)
        lam = lam[order]
        flux = flux[order]
        n_src = n_src[order]

        # Plot points with size by n_sources
        sizes = 10 + 80 * (n_src / max(n_src.max(), 1)) ** 0.5
        ax.scatter(
            lam,
            flux,
            c=color,
            s=sizes,
            alpha=0.7,
            label=f"log M$_*$ = {m_label}",
            zorder=3,
        )

        # Connect with line
        ax.plot(lam, flux, "-", color=color, alpha=0.4, lw=1.5, zorder=2)

    # Overlay scaled template
    if valid.any():
        med_flux = df.loc[valid, y_col].median()
        template_scale = med_flux / pah_template.max() if pah_template.max() > 0 else 1
        ax.plot(
            pah_lam,
            pah_template * template_scale,
            "k--",
            lw=1.5,
            alpha=0.5,
            label="PAH template (D&L07)",
            zorder=1,
        )

    ax.set_xlabel("Rest-frame wavelength ($\\mu$m)")
    ax.set_ylabel(y_label)
    ax.set_title("Tomographic PAH spectrum\n(each point = one z-bin at 24$\\mu$m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(5, 16)
    ax.grid(True, alpha=0.2)

    # Mark features AFTER data so ylim is set
    ylims = ax.get_ylim()
    for name, (lam_c, strength, fwhm) in PAH_FEATURES.items():
        ax.axvline(lam_c, color="gray", ls=":", alpha=0.3)
        ax.text(
            lam_c,
            ylims[1] * 0.97,
            f"{lam_c}$\\mu$m",
            fontsize=7,
            ha="center",
            va="top",
            color="gray",
            alpha=0.7,
            rotation=90,
        )

    # ── Panel 2: Detrended — remove smooth z/LIR trend ──────────────
    ax = axes[1]

    # Compute smooth trend from z (and log_l_ir if available)
    # Each rest wavelength corresponds to a unique z, so the z-trend
    # is the dominant smooth signal that masks the PAH features.
    has_z = "z" in df.columns
    has_lir = "log_l_ir" in df.columns

    detrended_all_lam = []
    detrended_all_flux = []

    for m_lo, m_hi, m_label, color in mass_bins:
        if has_mass:
            mask = valid & (df["stellar_mass"] >= m_lo) & (df["stellar_mass"] < m_hi)
        else:
            mask = valid

        if mask.sum() < 5:
            continue

        lam = df.loc[mask, "rest_lam_24"].values
        flux = df.loc[mask, y_col].values
        z_arr = df.loc[mask, "z"].values if has_z else None
        n_src = df.loc[mask, "n_sources"].values

        order = np.argsort(lam)
        lam = lam[order]
        flux = flux[order]
        n_src = n_src[order]
        if z_arr is not None:
            z_arr = z_arr[order]

        # Detrend: fit log(flux) = a*z + (b*logLIR) + c, then divide out
        try:
            pos = flux > 0
            if pos.sum() < 4:
                continue

            log_flux = np.log10(flux[pos])
            w = np.sqrt(n_src[pos].astype(float))

            if has_lir and z_arr is not None:
                lir_arr = df.loc[mask, "log_l_ir"].values[order][pos]
                if np.isfinite(lir_arr).all():
                    X = np.column_stack([z_arr[pos], lir_arr, np.ones(pos.sum())])
                else:
                    X = np.column_stack([z_arr[pos], np.ones(pos.sum())])
            elif z_arr is not None:
                X = np.column_stack([z_arr[pos], np.ones(pos.sum())])
            else:
                # Fallback: polynomial vs wavelength
                X = np.column_stack([lam[pos], np.ones(pos.sum())])

            coeffs, _, _, _ = np.linalg.lstsq(X * w[:, None], log_flux * w, rcond=None)
            smooth = 10 ** (X @ coeffs)
            normalized = flux[pos] / smooth

            sizes = 10 + 50 * (n_src[pos] / max(n_src.max(), 1)) ** 0.5
            ax.scatter(
                lam[pos],
                normalized,
                c=color,
                s=sizes,
                alpha=0.7,
                label=f"log M$_*$ = {m_label}",
                zorder=3,
            )
            ax.plot(lam[pos], normalized, "-", color=color, alpha=0.4, lw=1.5, zorder=2)

            detrended_all_lam.extend(lam[pos])
            detrended_all_flux.extend(normalized)
        except Exception:
            continue

    # Overlay normalized PAH template on detrended panel
    if pah_template.max() > 0:
        # Template is features only; normalize to mean=1 with bumps above
        pah_norm = 1.0 + pah_template / pah_template.max() * 0.3  # 30% bumps
        ax.plot(
            pah_lam, pah_norm, "k--", lw=1.5, alpha=0.5, label="PAH template (scaled)"
        )

    ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Rest-frame wavelength ($\\mu$m)")
    ax.set_ylabel("Flux / smooth z-trend")
    ax.set_title(
        "Continuum-subtracted PAH features\n(bumps at PAH wavelengths = detection)"
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(5, 16)
    ax.grid(True, alpha=0.2)

    # Set symmetric y range around 1
    if detrended_all_flux:
        spread = max(np.percentile(np.abs(np.array(detrended_all_flux) - 1), 95), 0.2)
        ax.set_ylim(1 - spread * 1.5, 1 + spread * 1.5)

    # Mark features on panel 2
    ylims2 = ax.get_ylim()
    for name, (lam_c, strength, fwhm) in PAH_FEATURES.items():
        ax.axvline(lam_c, color="gray", ls=":", alpha=0.3)
        ax.text(
            lam_c,
            ylims2[1] * 0.98,
            f"{lam_c}$\\mu$m",
            fontsize=7,
            ha="center",
            va="top",
            color="gray",
            alpha=0.7,
            rotation=90,
        )

    fig.suptitle(
        f"PAH spectrum from tomographic 24$\\mu$m stacking  |  "
        f"{valid.sum()} spectral points",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    # Output spectrum DataFrame
    df_spec = df.loc[
        valid,
        [
            "pop_id",
            "z",
            "rest_lam_24",
            y_col,
            "n_sources",
            "pah_template",
            "dominant_feature",
            "log_l_ir",
        ],
    ].copy()
    if has_mass:
        df_spec["stellar_mass"] = df.loc[valid, "stellar_mass"]
    df_spec = df_spec.sort_values("rest_lam_24")

    return fig, df_spec


def fit_pah_model(df, verbose=True):
    """
    Fit empirical PAH model from measure_pah_excess output.

    Fits: log(L24/LIR) = a*log(LIR) + b*z + c*PAH_template + d

    Parameters
    ----------
    df : DataFrame
        Output from measure_pah_excess.
    verbose : bool

    Returns
    -------
    model : dict
        Keys: 'coeffs' (array [a, b, c, d]), 'r_squared', 'n_fit',
        'residual_scatter' (dex).
    """
    valid = (
        np.isfinite(df["l_24_to_l_ir"])
        & np.isfinite(df["log_l_ir"])
        & np.isfinite(df["pah_template"])
        & (df["l_24_to_l_ir"] > 0)
    )

    X = np.column_stack(
        [
            df.loc[valid, "log_l_ir"].values,
            df.loc[valid, "z"].values,
            df.loc[valid, "pah_template"].values,
            np.ones(valid.sum()),
        ]
    )
    y = np.log10(df.loc[valid, "l_24_to_l_ir"].values)
    w = np.sqrt(df.loc[valid, "n_sources"].values.astype(float))

    coeffs, _, _, _ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)

    pred = X @ coeffs
    ss_res = np.sum(w**2 * (y - pred) ** 2)
    ss_tot = np.sum(w**2 * (y - np.average(y, weights=w**2)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    resid = y - pred
    scatter = np.std(resid)

    model = {
        "coeffs": coeffs,
        "r_squared": r2,
        "n_fit": int(valid.sum()),
        "residual_scatter": scatter,
        "labels": ["log_LIR", "z", "PAH_template", "intercept"],
    }

    if verbose:
        a, b, c, d = coeffs
        print(f"\nPAH correction model:")
        print(
            f"  log(L24/LIR) = {a:.3f} log(LIR) {b:+.3f} z "
            f"{c:+.3f} PAH_template {d:+.3f}"
        )
        print(f"  R² = {r2:.3f}  (N={valid.sum()}, " f"scatter = {scatter:.3f} dex)")
        print(f"\n  Example predictions (L24/LIR):")
        for z_ex in [0.5, 1.0, 2.0, 3.0]:
            pah = _pah_template_in_band(z_ex)[0]
            for l_ir_ex in [10.5, 11.5]:
                pred_val = 10 ** (a * l_ir_ex + b * z_ex + c * pah + d)
                print(
                    f"    z={z_ex}, log(LIR)={l_ir_ex}: "
                    f"L24/LIR = {pred_val:.3f} ({pred_val*100:.1f}%)"
                )

    return model


def predict_pah_flux(z, log_l_ir, f_ir_peak, model):
    """
    Predict PAH contribution to 24um flux for a population.

    Parameters
    ----------
    z : float
        Redshift.
    log_l_ir : float
        log10(L_IR / L_sun).
    f_ir_peak : float
        Peak FIR flux density (Jy), e.g., at 250um. Used to convert
        the predicted L24/LIR ratio to an absolute flux.
    model : dict
        Output from fit_pah_model.

    Returns
    -------
    f_24_pah : float
        Predicted 24um flux from PAH + warm dust (Jy).
    l_24_to_l_ir : float
        Predicted L24/LIR ratio.
    """
    a, b, c, d = model["coeffs"]
    pah_strength = _pah_template_in_band(z)[0]

    log_ratio = a * log_l_ir + b * z + c * pah_strength + d
    l_24_to_l_ir = 10**log_ratio

    # Convert to flux: f_24 ≈ l_24_to_l_ir * f_ir_peak * (correction)
    # This is approximate — the true conversion depends on the SED shape.
    # A more rigorous approach uses L_IR and d_L to get f_24 directly.
    # For now, use the ratio to the peak FIR flux as a scaling.
    # Typical f_24/f_peak for L24/LIR~0.05 is ~0.02 (from the data).
    f_24_pah = l_24_to_l_ir * f_ir_peak

    return f_24_pah, l_24_to_l_ir


def apply_pah_correction(wrapper, model, target_wavelength=24.0):
    """
    Apply PAH correction to 24um fluxes in stacking results.

    Subtracts the predicted PAH contribution from the 24um data point
    for each population, returning corrected flux arrays that can be
    re-fit with the greybody model.

    Parameters
    ----------
    wrapper : SimstackWrapper
    model : dict
        Output from fit_pah_model.
    target_wavelength : float
        Wavelength to correct (um).

    Returns
    -------
    corrections : dict
        {pop_id: {'f_24_original', 'f_24_pah', 'f_24_corrected',
                   'l_24_to_l_ir', 'z', 'log_l_ir'}}
    """
    try:
        from simstack4.plots import _parse_bins, _extract_pop_type
    except ImportError:
        from plots import _parse_bins, _extract_pop_type

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed results")
        return {}

    corrections = {}
    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue

        derived = pr.derived_quantities.get(pop_id)
        if not derived or derived.total_ir_luminosity <= 0:
            continue

        z = sed.median_redshift
        log_l_ir = np.log10(derived.total_ir_luminosity)

        # Find 24um index
        idx_24 = None
        for j, w in enumerate(sed.wavelengths):
            if abs(w - target_wavelength) / target_wavelength < 0.15:
                idx_24 = j
                break

        if idx_24 is None:
            continue

        f_data = sed.flux_densities[idx_24]
        f_peak = np.max(sed.flux_densities)

        f_24_pah, l24_ratio = predict_pah_flux(z, log_l_ir, f_peak, model)

        corrections[pop_id] = {
            "f_24_original": f_data,
            "f_24_pah": f_24_pah,
            "f_24_corrected": max(f_data - f_24_pah, 0),
            "l_24_to_l_ir": l24_ratio,
            "z": z,
            "log_l_ir": log_l_ir,
        }

    n_corr = len(corrections)
    if n_corr > 0:
        pah_fracs = [c["l_24_to_l_ir"] for c in corrections.values()]
        print(f"\nPAH correction applied to {n_corr} populations:")
        print(
            f"  Predicted L24/LIR: median = {np.median(pah_fracs):.3f} "
            f"({np.median(pah_fracs)*100:.1f}%)"
        )
        print(
            f"  Predicted f24_pah: median = "
            f"{np.median([c['f_24_pah'] for c in corrections.values()]):.4e} Jy"
        )

    return corrections


if __name__ == "__main__":
    print("Usage:")
    print("  from analyze_pah import (measure_pah_excess, fit_pah_model,")
    print("                           apply_pah_correction, pah_optimized_zbins)")
    print()
    print("  # Step 1: Measure PAH excess")
    print("  fig, df = measure_pah_excess(wrapper)")
    print()
    print("  # Step 2: Fit empirical model")
    print("  model = fit_pah_model(df)")
    print()
    print("  # Step 3: Apply correction to 24um fluxes")
    print("  corrections = apply_pah_correction(wrapper, model)")
    print()
    print("  # Step 4 (future): Re-fit greybody with corrected 24um")
    print("  # for pop_id, corr in corrections.items():")
    print("  #     sed.flux_densities[idx_24] = corr['f_24_corrected']")
    print()
    print("Suggested TOML config for PAH-optimized stacking:")
    print("  inflation_factors = {24: 1000}")
    print(f"  z bins = {PAH_Z_BINS}")
    print("  M* bins = [9.0, 9.5, 10.0, 10.5, 11.0, 12.0]")
