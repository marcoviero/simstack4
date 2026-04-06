#!/usr/bin/env python3
"""
Measure [CII] 158um (and other FIR line) intensity from stacked SEDs.

Tomographic line intensity mapping via broadband stacking: resolve the
aggregate line signal as a function of redshift AND galaxy physical
properties (Sigma_SFR, L_IR, M*, metallicity).

Method
------
For each stacked population, the greybody continuum fit provides a
model for the dust-only emission. Any excess in the photometric band
where a bright FIR line falls (at that population's redshift) is
attributed to line emission:

    F_line = F_data - F_continuum          (Jy, per source)
    L_line = F_line * 4 pi d_L^2 / (1+z)  (L_sun)
    L_line / L_FIR                          (the "[CII] deficit")

By binning galaxies in z x Sigma_SFR (or z x M*, z x metallicity),
we resolve how L_[CII]/L_FIR depends on the intensity of the
radiation field and the ISM properties — going beyond the z-only
measurement of Agrawal, Aguirre & Keenan (2026, A&A 705, A246).

Key correlations
----------------
- L_[CII]/L_FIR vs Sigma_SFR: the [CII] deficit (Diaz-Santos+2017)
- L_[CII]/L_FIR vs L_IR: classical deficit relation
- L_[CII] vs SFR: [CII]-SFR relation (De Looze+2014)
- L_[CII]/L_FIR vs metallicity: ISM cooling efficiency
- L_[CII]/L_FIR vs redshift: evolution of ISM conditions

References
----------
Agrawal, Aguirre & Keenan 2026 (A&A 705, A246): broadband [CII] stacking
Diaz-Santos+2017 (ApJL 846, L2): [CII] deficit vs Sigma_IR
De Looze+2014 (A&A 568, A62): [CII]-SFR relation
Smith+2017 (ApJ 834, 183): [CII]/FIR in local galaxies
Stacey+2010 (ApJ 724, 957): [CII] at high-z
Schaerer+2020 (A&A 643, A3): ALPINE [CII] at z~4.5
Zanella+2018 (MNRAS 481, 1976): [CII] deficit at z~2

Usage
-----
    from analyze_cii_lines import measure_line_intensity, measure_all_lines

    # [CII] measurement + 6-panel science plot
    fig, df = measure_line_intensity(wrapper)

    # Other lines
    fig, df = measure_line_intensity(wrapper, line="[NII] 205")

    # Summary of all bright FIR lines
    summary = measure_all_lines(wrapper)

    # Correlate with specific property
    fig, df = measure_line_intensity(wrapper, surface_color_by="metallicity")
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Line catalog ─────────────────────────────────────────────────────
LINES = {
    "[CII] 158":   157.74,
    "[NII] 205":   205.18,
    "[OI] 145":    145.53,
    "[NII] 122":   121.90,
    "[OIII] 88":    88.36,
    "[OI] 63":      63.18,
    "CO(3-2)":     866.96,
    "CO(4-3)":     650.25,
    "CO(5-4)":     520.23,
    "CO(6-5)":     433.56,
    "CO(7-6)":     371.65,
    "CI(1-0)":     609.14,
    "CI(2-1)":     370.42,
}

# Band edges (50% transmission, microns)
BAND_EDGES = {
    "PACS_100":    (85, 125),
    "PACS_160":    (130, 210),
    "SPIRE_250":   (194, 313),
    "SPIRE_350":   (283, 413),
    "SPIRE_500":   (383, 693),
    "SCUBA2_850":  (770, 940),
}

# Solar luminosity
L_SUN = 3.828e33  # erg/s
MPC_TO_CM = 3.0857e24
JY_TO_CGS = 1e-23  # Jy -> erg/s/cm^2/Hz


def _cmb_correction_factor(z, T_dust, beta_emis=1.8):
    """
    CMB correction factor for observed L_IR (da Cunha+2013).

    The observed dust flux is B_nu(T_dust) - B_nu(T_CMB), not B_nu(T_dust).
    We compute the flux-weighted mean correction over rest-frame 100-500um,
    weighted by the modified blackbody flux (proxy for contribution to L_IR).

    For T_dust ~ 30K:
        z=1: correction ~ 0.1%
        z=2: correction ~ 0.5%
        z=3: correction ~ 2%
        z=4: correction ~ 5%

    The correction is small because L_IR is dominated by wavelengths
    near the SED peak (~100um) where the CMB is negligible. The larger
    corrections (~10-20%) reported at specific long-wavelength bands
    (e.g., 500um) do not apply to the integrated luminosity.

    Parameters
    ----------
    z : float or array
        Redshift.
    T_dust : float or array
        Rest-frame dust temperature (K).
    beta_emis : float
        Dust emissivity index.

    Returns
    -------
    correction : float or array
        Multiply L_IR_observed by this to get L_IR_true. Always >= 1.

    References
    ----------
    da Cunha+2013 (ApJ 766, 13): CMB effects on high-z dust emission
    """
    if T_dust is None or np.any(np.asarray(T_dust) <= 0):
        return 1.0

    h = 6.626e-27   # erg s
    k = 1.381e-16   # erg/K
    c_cgs = 2.998e10  # cm/s

    z = np.asarray(z, dtype=float)
    T_dust = np.asarray(T_dust, dtype=float)
    T_cmb = 2.725 * (1 + z)

    # Rest-frame wavelength grid (um) spanning the FIR SED
    lam_rest = np.linspace(50, 500, 200)  # um
    nu = c_cgs / (lam_rest * 1e-4)  # Hz

    def _B_nu(nu_arr, T):
        x = h * nu_arr / (k * T)
        x = np.minimum(x, 500)
        return 2 * h * nu_arr**3 / c_cgs**2 / (np.exp(x) - 1)

    # Handle scalar inputs
    scalar = z.ndim == 0
    if scalar:
        z = z.reshape(1)
        T_dust = T_dust.reshape(1)
        T_cmb = T_cmb.reshape(1)

    corrections = np.ones(len(z))
    for i in range(len(z)):
        flux_dust = nu**beta_emis * _B_nu(nu, T_dust[i])
        flux_cmb = nu**beta_emis * _B_nu(nu, T_cmb[i])

        # Per-frequency correction, weighted by dust flux
        ratio = flux_cmb / flux_dust
        per_freq_corr = 1.0 / (1.0 - ratio)
        per_freq_corr = np.clip(per_freq_corr, 1.0, 10.0)

        # Flux-weighted mean
        corrections[i] = np.average(per_freq_corr, weights=flux_dust)

    if scalar:
        return float(corrections[0])
    return corrections


def _lines_in_band(z, band_lo, band_hi):
    """Return list of (line_name, rest_lam, obs_lam) in band at redshift z."""
    hits = []
    for name, lam_rest in LINES.items():
        lam_obs = lam_rest * (1 + z)
        if band_lo <= lam_obs <= band_hi:
            hits.append((name, lam_rest, lam_obs))
    return hits


def _band_for_wavelength(wave_um):
    """Return band name and edges for an observed wavelength."""
    for band, (lo, hi) in BAND_EDGES.items():
        center = (lo + hi) / 2
        if abs(wave_um - center) / center < 0.3:
            return band, lo, hi
    return None, None, None


def _excess_to_luminosity(excess_jy, z, dl_mpc, band_lo_um, band_hi_um):
    """
    Convert band-averaged excess flux to line luminosity.

    The excess flux (Jy) is spread across the band. To get line luminosity:
        L_line = F_excess * delta_nu * 4 pi dL^2 / (1+z)

    where delta_nu is the band width in Hz (the line flux is diluted
    across the full band).

    Returns L_line in L_sun.
    """
    c_cgs = 2.998e10  # cm/s
    # Band edges in Hz (nu = c/lambda)
    nu_hi = c_cgs / (band_lo_um * 1e-4)  # shorter wavelength = higher freq
    nu_lo = c_cgs / (band_hi_um * 1e-4)
    delta_nu = nu_hi - nu_lo  # Hz

    dl_cm = dl_mpc * MPC_TO_CM
    f_cgs = excess_jy * JY_TO_CGS  # erg/s/cm^2/Hz

    # Line luminosity: integrate excess over bandwidth
    L_line = f_cgs * delta_nu * 4 * np.pi * dl_cm**2 / (1 + z)

    return L_line / L_SUN  # in L_sun


def measure_line_intensity(
    wrapper,
    *,
    line="[CII] 158",
    min_tier="B",
    split_filter=None,
    surface_color_by="metallicity",
    apply_cmb_correction=True,
    save_path=None,
):
    """
    Measure FIR line intensity from stacked SED residuals.

    Parameters
    ----------
    wrapper : SimstackWrapper
    line : str
        Line to measure. Default "[CII] 158".
    min_tier : str
        Minimum fit quality tier.
    split_filter : list of int, optional
        Population class filter.
    surface_color_by : str
        Controls panel 5 (3-variable surface on Sigma_SFR axis):
        'metallicity' = color by Z, linestyle by z (default)
        'redshift' = color by z, linestyle by Z
    apply_cmb_correction : bool
        Apply da Cunha+2013 CMB correction to L_IR (default True).
        At z>2, the CMB floor suppresses observed dust emission by
        ~2-18%, biasing L_IR low and L_[CII]/L_IR high.
    save_path : str or Path, optional

    Returns
    -------
    fig, df : figure and DataFrame with line measurements
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

    line_rest_lam = LINES.get(line)
    if line_rest_lam is None:
        print(f"Unknown line '{line}'. Available: {list(LINES.keys())}")
        return None, None

    print(f"Measuring {line} ({line_rest_lam:.1f} um) in stacked SEDs")

    # Diagnostic: check model wavelength frame
    sample_sed = next((s for s in pr.sed_results.values() if s.greybody_fit_success 
                       and s.model_wavelengths is not None), None)
    _model_is_rest_frame = False
    if sample_sed is not None:
        mw = sample_sed.model_wavelengths
        mf = sample_sed.model_fluxes
        z_s = sample_sed.median_redshift

        # Where does the model peak vs the data peak?
        model_peak = mw[np.argmax(mf)]
        data_peak = sample_sed.wavelengths[np.argmax(sample_sed.flux_densities)]

        print(f"\n  Model wavelength check (first good population):")
        print(f"    Data wavelengths (observed):  {sample_sed.wavelengths}")
        print(f"    Model wavelengths range:      [{mw.min():.0f}, {mw.max():.0f}] um")
        print(f"    Model SED peak:               {model_peak:.0f} um")
        print(f"    Data SED peak:                {data_peak:.0f} um")
        print(f"    Median redshift:              z = {z_s:.2f}")
        print(f"    Model peak x (1+z):           {model_peak * (1 + z_s):.0f} um")

        # If model peak is much bluer than data peak, model is rest-frame
        # Expected: model_peak * (1+z) ≈ data_peak
        ratio_asis = model_peak / data_peak
        ratio_shifted = model_peak * (1 + z_s) / data_peak
        if abs(ratio_shifted - 1.0) < abs(ratio_asis - 1.0):
            _model_is_rest_frame = True
            print(f"    >>> MODEL IS IN REST FRAME — will multiply by (1+z)")
        else:
            print(f"    Model appears to be in observed frame")

    # ── extract measurements per population ──────────────────────────
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
        sfr = derived.star_formation_rate or 0

        # CMB correction: observed L_IR underestimates true L_IR at high z
        cmb_factor = 1.0
        if apply_cmb_correction and sed.dust_temperature_rest_frame:
            cmb_factor = _cmb_correction_factor(z, sed.dust_temperature_rest_frame)
            l_ir *= cmb_factor
            sfr *= cmb_factor  # SFR = 1e-10 * L_IR, scales together

        # Get bin_properties
        props = sed.bin_properties or {}
        if isinstance(props, str):
            import ast
            try:
                props = ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}

        # One-time diagnostic: show first population's keys
        if len(rows) == 0 and props:
            print(f"  bin_properties keys (first pop): {list(props.keys())}")

        extra = {}
        for key, val in props.items():
            kl = key.lower()
            if "mass" in kl and "delta" not in kl:
                extra["stellar_mass"] = val
            elif "sigma" in kl and "sfr" in kl:
                extra["log_sigma_sfr"] = val
            elif "metal" in kl:
                extra["metallicity"] = val
            elif "ebv" in kl:
                extra["ebv"] = val
            elif "ssfr" in kl:
                extra["log_ssfr"] = val
            elif "delta" in kl:
                extra["log_delta_ms"] = val
            elif "beta" in kl:
                extra["beta_uv"] = val

        # Derive log_ssfr from sfr_med and mass_med if not already present
        if "log_ssfr" not in extra:
            sfr_prop = props.get("sfr_med", None)
            mass_prop = extra.get("stellar_mass", None)
            if sfr_prop is not None and mass_prop is not None:
                if sfr_prop > 0 and mass_prop > 0:
                    # mass_med is log10(M*), sfr_med is linear SFR
                    extra["log_ssfr"] = np.log10(sfr_prop) - mass_prop

        if "stellar_mass" not in extra:
            extra["stellar_mass"] = sed.median_mass

        # Which band has the line?
        lam_line_obs = line_rest_lam * (1 + z)
        line_band_idx = None
        line_band_name = None
        band_lo, band_hi = None, None

        for i, w in enumerate(sed.wavelengths):
            b, blo, bhi = _band_for_wavelength(w)
            if b and blo <= lam_line_obs <= bhi:
                line_band_idx = i
                line_band_name = b
                band_lo, band_hi = blo, bhi
                break

        # Compute residuals in ALL bands
        all_residuals = {}
        if sed.model_wavelengths is not None and sed.model_fluxes is not None:
            # Detect if model wavelengths are rest-frame or observed
            model_wave = sed.model_wavelengths
            if _model_is_rest_frame:
                model_wave = model_wave * (1 + z)  # convert to observed

            for j in range(len(sed.wavelengths)):
                fm = np.interp(sed.wavelengths[j], model_wave, sed.model_fluxes)
                if fm > 0:
                    all_residuals[j] = {
                        "wave": sed.wavelengths[j],
                        "excess_jy": sed.flux_densities[j] - fm,
                        "excess_frac": (sed.flux_densities[j] - fm) / fm,
                        "flux_data": sed.flux_densities[j],
                        "flux_model": fm,
                        "flux_err": sed.flux_errors[j],
                    }

        # Baseline: mean residual in bands WITHOUT the target line
        baseline_fracs = []
        for j, res in all_residuals.items():
            if j == line_band_idx:
                continue
            b_j, blo_j, bhi_j = _band_for_wavelength(res["wave"])
            if blo_j is not None:
                lines_j = _lines_in_band(z, blo_j, bhi_j)
                has_target = any(n == line for n, _, _ in lines_j)
                if not has_target:
                    baseline_fracs.append(res["excess_frac"])
        baseline_mean = np.mean(baseline_fracs) if baseline_fracs else 0

        # Base row (always recorded)
        row = {
            "pop_id": pop_id,
            "z": z,
            "l_ir": l_ir,
            "log_l_ir": np.log10(l_ir) if l_ir > 0 else np.nan,
            "sfr": sfr,
            "n_sources": sed.n_sources,
            "tier": tier,
            "cmb_correction": cmb_factor,
            "line_in_band": line_band_idx is not None,
            "line_band": line_band_name or "none",
            "lam_line_obs": lam_line_obs,
            **extra,
        }

        if line_band_idx is not None and line_band_idx in all_residuals:
            res = all_residuals[line_band_idx]
            excess_jy = res["excess_jy"]
            excess_frac = res["excess_frac"]
            excess_corrected = excess_frac - baseline_mean
            snr = excess_jy / res["flux_err"] if res["flux_err"] > 0 else 0

            # Convert to luminosity
            dl_mpc = sed.luminosity_distances[line_band_idx] if hasattr(sed, 'luminosity_distances') else None
            if dl_mpc is None:
                from astropy.cosmology import Planck18
                dl_mpc = Planck18.luminosity_distance(z).value

            l_line = _excess_to_luminosity(excess_jy, z, dl_mpc, band_lo, band_hi)
            l_line_fir = l_line / l_ir if l_ir > 0 else np.nan

            # Other lines in the same band (blending)
            all_lines = _lines_in_band(z, band_lo, band_hi)

            row.update({
                "excess_jy": excess_jy,
                "excess_frac": excess_frac,
                "excess_frac_corrected": excess_corrected,
                "excess_snr": snr,
                "baseline_frac": baseline_mean,
                "l_line": l_line,
                "log_l_line": np.log10(abs(l_line)) * np.sign(l_line) if l_line != 0 else np.nan,
                "l_line_over_l_ir": l_line_fir,
                "log_l_line_over_l_ir": np.log10(l_line_fir) if l_line_fir > 0 else np.nan,
                "flux_data": res["flux_data"],
                "flux_model": res["flux_model"],
                "flux_err": res["flux_err"],
                "n_lines_in_band": len(all_lines),
                "lines_in_band": " + ".join([n for n, _, _ in all_lines]),
            })
        else:
            row.update({
                "excess_jy": np.nan, "excess_frac": np.nan,
                "excess_frac_corrected": np.nan, "excess_snr": np.nan,
                "baseline_frac": np.nan, "l_line": np.nan,
                "log_l_line": np.nan, "l_line_over_l_ir": np.nan,
                "log_l_line_over_l_ir": np.nan,
                "n_lines_in_band": 0, "lines_in_band": "",
            })

        rows.append(row)

    if not rows:
        print("No populations found")
        return None, None

    df = pd.DataFrame(rows)
    df_det = df[df["line_in_band"]].copy()
    df_ctrl = df[~df["line_in_band"]].copy()

    # ── summary statistics ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{line} LINE INTENSITY MEASUREMENT")
    print(f"{'='*65}")
    print(f"  Populations with {line} in a band: {len(df_det)}")
    print(f"  Control (no line in band):         {len(df_ctrl)}")

    if len(df_det) > 0:
        valid = np.isfinite(df_det["excess_frac"])
        if valid.any():
            ef = df_det.loc[valid, "excess_frac"]
            efc = df_det.loc[valid, "excess_frac_corrected"]
            ll = df_det.loc[valid, "l_line"]
            llfir = df_det.loc[valid, "l_line_over_l_ir"]

            print(f"\n  Fractional excess (data-model)/model:")
            print(f"    Raw:       median = {ef.median()*100:+.3f}%,  "
                  f"mean = {ef.mean()*100:+.3f}%")
            print(f"    Corrected: median = {efc.median()*100:+.3f}%")

            valid_l = np.isfinite(llfir) & (llfir > 0)
            if valid_l.any():
                print(f"\n  L_{line} / L_FIR:")
                print(f"    median = {llfir[valid_l].median():.4f} "
                      f"({llfir[valid_l].median()*100:.2f}%)")
                print(f"    range  = [{llfir[valid_l].min():.4f}, "
                      f"{llfir[valid_l].max():.4f}]")

            valid_ll = np.isfinite(ll) & (ll > 0)
            if valid_ll.any():
                print(f"\n  L_{line}:")
                print(f"    median = {ll[valid_ll].median():.2e} L_sun")
                print(f"    range  = [{ll[valid_ll].min():.2e}, "
                      f"{ll[valid_ll].max():.2e}]")

            # By band
            print(f"\n  By band:")
            for band in sorted(df_det["line_band"].unique()):
                bm = (df_det["line_band"] == band) & valid
                if bm.any():
                    med = ef[bm].median()
                    med_ratio = llfir[bm & np.isfinite(llfir)].median() if (bm & np.isfinite(llfir)).any() else np.nan
                    print(f"    {band:>12}: {bm.sum():>3} pops, "
                          f"excess = {med*100:+.3f}%, "
                          f"L_line/L_FIR = {med_ratio:.4f}")

            # Detection significance
            mean_excess = ef.mean()
            sem = ef.std() / np.sqrt(len(ef))
            significance = mean_excess / sem if sem > 0 else 0
            print(f"\n  Aggregate detection: {significance:.1f} sigma")

            # CMB correction info
            if apply_cmb_correction:
                cmb_vals = df_det.loc[valid, "cmb_correction"]
                print(f"\n  CMB correction applied (da Cunha+2013):")
                print(f"    L_IR multiplied by {cmb_vals.min():.3f} - {cmb_vals.max():.3f}")
                print(f"    median correction: x{cmb_vals.median():.3f}")
            else:
                print(f"\n  CMB correction: OFF (set apply_cmb_correction=True)")

            # Diagnostic: if excess is suspiciously large, show examples
            if ef.median() > 0.1:  # > 10% is too high for any line
                print(f"\n  WARNING: median excess = {ef.median()*100:.1f}% is "
                      f"unrealistically high for {line}.")
                print(f"  Expected: 0.1-1% for [CII], <0.1% for other lines.")
                print(f"  Likely cause: model interpolation mismatch.")
                print(f"\n  Diagnostic — first 3 populations:")
                for _, row in df_det[valid].head(3).iterrows():
                    print(f"    z={row['z']:.2f}, band={row['line_band']}, "
                          f"lam_obs={row['lam_line_obs']:.0f}um")
                    if "flux_data" in row and "flux_model" in row:
                        print(f"      data={row['flux_data']:.4e} Jy, "
                              f"model={row['flux_model']:.4e} Jy, "
                              f"excess={row['excess_frac']*100:+.1f}%")

    # ── 9-panel science plot (3x3) ───────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    cmap = "plasma"

    valid_det = df_det["line_in_band"] & np.isfinite(df_det["excess_frac"])
    valid_ratio = valid_det & np.isfinite(df_det["l_line_over_l_ir"]) & (df_det["l_line_over_l_ir"] > 0)

    # Helper: binned medians with errors (IQR)
    def _binned_stats(x, y, edges, min_count=2):
        centers, medians, lo_err, hi_err, counts = [], [], [], [], []
        for i in range(len(edges) - 1):
            mask = (x >= edges[i]) & (x < edges[i+1])
            if mask.sum() >= min_count:
                vals = y[mask]
                med = np.median(vals)
                p16, p84 = np.percentile(vals, [16, 84])
                centers.append((edges[i] + edges[i+1]) / 2)
                medians.append(med)
                lo_err.append(med - p16)
                hi_err.append(p84 - med)
                counts.append(mask.sum())
        return (np.array(centers), np.array(medians),
                np.array(lo_err), np.array(hi_err), np.array(counts))

    def _fit_and_plot(ax, x, y, color="k", label="fit", ls="-", lw=2,
                      x_is_log=False, weights=None):
        """Fit log(y) = a*x + b. weights: higher = more trusted."""
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if weights is not None:
            valid &= np.isfinite(weights) & (weights > 0)
        if valid.sum() < 4:
            return None
        log_y = np.log10(y[valid])
        x_v = x[valid]
        try:
            if weights is not None:
                coeffs = np.polyfit(x_v, log_y, 1, w=np.sqrt(weights[valid]))
            else:
                coeffs = np.polyfit(x_v, log_y, 1)
            x_grid = np.linspace(x_v.min(), x_v.max(), 100)
            y_fit = 10 ** np.polyval(coeffs, x_grid)
            x_plot = 10 ** x_grid if x_is_log else x_grid
            ax.plot(x_plot, y_fit * 100, ls=ls, color=color, lw=lw,
                    alpha=0.8, label=label, zorder=4)
            return coeffs
        except Exception:
            return None

    def _scatter_weighted(ax, df_sub, x_col, y_col, c_col, cmap_name,
                          x_is_log=False, **kwargs):
        """Scatter with point size proportional to n_sources."""
        ns = df_sub["n_sources"].values.astype(float)
        sizes = 10 + 60 * (ns / max(ns.max(), 1)) ** 0.5
        x_vals = df_sub[x_col].values
        return ax.scatter(x_vals, df_sub[y_col].values * 100,
                          c=df_sub[c_col].values, cmap=cmap_name,
                          s=sizes, alpha=0.7, **kwargs)

    # ────────────────────────────────────────────────────────────────
    # ROW 1: Tomographic, classical deficit, CII-SFR
    # ────────────────────────────────────────────────────────────────

    # Panel 1: L_line/L_FIR vs redshift (colored by band)
    ax = axes[0, 0]
    band_colors = {
        "PACS_160": "C0", "SPIRE_250": "C1",
        "SPIRE_350": "C2", "SPIRE_500": "C3",
        "SCUBA2_850": "C4",
    }
    for band, color in band_colors.items():
        bm = (df_det["line_band"] == band) & valid_ratio
        if bm.any():
            ax.scatter(
                df_det.loc[bm, "z"],
                df_det.loc[bm, "l_line_over_l_ir"] * 100,
                c=color, s=30, alpha=0.7,
                label=band.replace("_", " "),
            )
    if valid_ratio.any():
        z_edges = np.arange(0, 5.5, 0.5)
        zc, zm, zlo, zhi, _ = _binned_stats(
            df_det.loc[valid_ratio, "z"].values,
            df_det.loc[valid_ratio, "l_line_over_l_ir"].values,
            z_edges,
        )
        if len(zc) > 0:
            ax.errorbar(zc, zm * 100, yerr=[zlo * 100, zhi * 100],
                        fmt="ks", ms=9, mfc="white", mew=2,
                        capsize=3, elinewidth=1.5, zorder=5,
                        label="Binned median")
        _fit_and_plot(ax,
                      df_det.loc[valid_ratio, "z"].values,
                      df_det.loc[valid_ratio, "l_line_over_l_ir"].values,
                      color="k", label="Fit (weighted)", ls="--",
                      weights=df_det.loc[valid_ratio, "n_sources"].values.astype(float))
    ax.set_xlabel("Redshift")
    ax.set_ylabel(f"L$_{{{line}}}$ / L$_{{FIR}}$ (%)")
    ax.set_title(f"{line} intensity vs redshift")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Panel 2: L_line/L_FIR vs L_IR (classical deficit)
    ax = axes[0, 1]
    if valid_ratio.any():
        ns = df_det.loc[valid_ratio, "n_sources"].values.astype(float)
        sizes = 10 + 60 * (ns / ns.max()) ** 0.5
        sc = ax.scatter(
            df_det.loc[valid_ratio, "l_ir"],
            df_det.loc[valid_ratio, "l_line_over_l_ir"] * 100,
            c=df_det.loc[valid_ratio, "z"], cmap=cmap,
            s=sizes, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Redshift")
        lir_edges = np.logspace(
            np.floor(np.log10(df_det.loc[valid_ratio, "l_ir"].min())),
            np.ceil(np.log10(df_det.loc[valid_ratio, "l_ir"].max())), 8,
        )
        lc, lm, llo, lhi, _ = _binned_stats(
            df_det.loc[valid_ratio, "l_ir"].values,
            df_det.loc[valid_ratio, "l_line_over_l_ir"].values,
            lir_edges,
        )
        if len(lc) > 0:
            ax.errorbar(lc, lm * 100, yerr=[llo * 100, lhi * 100],
                        fmt="ks", ms=9, mfc="white", mew=2,
                        capsize=3, elinewidth=1.5, zorder=5)
        fit_w = df_det.loc[valid_ratio, "n_sources"].values.astype(float)
        coeffs = _fit_and_plot(ax,
                               np.log10(df_det.loc[valid_ratio, "l_ir"].values),
                               df_det.loc[valid_ratio, "l_line_over_l_ir"].values,
                               color="navy", label="Deficit fit", ls="-",
                               x_is_log=True, weights=fit_w)
        if coeffs is not None:
            ax.text(0.05, 0.05, f"slope = {coeffs[0]:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(fc="white", alpha=0.8))
        lir_ref = np.logspace(10, 13, 50)
        cii_pct = 0.5 * (lir_ref / 1e11) ** (-0.35)
        ax.plot(lir_ref, cii_pct, "r--", lw=1.5, alpha=0.5,
                label="D-S+17 (approx)")
        ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("L$_{IR}$ (L$_\\odot$)")
    ax.set_ylabel(f"L$_{{{line}}}$ / L$_{{FIR}}$ (%)")
    ax.set_title(f"Classical deficit (vs L$_{{IR}}$)")
    ax.grid(True, alpha=0.2)

    # Panel 3: L_line vs SFR (De Looze relation)
    ax = axes[0, 2]
    valid_ll = valid_det & np.isfinite(df_det["l_line"]) & (df_det["l_line"] > 0)
    if valid_ll.any() and "sfr" in df_det.columns:
        sc = ax.scatter(
            df_det.loc[valid_ll, "sfr"],
            df_det.loc[valid_ll, "l_line"],
            c=df_det.loc[valid_ll, "z"], cmap=cmap, s=30, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Redshift")
        sfr_grid = np.logspace(-1, 3, 50)
        l_cii_dl14 = 10**(1.0 * np.log10(sfr_grid) + 7.06)
        ax.plot(sfr_grid, l_cii_dl14, "r--", lw=2, alpha=0.6,
                label="De Looze+2014")
        ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SFR (M$_\\odot$/yr)")
    ax.set_ylabel(f"L$_{{{line}}}$ (L$_\\odot$)")
    ax.set_title(f"L$_{{{line}}}$ - SFR relation")
    ax.grid(True, alpha=0.2)

    # ────────────────────────────────────────────────────────────────
    # ROW 2: Physical driver (Sigma_SFR) + user correlate
    # ────────────────────────────────────────────────────────────────

    # Panel 4: L_line/L_FIR vs Sigma_SFR (2D surface fit)
    ax = axes[1, 0]
    has_sigma = "log_sigma_sfr" in df_det.columns and np.isfinite(df_det["log_sigma_sfr"]).any()
    if has_sigma and valid_ratio.any():
        valid_s = valid_ratio & np.isfinite(df_det["log_sigma_sfr"])
        if valid_s.any():
            sc = ax.scatter(
                df_det.loc[valid_s, "log_sigma_sfr"],
                df_det.loc[valid_s, "l_line_over_l_ir"] * 100,
                c=df_det.loc[valid_s, "z"], cmap=cmap, s=30, alpha=0.5,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")
            sig_edges = np.arange(-2, 2.5, 0.5)
            sc_, sm, slo, shi, _ = _binned_stats(
                df_det.loc[valid_s, "log_sigma_sfr"].values,
                df_det.loc[valid_s, "l_line_over_l_ir"].values,
                sig_edges,
            )
            if len(sc_) > 0:
                ax.errorbar(sc_, sm * 100, yerr=[slo * 100, shi * 100],
                            fmt="ks", ms=9, mfc="white", mew=2,
                            capsize=3, elinewidth=1.5, zorder=5,
                            label="All z")
            # Joint 2D fit: log(L_line/L_IR) = a*log_sigma + b*z + c
            sig_v = df_det.loc[valid_s, "log_sigma_sfr"].values
            z_v = df_det.loc[valid_s, "z"].values
            r_v = df_det.loc[valid_s, "l_line_over_l_ir"].values
            w_v = df_det.loc[valid_s, "n_sources"].values.astype(float)
            pos = (r_v > 0) & np.isfinite(w_v) & (w_v > 0)
            if pos.sum() >= 6:
                log_r = np.log10(r_v[pos])
                w = np.sqrt(w_v[pos])
                X = np.column_stack([sig_v[pos], z_v[pos], np.ones(pos.sum())])
                try:
                    Xw = X * w[:, None]
                    yw = log_r * w
                    c2d, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
                    a_sig, b_z, c_int = c2d

                    # Weighted R²
                    pred = X @ c2d
                    ss_res = np.sum(w**2 * (log_r - pred)**2)
                    ss_tot = np.sum(w**2 * (log_r - np.average(log_r, weights=w**2))**2)
                    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                    sig_grid = np.linspace(sig_v.min(), sig_v.max(), 100)
                    z_show = [0.5, 1.5, 2.5, 3.5]
                    z_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                    for z_fix, zcol in zip(z_show, z_colors):
                        y_fit = 10 ** (a_sig * sig_grid + b_z * z_fix + c_int)
                        ax.plot(sig_grid, y_fit * 100, "-", color=zcol, lw=2,
                                alpha=0.8, label=f"z={z_fix:.1f}", zorder=4)

                    # Black fit through binned medians
                    if len(sc_) >= 3:
                        valid_med = sm > 0
                        if valid_med.sum() >= 2:
                            med_coeffs = np.polyfit(sc_[valid_med], np.log10(sm[valid_med]), 1)
                            ax.plot(sc_[valid_med],
                                    10**np.polyval(med_coeffs, sc_[valid_med]) * 100,
                                    "k-", lw=2.5, alpha=0.9, zorder=6,
                                    label="Median fit")

                    ax.text(0.03, 0.95,
                            f"log(L/L_IR) = {a_sig:.2f} log$\\Sigma$ "
                            f"{b_z:+.2f} z {c_int:+.2f}\n"
                            f"R$^2$ = {r_sq:.3f}  (N={pos.sum()})",
                            transform=ax.transAxes, fontsize=7, va="top",
                            bbox=dict(fc="white", alpha=0.9))
                except np.linalg.LinAlgError:
                    pass
            ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("log $\\Sigma_{SFR}$ (M$_\\odot$/yr/kpc$^2$)")
    ax.set_ylabel(f"L$_{{{line}}}$ / L$_{{FIR}}$ (%)")
    ax.set_title(f"Deficit vs $\\Sigma_{{SFR}}$ (key driver)")
    ax.grid(True, alpha=0.2)

    # Panel 5: Deficit vs Sigma_SFR — 3-variable surface (Sigma + Z + z)
    # Color by one variable, linestyle by the other
    ax = axes[1, 1]
    has_Z = "metallicity" in df_det.columns and np.isfinite(df_det["metallicity"]).any()
    if has_sigma and has_Z and valid_ratio.any():
        valid_szm = (valid_ratio
                     & np.isfinite(df_det["log_sigma_sfr"])
                     & np.isfinite(df_det["metallicity"])
                     & (df_det["metallicity"] > 0))
        if valid_szm.sum() >= 6:
            sig_v = df_det.loc[valid_szm, "log_sigma_sfr"].values
            z_v = df_det.loc[valid_szm, "z"].values
            logZ_v = np.log10(df_det.loc[valid_szm, "metallicity"].values)
            r_v = df_det.loc[valid_szm, "l_line_over_l_ir"].values
            w_v = np.sqrt(df_det.loc[valid_szm, "n_sources"].values.astype(float))
            log_r = np.log10(r_v)

            # Fit: log(L/L_IR) = a*log_sigma + b*log_Z + c*z + d
            X = np.column_stack([sig_v, logZ_v, z_v, np.ones(len(sig_v))])
            try:
                c3, _, _, _ = np.linalg.lstsq(X * w_v[:, None], log_r * w_v, rcond=None)
                a_sig, b_Z, c_z, d_int = c3

                # Weighted R²
                pred = X @ c3
                ss_res = np.sum(w_v**2 * (log_r - pred)**2)
                ss_tot = np.sum(w_v**2 * (log_r - np.average(log_r, weights=w_v**2))**2)
                r_sq_3 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                # Scatter points
                if surface_color_by == "metallicity":
                    sc = ax.scatter(
                        df_det.loc[valid_szm, "log_sigma_sfr"],
                        df_det.loc[valid_szm, "l_line_over_l_ir"] * 100,
                        c=logZ_v, cmap="viridis", s=30, alpha=0.5,
                    )
                    plt.colorbar(sc, ax=ax, label="log(Z)")
                else:
                    sc = ax.scatter(
                        df_det.loc[valid_szm, "log_sigma_sfr"],
                        df_det.loc[valid_szm, "l_line_over_l_ir"] * 100,
                        c=z_v, cmap="plasma", s=30, alpha=0.5,
                    )
                    plt.colorbar(sc, ax=ax, label="Redshift")

                sig_grid = np.linspace(sig_v.min(), sig_v.max(), 100)

                from matplotlib.lines import Line2D as _Line2D
                if surface_color_by == "metallicity":
                    # Color = metallicity, linestyle = redshift
                    Z_show = np.array([0.004, 0.008, 0.02, 0.04])
                    Z_colors = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]
                    z_lines = [0.5, 1.5, 2.5]
                    line_styles = ["-", "--", ":"]
                    for logZ_fix, zcol in zip(np.log10(Z_show), Z_colors):
                        for z_fix, ls in zip(z_lines, line_styles):
                            y_fit = 10 ** (a_sig * sig_grid + b_Z * logZ_fix + c_z * z_fix + d_int)
                            ax.plot(sig_grid, y_fit * 100, ls=ls, color=zcol,
                                    lw=2, alpha=0.8, zorder=4)
                    style_handles = [
                        _Line2D([0], [0], color="gray", ls=ls, lw=1.5, label=f"z={zf}")
                        for zf, ls in zip(z_lines, line_styles)
                    ]
                    color_handles = [
                        _Line2D([0], [0], color=zcol, ls="-", lw=3,
                               label=f"Z={10**logZf:.3f}")
                        for logZf, zcol in zip(np.log10(Z_show), Z_colors)
                    ]
                    ax.legend(handles=color_handles + style_handles,
                              fontsize=6, loc="upper right", ncol=2)
                else:
                    # Color = redshift, linestyle = metallicity
                    z_show = [0.5, 1.5, 2.5, 3.5]
                    z_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                    Z_lines = [0.004, 0.02]
                    Z_styles = ["-", "--"]
                    Z_labels = ["0.2 Z_sun", "Z_sun"]
                    for z_fix, zcol in zip(z_show, z_colors):
                        for logZf, ls in zip(np.log10(Z_lines), Z_styles):
                            y_fit = 10 ** (a_sig * sig_grid + b_Z * logZf + c_z * z_fix + d_int)
                            ax.plot(sig_grid, y_fit * 100, ls=ls, color=zcol,
                                    lw=2, alpha=0.8, zorder=4)
                    style_handles = [
                        _Line2D([0], [0], color="gray", ls=ls, lw=1.5, label=zlbl)
                        for ls, zlbl in zip(Z_styles, Z_labels)
                    ]
                    color_handles = [
                        _Line2D([0], [0], color=zcol, ls="-", lw=3, label=f"z={zf}")
                        for zf, zcol in zip(z_show, z_colors)
                    ]
                    ax.legend(handles=color_handles + style_handles,
                              fontsize=6, loc="upper right", ncol=2)

                ax.text(0.03, 0.95,
                        f"log(L/L_IR) = {a_sig:.2f} log$\Sigma$ "
                        f"{b_Z:+.2f} logZ {c_z:+.2f} z {d_int:+.1f}\n"
                        f"R$^2$ = {r_sq_3:.3f}  (N={len(sig_v)})",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(fc="white", alpha=0.9))
            except np.linalg.LinAlgError:
                pass
    elif has_sigma and valid_ratio.any():
        ax.text(0.5, 0.5, "Add 'metallicity' to\nbin_property_columns",
                transform=ax.transAxes, ha="center", fontsize=10)
    ax.set_xlabel("log $\Sigma_{SFR}$ (M$_\odot$/yr/kpc$^2$)")
    ax.set_ylabel(f"L$_{{{line}}}$ / L$_{{FIR}}$ (%)")
    ax.set_title(f"3-variable surface ($\Sigma$, Z, z)")
    ax.grid(True, alpha=0.2)

    # Panel 6: Liang+2024 regime test — t_dep vs metallicity
    # Two regimes: H2-rich (low-z, high LIR) → deficit driven by t_dep (∝ 1/sSFR)
    #              H2-poor (high-z, low M*) → deficit driven by metallicity
    ax = axes[1, 2]
    has_Z = "metallicity" in df_det.columns and np.isfinite(df_det["metallicity"]).any()
    has_ssfr = "log_ssfr" in df_det.columns and np.isfinite(df_det["log_ssfr"]).any()
    has_sigma = "log_sigma_sfr" in df_det.columns and np.isfinite(df_det["log_sigma_sfr"]).any()

    # Diagnostic
    print(f"\n  Panel 6 (Liang+2024) diagnostics:")
    print(f"    Columns in df_det: {sorted(df_det.columns.tolist())}")
    print(f"    has_sigma={has_sigma}, has_Z={has_Z}, has_ssfr={has_ssfr}")
    if "log_ssfr" in df_det.columns:
        n_finite = np.isfinite(df_det["log_ssfr"]).sum()
        print(f"    log_ssfr: {n_finite}/{len(df_det)} finite values")
        if n_finite > 0:
            print(f"    log_ssfr range: [{df_det['log_ssfr'].min():.2f}, {df_det['log_ssfr'].max():.2f}]")
    else:
        print(f"    log_ssfr NOT in columns")
        print(f"    Available columns with 'ss': {[c for c in df_det.columns if 'ss' in c.lower()]}")

    if has_sigma and valid_ratio.any() and (has_Z or has_ssfr):
        # Split at z = 2 (cosmic noon boundary)
        z_split = 2.0
        lo_z = valid_ratio & (df_det["z"] < z_split) & np.isfinite(df_det["log_sigma_sfr"])
        hi_z = valid_ratio & (df_det["z"] >= z_split) & np.isfinite(df_det["log_sigma_sfr"])

        def _weighted_r2(X, y, w):
            """Weighted R2 from design matrix X, log(y), weights w."""
            try:
                Xw = X * w[:, None]
                yw = y * w
                c, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
                pred = X @ c
                ss_res = np.sum(w**2 * (y - pred)**2)
                ss_tot = np.sum(w**2 * (y - np.average(y, weights=w**2))**2)
                return 1 - ss_res / ss_tot if ss_tot > 0 else 0, c
            except Exception:
                return np.nan, None

        results = {}  # {(regime, model): (R2, coeffs)}

        for label, mask, z_label in [("z < 2", lo_z, "Low-z\n(H$_2$-rich)"),
                                      ("z >= 2", hi_z, "High-z\n(H$_2$-poor)")]:
            if mask.sum() < 6:
                continue
            sig = df_det.loc[mask, "log_sigma_sfr"].values
            r = df_det.loc[mask, "l_line_over_l_ir"].values
            w = np.sqrt(df_det.loc[mask, "n_sources"].values.astype(float))
            pos = r > 0
            if pos.sum() < 6:
                continue
            log_r = np.log10(r[pos])
            sig_p = sig[pos]
            w_p = w[pos]

            # Model 1: Sigma only (baseline)
            X1 = np.column_stack([sig_p, np.ones(pos.sum())])
            r2_1, _ = _weighted_r2(X1, log_r, w_p)
            results[(label, "$\Sigma$ only")] = r2_1

            # Model 2: Sigma + sSFR (t_dep proxy)
            if has_ssfr:
                ssfr = df_det.loc[mask, "log_ssfr"].values[pos]
                valid_ss = np.isfinite(ssfr)
                if valid_ss.sum() >= 6:
                    X2 = np.column_stack([sig_p[valid_ss], ssfr[valid_ss],
                                          np.ones(valid_ss.sum())])
                    r2_2, c2 = _weighted_r2(X2, log_r[valid_ss], w_p[valid_ss])
                    results[(label, "$\Sigma$ + sSFR")] = r2_2
                else:
                    results[(label, "$\Sigma$ + sSFR")] = np.nan

            # Model 3: Sigma + Z (metallicity)
            if has_Z:
                Z = df_det.loc[mask, "metallicity"].values[pos]
                valid_Z = np.isfinite(Z) & (Z > 0)
                if valid_Z.sum() >= 6:
                    logZ = np.log10(Z[valid_Z])
                    X3 = np.column_stack([sig_p[valid_Z], logZ,
                                          np.ones(valid_Z.sum())])
                    r2_3, c3 = _weighted_r2(X3, log_r[valid_Z], w_p[valid_Z])
                    results[(label, "$\Sigma$ + Z")] = r2_3
                else:
                    results[(label, "$\Sigma$ + Z")] = np.nan

        # Plot as grouped bar chart
        if results:
            regimes = ["z < 2", "z >= 2"]
            models = ["$\Sigma$ only", "$\Sigma$ + sSFR", "$\Sigma$ + Z"]
            model_colors = ["#7f7f7f", "#d62728", "#2ca02c"]

            x_pos = np.arange(len(regimes))
            n_models = len(models)
            width = 0.8 / n_models

            for j, (model, color) in enumerate(zip(models, model_colors)):
                vals = [results.get((reg, model), np.nan) for reg in regimes]
                offset = (j - (n_models - 1) / 2) * width
                bars = ax.bar(x_pos + offset, 
                              [v if np.isfinite(v) else 0 for v in vals],
                              width, label=model, color=color, alpha=0.85)
                for bar, val in zip(bars, vals):
                    if np.isfinite(val):
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.005,
                                f"{val:.3f}", ha="center", fontsize=7,
                                fontweight="bold")

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"z < {z_split:.0f}\n(N={lo_z.sum()})",
                                f"z >= {z_split:.0f}\n(N={hi_z.sum()})"],
                               fontsize=9)
            ax.set_ylabel("Weighted R$^2$")
            ax.set_ylim(0, max(v for v in results.values() if np.isfinite(v)) * 1.3 + 0.01)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.15, axis="y")

            # Liang+2024 prediction annotation
            ax.text(0.03, 0.95,
                    "Liang+2024 prediction:\n"
                    "sSFR dominates at low-z\n"
                    "Z dominates at high-z",
                    transform=ax.transAxes, fontsize=7, va="top",
                    style="italic",
                    bbox=dict(fc="lightyellow", alpha=0.9))
    else:
        missing = []
        if not has_ssfr:
            missing.append("log_ssfr")
        if not has_Z:
            missing.append("metallicity")
        ax.text(0.5, 0.5, f"Need {' + '.join(missing)}\nin bin_property_columns",
                transform=ax.transAxes, ha="center", fontsize=10)
    ax.set_title("Liang+2024 regime test", fontsize=10)

    # ────────────────────────────────────────────────────────────────
    # ROW 3: New panels — L_CII vs Sigma, deficit vs M*, deficit vs E(B-V)
    # ────────────────────────────────────────────────────────────────

    # Panel 7: L_line (absolute) vs Sigma_SFR
    ax = axes[2, 0]
    if has_sigma and valid_ratio.any():
        valid_sl = valid_ratio & np.isfinite(df_det["log_sigma_sfr"]) & np.isfinite(df_det["l_line"]) & (df_det["l_line"] > 0)
        if valid_sl.any():
            sc = ax.scatter(
                df_det.loc[valid_sl, "log_sigma_sfr"],
                df_det.loc[valid_sl, "l_line"],
                c=df_det.loc[valid_sl, "z"], cmap=cmap, s=30, alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")
            # Manual fit (can't use _fit_and_plot, it multiplies by 100)
            x_f = df_det.loc[valid_sl, "log_sigma_sfr"].values
            y_f = df_det.loc[valid_sl, "l_line"].values
            w_f = df_det.loc[valid_sl, "n_sources"].values.astype(float)
            try:
                coeffs = np.polyfit(x_f, np.log10(y_f), 1, w=np.sqrt(w_f))
                x_grid = np.linspace(x_f.min(), x_f.max(), 100)
                ax.plot(x_grid, 10 ** np.polyval(coeffs, x_grid),
                        "-", color="navy", lw=2, alpha=0.8, label="Fit")
                ax.text(0.05, 0.05, f"slope = {coeffs[0]:.2f}",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(fc="white", alpha=0.8))
            except Exception:
                pass
            ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("log $\\Sigma_{SFR}$ (M$_\\odot$/yr/kpc$^2$)")
    ax.set_ylabel(f"L$_{{{line}}}$ (L$_\\odot$)")
    ax.set_title(f"L$_{{{line}}}$ vs $\\Sigma_{{SFR}}$ (absolute)")
    ax.grid(True, alpha=0.2)

    # Panel 8: L_line/L_FIR vs M*
    ax = axes[2, 1]
    has_mass = "stellar_mass" in df_det.columns and np.isfinite(df_det["stellar_mass"]).any()
    if has_mass and valid_ratio.any():
        valid_m = valid_ratio & np.isfinite(df_det["stellar_mass"])
        if valid_m.any():
            sc = ax.scatter(
                df_det.loc[valid_m, "stellar_mass"],
                df_det.loc[valid_m, "l_line_over_l_ir"] * 100,
                c=df_det.loc[valid_m, "z"], cmap=cmap, s=30, alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")
            mass_edges = np.arange(9.0, 12.0, 0.5)
            mc, mm, mlo, mhi, _ = _binned_stats(
                df_det.loc[valid_m, "stellar_mass"].values,
                df_det.loc[valid_m, "l_line_over_l_ir"].values,
                mass_edges,
            )
            if len(mc) > 0:
                ax.errorbar(mc, mm * 100, yerr=[mlo * 100, mhi * 100],
                            fmt="ks", ms=9, mfc="white", mew=2,
                            capsize=3, elinewidth=1.5, zorder=5)
            _fit_and_plot(ax,
                          df_det.loc[valid_m, "stellar_mass"].values,
                          df_det.loc[valid_m, "l_line_over_l_ir"].values,
                          color="navy", label="Fit",
                          weights=df_det.loc[valid_m, "n_sources"].values.astype(float))
    ax.set_xlabel("log M$_*$ (M$_\\odot$)")
    ax.set_ylabel(f"L$_{{{line}}}$ / L$_{{FIR}}$ (%)")
    ax.set_title(f"Deficit vs M$_*$")
    ax.grid(True, alpha=0.2)

    # Panel 9: Metallicity decomposition — what drives the z-evolution?
    ax = axes[2, 2]
    has_Z = "metallicity" in df_det.columns and np.isfinite(df_det["metallicity"]).any()
    if has_sigma and has_Z and valid_ratio.any():
        valid_szm = (valid_ratio
                     & np.isfinite(df_det["log_sigma_sfr"])
                     & np.isfinite(df_det["metallicity"])
                     & (df_det["metallicity"] > 0))
        if valid_szm.sum() >= 6:
            sig_v = df_det.loc[valid_szm, "log_sigma_sfr"].values
            z_v = df_det.loc[valid_szm, "z"].values
            logZ_v = np.log10(df_det.loc[valid_szm, "metallicity"].values)
            r_v = df_det.loc[valid_szm, "l_line_over_l_ir"].values
            w_v = np.sqrt(df_det.loc[valid_szm, "n_sources"].values.astype(float))
            log_r = np.log10(r_v)

            # Fit 1: Sigma + z (no metallicity)
            X_sz = np.column_stack([sig_v, z_v, np.ones(len(sig_v))])
            r2_sz = np.nan
            try:
                c_sz, _, _, _ = np.linalg.lstsq(X_sz * w_v[:, None], log_r * w_v, rcond=None)
                pred = X_sz @ c_sz
                ss_res = np.sum(w_v**2 * (log_r - pred)**2)
                ss_tot = np.sum(w_v**2 * (log_r - np.average(log_r, weights=w_v**2))**2)
                r2_sz = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            except Exception:
                c_sz = [np.nan, np.nan, np.nan]

            # Fit 2: Sigma + Z (no z)
            X_sZ = np.column_stack([sig_v, logZ_v, np.ones(len(sig_v))])
            r2_sZ = np.nan
            try:
                c_sZ, _, _, _ = np.linalg.lstsq(X_sZ * w_v[:, None], log_r * w_v, rcond=None)
                pred = X_sZ @ c_sZ
                ss_res = np.sum(w_v**2 * (log_r - pred)**2)
                ss_tot = np.sum(w_v**2 * (log_r - np.average(log_r, weights=w_v**2))**2)
                r2_sZ = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            except Exception:
                c_sZ = [np.nan, np.nan, np.nan]

            # Fit 3: Sigma + Z + z (full)
            X_szZ = np.column_stack([sig_v, logZ_v, z_v, np.ones(len(sig_v))])
            r2_szZ = np.nan
            try:
                c_szZ, _, _, _ = np.linalg.lstsq(X_szZ * w_v[:, None], log_r * w_v, rcond=None)
                pred = X_szZ @ c_szZ
                ss_res = np.sum(w_v**2 * (log_r - pred)**2)
                ss_tot = np.sum(w_v**2 * (log_r - np.average(log_r, weights=w_v**2))**2)
                r2_szZ = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            except Exception:
                c_szZ = [np.nan, np.nan, np.nan, np.nan]

            # Bar chart of coefficients
            models = [
                f"$\\Sigma$ + z\nR$^2$={r2_sz:.3f}",
                f"$\\Sigma$ + Z\nR$^2$={r2_sZ:.3f}",
                f"$\\Sigma$ + Z + z\nR$^2$={r2_szZ:.3f}",
            ]
            coeff_sigma = [c_sz[0], c_sZ[0], c_szZ[0]]
            coeff_z = [c_sz[1], np.nan, c_szZ[2]]
            coeff_Z = [np.nan, c_sZ[1], c_szZ[1]]

            x_pos = np.arange(len(models))
            width = 0.25

            bars_sig = ax.bar(x_pos - width, coeff_sigma, width,
                              label="$\\Sigma_{SFR}$ coeff", color="C0", alpha=0.8)
            bars_z = ax.bar(x_pos, [v if np.isfinite(v) else 0 for v in coeff_z], width,
                            label="z coeff", color="C3", alpha=0.8)
            bars_Z = ax.bar(x_pos + width, [v if np.isfinite(v) else 0 for v in coeff_Z], width,
                            label="log(Z) coeff", color="C2", alpha=0.8)

            # Annotate values
            for bars, vals in [(bars_sig, coeff_sigma), (bars_z, coeff_z), (bars_Z, coeff_Z)]:
                for bar, val in zip(bars, vals):
                    if np.isfinite(val) and abs(val) > 0.01:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f"{val:.2f}", ha="center", va="bottom" if val > 0 else "top",
                                fontsize=7, fontweight="bold")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=8)
            ax.axhline(0, color="k", lw=0.8, alpha=0.5)
            ax.set_ylabel("Coefficient value")
            ax.set_title("Deficit decomposition")
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(True, alpha=0.15, axis="y")

            # Honest assessment
            dr2 = r2_szZ - r2_sz
            if np.isfinite(c_sz[1]) and np.isfinite(c_szZ[2]):
                reduction = (1 - abs(c_szZ[2]) / abs(c_sz[1])) * 100 if c_sz[1] != 0 else 0
                if abs(dr2) < 0.02:
                    verdict = "Z and z are degenerate\n(carry same information)"
                elif dr2 > 0.05:
                    verdict = f"Z adds {dr2:.3f} to R$^2$\n(independent information)"
                else:
                    verdict = f"Z adds {dr2:.3f} to R$^2$\n(marginal improvement)"
                ax.text(0.97, 0.95,
                        f"z coeff: {c_sz[1]:.2f} $\\rightarrow$ {c_szZ[2]:.2f}\n"
                        f"$\\Delta$R$^2$ = {dr2:+.3f}\n"
                        f"{verdict}",
                        transform=ax.transAxes, fontsize=7, ha="right", va="top",
                        bbox=dict(fc="lightyellow", alpha=0.9))
    else:
        missing = []
        if not has_sigma:
            missing.append("log_sigma_sfr")
        if not has_Z:
            missing.append("metallicity")
        ax.text(0.5, 0.5, f"Need {' + '.join(missing)}\nin bin_property_columns",
                transform=ax.transAxes, ha="center", fontsize=10)
        ax.set_title("Metallicity decomposition")
    ax.grid(True, alpha=0.2)

    # ────────────────────────────────────────────────────────────────
    n_det = valid_ratio.sum() if valid_ratio.any() else 0
    fig.suptitle(
        f"{line} line intensity: {n_det} measurements, "
        f"{(~df['line_in_band']).sum()} controls  (tier >= {min_tier})",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\n  Saved: {save_path}")

    return fig, df


def measure_all_lines(wrapper, min_tier="B", split_filter=None):
    """
    Measure intensity of all bright FIR lines and rank by strength.

    Returns a summary DataFrame sorted by L_line/L_FIR.
    """
    results = []
    for line_name in ["[CII] 158", "[NII] 205", "[OI] 145", "[NII] 122",
                      "[OIII] 88", "[OI] 63", "CO(5-4)", "CO(6-5)", "CO(7-6)"]:
        _, df = measure_line_intensity(
            wrapper, line=line_name,
            min_tier=min_tier, split_filter=split_filter,
        )
        plt.close()

        if df is not None:
            df_det = df[df["line_in_band"]]
            valid = np.isfinite(df_det["excess_frac"])
            valid_ratio = np.isfinite(df_det.get("l_line_over_l_ir", pd.Series())) & (df_det.get("l_line_over_l_ir", pd.Series()) > 0)

            if valid.any():
                ef = df_det.loc[valid, "excess_frac"]
                sem = ef.std() / np.sqrt(len(ef)) if len(ef) > 1 else np.inf
                row = {
                    "line": line_name,
                    "n_pops": valid.sum(),
                    "median_excess_pct": ef.median() * 100,
                    "mean_excess_pct": ef.mean() * 100,
                    "significance": ef.mean() / sem if sem > 0 else 0,
                }
                if valid_ratio.any():
                    row["median_L_line_L_FIR_pct"] = df_det.loc[valid_ratio, "l_line_over_l_ir"].median() * 100
                    row["median_L_line_Lsun"] = df_det.loc[valid_ratio, "l_line"].median()
                results.append(row)

    if results:
        summary = pd.DataFrame(results)
        summary = summary.sort_values("median_excess_pct", ascending=False)
        print("\n" + "=" * 75)
        print("FIR LINE INTENSITY SUMMARY")
        print("=" * 75)
        print(summary.to_string(index=False, float_format="%.3f"))
        return summary
    return None


if __name__ == "__main__":
    print("Usage:")
    print("  from analyze_cii_lines import measure_line_intensity, measure_all_lines")
    print()
    print("  # [CII] measurement with science plots")
    print("  fig, df = measure_line_intensity(wrapper)")
    print()
    print("  # Correlate with different properties")
    print("  fig, df = measure_line_intensity(wrapper, surface_color_by='metallicity')")
    print("  fig, df = measure_line_intensity(wrapper, surface_color_by='redshift')")
    print()
    print("  # Other lines")
    print("  fig, df = measure_line_intensity(wrapper, line='[NII] 205')")
    print("  fig, df = measure_line_intensity(wrapper, line='CO(6-5)')")
    print()
    print("  # Ranked summary of all lines")
    print("  summary = measure_all_lines(wrapper)")


# ── SED + Residual Grid ─────────────────────────────────────────────────

def plot_sed_residual_grid(
    wrapper,
    *,
    line="[CII] 158",
    col_dim=None,
    row_dim="auto",
    color_by="log_sigma_sfr",
    split_filter=None,
    min_tier="B",
    flux_unit="mJy",
    ylim=(1e-2, 5e2),
    resid_ylim=(-30, 30),
    figsize=None,
    save_path=None,
    dpi=150,
):
    """
    SED grid with residual sub-panels highlighting line excess.

    Each panel is split: top 3/4 shows the SED (data + model),
    bottom 1/4 shows (data-model)/model in percent, with the band
    containing the target line highlighted in color.

    Parameters
    ----------
    wrapper : SimstackWrapper
    line : str
        Line to highlight. Default "[CII] 158".
    col_dim, row_dim : str, optional
        Grid dimensions. Auto-detected if col_dim is None.
        row_dim='auto' (default): auto-detects second dimension.
        row_dim=None: forces single row, multiple SEDs overlap
        per panel colored by color_by.
    color_by : str
        Property to color overlapping SEDs by when multiple
        populations share a panel. Used when row_dim is None
        or when extra dimensions exist. Default 'log_sigma_sfr'.
        Also: 'stellar_mass', 'redshift', 'metallicity', etc.
    split_filter : list of int, optional
        Population class filter.
    min_tier : str
        Minimum fit quality tier.
    flux_unit : str
        'mJy' or 'Jy'.
    ylim : tuple
        y-axis limits for SED panel.
    resid_ylim : tuple
        y-axis limits for residual panel (percent).
    figsize : tuple, optional
    save_path : str or Path, optional
    dpi : int

    Returns
    -------
    fig
    """
    try:
        from simstack4.plots import _parse_bins, _extract_pop_type, _dim_label
    except ImportError:
        from plots import _parse_bins, _extract_pop_type, _dim_label

    pr = getattr(wrapper, "processed_results", None)
    sr = getattr(wrapper, "stacking_results", None)
    if pr is None or sr is None or not pr.sed_results:
        print("No results found")
        return None

    line_rest_lam = LINES.get(line)
    if line_rest_lam is None:
        print(f"Unknown line '{line}'")
        return None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    flux_scale = 1e3 if flux_unit == "mJy" else 1.0

    # Wavelengths from config
    wavelengths = np.array([
        wrapper.config.maps[m].wavelength for m in sr.map_names
    ])
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]

    # Detect model frame (same logic as measure_line_intensity)
    sample_sed = next((s for s in pr.sed_results.values()
                       if s.greybody_fit_success
                       and s.model_wavelengths is not None), None)
    model_is_rest = False
    if sample_sed is not None:
        mw = sample_sed.model_wavelengths
        mf = sample_sed.model_fluxes
        model_peak = mw[np.argmax(mf)]
        data_peak = sample_sed.wavelengths[np.argmax(sample_sed.flux_densities)]
        z_s = sample_sed.median_redshift
        ratio_asis = model_peak / data_peak
        ratio_shifted = model_peak * (1 + z_s) / data_peak
        if abs(ratio_shifted - 1.0) < abs(ratio_asis - 1.0):
            model_is_rest = True
            print(f"Model is rest-frame (peak {model_peak:.0f} um x (1+z) "
                  f"-> {model_peak*(1+z_s):.0f} um ~ data peak {data_peak:.0f} um)")

    # Filter populations
    pop_ids = list(pr.sed_results.keys())
    if split_filter is not None:
        allowed = {f"split_{i}" for i in split_filter}
        pop_ids = [p for p in pop_ids
                   if _extract_pop_type(p) in allowed
                   or _extract_pop_type(p) == "_all_"]

    # Filter by tier
    pop_ids = [p for p in pop_ids
               if pr.sed_results[p].greybody_fit_success
               and tier_rank.get(pr.sed_results[p].fit_quality_tier or "C", 2) <= min_rank]

    parsed = {pid: _parse_bins(pid) for pid in pop_ids}

    # Discover grid dimensions
    all_dims = set()
    for b in parsed.values():
        all_dims.update(b.keys())

    # Auto-detect col/row dims
    _priority = ["redshift", "z", "stellar_mass", "beta_uv", "log_sigma_sfr"]
    if col_dim is None:
        for d in _priority:
            if d in all_dims:
                col_dim = d
                break
        if col_dim is None:
            col_dim = sorted(all_dims)[0] if all_dims else None

    if row_dim == "auto":
        row_dim = None  # will be set below if a second dim exists
        for d in _priority:
            if d in all_dims and d != col_dim:
                row_dim = d
                break
    # If row_dim was passed as None explicitly, keep it as None (single row)

    if col_dim is None:
        print("Cannot determine grid dimensions")
        return None

    print(f"Grid: {col_dim} (cols) x {row_dim or 'none'} (rows)")

    # Unique bins per axis
    col_bins = sorted(set(parsed[p].get(col_dim) for p in pop_ids
                          if col_dim in parsed[p]))
    if row_dim:
        row_bins = sorted(set(parsed[p].get(row_dim) for p in pop_ids
                              if row_dim in parsed[p]))
    else:
        row_bins = [None]

    n_cols = len(col_bins)
    n_rows = len(row_bins)

    if figsize is None:
        figsize = (3.2 * n_cols, 3.5 * n_rows + 0.5)

    fig = plt.figure(figsize=figsize)

    # Create grid with shared x-axes, each cell = SED (top) + resid (bottom)
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize

    # Determine if we need a colorbar (multiple pops per panel)
    # Count max populations per panel
    max_pops_per_panel = 0
    for ri, row_bin in enumerate(row_bins):
        for ci, col_bin in enumerate(col_bins):
            n_match = sum(1 for pid in pop_ids
                          if parsed[pid].get(col_dim) == col_bin
                          and (row_dim is None or row_bin is None
                               or parsed[pid].get(row_dim) == row_bin))
            max_pops_per_panel = max(max_pops_per_panel, n_match)

    needs_colorbar = max_pops_per_panel > 1
    right_margin = 0.88 if needs_colorbar else 0.97

    outer_gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                 hspace=0.12, wspace=0.18,
                                 left=0.07, right=right_margin,
                                 top=0.93, bottom=0.05)

    line_band_color = "#e74c3c"  # red for line-contaminated band
    clean_color = "#2c3e50"      # dark for clean bands

    # Colormap for overlapping SEDs (when multiple pops per panel)
    pop_cmap = None
    pop_norm = None
    if needs_colorbar:
        # Collect color_by values from bin_properties or bin centers
        color_vals = []
        for pid in pop_ids:
            sed = pr.sed_results[pid]
            props = sed.bin_properties or {}
            if isinstance(props, str):
                import ast as _ast
                try:
                    props = _ast.literal_eval(props)
                except (ValueError, SyntaxError):
                    props = {}
            # Try bin_properties first
            val = None
            for key, v in (props if isinstance(props, dict) else {}).items():
                if color_by.replace("log_", "") in key.lower():
                    val = v
                    break
            # Fallback to bin centers
            if val is None:
                bins = parsed[pid]
                for dim, (lo, hi) in bins.items():
                    if color_by.replace("log_", "") in dim.lower():
                        val = (lo + hi) / 2
                        break
            if val is not None and np.isfinite(val):
                color_vals.append(val)

        if color_vals:
            pop_norm = Normalize(vmin=min(color_vals), vmax=max(color_vals))
            pop_cmap = mcm.get_cmap("viridis")
            print(f"Coloring overlapping SEDs by {color_by}: "
                  f"[{min(color_vals):.2f}, {max(color_vals):.2f}]")

    for ri, row_bin in enumerate(row_bins):
        for ci, col_bin in enumerate(col_bins):
            # Find matching population(s)
            matches = []
            for pid in pop_ids:
                b = parsed[pid]
                if b.get(col_dim) != col_bin:
                    continue
                if row_dim and b.get(row_dim) != row_bin:
                    continue
                matches.append(pid)

            # Create split panel
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                4, 1, subplot_spec=outer_gs[ri, ci], hspace=0.0,
            )
            ax_sed = fig.add_subplot(inner_gs[:3, 0])
            ax_res = fig.add_subplot(inner_gs[3, 0], sharex=ax_sed)

            if not matches:
                ax_sed.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax_sed.transAxes, fontsize=14, color="gray")
                ax_sed.set_xlim(10, 1200)
                ax_res.set_xlim(10, 1200)
                plt.setp(ax_sed.get_xticklabels(), visible=False)
                continue

            for pid in matches:
                sed = pr.sed_results[pid]
                z = sed.median_redshift
                lam_line_obs = line_rest_lam * (1 + z)

                # Get color from color_by property
                pop_color = clean_color
                if pop_cmap and pop_norm:
                    props = sed.bin_properties or {}
                    if isinstance(props, str):
                        import ast as _ast2
                        try:
                            props = _ast2.literal_eval(props)
                        except (ValueError, SyntaxError):
                            props = {}
                    cval = None
                    for key, v in (props if isinstance(props, dict) else {}).items():
                        if color_by.replace("log_", "") in key.lower():
                            cval = v
                            break
                    if cval is None:
                        for dim, (lo, hi) in parsed[pid].items():
                            if color_by.replace("log_", "") in dim.lower():
                                cval = (lo + hi) / 2
                                break
                    if cval is not None and np.isfinite(cval):
                        pop_color = pop_cmap(pop_norm(cval))

                # Identify which band has the line
                line_mask = np.zeros(len(sed.wavelengths), dtype=bool)
                for j, w in enumerate(sed.wavelengths):
                    b, blo, bhi = _band_for_wavelength(w)
                    if b and blo <= lam_line_obs <= bhi:
                        line_mask[j] = True

                # Model curve (corrected for frame)
                if sed.model_wavelengths is not None:
                    mw = sed.model_wavelengths
                    if model_is_rest:
                        mw = mw * (1 + z)
                    ax_sed.plot(mw, sed.model_fluxes * flux_scale,
                                "-", color=pop_color, alpha=0.6, lw=1.5, zorder=1)

                # Data points
                for j in range(len(sed.wavelengths)):
                    if line_mask[j]:
                        color = line_band_color
                        marker = "s"
                        ms = 7
                    else:
                        color = pop_color
                        marker = "o"
                        ms = 5
                    ax_sed.errorbar(
                        sed.wavelengths[j],
                        sed.flux_densities[j] * flux_scale,
                        yerr=sed.flux_errors[j] * flux_scale,
                        fmt=marker, color=color, ms=ms,
                        capsize=2, elinewidth=0.8, zorder=3,
                    )

                # Residuals
                if sed.model_wavelengths is not None:
                    mw = sed.model_wavelengths
                    if model_is_rest:
                        mw = mw * (1 + z)
                    for j in range(len(sed.wavelengths)):
                        fm = np.interp(sed.wavelengths[j], mw, sed.model_fluxes)
                        if fm > 0:
                            resid_pct = (sed.flux_densities[j] - fm) / fm * 100
                            err_pct = sed.flux_errors[j] / fm * 100
                            if line_mask[j]:
                                color = line_band_color
                                marker = "s"
                                ms = 7
                            else:
                                color = pop_color
                                marker = "o"
                                ms = 5
                            ax_res.errorbar(
                                sed.wavelengths[j], resid_pct,
                                yerr=err_pct,
                                fmt=marker, color=color, ms=ms,
                                capsize=2, elinewidth=0.8, zorder=3,
                            )

                # Mark line wavelength
                ax_sed.axvline(lam_line_obs, color=line_band_color,
                               ls=":", alpha=0.3, lw=1)
                ax_res.axvline(lam_line_obs, color=line_band_color,
                               ls=":", alpha=0.3, lw=1)

            # Formatting
            ax_sed.set_xscale("log")
            ax_sed.set_yscale("log")
            ax_sed.set_ylim(ylim)
            ax_sed.set_xlim(15, 1200)
            ax_sed.tick_params(axis="both", labelsize=6)
            plt.setp(ax_sed.get_xticklabels(), visible=False)
            # Suppress y-ticks on non-leftmost panels
            if ci > 0:
                plt.setp(ax_sed.get_yticklabels(), visible=False)

            ax_res.set_xscale("log")
            ax_res.set_ylim(resid_ylim)
            ax_res.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax_res.set_xlim(15, 1200)
            ax_res.tick_params(axis="both", labelsize=6)
            # Suppress x-ticks on non-bottom rows
            if ri < n_rows - 1:
                plt.setp(ax_res.get_xticklabels(), visible=False)
            if ci > 0:
                plt.setp(ax_res.get_yticklabels(), visible=False)

            # Labels
            tier = sed.fit_quality_tier or "?"
            n_src = sed.n_sources
            T = sed.dust_temperature_rest_frame
            label_parts = []
            if T:
                label_parts.append(f"T={T:.0f}K")
            label_parts.append(f"N={n_src:,}")
            label_parts.append(f"[{tier}]")
            ax_sed.text(0.97, 0.95, " ".join(label_parts),
                        transform=ax_sed.transAxes, fontsize=6,
                        ha="right", va="top",
                        bbox=dict(fc="white", alpha=0.8, pad=1))

            # Column header
            if ri == 0:
                lo, hi = col_bin
                ax_sed.set_title(f"{lo:.2g}–{hi:.2g}", fontsize=7, pad=2)

            # Row label
            if ci == 0 and row_bin is not None:
                lo, hi = row_bin
                ax_sed.set_ylabel(f"{lo:.2g}–{hi:.2g}", fontsize=7)

            # Bottom row x-label
            if ri == n_rows - 1:
                ax_res.set_xlabel("$\\lambda_{obs}$ ($\\mu$m)", fontsize=7)

            # Residual y-label
            if ci == 0:
                ax_res.set_ylabel("%", fontsize=6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color=line_band_color, ls="none",
               ms=7, label=f"{line} in band"),
        Line2D([0], [0], marker="o", color="gray", ls="none",
               ms=5, label="Continuum bands"),
        Line2D([0], [0], color=line_band_color, ls=":", lw=1,
               label=f"Obs. {line} wavelength"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.99))

    # Colorbar when multiple pops per panel
    if needs_colorbar and pop_cmap and pop_norm:
        sm = plt.cm.ScalarMappable(cmap=pop_cmap, norm=pop_norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(color_by, fontsize=10)
        cbar.ax.tick_params(labelsize=7)

    # Dimension labels at figure edges
    try:
        col_label = _dim_label(col_dim)
    except Exception:
        col_label = col_dim
    fig.text(0.5, 0.97, col_label, ha="center", fontsize=10, fontweight="bold")
    if row_dim:
        try:
            row_label = _dim_label(row_dim)
        except Exception:
            row_label = row_dim
        fig.text(0.01, 0.5, row_label, va="center", rotation=90,
                 fontsize=10, fontweight="bold")

    fig.suptitle(
        f"{line} | {len(pop_ids)} pops, tier >= {min_tier}",
        fontsize=10, y=1.00,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
