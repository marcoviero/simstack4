#!/usr/bin/env python3
"""
Measure warm dust contamination of PACS bands from stacked SEDs.

At z > 2, PACS 100 probes rest-frame ~30um where stochastically
heated small grains produce emission above the cool dust greybody.
This biases T_dust high when PACS bands are included in the fit.

Method
------
The greybody model is fit to SPIRE+SCUBA2 only (the "cool dust"
bands). The PACS excess above this extrapolation is attributed to
warm dust. By measuring this excess as a function of redshift, we
trace the transition from "PACS probes the greybody peak" (z < 1)
to "PACS probes warm dust" (z > 2).

Key diagnostic: if the warm dust fraction in PACS 100 correlates
with the apparent T_dust increase at high z, then part of the
T_dust evolution is an artifact of fitting a single-component
greybody to a two-component (cool + warm) SED.

Usage
-----
    from simstack4.analyze_warm_dust import measure_warm_dust

    fig, df = measure_warm_dust(wrapper)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Bands classified by dust component
COOL_DUST_BANDS = {"SPIRE_250", "SPIRE_350", "SPIRE_500", "SCUBA2_850"}
WARM_DUST_BANDS = {"PACS_100", "PACS_160"}
MID_IR_BANDS = {"MIPS_24"}

BAND_EDGES = {
    "MIPS_24": (20.5, 30.0),
    "PACS_100": (85, 125),
    "PACS_160": (130, 210),
    "SPIRE_250": (194, 313),
    "SPIRE_350": (283, 413),
    "SPIRE_500": (383, 693),
    "SCUBA2_850": (770, 940),
}


def _identify_band(wave_um):
    """Return band name for an observed wavelength."""
    for band, (lo, hi) in BAND_EDGES.items():
        center = (lo + hi) / 2
        if abs(wave_um - center) / center < 0.3:
            return band
    return None


def measure_warm_dust(
    wrapper,
    *,
    min_tier="B",
    split_filter=None,
    save_path=None,
):
    """
    Measure warm dust excess in PACS bands above the cool-dust greybody.

    Parameters
    ----------
    wrapper : SimstackWrapper
    min_tier : str
    split_filter : list of int, optional
    save_path : str or Path, optional

    Returns
    -------
    fig, df
    """
    try:
        from simstack4.plots import _parse_bins, _extract_pop_type
    except ImportError:
        from plots import _parse_bins, _extract_pop_type

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed results")
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
        if abs(model_peak * (1 + z_s) / data_peak - 1.0) < abs(
            model_peak / data_peak - 1.0
        ):
            model_is_rest = True
            print("Model wavelengths in rest frame — applying (1+z) correction")

    # ── Extract per-population measurements ──────────────────────────
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

        # bin properties
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
        if "stellar_mass" not in extra:
            extra["stellar_mass"] = sed.median_mass

        if sed.model_wavelengths is None or sed.model_fluxes is None:
            continue

        # Model in observed frame
        mw = sed.model_wavelengths
        if model_is_rest:
            mw = mw * (1 + z)

        # Compute residuals per band
        band_data = {}
        for j in range(len(sed.wavelengths)):
            band = _identify_band(sed.wavelengths[j])
            if band is None:
                continue

            # Only interpolate if within model grid
            if mw.min() <= sed.wavelengths[j] <= mw.max():
                f_model = np.interp(sed.wavelengths[j], mw, sed.model_fluxes)
            else:
                f_model = 0.0  # outside grid (e.g., 24um)

            f_data = sed.flux_densities[j]
            f_err = sed.flux_errors[j]
            rest_lam = sed.wavelengths[j] / (1 + z)

            band_data[band] = {
                "f_data": f_data,
                "f_model": f_model,
                "f_err": f_err,
                "excess": f_data - f_model,
                "excess_frac": (f_data - f_model) / f_model if f_model > 0 else np.nan,
                "rest_lam": rest_lam,
            }

        # Compute warm dust metrics
        row = {
            "pop_id": pop_id,
            "z": z,
            "l_ir": l_ir,
            "log_l_ir": np.log10(l_ir),
            "T_dust": T_dust,
            "n_sources": sed.n_sources,
            "tier": tier,
            **extra,
        }

        # PACS excess fractions
        for pacs_band in ["PACS_100", "PACS_160"]:
            if pacs_band in band_data:
                bd = band_data[pacs_band]
                prefix = pacs_band.lower().replace("pacs_", "p")
                row[f"{prefix}_excess_frac"] = bd["excess_frac"]
                row[f"{prefix}_excess_jy"] = bd["excess"]
                row[f"{prefix}_rest_lam"] = bd["rest_lam"]
                row[f"{prefix}_snr"] = (
                    bd["excess"] / bd["f_err"] if bd["f_err"] > 0 else 0
                )
            else:
                prefix = pacs_band.lower().replace("pacs_", "p")
                row[f"{prefix}_excess_frac"] = np.nan
                row[f"{prefix}_excess_jy"] = np.nan
                row[f"{prefix}_rest_lam"] = np.nan
                row[f"{prefix}_snr"] = np.nan

        # Mean SPIRE residual (baseline — how well does the greybody fit?)
        spire_fracs = []
        for band in COOL_DUST_BANDS:
            if band in band_data and np.isfinite(band_data[band]["excess_frac"]):
                spire_fracs.append(band_data[band]["excess_frac"])
        row["spire_baseline_frac"] = np.mean(spire_fracs) if spire_fracs else np.nan

        # Corrected PACS excess (subtract baseline)
        for pacs_band, prefix in [("PACS_100", "p100"), ("PACS_160", "p160")]:
            raw = row.get(f"{prefix}_excess_frac", np.nan)
            bl = row["spire_baseline_frac"]
            if np.isfinite(raw) and np.isfinite(bl):
                row[f"{prefix}_excess_corrected"] = raw - bl
            else:
                row[f"{prefix}_excess_corrected"] = np.nan

        rows.append(row)

    if not rows:
        print("No populations found")
        return None, None

    df = pd.DataFrame(rows)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("WARM DUST CONTAMINATION OF PACS BANDS")
    print(f"{'='*65}")
    print(f"  Populations: {len(df)}")

    for prefix, band in [("p100", "PACS 100"), ("p160", "PACS 160")]:
        col = f"{prefix}_excess_corrected"
        valid = np.isfinite(df[col])
        if valid.any():
            print(f"\n  {band} excess above greybody (baseline-corrected):")
            print(f"    median = {df.loc[valid, col].median()*100:+.1f}%")

            # By redshift
            z_edges = np.arange(0, min(df["z"].max() + 1, 12), 0.5)
            for i in range(len(z_edges) - 1):
                zbin = valid & (df["z"] >= z_edges[i]) & (df["z"] < z_edges[i + 1])
                if zbin.sum() > 0:
                    med = df.loc[zbin, col].median() * 100
                    rest = df.loc[zbin, f"{prefix}_rest_lam"].median()
                    flag = " *** WARM DUST" if med > 10 else ""
                    print(
                        f"      z={z_edges[i]:.1f}-{z_edges[i+1]:.1f}: "
                        f"excess = {med:+.1f}%  "
                        f"(rest {rest:.0f}um, N={zbin.sum()}){flag}"
                    )

    # ── 6-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    cmap = "plasma"

    # Panel 1: PACS 100 excess vs redshift
    ax = axes[0, 0]
    col = "p100_excess_corrected"
    valid = np.isfinite(df[col])
    if valid.any():
        sc = ax.scatter(
            df.loc[valid, "z"],
            df.loc[valid, col] * 100,
            c=df.loc[valid, "T_dust"],
            cmap="inferno",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="T$_{dust}$ (K)")

        # Binned medians with error bars
        z_edges = np.arange(0, min(df["z"].max() + 1, 12), 0.5)
        for i in range(len(z_edges) - 1):
            zbin = valid & (df["z"] >= z_edges[i]) & (df["z"] < z_edges[i + 1])
            if zbin.sum() > 2:
                med = df.loc[zbin, col].median() * 100
                p16 = np.percentile(df.loc[zbin, col] * 100, 16)
                p84 = np.percentile(df.loc[zbin, col] * 100, 84)
                ax.errorbar(
                    (z_edges[i] + z_edges[i + 1]) / 2,
                    med,
                    yerr=[[med - p16], [p84 - med]],
                    fmt="ks",
                    ms=9,
                    mfc="white",
                    mew=2,
                    capsize=3,
                    zorder=5,
                )

        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.axhline(10, color="r", ls=":", lw=1, alpha=0.5, label=">10% contamination")

        # Rest-frame wavelength axis on top
        ax2 = ax.twiny()
        z_ticks = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
        rest_100 = 100 / (1 + z_ticks)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(z_ticks)
        ax2.set_xticklabels([f"{r:.0f}" for r in rest_100], fontsize=7)
        ax2.set_xlabel("Rest-frame $\\lambda$ at 100$\\mu$m (${\\mu}$m)", fontsize=8)

    ax.set_xlabel("Redshift")
    ax.set_ylabel("PACS 100 excess (%)")
    ax.set_title("Warm dust in PACS 100")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: PACS 160 excess vs redshift
    ax = axes[0, 1]
    col = "p160_excess_corrected"
    valid = np.isfinite(df[col])
    if valid.any():
        sc = ax.scatter(
            df.loc[valid, "z"],
            df.loc[valid, col] * 100,
            c=df.loc[valid, "T_dust"],
            cmap="inferno",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="T$_{dust}$ (K)")

        z_edges = np.arange(0, min(df["z"].max() + 1, 12), 0.5)
        for i in range(len(z_edges) - 1):
            zbin = valid & (df["z"] >= z_edges[i]) & (df["z"] < z_edges[i + 1])
            if zbin.sum() > 2:
                med = df.loc[zbin, col].median() * 100
                p16 = np.percentile(df.loc[zbin, col] * 100, 16)
                p84 = np.percentile(df.loc[zbin, col] * 100, 84)
                ax.errorbar(
                    (z_edges[i] + z_edges[i + 1]) / 2,
                    med,
                    yerr=[[med - p16], [p84 - med]],
                    fmt="ks",
                    ms=9,
                    mfc="white",
                    mew=2,
                    capsize=3,
                    zorder=5,
                )

        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.axhline(10, color="r", ls=":", lw=1, alpha=0.5)

        ax2 = ax.twiny()
        rest_160 = 160 / (1 + z_ticks)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(z_ticks)
        ax2.set_xticklabels([f"{r:.0f}" for r in rest_160], fontsize=7)
        ax2.set_xlabel("Rest-frame $\\lambda$ at 160$\\mu$m ($\\mu$m)", fontsize=8)

    ax.set_xlabel("Redshift")
    ax.set_ylabel("PACS 160 excess (%)")
    ax.set_title("Warm dust in PACS 160")
    ax.grid(True, alpha=0.2)

    # Panel 3: PACS 100 excess vs T_dust (is T_dust biased?)
    ax = axes[0, 2]
    col = "p100_excess_corrected"
    valid = np.isfinite(df[col]) & np.isfinite(df["T_dust"])
    if valid.any():
        sc = ax.scatter(
            df.loc[valid, "T_dust"],
            df.loc[valid, col] * 100,
            c=df.loc[valid, "z"],
            cmap=cmap,
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Redshift")

        # Fit
        x = df.loc[valid, "T_dust"].values
        y = df.loc[valid, col].values * 100
        w = np.sqrt(df.loc[valid, "n_sources"].values.astype(float))
        try:
            coeffs = np.polyfit(x, y, 1, w=w)
            x_grid = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                x_grid,
                np.polyval(coeffs, x_grid),
                "k--",
                lw=2,
                label=f"slope={coeffs[0]:.1f}%/K",
            )
            ax.legend(fontsize=8)
        except Exception:
            pass

        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("T$_{dust}$ (K)")
    ax.set_ylabel("PACS 100 excess (%)")
    ax.set_title("Does warm dust bias T$_{dust}$?")
    ax.grid(True, alpha=0.2)

    # Panel 4: PACS excess vs L_IR
    ax = axes[1, 0]
    col = "p100_excess_corrected"
    valid = np.isfinite(df[col]) & np.isfinite(df["log_l_ir"])
    if valid.any():
        sc = ax.scatter(
            df.loc[valid, "l_ir"],
            df.loc[valid, col] * 100,
            c=df.loc[valid, "z"],
            cmap=cmap,
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Redshift")
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("L$_{IR}$ (L$_\\odot$)")
    ax.set_ylabel("PACS 100 excess (%)")
    ax.set_title("Warm dust vs L$_{IR}$")
    ax.grid(True, alpha=0.2)

    # Panel 5: PACS excess vs Sigma_SFR
    ax = axes[1, 1]
    has_sigma = "log_sigma_sfr" in df.columns and np.isfinite(df["log_sigma_sfr"]).any()
    if has_sigma:
        valid_s = np.isfinite(df["p100_excess_corrected"]) & np.isfinite(
            df["log_sigma_sfr"]
        )
        if valid_s.any():
            sc = ax.scatter(
                df.loc[valid_s, "log_sigma_sfr"],
                df.loc[valid_s, "p100_excess_corrected"] * 100,
                c=df.loc[valid_s, "z"],
                cmap=cmap,
                s=30,
                alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")
            ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("log $\\Sigma_{SFR}$ (M$_\\odot$/yr/kpc$^2$)")
    ax.set_ylabel("PACS 100 excess (%)")
    ax.set_title("Warm dust vs $\\Sigma_{SFR}$")
    ax.grid(True, alpha=0.2)

    # Panel 6: Recommended inflation factor vs z
    ax = axes[1, 2]
    col100 = "p100_excess_corrected"
    col160 = "p160_excess_corrected"
    v100 = np.isfinite(df[col100])
    v160 = np.isfinite(df[col160])

    z_edges = np.arange(0, min(df["z"].max() + 1, 12), 0.5)
    z_mids, inf100, inf160 = [], [], []
    for i in range(len(z_edges) - 1):
        z_mid = (z_edges[i] + z_edges[i + 1]) / 2
        z_mids.append(z_mid)

        for vmask, col, store in [(v100, col100, inf100), (v160, col160, inf160)]:
            zbin = vmask & (df["z"] >= z_edges[i]) & (df["z"] < z_edges[i + 1])
            if zbin.sum() > 0:
                med_excess = abs(df.loc[zbin, col].median())
                # If excess is X%, then the band has X% warm dust contamination.
                # Suggested inflation: scale error by (1 + excess_fraction)
                # so the band weight drops proportionally.
                store.append(max(1 + med_excess * 5, 1.0))  # 5x multiplier for safety
            else:
                store.append(1.0)

    if z_mids:
        ax.plot(z_mids, inf100, "o-", color="C0", lw=2, label="PACS 100")
        ax.plot(z_mids, inf160, "s-", color="C1", lw=2, label="PACS 160")
        ax.axhline(1, color="k", ls="--", lw=1, alpha=0.5)
        ax.axhline(10, color="r", ls=":", lw=1, alpha=0.3, label="Current default (10)")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Suggested inflation factor")
    ax.set_title("Adaptive error inflation\n(based on warm dust excess)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Warm dust contamination of PACS bands  |  "
        f"{len(df)} populations (tier >= {min_tier})",
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


def fit_warm_dust_model(df, verbose=True):
    """
    Fit empirical warm dust model from PACS 100 excess measurements.

    The warm dust amplitude is parameterized as:
        log10(f_warm / f_peak) = a*logM* + b*z + c

    or with Sigma_SFR:
        log10(f_warm / f_peak) = a*logM* + b*z + c*logSigma_SFR + d

    Parameters
    ----------
    df : DataFrame
        Output from measure_warm_dust. Must contain 'p100_excess_corrected',
        'z', and 'stellar_mass'. Optionally 'log_sigma_sfr'.

    Returns
    -------
    model : dict
        Keys: 'coeffs', 'r_squared', 'n_fit', 'labels'.
        'coeffs' is ready to paste into _pah_flux as _warm_coeffs.
    """
    # The PACS 100 excess (baseline-corrected) is the warm dust fraction:
    # f_warm / f_greybody ≈ excess_corrected
    # We want f_warm / f_peak (greybody peak), which is similar since
    # PACS 100 is near the greybody peak at z < 1.

    col = "p100_excess_corrected"
    valid = (
        np.isfinite(df[col])
        & (df[col] > 0)  # only positive excess = warm dust
        & np.isfinite(df["z"])
        & np.isfinite(df.get("stellar_mass", pd.Series(dtype=float)))
    )

    if valid.sum() < 5:
        print(f"Only {valid.sum()} populations with positive PACS 100 excess")
        return None

    y = np.log10(df.loc[valid, col].values)
    z_arr = df.loc[valid, "z"].values
    mass_arr = df.loc[valid, "stellar_mass"].values
    w = np.sqrt(df.loc[valid, "n_sources"].values.astype(float))

    # Try with Sigma_SFR if available
    has_sigma = (
        "log_sigma_sfr" in df.columns
        and np.isfinite(df.loc[valid, "log_sigma_sfr"]).all()
    )

    if has_sigma:
        sigma_arr = df.loc[valid, "log_sigma_sfr"].values
        X = np.column_stack([mass_arr, z_arr, sigma_arr, np.ones(valid.sum())])
        labels = ["logM*", "z", "logSigma_SFR", "const"]
    else:
        X = np.column_stack([mass_arr, z_arr, np.ones(valid.sum())])
        labels = ["logM*", "z", "const"]

    coeffs, _, _, _ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)

    pred = X @ coeffs
    ss_res = np.sum(w**2 * (y - pred) ** 2)
    ss_tot = np.sum(w**2 * (y - np.average(y, weights=w**2)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    model = {
        "coeffs": coeffs,
        "r_squared": r2,
        "n_fit": int(valid.sum()),
        "labels": labels,
    }

    if verbose:
        terms = " + ".join(f"{c:.3f}*{l}" for c, l in zip(coeffs, labels))
        print(f"\nWarm dust model:")
        print(f"  log10(f_warm/f_peak) = {terms}")
        print(f"  R² = {r2:.3f}  (N={valid.sum()})")
        print(f"\n  Example predictions:")
        for z_ex in [0.5, 1.0, 2.0, 3.0, 5.0]:
            if has_sigma:
                for sig in [-0.5, 0.5]:
                    pred_val = 10 ** (
                        coeffs[0] * 10.5
                        + coeffs[1] * z_ex
                        + coeffs[2] * sig
                        + coeffs[3]
                    )
                    print(
                        f"    z={z_ex}, logM*=10.5, logΣ={sig:.1f}: "
                        f"f_warm/f_peak = {pred_val:.4f} ({pred_val*100:.1f}%)"
                    )
            else:
                pred_val = 10 ** (coeffs[0] * 10.5 + coeffs[1] * z_ex + coeffs[-1])
                print(
                    f"    z={z_ex}, logM*=10.5: "
                    f"f_warm/f_peak = {pred_val:.4f} ({pred_val*100:.1f}%)"
                )

        print(f"\n  Paste into _pah_flux:")
        coeffs_str = ", ".join(f"{c:.3f}" for c in coeffs)
        print(f"    _warm_coeffs = np.array([{coeffs_str}])")

    return model


if __name__ == "__main__":
    print("Usage:")
    print("  from simstack4.analyze_warm_dust import measure_warm_dust, fit_warm_dust_model")
    print()
    print("  # Step 1: Measure PACS excess")
    print("  fig, df = measure_warm_dust(wrapper)")
    print()
    print("  # Step 2: Fit warm dust coefficients")
    print("  model = fit_warm_dust_model(df)")
    print("  # → prints coefficients to paste into _pah_flux")
    print()
    print("  # Step 3: Paste _warm_coeffs into greybody_pah_extension.py")
    print("  # _warm_coeffs = np.array([...])  # from model['coeffs']")
