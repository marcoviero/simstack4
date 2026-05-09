"""
PAH + warm dust model for the Wien side of the FIR SED.

Replaces the empirical ν^(-α) power-law extension with a physically
motivated model: sum of Gaussian PAH features + warm dust continuum,
calibrated from stacking measurements.

Two modes:
    1. Predicted: PAH amplitude from empirical model (no free params)
    2. Free: PAH amplitude as a fitting parameter

Usage in greybody.py:
    from pah_model import PAHModel

    pah = PAHModel()
    # or with custom empirical coefficients:
    pah = PAHModel(coeffs=[0.098, -0.321, 0.094, -2.060])

    # Get PAH flux at rest-frame wavelengths
    f_pah = pah.flux(wavelength_rest_um, amplitude_log10, z=1.5, log_l_ir=11.0)

    # Full SED: greybody + PAH
    f_total = greybody_flux + pah.flux(...)
"""

import numpy as np
from .plots import (
    plot_pah_fit,
    plot_pah_forward_fit,
    plot_pah_vs_property,
    plot_pah_multibin_forward_fit,
)

# numpy 2.x compatibility
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# PAH feature parameters (rest-frame)
# (center_um, relative_strength, fwhm_um)
PAH_FEATURES = [
    (6.2, 0.1262, 0.19),  # C-C stretch
    (7.7, 0.4577, 0.70),  # C-C stretch (strongest)
    (8.6, 0.6089, 0.34),  # C-H in-plane bend
    (11.3, 0.0000, 0.24),  # C-H out-of-plane bend (not detected)
    (12.7, 0.5187, 0.45),  # C-H out-of-plane bend
]

# Minor features (optional, for high-resolution work)
PAH_FEATURES_MINOR = [
    (3.3, 0.08, 0.05),  # C-H stretch
    (5.25, 0.05, 0.10),  # combination band
    (5.7, 0.05, 0.10),  # combination band
    (16.4, 0.10, 0.20),  # C-H/C-C bend
    (17.0, 0.08, 0.30),  # C-C-C bend
]


class PAHModel:
    """
    PAH + warm dust emission model for the Wien side of the FIR SED.

    The model has two components:
        1. PAH features: sum of Gaussians at known rest wavelengths
        2. Warm dust continuum: modified blackbody at T_warm (~60K)

    The overall amplitude is set by an empirical relation:
        log10(L_PAH / L_IR) = a * log10(L_IR) + b * z + c * PAH_template + d

    Parameters
    ----------
    coeffs : array-like of length 4, optional
        Empirical model coefficients [a, b, c, d] from fit_pah_model.
        Default: [0.098, -0.321, 0.094, -2.060] (from COSMOS stacking).
    T_warm : float
        Warm dust continuum temperature (K). Default 60.
    warm_frac : float
        Fraction of mid-IR luminosity in the warm continuum vs PAH features.
        Default 0.3 (30% continuum, 70% features).
    include_minor : bool
        Include minor PAH features (3.3, 5.25, 5.7, 16.4, 17.0 um).
    """

    def __init__(
        self,
        coeffs=None,
        T_warm=60.0,
        warm_frac=0.3,
        include_minor=False,
    ):
        if coeffs is not None:
            self.coeffs = np.asarray(coeffs, dtype=float)
        else:
            # Default from COSMOS stacking (measure_pah_excess + fit_pah_model)
            self.coeffs = np.array([0.098, -0.321, 0.094, -2.060])

        self.T_warm = T_warm
        self.warm_frac = warm_frac

        self.features = list(PAH_FEATURES)
        if include_minor:
            self.features.extend(PAH_FEATURES_MINOR)

        # Physical constants (CGS)
        self.h = 6.626e-27  # erg s
        self.k_B = 1.381e-16  # erg/K
        self.c = 2.998e10  # cm/s

    def pah_template_strength(self, z, band_lo=20.5, band_hi=30.0):
        """
        Compute relative PAH feature strength in a broadband filter.

        Used as input to the empirical amplitude model.
        """
        from scipy.special import erf

        total = 0.0
        for lam_c, strength, fwhm in self.features[:5]:  # major only
            lam_obs = lam_c * (1 + z)
            sigma = fwhm * (1 + z) / 2.355
            frac = 0.5 * (
                erf((band_hi - lam_obs) / (sigma * np.sqrt(2)))
                - erf((band_lo - lam_obs) / (sigma * np.sqrt(2)))
            )
            total += strength * max(frac, 0)
        return total

    def predict_amplitude(self, z, log_l_ir):
        """
        Predict log10(L_PAH / L_IR) from the empirical model.

        Parameters
        ----------
        z : float
        log_l_ir : float
            log10(L_IR / L_sun)

        Returns
        -------
        log_ratio : float
            log10(L_PAH / L_IR)
        """
        a, b, c, d = self.coeffs
        pah_strength = self.pah_template_strength(z)
        return a * log_l_ir + b * z + c * pah_strength + d

    def feature_spectrum(self, wavelength_um):
        """
        PAH feature spectrum (sum of Gaussians) at rest-frame wavelengths.

        Returns normalized spectrum (peak = 1 at 7.7um feature center).

        Parameters
        ----------
        wavelength_um : array
            Rest-frame wavelengths in microns.

        Returns
        -------
        spectrum : array
            Relative flux (arbitrary units, peak ~ 1).
        """
        lam = np.asarray(wavelength_um, dtype=float)
        spectrum = np.zeros_like(lam)

        for lam_c, strength, fwhm in self.features:
            sigma = fwhm / 2.355
            spectrum += strength * np.exp(-0.5 * ((lam - lam_c) / sigma) ** 2)

        return spectrum

    def warm_continuum(self, wavelength_um):
        """
        Warm dust continuum (modified blackbody at T_warm).

        Parameters
        ----------
        wavelength_um : array
            Rest-frame wavelengths in microns.

        Returns
        -------
        spectrum : array
            Relative flux (normalized to match PAH feature integral).
        """
        lam = np.asarray(wavelength_um, dtype=float)
        nu = self.c / (lam * 1e-4)  # Hz

        x = self.h * nu / (self.k_B * self.T_warm)
        x = np.minimum(x, 500)

        # Modified blackbody with beta=1.5 (warmer component, less steep)
        bb = nu**1.5 * 2 * self.h * nu**3 / self.c**2 / (np.exp(x) - 1)

        # Normalize so integral roughly matches PAH feature spectrum
        pah_spec = self.feature_spectrum(lam)
        if bb.max() > 0 and pah_spec.max() > 0:
            bb *= pah_spec.sum() / bb.sum() * self.warm_frac / (1 - self.warm_frac)

        return bb

    def flux(
        self,
        wavelength_um,
        greybody_amplitude_log10,
        *,
        z=None,
        log_l_ir=None,
        pah_amplitude=None,
    ):
        """
        PAH + warm dust flux density at rest-frame wavelengths.

        The amplitude can be set in two ways:
            1. Predicted: provide z and log_l_ir (uses empirical model)
            2. Free: provide pah_amplitude directly

        The returned flux is in the same units as the greybody amplitude:
        if greybody_amplitude is log10(A) where A gives mJy, then the
        PAH flux is also in mJy.

        Parameters
        ----------
        wavelength_um : array
            Rest-frame wavelengths (microns).
        greybody_amplitude_log10 : float
            log10 of the greybody amplitude (same as in greybody_model).
            Used to scale PAH flux relative to the cool dust.
        z : float, optional
            Redshift (for predicted amplitude).
        log_l_ir : float, optional
            log10(L_IR/L_sun) (for predicted amplitude).
        pah_amplitude : float, optional
            Direct log10(PAH/greybody) amplitude. Overrides prediction.

        Returns
        -------
        flux_density : array
            PAH + warm dust flux density (same units as greybody).
        """
        lam = np.asarray(wavelength_um, dtype=float)

        # PAH feature spectrum (normalized)
        pah_spec = self.feature_spectrum(lam)

        # Warm dust continuum
        warm_spec = self.warm_continuum(lam)

        # Combined template
        template = pah_spec + warm_spec

        # Determine amplitude
        if pah_amplitude is not None:
            log_ratio = pah_amplitude
        elif z is not None and log_l_ir is not None:
            log_ratio = self.predict_amplitude(z, log_l_ir)
        else:
            raise ValueError("Provide either (z, log_l_ir) or pah_amplitude")

        # Scale: the ratio is L_PAH/L_IR, and we need flux units
        # The greybody amplitude A gives the overall flux scale.
        # L_PAH/L_IR ≈ integral(PAH) / integral(greybody)
        # We scale the template so that its integral relative to a
        # reference greybody gives the predicted ratio.
        #
        # Simple scaling: pah_flux = 10^(amplitude) * ratio * template_normalized
        # where template is normalized to peak = 1
        A_gb = 10**greybody_amplitude_log10
        ratio = 10**log_ratio

        # Normalize template to unit integral (over the wavelength grid)
        if template.max() > 0:
            template_norm = template / _trapz(template, lam)
        else:
            return np.zeros_like(lam)

        # The greybody integral over the same wavelength range is ~A_gb * bandwidth
        # We want: integral(pah_flux) / integral(greybody) = ratio
        # Approximate: pah_peak ≈ ratio * A_gb * (greybody_bandwidth / pah_bandwidth)
        # For a simpler approach: scale so that the PAH flux at the
        # 7.7um peak gives the right L_PAH/L_IR when integrated.
        scale = A_gb * ratio * _trapz(np.ones_like(lam), lam)
        flux_density = scale * template_norm

        return flux_density

    def fit_from_spectrum(
        self,
        rest_wavelengths,
        flux_values,
        flux_errors=None,
        weights=None,
        redshifts=None,
        log_l_ir_values=None,
        fix_centers=True,
        fix_widths=False,
        detrend=True,
        verbose=True,
    ):
        """
        Fit PAH feature parameters to a reconstructed spectrum.

        Each PAH feature is a Gaussian:
            G_i(λ) = A_i × exp(-0.5 × ((λ - λ_i) / σ_i)²)

        Parameters
        ----------
        rest_wavelengths : array
            Rest-frame wavelengths (μm) from tomographic stacking.
        flux_values : array
            Measured flux ratios (e.g., L_24/L_IR).
        flux_errors : array, optional
        weights : array, optional
            Fitting weights (e.g., sqrt(n_sources)).
        redshifts : array, optional
        log_l_ir_values : array, optional
            log10(L_IR) per point. Used for detrending.
        fix_centers : bool
            Fix feature centers to lab wavelengths.
        fix_widths : bool
            Fix intrinsic widths to literature values.
        detrend : bool
            Remove smooth z/L_IR trend before fitting features.
        verbose : bool

        Returns
        -------
        result : dict
        """
        from scipy.optimize import least_squares

        lam = np.asarray(rest_wavelengths, dtype=float)
        flux = np.asarray(flux_values, dtype=float)

        valid = np.isfinite(lam) & np.isfinite(flux) & (flux > 0)
        if redshifts is not None:
            z_arr = np.asarray(redshifts, dtype=float)
            valid &= np.isfinite(z_arr)
        else:
            z_arr = None

        lam = lam[valid]
        flux = flux[valid]
        if z_arr is not None:
            z_arr = z_arr[valid]

        if weights is not None:
            w = np.asarray(weights, dtype=float)[valid]
        elif flux_errors is not None:
            err = np.asarray(flux_errors, dtype=float)[valid]
            w = 1.0 / np.maximum(err, 1e-10)
        else:
            w = np.ones_like(flux)

        # ── Detrending ───────────────────────────────────────────────
        smooth_trend = np.ones_like(flux)
        if detrend and z_arr is not None:
            X_trend = [z_arr, np.ones_like(z_arr)]
            if log_l_ir_values is not None:
                lir = np.asarray(log_l_ir_values, dtype=float)
                if len(lir) > np.sum(valid):
                    lir = lir[valid]
                if len(lir) == len(z_arr) and np.isfinite(lir).all():
                    X_trend.insert(1, lir)

            X_t = np.column_stack(X_trend)
            log_flux = np.log10(flux)
            try:
                c_trend, _, _, _ = np.linalg.lstsq(
                    X_t * w[:, None], log_flux * w, rcond=None
                )
                smooth_trend = 10 ** (X_t @ c_trend)
                if verbose:
                    labels = ["z"]
                    if len(c_trend) == 3:
                        labels = ["z", "logLIR"]
                    labels.append("const")
                    terms = " + ".join(f"{c:.3f}*{l}" for c, l in zip(c_trend, labels))
                    print(f"  Detrending: log(flux) = {terms}")
            except Exception:
                if verbose:
                    print("  Detrending failed, using raw flux")

        flux_dt = flux / smooth_trend
        if verbose and detrend:
            print(f"  Detrended flux range: {flux_dt.min():.3f} to {flux_dt.max():.3f}")

        n_feat = len(self.features)
        feature_names = []
        _all_names = [
            "6.2 C-C",
            "7.7 C-C",
            "8.6 C-H",
            "11.3 C-H",
            "12.7 C-H",
            "3.3 C-H",
            "5.25",
            "5.7",
            "16.4 C-H",
            "17.0 C-C",
        ]
        for i in range(n_feat):
            feature_names.append(_all_names[i] if i < len(_all_names) else f"F{i}")

        # ── Parameter vector ─────────────────────────────────────────
        p0, bounds_lo, bounds_hi = [], [], []
        med_dt = np.median(flux_dt)

        for lam_c, rel_strength, fwhm in self.features:
            p0.append(med_dt * 0.2 * max(rel_strength, 0.01))
            bounds_lo.append(0)
            bounds_hi.append(med_dt * 5)

        if not fix_widths:
            for lam_c, rel_strength, fwhm in self.features:
                sigma = fwhm / 2.355
                p0.append(sigma)
                bounds_lo.append(max(min_s, 0.05))
                bounds_hi.append(1.5)

        p0.append(med_dt * 0.8)
        bounds_lo.append(0)
        bounds_hi.append(med_dt * 5)
        p0 = np.array(p0)

        def _model(params):
            idx = 0
            amps = params[idx : idx + n_feat]
            idx += n_feat
            if fix_widths:
                sigmas_int = np.array([fw / 2.355 for _, _, fw in self.features])
            else:
                sigmas_int = params[idx : idx + n_feat]
                idx += n_feat
            centers = np.array([lc for lc, _, _ in self.features])
            C0 = params[idx]

            model = np.full_like(lam, C0)
            for A, sigma, center in zip(amps, sigmas_int, centers):
                model += A * np.exp(-0.5 * ((lam - center) / sigma) ** 2)
            return model

        def _residuals(params):
            return w * (flux_dt - _model(params))

        result = least_squares(
            _residuals, p0, bounds=(bounds_lo, bounds_hi), method="trf", max_nfev=10000
        )

        # ── Extract results ──────────────────────────────────────────
        idx = 0
        amplitudes = result.x[idx : idx + n_feat]
        idx += n_feat
        if fix_widths:
            widths_sigma = np.array([fw / 2.355 for _, _, fw in self.features])
        else:
            widths_sigma = result.x[idx : idx + n_feat]
            idx += n_feat
        centers = np.array([lc for lc, _, _ in self.features])
        C0 = result.x[idx]
        widths_fwhm = widths_sigma * 2.355

        model_dt = _model(result.x)
        model_spectrum = model_dt * smooth_trend
        residuals = flux - model_spectrum
        n_params = len(result.x)
        chi2_red = np.sum(_residuals(result.x) ** 2) / max(len(lam) - n_params, 1)

        amp_max = amplitudes.max() if amplitudes.max() > 0 else 1
        self.features = [
            (c, float(a / amp_max), fw)
            for c, a, fw in zip(centers, amplitudes, widths_fwhm)
        ]
        self._fitted_amplitudes = amplitudes
        self._fitted_continuum_level = C0

        fit_result = {
            "amplitudes": amplitudes,
            "widths_fwhm": widths_fwhm,
            "centers": centers,
            "continuum_level": C0,
            "model_spectrum": model_spectrum,
            "model_detrended": model_dt,
            "residuals": residuals,
            "chi2_red": chi2_red,
            "feature_names": feature_names[:n_feat],
            "wavelengths": lam,
            "data": flux,
            "detrended_flux": flux_dt,
            "smooth_trend": smooth_trend,
        }

        if verbose:
            print(
                f"\nPAH spectrum fit ({'fixed' if fix_centers else 'free'} centers, "
                f"{'fixed' if fix_widths else 'free'} widths, "
                f"{'detrended' if detrend else 'raw'}):"
            )
            print(f"  chi2_red = {chi2_red:.3f}  (N={len(lam)}, params={n_params})")
            print(f"  Baseline: {C0:.4f}")
            print(
                f"\n  {'Feature':<12} {'Center':>7} {'FWHM':>6} "
                f"{'Amp':>10} {'Rel':>5} {'Bump%':>6}"
            )
            print(f"  {'-'*52}")
            for i in range(n_feat):
                rel = amplitudes[i] / amp_max
                bump = amplitudes[i] / C0 * 100 if C0 > 0 else 0
                print(
                    f"  {feature_names[i]:<12} {centers[i]:7.2f}μm "
                    f"{widths_fwhm[i]:5.2f}  "
                    f"{amplitudes[i]:10.4f} {rel:5.2f} {bump:5.1f}%"
                )

        return fit_result

    def plot_fit(self, fit_result, ax=None, show_components=True):
        """Plot a single PAH fit. Delegates to plots.plot_pah_fit."""
        return plot_pah_fit(fit_result, self.features, ax=ax, show_components=show_components)

    def fit_forward_model(
        self,
        z_values,
        flux_values,
        flux_errors=None,
        weights=None,
        log_l_ir_values=None,
        fix_widths=True,
        feature_groups=None,
        verbose=True,
    ):
        """
        Recover intrinsic PAH amplitudes via forward-modeling through
        the real MIPS 24um bandpass.

        JOINTLY fits the smooth z-trend AND PAH amplitudes:
            flux(z) = 10^(a*z + b*logLIR + c) × [1 + Σ A_j × BP(z, feature_j)]

        This avoids the detrend-then-fit problem where sequential detrending
        absorbs PAH signal.

        Parameters
        ----------
        z_values : array
            Redshift of each measurement.
        flux_values : array
            Measured flux ratios (e.g., L_24/L_IR).
        flux_errors : array, optional
        weights : array, optional
            sqrt(n_sources) recommended.
        log_l_ir_values : array, optional
            If provided, trend includes logLIR term.
        fix_widths : bool
            Fix feature widths to literature values.
        feature_groups : list of lists, optional
            Group correlated features to reduce free parameters.
            Each group shares one free amplitude; features within a
            group maintain their relative ratios from self.features.

            Default (None): each feature is independent (5 free amps).

            Recommended groupings:
              [[0], [1,2], [4]]         — 3 params: 6.2, 7.7+8.6, 12.7
                                          (drops 11.3, links the blend)
              [[0], [1,2], [3], [4]]    — 4 params: keeps 11.3
              [[0,1,2,3,4]]             — 1 param: one overall PAH scale

            Feature indices: 0=6.2, 1=7.7, 2=8.6, 3=11.3, 4=12.7
        verbose : bool

        Returns
        -------
        result : dict
        """
        from scipy.optimize import least_squares

        # ── MIPS 24um bandpass (from IRSA calibration) ───────────
        bp_lam = np.array(
            [
                18.005,
                19.134,
                19.716,
                19.944,
                20.177,
                20.415,
                20.577,
                20.742,
                20.909,
                20.993,
                21.079,
                21.252,
                21.427,
                21.606,
                21.787,
                21.879,
                21.972,
                22.160,
                22.351,
                22.545,
                22.743,
                22.944,
                23.149,
                23.358,
                23.570,
                23.786,
                24.006,
                24.231,
                24.459,
                24.692,
                24.930,
                25.172,
                25.419,
                25.670,
                25.927,
                26.189,
                26.456,
                26.729,
                27.007,
                27.292,
                27.582,
                27.878,
                28.181,
                28.491,
                28.808,
                29.131,
                29.462,
                29.801,
                30.148,
                30.865,
                31.618,
                32.207,
            ]
        )
        bp_resp = np.array(
            [
                0.000237,
                0.000644,
                0.004662,
                0.012914,
                0.024468,
                0.076447,
                0.177656,
                0.377777,
                0.735924,
                0.813463,
                0.857335,
                0.907091,
                0.957009,
                0.984957,
                0.997749,
                1.000000,
                0.998522,
                0.970058,
                0.926258,
                0.880798,
                0.856114,
                0.856779,
                0.834520,
                0.795473,
                0.752764,
                0.777653,
                0.839877,
                0.911819,
                0.924876,
                0.897806,
                0.859096,
                0.803216,
                0.736609,
                0.649479,
                0.558191,
                0.486209,
                0.428693,
                0.381986,
                0.326682,
                0.261178,
                0.195445,
                0.148445,
                0.117945,
                0.097827,
                0.080512,
                0.060997,
                0.042525,
                0.029764,
                0.021807,
                0.011002,
                0.003471,
                0.000727,
            ]
        )
        bp_lam_f = np.linspace(18, 33, 500)
        bp_resp_f = np.interp(bp_lam_f, bp_lam, bp_resp, left=0, right=0)
        bp_norm = _trapz(bp_resp_f, bp_lam_f)

        z = np.asarray(z_values, dtype=float)
        flux = np.asarray(flux_values, dtype=float)
        valid = np.isfinite(z) & np.isfinite(flux) & (flux > 0)
        z = z[valid]
        flux = flux[valid]

        if flux_errors is not None:
            ferr = np.asarray(flux_errors, dtype=float)[valid]
            ferr = np.maximum(ferr, np.median(flux) * 1e-4)  # floor
            w = 1.0 / ferr
        elif weights is not None:
            w = np.asarray(weights, dtype=float)[valid]
        else:
            # Estimate errors from scatter: MAD of flux in narrow z-windows
            w = np.ones_like(flux)
            if verbose:
                print(
                    "  WARNING: no flux_errors or weights — fit may be poorly constrained"
                )

        has_lir = False
        lir = None
        if log_l_ir_values is not None:
            lir = np.asarray(log_l_ir_values, dtype=float)
            if len(lir) > np.sum(valid):
                lir = lir[valid]
            if len(lir) == len(z) and np.isfinite(lir).all():
                has_lir = True
            else:
                lir = None

        n_feat = len(self.features)
        feature_names = [
            "6.2 C-C",
            "7.7 C-C",
            "8.6 C-H",
            "11.3 C-H",
            "12.7 C-H",
            "3.3 C-H",
            "5.25",
            "5.7",
            "16.4",
            "17.0",
        ][:n_feat]

        # ── Feature groups ───────────────────────────────────────
        # Each group gets one free amplitude; features within a group
        # maintain their relative ratios from self.features.
        if feature_groups is None:
            # Default: each feature independent
            groups = [[i] for i in range(n_feat)]
        else:
            groups = feature_groups

        n_groups = len(groups)

        # Reference amplitudes for scaling within groups
        ref_amps = np.array([max(rs, 1e-6) for _, rs, _ in self.features])
        # For each group, the "lead" feature is the strongest one
        group_ref = []
        for g in groups:
            lead = g[np.argmax(ref_amps[g])]
            group_ref.append(ref_amps[lead])

        # Map: group amplitude → individual feature amplitudes
        # feature_j amplitude = group_scale × (ref_amp_j / ref_amp_lead)
        group_of_feat = np.full(n_feat, -1, dtype=int)
        feat_scale_in_group = np.zeros(n_feat)
        for gi, g in enumerate(groups):
            lead_amp = ref_amps[g[np.argmax(ref_amps[g])]]
            for j in g:
                group_of_feat[j] = gi
                feat_scale_in_group[j] = ref_amps[j] / lead_amp

        # Features not in any group get amplitude = 0
        active_feats = [j for j in range(n_feat) if group_of_feat[j] >= 0]

        if verbose and feature_groups is not None:
            dropped = [j for j in range(n_feat) if group_of_feat[j] < 0]
            print(f"  Feature groups ({n_groups} free amplitudes):")
            for gi, g in enumerate(groups):
                names = [feature_names[j] for j in g]
                scales = [f"{feat_scale_in_group[j]:.2f}" for j in g]
                print(f"    Group {gi}: {names} (relative: {scales})")
            if dropped:
                print(f"    Dropped: {[feature_names[j] for j in dropped]}")

        # ── Precompute bandpass-integrated PAH template per feature per z ─
        widths_sigma_default = np.array([fw / 2.355 for _, _, fw in self.features])

        def _compute_templates(widths_sigma):
            templates = np.zeros((n_feat, len(z)))
            for i, zi in enumerate(z):
                rest_lam = bp_lam_f / (1 + zi)
                for j, (lc, _, _) in enumerate(self.features):
                    sigma = widths_sigma[j]
                    feat_spec = np.exp(-0.5 * ((rest_lam - lc) / sigma) ** 2)
                    templates[j, i] = _trapz(feat_spec * bp_resp_f, bp_lam_f) / bp_norm
            return templates

        # ── Joint model ──────────────────────────────────────────
        # flux(z_i) = 10^(trend) × [1 + Σ_group A_g × Σ_j∈g scale_j × template_j(z_i)]

        # Trend design matrix: z, z², (logLIR,) const
        def _trend_X(z_arr, lir_arr=None):
            cols = [z_arr, z_arr**2, np.ones_like(z_arr)]
            if lir_arr is not None:
                cols.insert(2, lir_arr)  # z, z², logLIR, const
            return np.column_stack(cols)

        # Initial trend estimate
        X_init = _trend_X(z, lir if has_lir else None)
        try:
            c_init, _, _, _ = np.linalg.lstsq(X_init, np.log10(flux), rcond=None)
        except Exception:
            c_init = np.zeros(X_init.shape[1])
            c_init[-1] = np.log10(np.median(flux))

        # Build parameter vector: [group_amplitudes, (widths,) trend_coeffs]
        p0, bounds_lo, bounds_hi = [], [], []

        for gi, g in enumerate(groups):
            # Initial guess from reference amplitude of lead feature
            p0.append(0.1)
            bounds_lo.append(0)
            bounds_hi.append(5.0)

        if not fix_widths:
            for lc, rel_s, fwhm in self.features:
                p0.append(fwhm / 2.355)
                bounds_lo.append(0.05)
                bounds_hi.append(1.5)

        n_trend = len(c_init)
        for c in c_init:
            p0.append(float(c))
            bounds_lo.append(-10)
            bounds_hi.append(10)

        p0 = np.array(p0)

        def _unpack(params):
            idx = 0
            group_amps = params[idx : idx + n_groups]
            idx += n_groups
            if fix_widths:
                sigmas = widths_sigma_default.copy()
            else:
                sigmas = params[idx : idx + n_feat]
                idx += n_feat
            trend_coeffs = params[idx : idx + n_trend]
            return group_amps, sigmas, trend_coeffs

        def _expand_amps(group_amps):
            """Expand group amplitudes to per-feature amplitudes."""
            amps = np.zeros(n_feat)
            for gi, g in enumerate(groups):
                for j in g:
                    amps[j] = group_amps[gi] * feat_scale_in_group[j]
            return amps

        # Template cache
        _cache = [None, None]

        def _get_templates(sigmas):
            key = tuple(np.round(sigmas, 5))
            if _cache[0] != key:
                _cache[1] = _compute_templates(sigmas)
                _cache[0] = key
            return _cache[1]

        def _model(params):
            group_amps, sigmas, trend_coeffs = _unpack(params)
            amps = _expand_amps(group_amps)
            templates = _get_templates(sigmas)

            X = _trend_X(z, lir if has_lir else None)
            smooth = 10 ** (X @ trend_coeffs)

            pah_mod = np.ones_like(z)
            for j in range(n_feat):
                pah_mod += amps[j] * templates[j]

            return smooth * pah_mod

        def _residuals(params):
            return w * (flux - _model(params))

        result = least_squares(
            _residuals, p0, bounds=(bounds_lo, bounds_hi), method="trf", max_nfev=20000
        )

        # ── Extract results ──────────────────────────────────────
        group_amps, widths_sigma, trend_coeffs = _unpack(result.x)
        amplitudes = _expand_amps(group_amps)
        widths_fwhm = widths_sigma * 2.355
        centers = np.array([lc for lc, _, _ in self.features])

        X_trend = _trend_X(z, lir if has_lir else None)
        smooth_trend = 10 ** (X_trend @ trend_coeffs)

        templates = _get_templates(widths_sigma)
        pah_modulation = np.ones_like(z)
        for j in range(n_feat):
            pah_modulation += amplitudes[j] * templates[j]

        model_full = smooth_trend * pah_modulation
        model_dt = pah_modulation
        flux_dt = flux / smooth_trend

        n_params = len(result.x)
        chi2_red = np.sum(_residuals(result.x) ** 2) / max(len(z) - n_params, 1)

        amp_max = amplitudes.max() if amplitudes.max() > 0 else 1
        self.features = [
            (c, float(a / amp_max), fw)
            for c, a, fw in zip(centers, amplitudes, widths_fwhm)
        ]

        fit_result = {
            "amplitudes": amplitudes,
            "widths_fwhm": widths_fwhm,
            "centers": centers,
            "trend_coeffs": trend_coeffs,
            "group_amplitudes": group_amps,
            "feature_groups": groups,
            "model_flux_vs_z": model_full,
            "model_detrended": model_dt,
            "z_values": z,
            "data": flux,
            "detrended_flux": flux_dt,
            "smooth_trend": smooth_trend,
            "chi2_red": chi2_red,
            "feature_names": feature_names,
            "success": result.success,
            "wavelengths": 24.0 / (1 + z),
            "model_spectrum": model_full,
            "continuum_level": 1.0,
        }

        if verbose:
            grp_label = f", {n_groups} PAH groups" if feature_groups else ""
            print(f"\nPAH forward-model fit (joint trend + PAH{grp_label}):")
            print(f"  chi2_red = {chi2_red:.3f}  (N={len(z)}, params={n_params})")
            trend_labels = (
                ["z", "z²", "logLIR", "const"] if has_lir else ["z", "z²", "const"]
            )
            trend_str = " + ".join(
                f"{c:.3f}*{l}" for c, l in zip(trend_coeffs, trend_labels)
            )
            print(f"  Trend: log10(baseline) = {trend_str}")
            print(
                f"\n  {'Feature':<12} {'Center':>7} {'FWHM':>6} "
                f"{'Amplitude':>10} {'Rel':>5} {'Group':>6}"
            )
            print(f"  {'-'*52}")
            for i in range(n_feat):
                rel = amplitudes[i] / amp_max if amp_max > 0 else 0
                gi = group_of_feat[i]
                grp_str = f"G{gi}" if gi >= 0 else "  --"
                print(
                    f"  {feature_names[i]:<12} {centers[i]:7.2f}um "
                    f"{widths_fwhm[i]:5.2f}  {amplitudes[i]:10.4f} {rel:5.2f} {grp_str:>6}"
                )

        return fit_result

    def plot_forward_fit(self, fit_result, save_path=None):
        """Plot forward-model PAH fit results. Delegates to plots.plot_pah_forward_fit."""
        return plot_pah_forward_fit(fit_result, self.features, save_path=save_path)


    def fit_forward_per_bin(
        self,
        df,
        *,
        group_by="stellar_mass",
        flux_col="f24_to_fpeak",
        fix_widths=True,
        feature_groups=None,
        verbose=True,
    ):
        """
        Run fit_forward_model separately per bin of a grouping variable.

        Recovers intrinsic PAH feature amplitudes per (e.g.) stellar mass
        bin, showing how PAH emission evolves with physical properties.

        Parameters
        ----------
        df : DataFrame
            Output from combine_pah_spectra or measure_pah_excess.
            Must contain 'z', flux_col, 'n_sources', and group_by column.
        group_by : str
            Column to split on (e.g., 'stellar_mass').
        flux_col : str
            Column with flux values ('f24_to_fpeak' or 'l_24_to_l_ir').
        fix_widths : bool
        feature_groups : list of lists, optional
            Passed to fit_forward_model. Recommended: [[0], [1,2], [4]]
        verbose : bool

        Returns
        -------
        results : dict
            {bin_label: fit_result, '_evolution': DataFrame, '_group_by': str}
        """
        import pandas as pd

        if group_by not in df.columns:
            print(f"'{group_by}' not in df. Available: {list(df.columns)}")
            return None

        # ── Detect bin edges ─────────────────────────────────────
        # Strategy 1: Extract from population IDs (most reliable)
        bin_edges = None
        if "pop_id" in df.columns:
            import re as _re
            # Build alternate match keys: strip common log_ prefix so that
            # group_by="log_sigma_sfr" also matches pop_id parts named "sigma_sfr"
            _gb_key = group_by.lower().replace("_", "")
            _gb_key_nolog = _re.sub(r'^log', '', _gb_key)
            edge_set = set()
            for pop_id in df["pop_id"].dropna().unique():
                # Population IDs contain bin ranges like "metallicity_0.005_0.01"
                parts = str(pop_id).split("__")
                for part in parts:
                    part_key = part.lower().replace("_", "")
                    if _gb_key in part_key or (
                        _gb_key_nolog and _gb_key_nolog in part_key
                    ):
                        # Strip the dimension name token to isolate the numbers
                        # Use the actual dimension substring present in part_key
                        dim_token = group_by if group_by in part else _re.sub(
                            r'^log_?', '', group_by, flags=_re.IGNORECASE
                        )
                        nums = []
                        for token in (
                            part.replace(dim_token, "").replace("_", " ").split()
                        ):
                            try:
                                nums.append(float(token))
                            except ValueError:
                                pass
                        if len(nums) == 2:
                            edge_set.add((min(nums), max(nums)))
            if edge_set:
                bin_edges = sorted(edge_set)

        # Strategy 2: Cluster unique values (fallback)
        if not bin_edges:
            vals = df[group_by].dropna().values
            unique = np.sort(np.unique(np.round(vals, 6)))

            if len(unique) < 2:
                print(f"Only {len(unique)} unique {group_by} values")
                return None

            # Use relative spacing: gap > 20% of range between adjacent values
            spacings = np.diff(unique)
            if len(spacings) > 1:
                median_spacing = np.median(spacings)
                # Group values that are much closer than the typical spacing
                centers = [unique[0]]
                for j in range(1, len(unique)):
                    if unique[j] - centers[-1] > median_spacing * 0.5:
                        centers.append(unique[j])
            else:
                centers = list(unique)

            edges = [
                (
                    centers[0] - spacings[0] * 0.5
                    if len(spacings) > 0
                    else centers[0] - 0.01
                )
            ]
            for i in range(len(centers) - 1):
                edges.append((centers[i] + centers[i + 1]) / 2)
            edges.append(
                centers[-1] + (spacings[-1] * 0.5 if len(spacings) > 0 else 0.01)
            )
            bin_edges = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

        # Smart label formatting based on value scale
        def _fmt(val):
            if abs(val) == 0:
                return "0"
            mag = np.floor(np.log10(abs(val))) if val != 0 else 0
            if mag >= 1:
                return f"{val:.1f}"
            elif mag >= 0:
                return f"{val:.2f}"
            elif mag >= -1:
                return f"{val:.3f}"
            else:
                return f"{val:.4f}"

        if verbose:
            print(f"\nForward-model PAH fit per {group_by}: {len(bin_edges)} bins")

        results = {}
        evo_rows = []

        for lo, hi in bin_edges:
            mask = (df[group_by] >= lo) & (df[group_by] < hi)
            sub = df.loc[mask]
            n_pts = len(sub)
            label = f"{_fmt(lo)}-{_fmt(hi)}"

            if n_pts < 8:
                if verbose:
                    print(f"\n  {group_by}={label}: {n_pts} pts (skip, need ≥8)")
                continue

            if verbose:
                print(
                    f"\n  {group_by}={label}: {n_pts} pts, "
                    f"z = {sub['z'].min():.2f}-{sub['z'].max():.2f}"
                )

            # Fresh PAHModel per bin
            pah_bin = PAHModel(
                coeffs=self.coeffs,
                include_minor=len(self.features) > 5,
            )

            try:
                lir_vals = sub["log_l_ir"].values if "log_l_ir" in sub.columns else None

                # Auto-detect flux errors
                err_col = flux_col + "_err"
                flux_err = sub[err_col].values if err_col in sub.columns else None

                r = pah_bin.fit_forward_model(
                    sub["z"].values,
                    sub[flux_col].values,
                    flux_errors=flux_err,
                    log_l_ir_values=lir_vals,
                    fix_widths=fix_widths,
                    feature_groups=feature_groups,
                    verbose=verbose,
                )
                r["bin_lo"] = lo
                r["bin_hi"] = hi
                r["bin_label"] = label
                r["bin_center"] = (lo + hi) / 2
                r["n_points"] = n_pts
                results[label] = r

                # Evolution row
                row = {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "bin_center": (lo + hi) / 2,
                    "n_points": n_pts,
                    "chi2_red": r["chi2_red"],
                }
                for i, name in enumerate(r["feature_names"]):
                    row[f"A_{name}"] = r["amplitudes"][i]
                evo_rows.append(row)

            except Exception as e:
                if verbose:
                    print(f"    Failed: {e}")

        if not results:
            print("No successful fits")
            return None

        df_evo = pd.DataFrame(evo_rows)
        bin_labels = [k for k in results if not k.startswith("_")]
        results["_bin_labels"] = bin_labels
        results["_evolution"] = df_evo
        results["_group_by"] = group_by

        if verbose and len(df_evo) > 0:
            a_cols = [c for c in df_evo.columns if c.startswith("A_")]
            print(f"\n{'='*65}")
            print(f"Intrinsic PAH amplitudes vs {group_by}")
            print(f"{'='*65}")
            print(f"  {'bin':>12} {'N':>4} {'χ²r':>5}", end="")
            for c in a_cols:
                print(f" {c.replace('A_',''):>8}", end="")
            print()
            for _, row in df_evo.iterrows():
                print(
                    f"  {_fmt(row['bin_lo'])}-{_fmt(row['bin_hi'])} "
                    f" {row['n_points']:4.0f} {row['chi2_red']:5.2f}",
                    end="",
                )
                for c in a_cols:
                    print(f" {row[c]:8.4f}", end="")
                print()

        return results

    def plot_pah_vs_property(self, per_bin_results, save_path=None):
        """Plot recovered PAH spectra per bin. Delegates to plots.plot_pah_vs_property."""
        return plot_pah_vs_property(per_bin_results, self.features, save_path=save_path)


    def fit_forward_model_multibin(
        self,
        df,
        group_col="stellar_mass",
        flux_col="f24_to_fpeak",
        feature_groups=None,
        fix_widths=True,
        bin_edges=None,
        verbose=True,
    ):
        """
        Jointly fit all stellar-mass (or other property) bins simultaneously.

        The PAH feature template SHAPE is shared across bins; only the
        overall PAH amplitude α_m varies per bin. This breaks the baseline-PAH
        degeneracy that plagues independent per-bin fitting.

        Model per bin m:
            flux_m(z) = baseline_m(z) × (1 + α_m × Σ_g r_g × T_g(z))
        where:
            baseline_m(z) = 10^(a_m·z + b_m·z² + c_m)  — per-bin, independent
            α_m           — per-bin overall PAH amplitude, ≥ 0
            r_g           — shared relative feature-group ratio (r_0 ≡ 1)
            T_g(z)        — bandpass-integrated template for feature group g

        Parameters
        ----------
        df : DataFrame
            Output from combine_pah_spectra. Must have 'z', flux_col, group_col,
            and optionally flux_col + '_err'.
        group_col : str
            Column to split on (e.g., 'stellar_mass').
        flux_col : str
            Flux column ('f24_to_fpeak' or 'l_24_to_l_ir').
        feature_groups : list of lists, optional
            Same semantics as fit_forward_model.
            Recommended: [[0], [1, 2], [4]]  (6.2, 7.7+8.6, 12.7)
        fix_widths : bool
            Fix feature widths to literature values (recommended).
        verbose : bool

        Returns
        -------
        result : dict
            'bin_labels', 'bin_centers',
            'alpha_per_bin', 'alpha_err_per_bin',
            'group_ratios', 'ratio_errors',
            'baseline_coeffs', 'baseline_err',
            'model_per_bin', 'chi2_red', 'success',
            'n_data', 'n_params', 'feature_groups', 'feature_names'.
        """
        from scipy.optimize import least_squares

        # ── MIPS 24um bandpass ─────────────────────────────────────
        bp_lam = np.array(
            [
                18.005,
                19.134,
                19.716,
                19.944,
                20.177,
                20.415,
                20.577,
                20.742,
                20.909,
                20.993,
                21.079,
                21.252,
                21.427,
                21.606,
                21.787,
                21.879,
                21.972,
                22.160,
                22.351,
                22.545,
                22.743,
                22.944,
                23.149,
                23.358,
                23.570,
                23.786,
                24.006,
                24.231,
                24.459,
                24.692,
                24.930,
                25.172,
                25.419,
                25.670,
                25.927,
                26.189,
                26.456,
                26.729,
                27.007,
                27.292,
                27.582,
                27.878,
                28.181,
                28.491,
                28.808,
                29.131,
                29.462,
                29.801,
                30.148,
                30.865,
                31.618,
                32.207,
            ]
        )
        bp_resp = np.array(
            [
                0.000237,
                0.000644,
                0.004662,
                0.012914,
                0.024468,
                0.076447,
                0.177656,
                0.377777,
                0.735924,
                0.813463,
                0.857335,
                0.907091,
                0.957009,
                0.984957,
                0.997749,
                1.000000,
                0.998522,
                0.970058,
                0.926258,
                0.880798,
                0.856114,
                0.856779,
                0.834520,
                0.795473,
                0.752764,
                0.777653,
                0.839877,
                0.911819,
                0.924876,
                0.897806,
                0.859096,
                0.803216,
                0.736609,
                0.649479,
                0.558191,
                0.486209,
                0.428693,
                0.381986,
                0.326682,
                0.261178,
                0.195445,
                0.148445,
                0.117945,
                0.097827,
                0.080512,
                0.060997,
                0.042525,
                0.029764,
                0.021807,
                0.011002,
                0.003471,
                0.000727,
            ]
        )
        bp_lam_f = np.linspace(18, 33, 500)
        bp_resp_f = np.interp(bp_lam_f, bp_lam, bp_resp, left=0, right=0)
        bp_norm = _trapz(bp_resp_f, bp_lam_f)

        # ── Feature groups ─────────────────────────────────────────
        n_feat = len(self.features)
        feature_names = [
            "6.2 C-C",
            "7.7 C-C",
            "8.6 C-H",
            "11.3 C-H",
            "12.7 C-H",
            "3.3 C-H",
            "5.25",
            "5.7",
            "16.4",
            "17.0",
        ][:n_feat]

        if feature_groups is None:
            groups = [[i] for i in range(n_feat)]
        else:
            groups = feature_groups
        n_groups = len(groups)

        ref_amps = np.array([max(rs, 1e-6) for _, rs, _ in self.features])
        widths_sigma = np.array([fw / 2.355 for _, _, fw in self.features])

        # Within-group scaling relative to lead feature
        feat_scale = np.zeros(n_feat)
        for g in groups:
            lead_amp = ref_amps[g[np.argmax(ref_amps[g])]]
            for j in g:
                feat_scale[j] = ref_amps[j] / lead_amp

        # ── Precompute group templates over a dense z grid ─────────
        def _compute_group_templates(z_arr):
            T = np.zeros((n_groups, len(z_arr)))
            for i, zi in enumerate(z_arr):
                rest_lam = bp_lam_f / (1 + zi)
                for gi, g in enumerate(groups):
                    for j in g:
                        lc = self.features[j][0]
                        sigma = widths_sigma[j]
                        feat_spec = np.exp(-0.5 * ((rest_lam - lc) / sigma) ** 2)
                        T[gi, i] += (
                            feat_scale[j]
                            * _trapz(feat_spec * bp_resp_f, bp_lam_f)
                            / bp_norm
                        )
            return T

        # ── Split df into per-bin datasets ─────────────────────────
        if group_col not in df.columns:
            print(f"'{group_col}' not in df. Available: {list(df.columns)}")
            return None

        if bin_edges is None:
            # Strategy 1: extract bin edges from pop_id (most reliable)
            if "pop_id" in df.columns:
                edge_set = set()
                for pop_id in df["pop_id"].dropna().unique():
                    parts = str(pop_id).split("__")
                    for part in parts:
                        if group_col.lower().replace("_", "") in part.lower().replace(
                            "_", ""
                        ):
                            nums = []
                            for token in (
                                part.replace(group_col, "").replace("_", " ").split()
                            ):
                                try:
                                    nums.append(float(token))
                                except ValueError:
                                    pass
                            if len(nums) == 2:
                                edge_set.add((min(nums), max(nums)))
                if edge_set:
                    bin_edges = sorted(edge_set)

            # Strategy 2: cluster unique values (fallback)
            if not bin_edges:
                vals = df[group_col].dropna().values
                unique = np.sort(np.unique(np.round(vals, 6)))
                if len(unique) < 2:
                    print(
                        f"Only {len(unique)} unique {group_col} values — cannot detect bins"
                    )
                    return None
                spacings = np.diff(unique)
                med_sp = np.median(spacings)
                centers_v = [unique[0]]
                for j in range(1, len(unique)):
                    if unique[j] - centers_v[-1] > med_sp * 0.5:
                        centers_v.append(unique[j])
                sp = spacings if len(spacings) > 0 else [0.01]
                edges = [centers_v[0] - sp[0] * 0.5]
                for i in range(len(centers_v) - 1):
                    edges.append((centers_v[i] + centers_v[i + 1]) / 2)
                edges.append(centers_v[-1] + sp[-1] * 0.5)
                bin_edges = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

        err_col = flux_col + "_err"
        bins_data = []
        for lo, hi in bin_edges:
            mask = (df[group_col] >= lo) & (df[group_col] < hi)
            sub = df.loc[mask]
            valid = (
                np.isfinite(sub["z"]) & np.isfinite(sub[flux_col]) & (sub[flux_col] > 0)
            )
            sub = sub.loc[valid]
            if len(sub) < 8:
                continue
            z_m = sub["z"].values
            flux_m = sub[flux_col].values
            if err_col in sub.columns:
                ferr = np.maximum(sub[err_col].values, np.median(flux_m) * 1e-4)
                w_m = 1.0 / ferr
            else:
                w_m = np.ones_like(flux_m)
            label = f"{lo:.2f}-{hi:.2f}"
            bins_data.append((label, z_m, flux_m, w_m, (lo + hi) / 2))

        M = len(bins_data)
        if M < 2:
            print(f"Need ≥2 bins with ≥8 points each, got {M}")
            return None

        if verbose:
            print(f"\nMulti-bin PAH forward model: {M} bins, {n_groups} feature groups")
            for i, (lbl, z_m, flux_m, _, ctr) in enumerate(bins_data):
                print(
                    f"  Bin {i}: {lbl}, {len(z_m)} pts, "
                    f"z={z_m.min():.2f}-{z_m.max():.2f}, center={ctr:.2f}"
                )

        # Precompute templates at the union of all z-values
        z_union = np.sort(np.unique(np.concatenate([b[1] for b in bins_data])))
        if verbose:
            print(
                f"  Precomputing templates at {len(z_union)} z values... ",
                end="",
                flush=True,
            )
        T_grid = _compute_group_templates(z_union)
        if verbose:
            print("done")

        def _T_for(z_m):
            return np.array(
                [np.interp(z_m, z_union, T_grid[gi]) for gi in range(n_groups)]
            )

        # ── Parameter layout ───────────────────────────────────────
        # [α_0..α_{M-1}, r_1..r_{G-1}, a_0,b_0,c_0, a_1,b_1,c_1, ...]
        n_alpha = M
        n_ratios = n_groups - 1  # r_0 fixed to 1
        n_base = M * 3
        n_params = n_alpha + n_ratios + n_base

        def _unpack(p):
            alpha = p[:M]
            ratios = np.concatenate([[1.0], p[M : M + n_ratios]])
            baseline = p[M + n_ratios :].reshape(M, 3)
            return alpha, ratios, baseline

        def _pred(alpha_m, ratios, bcoeffs, z_m, T_m):
            bl = 10 ** (bcoeffs[0] * z_m + bcoeffs[1] * z_m**2 + bcoeffs[2])
            pah = np.ones(len(z_m))
            for gi in range(n_groups):
                pah += alpha_m * ratios[gi] * T_m[gi]
            return bl * pah

        def _residuals(p):
            alpha, ratios, baseline = _unpack(p)
            parts = []
            for m, (_, z_m, flux_m, w_m, _) in enumerate(bins_data):
                T_m = _T_for(z_m)
                parts.append(
                    w_m * (flux_m - _pred(alpha[m], ratios, baseline[m], z_m, T_m))
                )
            return np.concatenate(parts)

        # ── Initialization ─────────────────────────────────────────
        # Per-bin baseline from log-linear fit (α = 0)
        p0_base = []
        for _, z_m, flux_m, _, _ in bins_data:
            X = np.column_stack([z_m, z_m**2, np.ones_like(z_m)])
            try:
                c0, _, _, _ = np.linalg.lstsq(X, np.log10(flux_m), rcond=None)
            except Exception:
                c0 = np.array([0.0, 0.0, np.log10(max(np.median(flux_m), 1e-10))])
            p0_base.extend(c0.tolist())

        # Per-bin α from projection of residuals onto composite template
        p0_alpha = []
        for m, (_, z_m, flux_m, _, _) in enumerate(bins_data):
            bc = np.array(p0_base[m * 3 : (m + 1) * 3])
            bl_est = 10 ** (bc[0] * z_m + bc[1] * z_m**2 + bc[2])
            excess = flux_m / bl_est - 1.0
            T_m = _T_for(z_m)
            T_comp = np.sum(T_m, axis=0)  # sum all groups equally for init
            denom = float(np.dot(T_comp, T_comp))
            alpha_est = float(np.dot(excess, T_comp)) / denom if denom > 0 else 0.1
            p0_alpha.append(max(alpha_est, 0.01))

        p0_ratios = np.ones(n_ratios) * 0.5
        p0 = np.array(p0_alpha + list(p0_ratios) + p0_base)

        bounds_lo = np.concatenate(
            [
                np.zeros(M),
                np.zeros(n_ratios),
                np.full(n_base, -10.0),
            ]
        )
        bounds_hi = np.concatenate(
            [
                np.full(M, 5.0),
                np.full(n_ratios, 5.0),
                np.full(n_base, 10.0),
            ]
        )

        # ── Optimize ───────────────────────────────────────────────
        opt = least_squares(
            _residuals,
            p0,
            bounds=(bounds_lo, bounds_hi),
            method="trf",
            max_nfev=30000,
            x_scale="jac",
        )

        # ── Formal uncertainties from Jacobian ─────────────────────
        try:
            JtJ = opt.jac.T @ opt.jac
            pcov = np.linalg.inv(JtJ)
            perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        except np.linalg.LinAlgError:
            perr = np.full(n_params, np.nan)

        alpha_opt, ratios_opt, baseline_opt = _unpack(opt.x)
        alpha_err = perr[:M]
        ratio_err = perr[M : M + n_ratios]

        n_data = sum(len(b[1]) for b in bins_data)
        chi2_red = float(np.sum(opt.fun**2) / max(n_data - n_params, 1))

        # ── Build per-bin output ────────────────────────────────────
        model_per_bin = {}
        for m, (label, z_m, flux_m, w_m, ctr) in enumerate(bins_data):
            T_m = _T_for(z_m)
            bl_m = 10 ** (
                baseline_opt[m, 0] * z_m
                + baseline_opt[m, 1] * z_m**2
                + baseline_opt[m, 2]
            )
            mdl_m = _pred(alpha_opt[m], ratios_opt, baseline_opt[m], z_m, T_m)
            ferr_m = np.where(w_m > 0, 1.0 / w_m, np.median(flux_m) * 0.03)
            model_per_bin[label] = {
                "z": z_m,
                "flux": flux_m,
                "flux_err": ferr_m,
                "model": mdl_m,
                "baseline": bl_m,
                "detrended_data": flux_m / bl_m,
                "detrended_model": mdl_m / bl_m,
                "detrended_err": ferr_m / bl_m,
                "bin_center": ctr,
            }

        if verbose:
            print(f"\n  chi2_red = {chi2_red:.3f}  (N={n_data}, params={n_params})")
            print(f"  Shared group ratios (group 0 fixed to 1.0):")
            for gi in range(n_groups):
                names = [feature_names[j] for j in groups[gi]]
                ri = ratios_opt[gi]
                ei = ratio_err[gi - 1] if gi > 0 else 0.0
                print(f"    Group {gi} {names}: r = {ri:.3f} ± {ei:.3f}")
            print(f"  Per-bin PAH amplitudes:")
            for m, (label, _, _, _, ctr) in enumerate(bins_data):
                print(f"    {label}: α = {alpha_opt[m]:.4f} ± {alpha_err[m]:.4f}")

        return {
            "bin_labels": [b[0] for b in bins_data],
            "bin_centers": np.array([b[4] for b in bins_data]),
            "alpha_per_bin": alpha_opt,
            "alpha_err_per_bin": alpha_err,
            "group_ratios": ratios_opt,
            "ratio_errors": np.concatenate([[0.0], ratio_err]),
            "baseline_coeffs": baseline_opt,
            "baseline_err": perr[M + n_ratios :].reshape(M, 3),
            "model_per_bin": model_per_bin,
            "chi2_red": chi2_red,
            "success": opt.success,
            "n_data": n_data,
            "n_params": n_params,
            "feature_groups": groups,
            "feature_names": feature_names,
            "feat_scale": feat_scale,
            "_group_col": group_col,
            "_z_union": z_union,
            "_T_grid": T_grid,
        }

    def plot_multibin_forward_fit(self, result, save_path=None):
        """Plot multi-bin forward model results. Delegates to plots.plot_pah_multibin_forward_fit."""
        return plot_pah_multibin_forward_fit(result, self.features, save_path=save_path)

    def __repr__(self):
        a, b, c, d = self.coeffs
        return (
            f"PAHModel(log(L_PAH/L_IR) = {a:.3f}*logLIR {b:+.3f}*z "
            f"{c:+.3f}*PAH {d:+.3f}, T_warm={self.T_warm}K)"
        )


def greybody_plus_pah(
    greybody_model_func,
    wavelength_um,
    amplitude,
    temperature,
    beta=1.8,
    alpha=2.0,
    z=None,
    log_l_ir=None,
    pah_model=None,
    pah_amplitude=None,
):
    """
    Greybody + PAH combined SED model.

    Drop-in replacement for greybody_model that adds PAH emission
    on the Wien side instead of (or in addition to) the power-law.

    Parameters
    ----------
    greybody_model_func : callable
        The original greybody_model method (bound to GreybodyFitter).
    wavelength_um : array
        Rest-frame wavelengths.
    amplitude, temperature, beta, alpha : float
        Greybody parameters (same as greybody_model).
    z : float, optional
        Redshift (for PAH amplitude prediction).
    log_l_ir : float, optional
        log10(L_IR/L_sun).
    pah_model : PAHModel, optional
        PAH model instance. If None, no PAH is added.
    pah_amplitude : float, optional
        Direct PAH amplitude (overrides z/log_l_ir prediction).

    Returns
    -------
    flux_density : array
        Combined greybody + PAH flux density.
    """
    # Original greybody (with power-law Wien extension)
    flux_gb = greybody_model_func(wavelength_um, amplitude, temperature, beta, alpha)

    if pah_model is None:
        return flux_gb

    # PAH component
    flux_pah = pah_model.flux(
        wavelength_um,
        amplitude,
        z=z,
        log_l_ir=log_l_ir,
        pah_amplitude=pah_amplitude,
    )

    return flux_gb + flux_pah
