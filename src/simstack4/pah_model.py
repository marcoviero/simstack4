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

# numpy 2.x compatibility
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# PAH feature parameters (rest-frame)
# (center_um, relative_strength, fwhm_um)
PAH_FEATURES = [
    (6.2,  0.40, 0.19),   # C-C stretch
    (7.7,  1.00, 0.70),   # C-C stretch (strongest)
    (8.6,  0.30, 0.34),   # C-H in-plane bend
    (11.3, 0.60, 0.24),   # C-H out-of-plane bend
    (12.7, 0.30, 0.45),   # C-H out-of-plane bend
]

# Minor features (optional, for high-resolution work)
PAH_FEATURES_MINOR = [
    (3.3,  0.08, 0.05),   # C-H stretch
    (5.25, 0.05, 0.10),   # combination band
    (5.7,  0.05, 0.10),   # combination band
    (16.4, 0.10, 0.20),   # C-H/C-C bend
    (17.0, 0.08, 0.30),   # C-C-C bend
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
        self.h = 6.626e-27      # erg s
        self.k_B = 1.381e-16    # erg/K
        self.c = 2.998e10       # cm/s

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
            frac = 0.5 * (erf((band_hi - lam_obs) / (sigma * np.sqrt(2)))
                          - erf((band_lo - lam_obs) / (sigma * np.sqrt(2))))
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
        A_gb = 10 ** greybody_amplitude_log10
        ratio = 10 ** log_ratio

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

        Uses the output of reconstruct_pah_spectrum: rest-frame wavelengths
        and L_24/L_IR (or f_24/f_peak) values that trace the PAH complex.

        Each PAH feature is a Gaussian:
            G_i(λ) = A_i × exp(-0.5 × ((λ - λ_i) / σ_i)²)

        Plus a smooth power-law continuum:
            C(λ) = C_0 × (λ / 10μm)^γ

        Parameters
        ----------
        rest_wavelengths : array
            Rest-frame wavelengths (μm) from tomographic stacking.
        flux_values : array
            Measured flux ratios (e.g., L_24/L_IR) at each wavelength.
        flux_errors : array, optional
            Uncertainties on flux_values. If None, uses uniform weights.
        weights : array, optional
            Fitting weights (e.g., sqrt(n_sources)). Overrides flux_errors.
        redshifts : array, optional
            Redshift of each point. Required if detrend=True.
        log_l_ir_values : array, optional
            log10(L_IR) of each point. Used for detrending if provided.
        fix_centers : bool
            Fix feature centers to lab wavelengths (recommended).
        fix_widths : bool
            Fix widths to literature values. If False, widths are free.
        detrend : bool
            If True, remove the smooth z/L_IR trend before fitting
            PAH features. This separates the PAH bumps from the
            population-evolution signal. Requires redshifts.
        verbose : bool

        Returns
        -------
        result : dict with keys:
            'amplitudes': fitted amplitude per feature
            'widths': fitted FWHM per feature (μm)
            'centers': feature centers (μm)
            'continuum': (C_0, gamma) power-law continuum params
            'model_spectrum': fitted model evaluated at input wavelengths
            'residuals': data - model
            'chi2_red': reduced chi-squared
            'feature_names': list of feature names
            'smooth_trend': the removed trend (if detrend=True)
            'detrended_flux': flux after trend removal
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
            # Fit smooth trend: log(flux) = a*z + b*log(LIR) + c
            # or just log(flux) = a*z + c if no LIR
            X_trend = [z_arr, np.ones_like(z_arr)]
            if log_l_ir_values is not None:
                lir = np.asarray(log_l_ir_values, dtype=float)[valid[:len(log_l_ir_values)] if len(valid) > len(log_l_ir_values) else valid]
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
                    print(f"  Trend range: {smooth_trend.min():.4f} to {smooth_trend.max():.4f}")
            except Exception:
                if verbose:
                    print("  Detrending failed, using raw flux")
                smooth_trend = np.ones_like(flux)

        # Detrended flux: normalized so mean ~ 1
        flux_dt = flux / smooth_trend
        if verbose and detrend:
            print(f"  Detrended flux range: {flux_dt.min():.3f} to {flux_dt.max():.3f}")
            print(f"  Mean: {flux_dt.mean():.3f}")

        n_feat = len(self.features)
        feature_names = []
        _all_names = ["6.2 C-C", "7.7 C-C", "8.6 C-H", "11.3 C-H", "12.7 C-H",
                       "3.3 C-H", "5.25", "5.7", "16.4 C-H", "17.0 C-C"]
        for i, (lam_c, _, _) in enumerate(self.features):
            feature_names.append(_all_names[i] if i < len(_all_names) else f"F{i}")

        # ── Build parameter vector ───────────────────────────────────
        p0 = []
        bounds_lo = []
        bounds_hi = []

        # Feature amplitudes (linear scale on detrended flux)
        med_dt = np.median(flux_dt)
        for lam_c, rel_strength, fwhm in self.features:
            # Initial guess: bump height ~ 20% of median × relative strength
            p0.append(med_dt * 0.2 * rel_strength)
            bounds_lo.append(0)         # amplitudes must be positive
            bounds_hi.append(med_dt * 5)

        # Feature widths (if free)
        if not fix_widths:
            for lam_c, rel_strength, fwhm in self.features:
                sigma = fwhm / 2.355
                p0.append(sigma)
                bounds_lo.append(0.05)
                bounds_hi.append(1.5)

        # Continuum: C_0 (flat baseline in detrended space)
        p0.append(med_dt * 0.8)
        bounds_lo.append(0)
        bounds_hi.append(med_dt * 5)

        p0 = np.array(p0)

        def _model(params):
            idx = 0
            amps = params[idx:idx + n_feat]
            idx += n_feat

            if fix_widths:
                sigmas = np.array([fw / 2.355 for _, _, fw in self.features])
            else:
                sigmas = params[idx:idx + n_feat]
                idx += n_feat

            if fix_centers:
                centers = np.array([lc for lc, _, _ in self.features])
            else:
                centers = params[idx:idx + n_feat]
                idx += n_feat

            C0 = params[idx]

            model = np.full_like(lam, C0)
            for A, sigma, center in zip(amps, sigmas, centers):
                model += A * np.exp(-0.5 * ((lam - center) / sigma) ** 2)

            return model

        def _residuals(params):
            model = _model(params)
            return w * (flux_dt - model)

        result = least_squares(
            _residuals, p0,
            bounds=(bounds_lo, bounds_hi),
            method="trf",
            max_nfev=10000,
        )

        # ── Extract results ──────────────────────────────────────────
        idx = 0
        amplitudes = result.x[idx:idx + n_feat]
        idx += n_feat

        if fix_widths:
            widths_sigma = np.array([fw / 2.355 for _, _, fw in self.features])
        else:
            widths_sigma = result.x[idx:idx + n_feat]
            idx += n_feat

        if fix_centers:
            centers = np.array([lc for lc, _, _ in self.features])
        else:
            centers = result.x[idx:idx + n_feat]
            idx += n_feat

        C0 = result.x[idx]
        widths_fwhm = widths_sigma * 2.355

        model_dt = _model(result.x)
        model_spectrum = model_dt * smooth_trend  # back to original units
        residuals = flux - model_spectrum
        n_params = len(result.x)
        chi2_red = np.sum(_residuals(result.x)**2) / max(len(lam) - n_params, 1)

        # Update self.features with fitted values
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
            "success": result.success,
        }

        if verbose:
            print(f"\nPAH spectrum fit ({'fixed' if fix_centers else 'free'} centers, "
                  f"{'fixed' if fix_widths else 'free'} widths, "
                  f"{'detrended' if detrend else 'raw'}):")
            print(f"  chi2_red = {chi2_red:.3f}  "
                  f"(N_data={len(lam)}, N_params={n_params})")
            print(f"  Baseline level: {C0:.4f}")
            print(f"\n  {'Feature':<12} {'Center':>7} {'FWHM':>6} "
                  f"{'Amplitude':>10} {'Rel':>5} {'Bump %':>7}")
            print(f"  {'-'*52}")
            for i in range(n_feat):
                name = feature_names[i]
                rel = amplitudes[i] / amp_max
                bump_pct = amplitudes[i] / C0 * 100 if C0 > 0 else 0
                print(f"  {name:<12} {centers[i]:7.2f}um "
                      f"{widths_fwhm[i]:5.2f}um "
                      f"{amplitudes[i]:10.4f} {rel:5.2f} {bump_pct:6.1f}%")

        return fit_result

    def plot_fit(self, fit_result, ax=None, show_components=True):
        """
        Plot the fitted PAH spectrum with components.

        Shows two panels: raw data + model (left), detrended + fit (right).
        """
        import matplotlib.pyplot as plt

        has_detrend = "detrended_flux" in fit_result and fit_result["detrended_flux"] is not None

        if ax is not None:
            fig = ax.figure
            axes = [ax]
        else:
            n_panels = 2 if has_detrend else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
            if n_panels == 1:
                axes = [axes]

        lam = fit_result["wavelengths"]
        data = fit_result["data"]
        model = fit_result["model_spectrum"]

        # Panel 1: Raw data + full model
        ax = axes[0]
        ax.scatter(lam, data, c="k", s=15, zorder=5, alpha=0.5, label="Data")
        ax.plot(lam, model, "r-", lw=2, label="PAH + trend", zorder=4)
        if has_detrend:
            ax.plot(lam, fit_result["smooth_trend"], "b--", lw=1.5, alpha=0.6,
                    label="Smooth z-trend")
        ax.set_xlabel("Rest-frame wavelength ($\\mu$m)")
        ax.set_ylabel("L$_{24}$/L$_{IR}$")
        ax.set_title("Raw data + model")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Panel 2: Detrended data + PAH features
        if has_detrend and len(axes) > 1:
            ax = axes[1]
            flux_dt = fit_result["detrended_flux"]
            model_dt = fit_result["model_detrended"]

            ax.scatter(lam, flux_dt, c="k", s=15, zorder=5, alpha=0.5,
                       label="Detrended data")
            ax.plot(lam, model_dt, "r-", lw=2, label="PAH model", zorder=4)

            # Baseline
            C0 = fit_result["continuum_level"]
            ax.axhline(C0, color="gray", ls="--", lw=1, alpha=0.5,
                       label=f"Baseline = {C0:.3f}")

            if show_components:
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                for i, (A, fwhm, center) in enumerate(
                    zip(fit_result["amplitudes"], fit_result["widths_fwhm"],
                        fit_result["centers"])):
                    sigma = fwhm / 2.355
                    feat = C0 + A * np.exp(-0.5 * ((lam - center) / sigma) ** 2)
                    name = fit_result["feature_names"][i] if i < len(fit_result["feature_names"]) else ""
                    bump_pct = A / C0 * 100 if C0 > 0 else 0
                    ax.plot(lam, feat, "-", color=colors[i % len(colors)],
                            lw=1.5, alpha=0.7,
                            label=f"{center:.1f}$\\mu$m ({bump_pct:.0f}%)")

            # Mark feature centers
            for lam_c, _, _ in self.features:
                ax.axvline(lam_c, color="gray", ls=":", alpha=0.2)

            ax.set_xlabel("Rest-frame wavelength ($\\mu$m)")
            ax.set_ylabel("Flux / smooth trend")
            ax.set_title(f"PAH features (detrended, "
                         f"$\\chi^2_{{red}}$ = {fit_result['chi2_red']:.2f})")
            ax.legend(fontsize=7, ncol=2, loc="upper right")
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        return fig

    def __repr__(self):
        a, b, c, d = self.coeffs
        return (f"PAHModel(log(L_PAH/L_IR) = {a:.3f}*logLIR {b:+.3f}*z "
                f"{c:+.3f}*PAH {d:+.3f}, T_warm={self.T_warm}K)")


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
