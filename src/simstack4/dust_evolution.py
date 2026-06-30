"""
Two-component dust SED model with hierarchical evolution fitter.

Science motivation
------------------
Viero+22 found T_dust evolving steeply with redshift to z~10, accelerating
at z>4.  ALMA studies mostly find cooler temperatures at high-z.  There are
two candidate explanations that the single-component greybody cannot separate:

  (a) The warm dust fraction f_w rises with redshift, biasing a single-
      component fit toward hotter apparent temperatures.
  (b) T_cold itself rises, and ALMA is biased because it doesn't straddle
      the SED peak.

This module disentangles these by fitting a *two-component* SED jointly
across all (z, M*, σ_SFR) bins, with the warm fraction and warm temperature
parameterized as smooth global functions:

    F_ν(λ_rest) = A_c · GB(λ, T_c, β) + f_w · A_c · GB(λ, T_w, β=1.5)

    f_w(z, log_M*)  = 10^(a0 + a_z·z + a_M·log_M*)   [warm fraction of A_c]
    T_w(log_σ_SFR)  = T_w0 + c_σ · log_σ_SFR          [warm temp vs ISRF proxy]

Per-bin amplitudes A_c[m] are solved analytically at each MCMC step
(same trick as Greybody._solve_amplitude_at_T), reducing free parameters
from ~6+4M to ~6 globals.  β is fixed at 1.8 for the cold component and
1.5 for warm (small-grain emissivity).

The SED peak wavelength is NOT assumed constant.  The simulator correctly
propagates T_c(z) → λ_peak(z) so that band dropout at high z is physically
motivated.  At z>4 the warm/cold decomposition is individually
under-constrained per bin, but the global f_w coefficients are still
recoverable because the z-trajectory of the SED shape is jointly informative
across all bins.
"""

import functools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Physical constants
_h = 6.62607015e-34   # J·s
_c = 2.99792458e8     # m/s
_k = 1.380649e-23     # J/K

# Band definitions: name → observed wavelength (µm)
# Matches the cosmos25_mass_sigma_sfr.toml configuration.
COSMOS_BANDS = {
    "MIPS_24":    24.0,
    "PACS_100":  100.0,
    "PACS_160":  160.0,
    "SPIRE_250": 250.0,
    "SPIRE_350": 350.0,
    "SPIRE_500": 500.0,
    "SCUBA2_850": 850.0,
}

# Typical stacking noise floors (mJy).  Rough values for realistic simulations;
# the actual noise varies by bin population size and depends on source confusion.
# Override via the noise_model argument to simulate_stacked_dataframe.
_DEFAULT_NOISE_MJY = {
    "MIPS_24":     0.002,
    "PACS_100":    0.10,
    "PACS_160":    0.15,
    "SPIRE_250":   0.20,
    "SPIRE_350":   0.20,
    "SPIRE_500":   0.25,
    "SCUBA2_850":  0.30,
}

# Bands that become unreliable at high redshift (mid-IR, not thermal dust at z>2)
_MIPS_ZMAX = 2.5   # MIPS 24 dropped above this z (rest < 9 µm — PAH, not continuum)


# ---------------------------------------------------------------------------
# SED primitives
# ---------------------------------------------------------------------------

def _planck_nu(nu_hz: np.ndarray, T_K: float) -> np.ndarray:
    """B_ν(T) in arbitrary linear units (normalised so peak ≈ 1 at T=30K)."""
    x = _h * nu_hz / (_k * T_K)
    x = np.clip(x, 1e-30, 700.0)
    return nu_hz**3 / (np.expm1(x))


# Fixed broad-grid for SED peak normalization (10 µm – 10 mm).
# Used so that _greybody_nu returns the correct value at ANY single wavelength.
# Normalization must not depend on the evaluation points.
_NORM_LAM_UM = np.logspace(1.0, 4.0, 400)
_NORM_NU_HZ  = _c * 1e6 / _NORM_LAM_UM


@functools.lru_cache(maxsize=8192)
def _norm_peak(T_K: float, beta: float) -> float:
    """Peak of ν^β·B_ν(T) over the fixed broad grid, cached by (T_K, beta)."""
    return float((_NORM_NU_HZ**beta * _planck_nu(_NORM_NU_HZ, T_K)).max())


def _greybody_nu(nu_hz: np.ndarray, T_K: float, beta: float) -> np.ndarray:
    """S_ν ∝ ν^β · B_ν(T), peak-normalised pure modified blackbody.

    Used throughout the MCMC likelihood and amplitude solver.  No Wien-side
    power law — that extension lives in Greybody.greybody_model in greybody.py.
    """
    peak = _norm_peak(T_K, beta)
    raw  = nu_hz**beta * _planck_nu(nu_hz, T_K)
    return raw / peak if peak > 0 else raw


def _greybody_sed(
    lambda_rest_um: np.ndarray,
    A_c: float,
    T_c: float,
    A_w: float,
    T_w: float,
    beta_c: float = 1.8,
    beta_w: float = 1.5,
) -> np.ndarray:
    """Two-component pure modified blackbody SED in mJy (arbitrary normalisation)."""
    nu = _c * 1e6 / lambda_rest_um   # Hz
    cold = A_c * _greybody_nu(nu, T_c, beta_c)
    warm = A_w * _greybody_nu(nu, T_w, beta_w)
    return cold + warm


def _warm_sed_nu(
    nu_hz: np.ndarray,
    T_K: float,
    beta: float,
    z: float,
    log_l_ir: float,
    pah_model=None,
) -> np.ndarray:
    """Warm SED = GB(T_K, beta) + PAH features; peak-normalised over the GB alone.

    PAH amplitude is treated as a peak ratio relative to the warm GB peak — not a
    rigorous integral luminosity ratio — so pah_model.predict_amplitude is an
    approximation for how much short-wavelength flux PAH contributes.
    """
    gb = _greybody_nu(nu_hz, T_K, beta)
    if pah_model is None or not np.isfinite(log_l_ir):
        return gb

    lam_um = _c * 1e6 / nu_hz  # rest-frame wavelengths in µm
    pah_spec = pah_model.feature_spectrum(lam_um)

    log_ratio = pah_model.predict_amplitude(z, log_l_ir)
    ratio = 10.0 ** float(np.clip(log_ratio, -4.0, 0.0))

    pah_peak = float(pah_spec.max()) if pah_spec.max() > 0 else 1.0
    pah_scaled = pah_spec * ratio / pah_peak
    return gb + pah_scaled


def _peak_wavelength_um(T_K: float, beta: float = 1.8) -> float:
    """Rest-frame SED peak of the pure modified blackbody via grid search."""
    lam = np.logspace(1, 4, 4000)   # 10 µm … 10 mm
    nu = _c * 1e6 / lam
    sed = _greybody_nu(nu, T_K, beta)
    return lam[np.argmax(sed)]


# ---------------------------------------------------------------------------
# Warm-fraction and warm-temperature parametrizations
# ---------------------------------------------------------------------------

def warm_fraction(
    z: float,
    log_M_star: float,
    log_sigma_sfr: float,
    theta_f: np.ndarray,
) -> float:
    """
    Fraction of cold-component amplitude carried by the warm component.

        f_w = 10^(a0 + a_z·z + a_M·log_M* + a_sigma·log_σ_SFR)

    theta_f = [a0, a_z, a_M, a_sigma]

    Returns a value in [0, ∞).  Values >1 are physically allowed (warm can
    dominate at extreme σ_SFR) but the prior penalises them.
    """
    a0, a_z, a_M, a_sigma = theta_f
    log_fw = a0 + a_z * z + a_M * log_M_star + a_sigma * log_sigma_sfr
    return 10.0 ** np.clip(log_fw, -6.0, 3.0)


def warm_temperature(log_sigma_sfr: float, T_w0: float, c_sigma: float) -> float:
    """
    Warm-dust temperature as a function of SFR surface density.

        T_w = T_w0 + c_sigma · log_σ_SFR

    Physical basis: stochastically heated small grains respond to the local ISRF
    intensity traced by σ_SFR.  Redshift dependence is captured by T_c(z) and
    the warm fraction f_w(z, M*, σ_SFR); T_w itself stays z-independent.
    """
    return T_w0 + c_sigma * log_sigma_sfr


def cold_temperature(z: float, T_c0: float, b_z: float = 0.0) -> float:
    """
    Cold-dust temperature as a function of redshift.

        T_c = T_c0 + b_z · z

    Cold dust heats with redshift due to the rising CMB floor, stronger ISRF,
    and more compact star-forming regions at high z.
    """
    return T_c0 + b_z * z


def main_sequence_ssfr(redshift, log_mstar, relation: str = "speagle2014"):
    """Star-forming main-sequence specific SFR, log10(sSFR / yr^-1).

    Maps a galaxy's ``(z, M*)`` onto an expected specific star-formation rate,
    the ISRF proxy that sets the PAH amplitude and ionized/neutral band ratios.
    Within a fixed stellar-mass bin sSFR climbs steeply with z, so this is the
    physical "evolution axis" used by
    :class:`~simstack4.pah_dither.TruthSpectrum` injection and as the per-point
    fallback in :meth:`PAHSpectrumModel.fit_evolving`.  Companion to
    :func:`warm_temperature` / :func:`cold_temperature`: those give dust
    properties vs an ISRF proxy, this supplies the proxy from ``(z, M*)``.

    Args:
        redshift: Redshift(s).
        log_mstar: log10(M* / M_sun), broadcast against ``redshift``.
        relation: ``"speagle2014"`` (Speagle+2014, ApJ 214, 15; default) or
            ``"schreiber2015"`` (Schreiber+2015, A&A 575, A74).

    Returns:
        log10(sSFR / yr^-1) = log10(SFR / M_sun yr^-1) - log10(M* / M_sun).
    """
    z = np.asarray(redshift, dtype=float)
    logM = np.asarray(log_mstar, dtype=float)

    if relation == "speagle2014":
        # log SFR = (0.84 - 0.026 t) logM - (6.51 - 0.11 t), t = age(z) in Gyr.
        from astropy import units as u
        from astropy.cosmology import Planck18

        t_gyr = Planck18.age(z).to(u.Gyr).value
        log_sfr = (0.84 - 0.026 * t_gyr) * logM - (6.51 - 0.11 * t_gyr)
    elif relation == "schreiber2015":
        # m = log10(M*/1e9), r = log10(1+z); bent main sequence (their eq. 9).
        m = logM - 9.0
        r = np.log10(1.0 + z)
        m0, a0, a1, m1, a2 = 0.5, 1.5, 0.3, 0.36, 2.5
        bend = np.maximum(0.0, m - m1 - a2 * r)
        log_sfr = m - m0 + a0 * r - a1 * bend**2
    else:
        raise ValueError(
            f"unknown main-sequence relation {relation!r}; "
            "use 'speagle2014' or 'schreiber2015'"
        )

    return log_sfr - logM


# ---------------------------------------------------------------------------
# Dataclasses for results
# ---------------------------------------------------------------------------

@dataclass
class _BinObs:
    """Pre-baked per-bin observations for the MCMC hot path (no pandas)."""
    z: float
    log_M_star: float
    log_sigma_sfr: float
    nu_rest: np.ndarray   # Hz, valid bands only
    f_obs: np.ndarray     # observed fluxes (mJy)
    inv_var: np.ndarray   # 1/σ²
    log_l_ir: float = np.nan  # log10(L_IR/L_sun) for PAH amplitude scaling


@dataclass
class DustEvolutionResult:
    """Output from fit_dust_evolution."""
    theta_global: np.ndarray       # fitted free params (5–9 depending on fixed params)
    theta_err: np.ndarray          # posterior std for each free param
    A_c_per_bin: np.ndarray        # (M,) cold amplitudes per property bin
    A_w_per_bin: np.ndarray        # (M,) warm amplitudes per property bin (derived)
    f_w_per_bin: np.ndarray        # (M,) warm fractions
    T_c_per_bin: np.ndarray        # (M,) T_c0 + b_z·z per bin
    T_w_grid: np.ndarray           # (M,) T_w per bin (T_w0 + c_σ·log_σ_SFR)
    T_c_fit: float                 # T_c0 anchor (K) — T_c at z=0
    sampler: object | None = None  # emcee sampler for post-processing
    acceptance_fraction: float = 0.0
    autocorr_time: float = np.nan
    n_bins: int = 0
    param_names: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DustEvolutionModel:
    """
    Two-component dust SED fitter with hierarchical evolution.

    Typical use
    -----------
    dem = DustEvolutionModel()

    # 1. Build a synthetic dataset (or load real stacking output)
    sim = dem.simulate_stacked_dataframe(bin_grid, theta_true=theta_true)

    # 2. Fit
    result = dem.fit_dust_evolution(sim["df"], bin_col="bin_id",
                                    sigma_sfr_per_bin=sim["sigma_sfr_per_bin"])

    # 3. Inspect
    print(result.theta_global)   # [T_c, T_w0, c_sigma, a0, a_z, a_M]
    """

    def __init__(
        self,
        beta_c: float = 1.8,
        beta_w: float = 1.5,
        T_c_min: float = 20.0,
        T_c_max: float = 60.0,
        T_w_min: float = 40.0,
        T_w_max: float = 100.0,
        T_c_prior: tuple[float, float] = (30.0, 5.0),     # (mean, sigma) — Schreiber+18 MS anchor
        T_w0_prior: tuple[float, float] = (55.0, 15.0),   # (mean, sigma)
        c_sigma_prior: tuple[float, float] = (5.0, 3.0),
        bands: dict | None = None,
        noise_model: dict | None = None,
        use_pah_warm: bool = True,
        log_l_ir_default: float = np.nan,
    ):
        self.beta_c = beta_c
        self.beta_w = beta_w
        self.T_c_min = T_c_min
        self.T_c_max = T_c_max
        self.T_w_min = T_w_min
        self.T_w_max = T_w_max
        self.T_c_prior = T_c_prior
        self.T_w0_prior = T_w0_prior
        self.c_sigma_prior = c_sigma_prior
        self.bands = bands or COSMOS_BANDS
        self.noise_model = noise_model or _DEFAULT_NOISE_MJY
        self.use_pah_warm = use_pah_warm
        self.log_l_ir_default = log_l_ir_default
        if use_pah_warm:
            from .pah_model import PAHModel
            self.pah_model = PAHModel()
        else:
            self.pah_model = None

    # ------------------------------------------------------------------ #
    # Forward model                                                        #
    # ------------------------------------------------------------------ #

    def sed_at_z(
        self,
        z: float,
        log_M_star: float,
        log_sigma_sfr: float,
        A_c: float,
        theta_global: np.ndarray,
        bands: list[str] | None = None,
        log_l_ir: float = np.nan,
    ) -> dict[str, float]:
        """
        Compute observed flux densities (mJy) for a single population at redshift z.

        theta_global = [T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma]
        log_l_ir: log10(L_IR/L_sun) for PAH amplitude scaling (nan → no PAH)
        """
        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global
        Tc = cold_temperature(z, T_c0, b_z)
        fw = warm_fraction(z, log_M_star, log_sigma_sfr, np.array([a0, a_z, a_M, a_sigma]))
        Tw = warm_temperature(log_sigma_sfr, T_w0, c_sigma)
        Tw = np.clip(Tw, self.T_w_min, self.T_w_max)

        bands_to_use = bands or list(self.bands.keys())
        fluxes = {}
        for band in bands_to_use:
            lam_obs = self.bands[band]
            lam_rest = lam_obs / (1.0 + z)
            nu_rest = np.array([_c * 1e6 / lam_rest])
            cold_flux = A_c * _greybody_nu(nu_rest, Tc, self.beta_c)
            warm_flux = fw * A_c * _warm_sed_nu(
                nu_rest, Tw, self.beta_w, z, log_l_ir,
                self.pah_model if self.use_pah_warm else None,
            )
            F = float((cold_flux + warm_flux)[0])
            fluxes[band] = F
        return fluxes

    # ------------------------------------------------------------------ #
    # Simulator                                                            #
    # ------------------------------------------------------------------ #

    def simulate_stacked_dataframe(
        self,
        bin_grid: list[dict],
        theta_true: np.ndarray,
        A_c_true: np.ndarray | None = None,
        noise_scale: float = 1.0,
        n_obs_per_bin: int = 1,
        seed: int = 42,
    ) -> dict:
        """
        Simulate stacked flux densities for a grid of (z, M*, σ_SFR) bins.

        Parameters
        ----------
        bin_grid : list of dicts, each with keys:
            'bin_id'        — integer label
            'z'             — median redshift of the property bin
            'log_M_star'    — median log stellar mass
            'log_sigma_sfr' — median log SFR surface density
        theta_true : array [T_c, T_w0, c_sigma, a0, a_z, a_M]
        A_c_true   : (M,) cold amplitudes; if None, defaults to 1.0 for all bins
        noise_scale : scale factor applied to default noise floors
        n_obs_per_bin : repeat flux measurements (unused — stacking produces one
                        mean flux per bin; kept for API consistency with pah_model)
        seed : RNG seed

        Returns
        -------
        dict with keys:
            'df'              — DataFrame with columns [bin_id, z, log_M_star,
                                log_sigma_sfr, <band>, <band>_err, ...]
            'true_params'     — dict of all true parameters
            'bin_grid'        — the input bin_grid echoed back
            'sigma_sfr_per_bin' — array (M,) of log_sigma_sfr values
        """
        rng = np.random.default_rng(seed)
        M = len(bin_grid)

        if A_c_true is None:
            A_c_true = np.ones(M)
        assert len(A_c_true) == M

        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_true

        rows = []
        for m, bdict in enumerate(bin_grid):
            z = bdict["z"]
            log_M = bdict["log_M_star"]
            log_sig = bdict["log_sigma_sfr"]
            bid = bdict.get("bin_id", m)

            # Which bands are available at this z?
            available = [
                b for b in self.bands
                if not (b == "MIPS_24" and z > _MIPS_ZMAX)
            ]

            # True fluxes
            true_fluxes = self.sed_at_z(
                z, log_M, log_sig, A_c_true[m], theta_true,
                bands=available, log_l_ir=self.log_l_ir_default,
            )

            row = {
                "bin_id": bid,
                "z": z,
                "log_M_star": log_M,
                "log_sigma_sfr": log_sig,
            }
            for band in self.bands:
                if band in true_fluxes:
                    noise = self.noise_model.get(band, 0.1) * noise_scale
                    row[band] = true_fluxes[band] + rng.normal(0, noise)
                    row[f"{band}_err"] = noise
                else:
                    row[band] = np.nan
                    row[f"{band}_err"] = np.nan

            rows.append(row)

        df = pd.DataFrame(rows)

        # Derived true quantities for inspection
        fw_true = np.array([
            warm_fraction(b["z"], b["log_M_star"], b["log_sigma_sfr"],
                          np.array([a0, a_z, a_M, a_sigma]))
            for b in bin_grid
        ])
        Tw_true = np.array([
            warm_temperature(b["log_sigma_sfr"], T_w0, c_sigma)
            for b in bin_grid
        ])
        Tc_true = np.array([cold_temperature(b["z"], T_c0, b_z) for b in bin_grid])

        return {
            "df": df,
            "true_params": {
                "theta_global": theta_true,
                "A_c_true": A_c_true,
                "fw_true": fw_true,
                "Tw_true": Tw_true,
                "Tc_true": Tc_true,
                "T_c0": T_c0,
                "b_z": b_z,
                "T_w0": T_w0,
                "c_sigma": c_sigma,
                "a0": a0,
                "a_z": a_z,
                "a_M": a_M,
                "a_sigma": a_sigma,
            },
            "bin_grid": bin_grid,
            "sigma_sfr_per_bin": np.array([b["log_sigma_sfr"] for b in bin_grid]),
        }

    # ------------------------------------------------------------------ #
    # MCMC fitter                                                          #
    # ------------------------------------------------------------------ #

    def _solve_amplitudes(
        self,
        df: pd.DataFrame,
        bin_col: str,
        theta_global: np.ndarray,
    ) -> np.ndarray:
        """
        Analytically solve cold amplitudes A_c[m] at fixed theta_global.

        For each bin m, A_c[m] = Σ(f_i · t_i / σ_i²) / Σ(t_i² / σ_i²)
        where t_i is the two-component template (evaluated at A_c=1) for
        band i in bin m.

        Returns (M,) array of A_c values; sets A_c=0 for degenerate bins.
        """
        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global
        theta_f = np.array([a0, a_z, a_M, a_sigma])

        bins = sorted(df[bin_col].unique())
        A_c = np.zeros(len(bins))

        for idx, bid in enumerate(bins):
            sub = df[df[bin_col] == bid].iloc[0]
            z = sub["z"]
            log_M = sub["log_M_star"]
            log_sig = sub["log_sigma_sfr"]

            Tc = cold_temperature(z, T_c0, b_z)
            fw = warm_fraction(z, log_M, log_sig, theta_f)
            Tw = np.clip(warm_temperature(log_sig, T_w0, c_sigma),
                         self.T_w_min, self.T_w_max)

            numer = denom = 0.0
            for band in self.bands:
                f_obs = sub.get(band, np.nan)
                f_err = sub.get(f"{band}_err", np.nan)
                if not (np.isfinite(f_obs) and np.isfinite(f_err) and f_err > 0):
                    continue
                if band == "MIPS_24" and z > _MIPS_ZMAX:
                    continue

                lam_rest = self.bands[band] / (1.0 + z)
                t_cold = _greybody_sed(
                    np.array([lam_rest]), 1.0, Tc, fw, Tw,
                    self.beta_c, self.beta_w,
                )[0]
                w = 1.0 / f_err**2
                numer += f_obs * t_cold * w
                denom += t_cold**2 * w

            A_c[idx] = numer / denom if denom > 0 else 0.0

        return A_c

    def _log_likelihood(
        self,
        theta_global: np.ndarray,
        df: pd.DataFrame,
        bin_col: str,
    ) -> float:
        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global
        theta_f = np.array([a0, a_z, a_M, a_sigma])

        # Analytic amplitudes at this theta
        A_c_arr = self._solve_amplitudes(df, bin_col, theta_global)
        bins = sorted(df[bin_col].unique())

        ll = 0.0
        for idx, bid in enumerate(bins):
            sub = df[df[bin_col] == bid].iloc[0]
            z = sub["z"]
            log_M = sub["log_M_star"]
            log_sig = sub["log_sigma_sfr"]

            fw = warm_fraction(z, log_M, log_sig, theta_f)
            Tw = np.clip(warm_temperature(log_sig, T_w0, c_sigma), self.T_w_min, self.T_w_max)
            A_c = A_c_arr[idx]
            if A_c <= 0:
                return -np.inf

            for band in self.bands:
                f_obs = sub.get(band, np.nan)
                f_err = sub.get(f"{band}_err", np.nan)
                if not (np.isfinite(f_obs) and np.isfinite(f_err) and f_err > 0):
                    continue
                if band == "MIPS_24" and z > _MIPS_ZMAX:
                    continue

                lam_rest = self.bands[band] / (1.0 + z)
                Tc = cold_temperature(z, T_c0, b_z)
                f_model = _greybody_sed(
                    np.array([lam_rest]), A_c, Tc, fw * A_c, Tw,
                    self.beta_c, self.beta_w,
                )[0]
                ll -= 0.5 * ((f_obs - f_model) / f_err) ** 2

        return ll

    def _log_prior(self, theta_global: np.ndarray) -> float:
        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global

        # Hard bounds on T_c0 anchor
        if not (self.T_c_min < T_c0 < self.T_c_max):
            return -np.inf
        if not (self.T_w_min < T_w0 < self.T_w_max):
            return -np.inf
        # T_c evolution: Schreiber+18 gives b_z ≈ 4.6 K/z; cap at 7 K/z
        if not (0.0 <= b_z <= 7.0):
            return -np.inf
        # c_sigma > 15 K/dex is unphysical (T_w saturates the clip first)
        if not (0.0 <= c_sigma <= 15.0):
            return -np.inf
        # Warm fraction grows with z (V22, Parente+), but cap to prevent runaway
        if not (0.0 <= a_z <= 1.5):
            return -np.inf
        # Physical cap: f_w < 1 at z=4 (warm can't exceed cold at realistic redshifts).
        # Hard constraint prevents runaway warm fraction in the absence of high-z data.
        if a0 + a_z * 4.0 > 0.0:
            return -np.inf

        # Gaussian priors on temperature anchors and sigma slope
        lp = 0.0
        mu_Tc, sig_Tc = self.T_c_prior
        lp -= 0.5 * ((T_c0 - mu_Tc) / sig_Tc) ** 2

        mu_Tw0, sig_Tw0 = self.T_w0_prior
        lp -= 0.5 * ((T_w0 - mu_Tw0) / sig_Tw0) ** 2

        mu_cs, sig_cs = self.c_sigma_prior
        lp -= 0.5 * ((c_sigma - mu_cs) / sig_cs) ** 2

        # Schreiber+18 expects b_z ≈ 4.6 K/z; weak Gaussian keeps chain from wandering
        lp -= 0.5 * (b_z / 5.0) ** 2

        # Broad Gaussian on log_fw coefficients to prevent runaway
        lp -= 0.5 * (a0 / 3.0) ** 2
        lp -= 0.5 * (a_z / 2.0) ** 2
        lp -= 0.5 * (a_M / 2.0) ** 2
        lp -= 0.5 * (a_sigma / 2.0) ** 2

        return lp

    def _log_posterior(
        self,
        theta_global: np.ndarray,
        df: pd.DataFrame,
        bin_col: str,
    ) -> float:
        lp = self._log_prior(theta_global)
        if not np.isfinite(lp):
            return -np.inf
        ll = self._log_likelihood(theta_global, df, bin_col)
        return lp + ll

    def _prepare_obs(self, df: pd.DataFrame, bin_col: str) -> list:
        """Convert DataFrame to a list of _BinObs (called once before MCMC)."""
        obs = []
        for bid in sorted(df[bin_col].unique()):
            sub = df[df[bin_col] == bid].iloc[0]
            z = float(sub["z"])
            log_l_ir = float(sub["log_l_ir"]) if "log_l_ir" in sub.index else self.log_l_ir_default
            nu_list, f_list, iv_list = [], [], []
            for band, lam_obs_um in self.bands.items():
                if band == "MIPS_24" and z > _MIPS_ZMAX:
                    continue
                f_obs = sub.get(band, np.nan)
                f_err = sub.get(f"{band}_err", np.nan)
                if not (np.isfinite(f_obs) and np.isfinite(f_err) and f_err > 0):
                    continue
                nu_list.append(_c * 1e6 / (lam_obs_um / (1.0 + z)))
                f_list.append(f_obs)
                iv_list.append(1.0 / f_err**2)
            obs.append(_BinObs(
                z=z,
                log_M_star=float(sub["log_M_star"]),
                log_sigma_sfr=float(sub["log_sigma_sfr"]),
                nu_rest=np.array(nu_list),
                f_obs=np.array(f_list),
                inv_var=np.array(iv_list),
                log_l_ir=log_l_ir,
            ))
        return obs

    def _per_bin_fits(self, obs_list: list) -> tuple:
        """
        Single-component greybody fit per bin via scipy curve_fit.

        Returns (T_arr, z_arr, sig_arr) for bins that converge.
        Used to bootstrap physics-informed MCMC initialisation.
        """
        from scipy.optimize import curve_fit

        T_list, z_list, sig_list = [], [], []
        for b in obs_list:
            if len(b.nu_rest) < 2:
                continue

            def sc(nu, A, T, _b=b):
                return A * _greybody_nu(nu, T, self.beta_c)

            try:
                popt, _ = curve_fit(
                    sc, b.nu_rest, b.f_obs,
                    p0=[float(np.median(b.f_obs)), 35.0],
                    bounds=([0.0, self.T_c_min], [1e12, 80.0]),
                    sigma=1.0 / np.sqrt(b.inv_var),
                    absolute_sigma=True,
                    maxfev=2000,
                )
                T_list.append(popt[1])
                z_list.append(b.z)
                sig_list.append(b.log_sigma_sfr)
            except Exception:
                pass

        return np.array(T_list), np.array(z_list), np.array(sig_list)

    def _compute_map_init(
        self,
        log_post_fn,
        default_init: np.ndarray,
        n_params: int,
        obs_list: list,
        param_names: list | None = None,
    ) -> np.ndarray:
        """
        MAP estimate via Nelder-Mead, warm-started from per-bin greybody fits.

        Strategy
        --------
        1. Fit each bin independently with a single-component greybody to get
           T_apparent(z, σ_SFR).  Low-z T_apparent ≈ T_c (warm subdominant);
           the slope vs σ_SFR gives c_sigma; high-z mean gives T_w0 ceiling.
        2. Use those physics-informed values as the Nelder-Mead starting point
           (plus a few random restarts for robustness).
        3. Return the theta with the highest log-posterior found.

        Falls back to default_init if fitting fails entirely.
        """
        from scipy.optimize import minimize

        T_arr, z_arr, sig_arr = self._per_bin_fits(obs_list)

        # --- Physics-informed starting point ---
        pnames = param_names or []
        def _idx(name):
            return pnames.index(name) if name in pnames else -1

        phys_start = default_init.copy()
        if len(T_arr) >= 3:
            # T_c0: low percentile of per-bin T at low-z (least warm contamination)
            low_z_mask = z_arr <= np.percentile(z_arr, 40)
            base = T_arr[low_z_mask] if low_z_mask.sum() > 2 else T_arr
            T_c0_est = float(np.clip(np.percentile(base, 20),
                                     self.T_c_min + 1, self.T_c_max - 5))
            i = _idx('T_c0')
            if i >= 0: phys_start[i] = T_c0_est

            # T_w0: upper envelope of high-z bins (more warm-component signal)
            high_z_mask = z_arr >= np.percentile(z_arr, 70)
            if high_z_mask.sum() > 1:
                T_w0_est = float(np.clip(
                    np.percentile(T_arr[high_z_mask], 70),
                    T_c0_est + 5, self.T_w_max - 5,
                ))
                i = _idx('T_w0')
                if i >= 0: phys_start[i] = T_w0_est

            # c_sigma: slope of T_apparent vs log σ_SFR
            if sig_arr.std() > 0.1 and len(sig_arr) > 3:
                slope = float(np.polyfit(sig_arr, T_arr, 1)[0])
                i = _idx('c_sigma')
                if i >= 0: phys_start[i] = float(np.clip(slope, 0.0, 10.0))

        logger.info("Per-bin init: T_c0≈%.1f K, T_w0≈%.1f K, c_σ≈%.1f K/dex",
                    phys_start[_idx('T_c0')] if _idx('T_c0') >= 0 else float('nan'),
                    phys_start[_idx('T_w0')] if _idx('T_w0') >= 0 else float('nan'),
                    phys_start[_idx('c_sigma')] if _idx('c_sigma') >= 0 else float('nan'))

        # --- Nelder-Mead MAP from phys_start + random restarts ---
        rng = np.random.default_rng(42)
        # Perturbation scale per parameter in the full 8-param order
        # [T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma]
        _full_perturb = np.array([2.0, 0.5, 3.0, 1.0, 0.5, 0.05, 0.1, 0.1])
        _pname_order = ['T_c0','b_z','T_w0','c_sigma','a0','a_z','a_M','a_sigma']
        perturb = np.array([
            _full_perturb[_pname_order.index(p)]
            if p in _pname_order
            else 0.1
            for p in pnames
        ]) if pnames else _full_perturb[:n_params]
        starts = [phys_start, default_init.copy()]
        for _ in range(2):
            starts.append(phys_start + perturb * rng.standard_normal(n_params) * 0.4)

        best_theta = phys_start.copy()
        best_val   = -np.inf

        for start in starts:
            try:
                def neg_lp(t):
                    v = log_post_fn(t)
                    return -v if np.isfinite(v) else 1e10

                res = minimize(
                    neg_lp, start,
                    method='Nelder-Mead',
                    options={'maxiter': 10_000, 'xatol': 0.05,
                             'fatol': 0.1, 'adaptive': True},
                )
                val = log_post_fn(res.x)
                if np.isfinite(val) and val > best_val:
                    best_val   = val
                    best_theta = res.x.copy()
            except Exception:
                pass

        logger.info("MAP init: %s  (log-post=%.1f)",
                    np.array2string(best_theta, precision=2, suppress_small=True),
                    best_val)
        return best_theta

    def _log_posterior_fast(self, theta: np.ndarray, obs_list: list) -> float:
        """Vectorized log-posterior using pre-baked obs data and cached peaks.

        Single pass per bin: solves A_c analytically then accumulates χ².
        """
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta
        theta_f = np.array([a0, a_z, a_M, a_sigma])
        beta_c, beta_w = self.beta_c, self.beta_w

        ll = 0.0
        for b in obs_list:
            if len(b.nu_rest) == 0:
                continue
            Tc = float(np.clip(cold_temperature(b.z, T_c0, b_z),
                               self.T_c_min, self.T_c_max))
            fw = warm_fraction(b.z, b.log_M_star, b.log_sigma_sfr, theta_f)
            Tw = float(np.clip(
                warm_temperature(b.log_sigma_sfr, T_w0, c_sigma),
                self.T_w_min, self.T_w_max,
            ))

            # Template at A_c=1 — pure modified blackbody (no Wien splice here;
            # Wien is used only for per-bin T_c prior and plotting)
            cold = _greybody_nu(b.nu_rest, Tc, beta_c)
            warm = fw * _warm_sed_nu(
                b.nu_rest, Tw, beta_w, b.z, b.log_l_ir,
                self.pah_model if self.use_pah_warm else None,
            )
            t = cold + warm

            # WLS solve for A_c
            denom = float((t * t * b.inv_var).sum())
            if denom <= 0.0:
                return -np.inf
            A_c = float((b.f_obs * t * b.inv_var).sum()) / denom
            if A_c <= 0.0:
                return -np.inf

            # χ² contribution (all bands at once)
            r = b.f_obs - A_c * t
            ll -= 0.5 * float((r * r * b.inv_var).sum())

        return lp + ll

    def fit_dust_evolution_lstsq(
        self,
        df: pd.DataFrame,
        bin_col: str = "bin_id",
        theta_init: np.ndarray | None = None,
        fix_a_M: bool = False,
        fix_a_sigma: bool = False,
        verbose: bool = True,
    ) -> DustEvolutionResult:
        """
        Least-squares (MAP) fit — returns the Nelder-Mead posterior mode.

        Same parameter structure and fixing conventions as fit_dust_evolution,
        but skips MCMC entirely.  Use this to: (a) diagnose the model before
        committing to MCMC, (b) get initial parameter guesses, (c) explore
        warm-fraction decomposition on real data quickly.

        Returns a DustEvolutionResult with sampler=None and theta_err from
        a numerical Hessian approximation (unreliable — treat as lower bound
        on uncertainties; use MCMC for proper posteriors).
        """
        from scipy.optimize import minimize

        obs_list = self._prepare_obs(df, bin_col)

        _saved_T_c_prior = self.T_c_prior
        try:
            T_arr_init, z_arr_init, _ = self._per_bin_fits(obs_list)
            if len(T_arr_init) >= 3:
                low_z_mask = z_arr_init <= np.percentile(z_arr_init, 40)
                base = T_arr_init[low_z_mask] if low_z_mask.sum() > 2 else T_arr_init
                T_c0_data = float(np.clip(np.percentile(base, 20),
                                          self.T_c_min + 1, self.T_c_max - 5))
                logger.info("Data-driven T_c0 prior centre: %.1f K (σ=3 K)", T_c0_data)
                self.T_c_prior = (T_c0_data, 3.0)
        except Exception:
            pass

        if fix_a_M and fix_a_sigma:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z"]
            def log_post_fn(t):
                T_c0, b_z, T_w0, c_sigma, a0, a_z = t
                return self._log_posterior_fast(
                    np.array([T_c0, b_z, T_w0, c_sigma, a0, a_z, 0.0, 0.0]), obs_list)
            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05])
        elif fix_a_M:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_sigma"]
            def log_post_fn(t):
                T_c0, b_z, T_w0, c_sigma, a0, a_z, a_sigma = t
                return self._log_posterior_fast(
                    np.array([T_c0, b_z, T_w0, c_sigma, a0, a_z, 0.0, a_sigma]), obs_list)
            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05, 0.0])
        elif fix_a_sigma:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_M"]
            def log_post_fn(t):
                T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M = t
                return self._log_posterior_fast(
                    np.array([T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, 0.0]), obs_list)
            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05, 0.0])
        else:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_M", "a_sigma"]
            def log_post_fn(t):
                return self._log_posterior_fast(t, obs_list)
            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05, 0.0, 0.0])

        n_params = len(param_names)
        if theta_init is not None:
            start = np.asarray(theta_init)[:n_params]
        else:
            start = default_init

        theta_map = self._compute_map_init(log_post_fn, start, n_params, obs_list, param_names)

        # Numerical Hessian for rough uncertainty estimate
        eps = 1e-3
        hess_diag = np.zeros(n_params)
        lp0 = log_post_fn(theta_map)
        for i in range(n_params):
            dv = np.zeros(n_params); dv[i] = eps
            hp = log_post_fn(theta_map + dv)
            hm = log_post_fn(theta_map - dv)
            h2 = (hp - 2.0 * lp0 + hm) / eps**2
            hess_diag[i] = max(-h2, 1e-10)

        theta_err = 1.0 / np.sqrt(hess_diag)

        # Reconstruct full 8-param theta
        fit_vals = dict(zip(param_names, theta_map))
        _def = {"T_c0": 30.0, "b_z": 0.0, "T_w0": 55.0, "c_sigma": 5.0,
                "a0": -0.5, "a_z": 0.05, "a_M": 0.0, "a_sigma": 0.0}
        _all_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_M", "a_sigma"]
        theta_med = np.array([fit_vals.get(n, _def[n]) for n in _all_names])

        A_c = self._solve_amplitudes(df, bin_col, theta_med)
        bins = sorted(df[bin_col].unique())
        M = len(bins)

        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_med
        theta_f = np.array([a0, a_z, a_M, a_sigma])
        fw_arr = np.zeros(M); Tw_arr = np.zeros(M); Tc_arr = np.zeros(M)
        for idx, bid in enumerate(bins):
            sub = df[df[bin_col] == bid].iloc[0]
            z = float(sub["z"])
            fw_arr[idx] = warm_fraction(z, sub["log_M_star"], sub["log_sigma_sfr"], theta_f)
            Tw_arr[idx] = float(np.clip(warm_temperature(sub["log_sigma_sfr"], T_w0, c_sigma),
                                        self.T_w_min, self.T_w_max))
            Tc_arr[idx] = float(np.clip(cold_temperature(z, T_c0, b_z),
                                        self.T_c_min, self.T_c_max))

        self.T_c_prior = _saved_T_c_prior

        if verbose:
            for name, val, err in zip(param_names, theta_map, theta_err):
                logger.info("  %-12s = %7.3f ± %.3f", name, val, err)

        return DustEvolutionResult(
            theta_global=theta_map,
            theta_err=theta_err,
            A_c_per_bin=A_c,
            A_w_per_bin=fw_arr * A_c,
            f_w_per_bin=fw_arr,
            T_c_per_bin=Tc_arr,
            T_w_grid=Tw_arr,
            T_c_fit=T_c0,
            sampler=None,
            acceptance_fraction=0.0,
            autocorr_time=np.nan,
            n_bins=M,
            param_names=param_names,
        )

    def fit_dust_evolution(
        self,
        df: pd.DataFrame,
        bin_col: str = "bin_id",
        n_walkers: int = 32,
        n_steps: int = 1000,
        n_burn: int = 300,
        progress: bool = True,
        verbose: bool = True,
        theta_init: np.ndarray | None = None,
        fix_a_M: bool = False,
        fix_a_sigma: bool = False,
        use_lstsq_init: bool = True,
    ) -> DustEvolutionResult | None:
        """
        Fit the two-component dust evolution model via emcee.

        Parameters
        ----------
        df : DataFrame from simulate_stacked_dataframe or real stacking output.
             Must contain columns: bin_id (or bin_col), z, log_M_star,
             log_sigma_sfr, plus <band> and <band>_err per COSMOS_BANDS.
        bin_col : column identifying the property bin index.
        n_walkers : emcee walkers (must be even, ≥ 2×n_params).
        n_steps, n_burn : MCMC steps after/before burn-in discard.
        theta_init : starting point [T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma].
                     If None and use_lstsq_init=True, derived from per-bin fits.
        fix_a_M : if True, fix a_M=0.  Use when log_M* is constant across bins.
        fix_a_sigma : if True, fix a_sigma=0.  Use when log_σ_SFR is constant
                      across bins (rare — most real grids vary σ_SFR).
        use_lstsq_init : if True (default), run per-bin greybody fits + Nelder-Mead
                         MAP optimisation before MCMC to seed walkers near the
                         posterior mode.  Strongly recommended for real data where
                         the default starting point may miss the dominant basin.

        Returns
        -------
        DustEvolutionResult or None if emcee is unavailable.
        """
        try:
            import emcee
        except ImportError:
            logger.error("emcee not installed — cannot run MCMC")
            return None

        # Pre-bake observations once — eliminates all pandas access from the hot path
        obs_list = self._prepare_obs(df, bin_col)

        # Tighten the T_c0 prior using low-z single-component fits (the "grey lines").
        # Those fits are reliable because the warm component is subdominant at low z,
        # so their T estimate is a good proxy for T_c.  We set the prior center to the
        # 20th percentile of low-z per-bin temperatures with σ=3 K so MCMC stays
        # anchored near what the data directly constrain without MCMC.
        _saved_T_c_prior = self.T_c_prior
        if use_lstsq_init:
            try:
                T_arr_init, z_arr_init, _ = self._per_bin_fits(obs_list)
                if len(T_arr_init) >= 3:
                    low_z_mask = z_arr_init <= np.percentile(z_arr_init, 40)
                    base = T_arr_init[low_z_mask] if low_z_mask.sum() > 2 else T_arr_init
                    T_c0_data = float(np.clip(np.percentile(base, 20),
                                              self.T_c_min + 1, self.T_c_max - 5))
                    logger.info("Data-driven T_c0 prior centre: %.1f K (σ=3 K)", T_c0_data)
                    self.T_c_prior = (T_c0_data, 3.0)
            except Exception:
                pass  # fall back to default prior

        # Build parameter list and closure based on which params are fixed.
        # Full theta order: [T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma]
        if fix_a_M and fix_a_sigma:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z"]

            def log_post_fn(t):
                T_c0, b_z, T_w0, c_sigma, a0, a_z = t
                return self._log_posterior_fast(
                    np.array([T_c0, b_z, T_w0, c_sigma, a0, a_z, 0.0, 0.0]), obs_list)

            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05])
            scales       = np.array([2.0,  0.3, 3.0,  1.0,  0.3,  0.05])

        elif fix_a_M:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_sigma"]

            def log_post_fn(t):
                T_c0, b_z, T_w0, c_sigma, a0, a_z, a_sigma = t
                return self._log_posterior_fast(
                    np.array([T_c0, b_z, T_w0, c_sigma, a0, a_z, 0.0, a_sigma]), obs_list)

            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05, 0.0])
            scales       = np.array([2.0,  0.3, 3.0,  1.0,  0.3,  0.05, 0.05])

        elif fix_a_sigma:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_M"]

            def log_post_fn(t):
                T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M = t
                return self._log_posterior_fast(
                    np.array([T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, 0.0]), obs_list)

            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05, 0.0])
            scales       = np.array([2.0,  0.3, 3.0,  1.0,  0.3,  0.05, 0.05])

        else:
            param_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_M", "a_sigma"]

            def log_post_fn(t):
                return self._log_posterior_fast(t, obs_list)

            default_init = np.array([30.0, 0.0, 55.0, 5.0, -0.5, 0.05, 0.0, 0.0])
            scales       = np.array([2.0,  0.3, 3.0,  1.0,  0.3,  0.05, 0.05, 0.05])

        n_params = len(param_names)
        n_walkers = max(n_walkers, 2 * n_params + 2)
        if n_walkers % 2:
            n_walkers += 1

        if theta_init is not None:
            theta_init_use = np.asarray(theta_init)[:n_params]
        elif use_lstsq_init:
            logger.info("Computing MAP initialisation from per-bin fits ...")
            theta_init_use = self._compute_map_init(
                log_post_fn, default_init, n_params, obs_list, param_names)
        else:
            theta_init_use = default_init

        rng = np.random.default_rng(0)
        p0 = theta_init_use + scales * rng.standard_normal((n_walkers, n_params))

        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_post_fn)

        if verbose:
            fixed = []
            if fix_a_M:    fixed.append("a_M=0")
            if fix_a_sigma: fixed.append("a_sigma=0")
            label = ("fix[" + ",".join(fixed) + "]") if fixed else "free_all"
            logger.info("Running MCMC: %d walkers × %d steps (%s)",
                        n_walkers, n_steps + n_burn, label)

        sampler.run_mcmc(p0, n_steps + n_burn, progress=progress)

        # Discard burn-in
        flat = sampler.get_chain(discard=n_burn, flat=True)
        theta_med_fit = np.median(flat, axis=0)
        theta_err_fit = flat.std(axis=0)

        # Reconstruct full 8-param theta for derived quantities
        fit_vals = dict(zip(param_names, theta_med_fit))
        _def = {"T_c0": 30.0, "b_z": 0.0, "T_w0": 55.0, "c_sigma": 5.0,
                "a0": -0.5, "a_z": 0.05, "a_M": 0.0, "a_sigma": 0.0}
        _all_names = ["T_c0", "b_z", "T_w0", "c_sigma", "a0", "a_z", "a_M", "a_sigma"]
        theta_med = np.array([fit_vals.get(n, _def[n]) for n in _all_names])

        # Derived quantities at the posterior median
        A_c = self._solve_amplitudes(df, bin_col, theta_med)
        bins = sorted(df[bin_col].unique())
        M = len(bins)

        T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_med
        theta_f = np.array([a0, a_z, a_M, a_sigma])

        fw_arr  = np.zeros(M)
        Tw_arr  = np.zeros(M)
        Tc_arr  = np.zeros(M)
        for idx, bid in enumerate(bins):
            sub = df[df[bin_col] == bid].iloc[0]
            z   = float(sub["z"])
            fw_arr[idx] = warm_fraction(
                z, sub["log_M_star"], sub["log_sigma_sfr"], theta_f)
            Tw_arr[idx] = float(np.clip(
                warm_temperature(sub["log_sigma_sfr"], T_w0, c_sigma),
                self.T_w_min, self.T_w_max,
            ))
            Tc_arr[idx] = float(np.clip(
                cold_temperature(z, T_c0, b_z),
                self.T_c_min, self.T_c_max,
            ))

        try:
            tau = sampler.get_autocorr_time(quiet=True)
            tau_max = float(np.nanmax(tau))
        except Exception:
            tau_max = np.nan

        acc_frac = float(np.mean(sampler.acceptance_fraction))

        if verbose:
            logger.info("Acceptance: %.2f  τ_max: %.1f", acc_frac, tau_max)
            for name, val, err in zip(param_names, theta_med_fit, theta_err_fit):
                logger.info("  %-12s = %7.3f ± %.3f", name, val, err)

        self.T_c_prior = _saved_T_c_prior  # restore after fit

        return DustEvolutionResult(
            theta_global=theta_med_fit,
            theta_err=theta_err_fit,
            A_c_per_bin=A_c,
            A_w_per_bin=fw_arr * A_c,
            f_w_per_bin=fw_arr,
            T_c_per_bin=Tc_arr,
            T_w_grid=Tw_arr,
            T_c_fit=T_c0,   # anchor at z=0 for backward compat
            sampler=sampler,
            acceptance_fraction=acc_frac,
            autocorr_time=tau_max,
            n_bins=M,
            param_names=param_names,
        )
