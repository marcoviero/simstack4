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
    """S_ν ∝ ν^β · B_ν(T), normalised to peak = 1 over a fixed broad grid."""
    peak = _norm_peak(T_K, beta)
    raw = nu_hz**beta * _planck_nu(nu_hz, T_K)
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
    """
    Two-component greybody SED in mJy (arbitrary normalisation).

    Parameters
    ----------
    lambda_rest_um : rest-frame wavelengths in µm
    A_c : cold-dust amplitude (linear, mJy at SED peak)
    T_c : cold-dust temperature (K)
    A_w : warm-dust amplitude (linear, mJy at warm-component peak)
    T_w : warm-dust temperature (K)
    beta_c, beta_w : emissivity indices
    """
    nu = _c * 1e6 / lambda_rest_um   # Hz
    cold = A_c * _greybody_nu(nu, T_c, beta_c)
    warm = A_w * _greybody_nu(nu, T_w, beta_w)
    return cold + warm


def _peak_wavelength_um(T_K: float, beta: float = 1.8) -> float:
    """
    Rest-frame SED peak of ν^β·B_ν(T) via a quick grid search.
    Faster and more accurate than the analytic approximation for β ≠ 0.
    """
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

    Physical basis: stochastically heated small grains, T ∝ ISRF^(1/(4+β)).
    σ_SFR traces ISRF, so a linear relationship in log_σ_SFR is a reasonable
    first-order model.
    """
    return T_w0 + c_sigma * log_sigma_sfr


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


@dataclass
class DustEvolutionResult:
    """Output from fit_dust_evolution."""
    theta_global: np.ndarray       # [T_c, T_w0, c_sigma, a0, a_z, a_M]
    theta_err: np.ndarray          # posterior std for each global param
    A_c_per_bin: np.ndarray        # (M,) cold amplitudes per property bin
    A_w_per_bin: np.ndarray        # (M,) warm amplitudes per property bin (derived)
    f_w_per_bin: np.ndarray        # (M,) warm fractions
    T_c_fit: float                 # fitted cold temperature (K)
    T_w_grid: np.ndarray           # (M,) warm temperature per bin (T_w0 + c_σ·σ_SFR)
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
        c_sigma_prior: tuple[float, float] = (5.0, 3.0),  # (mean, sigma) — tightened from 5.0
        bands: dict | None = None,
        noise_model: dict | None = None,
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
    ) -> dict[str, float]:
        """
        Compute observed flux densities (mJy) for a single population at redshift z.

        theta_global = [T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma]
        """
        T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global
        fw = warm_fraction(z, log_M_star, log_sigma_sfr, np.array([a0, a_z, a_M, a_sigma]))
        Tw = warm_temperature(log_sigma_sfr, T_w0, c_sigma)
        Tw = np.clip(Tw, self.T_w_min, self.T_w_max)

        bands_to_use = bands or list(self.bands.keys())
        fluxes = {}
        for band in bands_to_use:
            lam_obs = self.bands[band]
            lam_rest = lam_obs / (1.0 + z)
            # Two-component SED at rest wavelength
            F = _greybody_sed(
                np.array([lam_rest]),
                A_c=A_c,
                T_c=T_c,
                A_w=fw * A_c,
                T_w=Tw,
                beta_c=self.beta_c,
                beta_w=self.beta_w,
            )[0]
            # Apply (1+z) cosmological dimming proxy: flux ∝ 1/(1+z) for fixed L
            # (proper D_L scaling not applied — A_c absorbs it; this is relative)
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

        T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_true

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
                z, log_M, log_sig, A_c_true[m], theta_true, bands=available
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

        return {
            "df": df,
            "true_params": {
                "theta_global": theta_true,
                "A_c_true": A_c_true,
                "fw_true": fw_true,
                "Tw_true": Tw_true,
                "T_c": T_c,
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
        T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global
        theta_f = np.array([a0, a_z, a_M, a_sigma])

        bins = sorted(df[bin_col].unique())
        A_c = np.zeros(len(bins))

        for idx, bid in enumerate(bins):
            sub = df[df[bin_col] == bid].iloc[0]
            z = sub["z"]
            log_M = sub["log_M_star"]
            log_sig = sub["log_sigma_sfr"]

            fw = warm_fraction(z, log_M, log_sig, theta_f)
            Tw = np.clip(warm_temperature(log_sig, T_w0, c_sigma), self.T_w_min, self.T_w_max)

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
                    np.array([lam_rest]), 1.0, T_c, fw, Tw,
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
        T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global
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
                f_model = _greybody_sed(
                    np.array([lam_rest]), A_c, T_c, fw * A_c, Tw,
                    self.beta_c, self.beta_w,
                )[0]
                ll -= 0.5 * ((f_obs - f_model) / f_err) ** 2

        return ll

    def _log_prior(self, theta_global: np.ndarray) -> float:
        T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_global

        # Hard bounds
        if not (self.T_c_min < T_c < self.T_c_max):
            return -np.inf
        if not (self.T_w_min < T_w0 < self.T_w_max):
            return -np.inf
        # Physically motivated: warm fraction increases with redshift (V22, Parente+)
        if a_z < 0:
            return -np.inf

        # Gaussian priors on dust temperatures and sigma slope
        lp = 0.0
        mu_Tc, sig_Tc = self.T_c_prior
        lp -= 0.5 * ((T_c - mu_Tc) / sig_Tc) ** 2

        mu_Tw0, sig_Tw0 = self.T_w0_prior
        lp -= 0.5 * ((T_w0 - mu_Tw0) / sig_Tw0) ** 2

        mu_cs, sig_cs = self.c_sigma_prior
        lp -= 0.5 * ((c_sigma - mu_cs) / sig_cs) ** 2

        # Broad Gaussian on log_fw coefficients to prevent runaway
        lp -= 0.5 * (a0 / 3.0) ** 2       # |a0| < ~3 (log units)
        lp -= 0.5 * (a_z / 2.0) ** 2      # modest z evolution
        lp -= 0.5 * (a_M / 2.0) ** 2
        lp -= 0.5 * (a_sigma / 2.0) ** 2  # modest σ_SFR dependence

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
            ))
        return obs

    def _log_posterior_fast(self, theta: np.ndarray, obs_list: list) -> float:
        """Vectorized log-posterior using pre-baked obs data and cached peaks.

        Single pass per bin: solves A_c analytically then accumulates χ².
        """
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta
        theta_f = np.array([a0, a_z, a_M, a_sigma])
        beta_c, beta_w = self.beta_c, self.beta_w

        ll = 0.0
        for b in obs_list:
            if len(b.nu_rest) == 0:
                continue
            fw = warm_fraction(b.z, b.log_M_star, b.log_sigma_sfr, theta_f)
            Tw = float(np.clip(
                warm_temperature(b.log_sigma_sfr, T_w0, c_sigma),
                self.T_w_min, self.T_w_max,
            ))

            # Template at A_c=1 (vectorized over bands; peaks are cached)
            cold = b.nu_rest**beta_c * _planck_nu(b.nu_rest, T_c) / _norm_peak(T_c, beta_c)
            warm = fw * b.nu_rest**beta_w * _planck_nu(b.nu_rest, Tw) / _norm_peak(Tw, beta_w)
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
                     Defaults to a physically motivated starting point.
        fix_a_M : if True, fix a_M=0.  Use when log_M* is constant across bins.
        fix_a_sigma : if True, fix a_sigma=0.  Use when log_σ_SFR is constant
                      across bins (rare — most real grids vary σ_SFR).

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

        # Build parameter list and closure based on which params are fixed.
        # Full theta order: [T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma]
        if fix_a_M and fix_a_sigma:
            param_names = ["T_c", "T_w0", "c_sigma", "a0", "a_z"]

            def log_post_fn(t):
                return self._log_posterior_fast(
                    np.array([*t, 0.0, 0.0]), obs_list)

            default_init = np.array([30.0, 55.0, 5.0, -0.5, 0.05])
            scales       = np.array([2.0,  3.0,  1.0,  0.3,  0.05])

        elif fix_a_M:
            param_names = ["T_c", "T_w0", "c_sigma", "a0", "a_z", "a_sigma"]

            def log_post_fn(t):
                T_c, T_w0, c_sigma, a0, a_z, a_sigma = t
                return self._log_posterior_fast(
                    np.array([T_c, T_w0, c_sigma, a0, a_z, 0.0, a_sigma]), obs_list)

            default_init = np.array([30.0, 55.0, 5.0, -0.5, 0.05, 0.0])
            scales       = np.array([2.0,  3.0,  1.0,  0.3,  0.05, 0.05])

        elif fix_a_sigma:
            param_names = ["T_c", "T_w0", "c_sigma", "a0", "a_z", "a_M"]

            def log_post_fn(t):
                T_c, T_w0, c_sigma, a0, a_z, a_M = t
                return self._log_posterior_fast(
                    np.array([T_c, T_w0, c_sigma, a0, a_z, a_M, 0.0]), obs_list)

            default_init = np.array([30.0, 55.0, 5.0, -0.5, 0.05, 0.0])
            scales       = np.array([2.0,  3.0,  1.0,  0.3,  0.05, 0.05])

        else:
            param_names = ["T_c", "T_w0", "c_sigma", "a0", "a_z", "a_M", "a_sigma"]

            def log_post_fn(t):
                return self._log_posterior_fast(t, obs_list)

            default_init = np.array([30.0, 55.0, 5.0, -0.5, 0.05, 0.0, 0.0])
            scales       = np.array([2.0,  3.0,  1.0,  0.3,  0.05, 0.05, 0.05])

        n_params = len(param_names)
        n_walkers = max(n_walkers, 2 * n_params + 2)
        if n_walkers % 2:
            n_walkers += 1

        theta_init_use = default_init if theta_init is None else np.asarray(theta_init)[:n_params]

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

        # Reconstruct full 7-param theta for derived quantities
        fit_vals = dict(zip(param_names, theta_med_fit))
        fit_errs = dict(zip(param_names, theta_err_fit))
        _def = {"T_c": 30.0, "T_w0": 55.0, "c_sigma": 5.0,
                "a0": -0.5, "a_z": 0.05, "a_M": 0.0, "a_sigma": 0.0}
        _all_names = ["T_c", "T_w0", "c_sigma", "a0", "a_z", "a_M", "a_sigma"]
        theta_med = np.array([fit_vals.get(n, _def[n]) for n in _all_names])
        theta_err_full = np.array([fit_errs.get(n, 0.0) for n in _all_names])

        # Derived quantities at the posterior median
        A_c = self._solve_amplitudes(df, bin_col, theta_med)
        bins = sorted(df[bin_col].unique())
        M = len(bins)

        T_c, T_w0, c_sigma, a0, a_z, a_M, a_sigma = theta_med
        theta_f = np.array([a0, a_z, a_M, a_sigma])

        fw_arr = np.zeros(M)
        Tw_arr = np.zeros(M)
        for idx, bid in enumerate(bins):
            sub = df[df[bin_col] == bid].iloc[0]
            fw_arr[idx] = warm_fraction(
                sub["z"], sub["log_M_star"], sub["log_sigma_sfr"], theta_f)
            Tw_arr[idx] = np.clip(
                warm_temperature(sub["log_sigma_sfr"], T_w0, c_sigma),
                self.T_w_min, self.T_w_max,
            )

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

        return DustEvolutionResult(
            theta_global=theta_med_fit,   # fitted params only (5 or 6)
            theta_err=theta_err_fit,
            A_c_per_bin=A_c,
            A_w_per_bin=fw_arr * A_c,
            f_w_per_bin=fw_arr,
            T_c_fit=T_c,
            T_w_grid=Tw_arr,
            sampler=sampler,
            acceptance_fraction=acc_frac,
            autocorr_time=tau_max,
            n_bins=M,
            param_names=param_names,
        )
