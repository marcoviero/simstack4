"""
Flux-level simulator and strategy optimization for PAH dithered stacking.

Dithered stacking runs the simultaneous-stacking pipeline many times with
offset fine redshift bins, so PAH features sweep through the MIPS 24/70 µm
bandpasses and the stacked fluxes trace a rest-frame pseudo-spectrum. This
module answers "what is the best dithering strategy?" without touching
maps: it simulates stacked fluxes per dither bin directly from a redshift
distribution, photo-z scatter (with catastrophic outliers), and a
bootstrap-calibrated noise model — fast enough to sweep many strategies.

Components
----------
- DitherScheme    : staggered bin-edge sets + property bins; exports TOML bins
- TruthSpectrum   : injected rest-frame MIR spectrum (PAH Gaussians + warm
                    MBB continuum) with property-dependent amplitudes;
                    band fluxes by DIRECT bandpass integration, deliberately
                    independent of the pah_spectrum kernel path (Tier-1
                    cross-check between the two implementations)
- compute_pz_matrix : per-bin true-z probability masses on a z grid,
                    including photo-z smearing and the outlier pedestal —
                    feeds pah_spectrum.build_design_matrix
- NoiseModel      : σ_i = σ_ref·sqrt(n_ref/N_i) per band, plus the
                    shared-source covariance between staggered runs
                    (Cov_ij = σ_i σ_j N_shared/sqrt(N_i N_j)) that makes the
                    "n_stagger× free resolution" claim honest
- simulate_dithered_fluxes : per-source realization → stacked DataFrame
                    (same return convention as
                    DustEvolutionModel.simulate_stacked_dataframe)
- fisher_for_scheme / sweep_strategies : marginal Cramér–Rao bounds on the
                    feature-group amplitudes for ANY scheme, with the
                    continuum amplitude profiled out — the principled
                    figure of merit for strategy optimization
- injection_recovery_sweep : simulate → GLS fit over realizations,
                    confirming the Fisher predictions

The forward-model fitting lives in pah_spectrum.py.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr

from .dust_evolution import _greybody_nu
from .pah_spectrum import (
    DEFAULT_BANDS,
    DEFAULT_FEATURES,
    DEFAULT_GROUPS,
    PAHFeature,
    build_design_matrix,
    get_bandpass,
    group_weights,
    solve_linear_amplitudes,
    warm_continuum_kernel,
)

_c_um_hz = 2.998e14  # speed of light [µm·Hz]


# ---------------------------------------------------------------------------
# Dither scheme
# ---------------------------------------------------------------------------


def _default_property_bins() -> list[dict[str, float]]:
    return [{"log_M_star": 10.5, "log_sigma_sfr": 0.0}]


@dataclass
class DitherScheme:
    """A set of staggered redshift binnings plus property (sub-)bins.

    runs          : bin edges per staggered stacking run
    property_bins : representative property values per population split,
                    e.g. [{"log_M_star": 10.0, "log_sigma_sfr": 0.0}, ...];
                    each property bin is stacked independently
    bands         : bandpass names fluxes are simulated/fit in
    """

    runs: list[NDArray[np.float64]]
    property_bins: list[dict[str, float]] = field(
        default_factory=_default_property_bins
    )
    bands: tuple[str, ...] = DEFAULT_BANDS

    @classmethod
    def uniform(
        cls,
        z_min: float = 0.5,
        z_max: float = 3.5,
        dz: float = 0.15,
        n_stagger: int = 3,
        property_bins: list[dict[str, float]] | None = None,
        bands: tuple[str, ...] = DEFAULT_BANDS,
    ) -> "DitherScheme":
        """Uniform bins of width dz; run s is offset by s·dz/n_stagger."""
        runs = []
        for s in range(n_stagger):
            start = z_min + dz * s / n_stagger
            edges = np.arange(start, z_max + 1e-9, dz)
            if len(edges) < 2:
                raise ValueError("dz too large for the redshift range")
            runs.append(edges)
        return cls(
            runs=runs,
            property_bins=property_bins or _default_property_bins(),
            bands=bands,
        )

    @classmethod
    def adaptive(
        cls,
        redshifts: NDArray[np.float64],
        *,
        n_stagger: int = 3,
        z_min: float = 0.5,
        z_max: float = 3.5,
        min_sources: int = 500,
        max_dz: float = 0.15,
        min_dz: float = 0.03,
        property_bins: list[dict[str, float]] | None = None,
        bands: tuple[str, ...] = DEFAULT_BANDS,
    ) -> "DitherScheme":
        """Adaptive staggered bins via analyze_pah.staggered_pah_zbins."""
        from .analyze_pah import staggered_pah_zbins

        edge_sets = staggered_pah_zbins(
            redshifts,
            n_stagger=n_stagger,
            z_min=z_min,
            z_max=z_max,
            min_sources=min_sources,
            max_dz=max_dz,
            min_dz=min_dz,
            verbose=False,
        )
        return cls(
            runs=[np.asarray(e, dtype=float) for e in edge_sets],
            property_bins=property_bins or _default_property_bins(),
            bands=bands,
        )

    @property
    def z_min(self) -> float:
        return float(min(e[0] for e in self.runs))

    @property
    def z_max(self) -> float:
        return float(max(e[-1] for e in self.runs))

    @property
    def n_zbins(self) -> int:
        return sum(len(e) - 1 for e in self.runs)

    def bin_table(self) -> pd.DataFrame:
        """One row per (run, z-bin): run_id, zbin_id, z_lo, z_hi, z_mid."""
        rows = []
        for run_id, edges in enumerate(self.runs):
            for k in range(len(edges) - 1):
                rows.append(
                    {
                        "run_id": run_id,
                        "zbin_id": k,
                        "z_lo": float(edges[k]),
                        "z_hi": float(edges[k + 1]),
                        "z_mid": float(0.5 * (edges[k] + edges[k + 1])),
                    }
                )
        return pd.DataFrame(rows)

    def to_toml_bins(self, ndigits: int = 3) -> list[list[float]]:
        """Bin-edge lists ready to paste into the TOML configs (one per run)."""
        return [[round(float(e), ndigits) for e in edges] for edges in self.runs]


# ---------------------------------------------------------------------------
# Truth spectrum
# ---------------------------------------------------------------------------


@dataclass
class TruthSpectrum:
    """Injected rest-frame MIR spectrum: warm MBB continuum + PAH features.

    Feature-group amplitudes are peak feature-to-continuum ratios in f_ν
    (same convention as pah_spectrum) and scale with population properties:

        log10 A_g,m = log10(amp0_g) + beta_mass·(log M* − pivot_M)
                                    + beta_sigma·(log σ_SFR − pivot_σ)
    """

    features: list[PAHFeature] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    feature_groups: list[list[int]] = field(
        default_factory=lambda: [list(g) for g in DEFAULT_GROUPS]
    )
    amp0: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.6, 2.0, 0.3, 0.8, 0.5])
    )
    beta_mass: float = 0.0
    beta_sigma: float = 0.0
    pivot_log_mass: float = 10.5
    pivot_log_sigma_sfr: float = 0.0
    T_warm: float = 60.0
    beta_warm: float = 1.5
    continuum_amp: float = 1.0  # C: overall normalization (mJy at MBB peak)

    def amplitudes(self, prop_bin: dict[str, float] | None = None) -> NDArray:
        """Feature-group amplitudes A_g for one property bin."""
        log_a = np.log10(np.asarray(self.amp0, dtype=float))
        if prop_bin is not None:
            log_a = log_a + self.beta_mass * (
                prop_bin.get("log_M_star", self.pivot_log_mass) - self.pivot_log_mass
            )
            log_a = log_a + self.beta_sigma * (
                prop_bin.get("log_sigma_sfr", self.pivot_log_sigma_sfr)
                - self.pivot_log_sigma_sfr
            )
        return 10.0**log_a

    def rest_spectrum(
        self,
        lam_um: NDArray[np.float64],
        prop_bin: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """f_ν at rest wavelengths: continuum_amp · (MBB + Σ_g A_g · features)."""
        lam = np.asarray(lam_um, dtype=float)
        spec = _greybody_nu(_c_um_hz / lam, self.T_warm, self.beta_warm)
        amps = self.amplitudes(prop_bin)
        weights = group_weights(self.features, self.feature_groups)
        for a_g, grp, w in zip(amps, self.feature_groups, weights, strict=True):
            for j, wj in zip(grp, w, strict=True):
                center, _, fwhm = self.features[j]
                sigma = fwhm / 2.355
                spec = spec + a_g * wj * np.exp(-0.5 * ((lam - center) / sigma) ** 2)
        return self.continuum_amp * spec

    def band_flux_curve(
        self,
        z_grid: NDArray[np.float64],
        band: str,
        prop_bin: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """In-band flux vs z by direct integration of the full rest spectrum.

        Independent of the pah_spectrum kernel decomposition — Tier-1
        tests cross-check the two paths against each other.
        """
        bp = get_bandpass(band)
        lam_rest = bp.lam_fine[None, :] / (1.0 + np.asarray(z_grid)[:, None])
        spec = self.rest_spectrum(lam_rest, prop_bin)
        return np.trapezoid(spec * bp.resp_fine, bp.lam_fine, axis=1) / bp.norm

    def band_flux(
        self, z: float, band: str, prop_bin: dict[str, float] | None = None
    ) -> float:
        """In-band flux at a single sharp redshift."""
        return float(self.band_flux_curve(np.array([z]), band, prop_bin)[0])


# ---------------------------------------------------------------------------
# Redshift distribution and photo-z machinery
# ---------------------------------------------------------------------------


def make_dndz(
    kind: str = "cosmos_like", z_peak: float = 1.1
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Analytic catalog redshift distributions (unnormalized).

    "cosmos_like": n(z) ∝ z² exp(−2z/z_peak), peaking at z_peak.
    "flat": constant.
    """
    if kind == "cosmos_like":

        def dndz(z: NDArray[np.float64]) -> NDArray[np.float64]:
            z = np.asarray(z, dtype=float)
            return np.clip(z, 0.0, None) ** 2 * np.exp(-2.0 * z / z_peak)

        return dndz
    if kind == "flat":
        return lambda z: np.ones_like(np.asarray(z, dtype=float))
    raise ValueError(f"unknown dndz kind {kind!r}")


def compute_pz_matrix(
    scheme: DitherScheme,
    dndz: Callable | None = None,
    sigma_z0: float = 0.01,
    f_cat: float = 0.0,
    z_grid: NDArray[np.float64] | None = None,
    n_grid: int = 600,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """True-z probability masses per dither bin (rows of the kernel model).

    A source lands in bin [z_lo, z_hi) when its *photo-z* does, so the
    distribution of its TRUE redshift is

        p_i(z) ∝ dN/dz(z) · [(1−f_cat)·ΔΦ_i(z) + f_cat·(z_hi−z_lo)/Z_range]

    with ΔΦ_i(z) = Φ((z_hi−z)/σ_z) − Φ((z_lo−z)/σ_z), σ_z = sigma_z0·(1+z),
    and the second term the catastrophic-outlier pedestal (outliers scatter
    uniformly over the survey range). Rows sum to 1 over z_grid; row order
    matches scheme.bin_table(). Returns (pz_matrix, z_grid).
    """
    bt = scheme.bin_table()
    if z_grid is None:
        pad = 5.0 * sigma_z0 * (1.0 + scheme.z_max) + 0.05
        z_grid = np.linspace(max(1e-3, scheme.z_min - pad), scheme.z_max + pad, n_grid)
    dn = dndz(z_grid) if dndz is not None else np.ones_like(z_grid)
    sig = np.maximum(sigma_z0 * (1.0 + z_grid), 1e-8)
    z_range = scheme.z_max - scheme.z_min

    pz = np.zeros((len(bt), len(z_grid)))
    for i, row in enumerate(bt.itertuples()):
        sel = ndtr((row.z_hi - z_grid) / sig) - ndtr((row.z_lo - z_grid) / sig)
        pedestal = (row.z_hi - row.z_lo) / z_range
        p = dn * ((1.0 - f_cat) * sel + f_cat * pedestal)
        total = p.sum()
        pz[i] = p / total if total > 0 else p
    return pz, z_grid


def shared_fraction_matrix(
    scheme: DitherScheme,
    dndz: Callable | None = None,
    z_grid: NDArray[np.float64] | None = None,
    n_grid: int = 2000,
) -> NDArray[np.float64]:
    """Expected source-sharing between dither bins, as fractions of N_total.

    f_ij = (expected sources selected by BOTH bin i and bin j) / N_total,
    computed from photo-z interval overlap weighted by dN/dz. The diagonal
    is each bin's occupancy fraction. Bins of the same run are disjoint;
    bins of different runs overlap where their intervals intersect.
    """
    bt = scheme.bin_table()
    if z_grid is None:
        z_grid = np.linspace(max(1e-3, scheme.z_min), scheme.z_max, n_grid)
    dn = dndz(z_grid) if dndz is not None else np.ones_like(z_grid)
    dn = dn / np.trapezoid(dn, z_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dn[1:] + dn[:-1]) * np.diff(z_grid))])

    def mass(lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.interp(hi, z_grid, cdf) - np.interp(lo, z_grid, cdf))

    n = len(bt)
    lo = bt["z_lo"].to_numpy()
    hi = bt["z_hi"].to_numpy()
    run = bt["run_id"].to_numpy()
    f = np.zeros((n, n))
    for i in range(n):
        f[i, i] = mass(lo[i], hi[i])
        for j in range(i + 1, n):
            if run[i] == run[j]:
                continue
            m = mass(max(lo[i], lo[j]), min(hi[i], hi[j]))
            f[i, j] = f[j, i] = m
    return f


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------


def _default_sigma_ref() -> dict[str, float]:
    # stacked-flux 1σ (mJy) for a 1000-source bin, COSMOS-depth bootstrap scale
    return {"MIPS_24": 0.005, "MIPS_70": 0.5}


@dataclass
class NoiseModel:
    """Bootstrap-calibrated stacked-flux noise: σ_i = σ_ref·sqrt(n_ref/N_i)."""

    sigma_ref: dict[str, float] = field(default_factory=_default_sigma_ref)
    n_ref: int = 1000

    def sigma_per_bin(
        self, band: str, n_sources: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n = np.asarray(n_sources, dtype=float)
        with np.errstate(divide="ignore"):
            return self.sigma_ref[band] * np.sqrt(self.n_ref / np.maximum(n, 1e-300))

    def covariance(
        self,
        n_sources: NDArray[np.float64],
        shared: NDArray[np.float64],
        bands: Sequence[str] = DEFAULT_BANDS,
    ) -> NDArray[np.float64]:
        """Full covariance over the flattened (bin, band) data vector.

        Per band b: Σ_b[i,j] = σ_i σ_j · N_shared,ij / sqrt(N_i N_j) — the
        correlation induced by staggered runs reusing the same sources.
        Different bands come from different maps → cross-band blocks are 0.
        Flattening is row-major (bin, band), matching
        pah_spectrum.solve_linear_amplitudes. Empty bins get an uncorrelated
        placeholder diagonal (their fluxes are NaN and masked in fits).
        """
        n = np.asarray(n_sources, dtype=float)
        n_bins = len(n)
        n_bands = len(bands)
        cov = np.zeros((n_bins * n_bands, n_bins * n_bands))
        valid = n > 0
        n_safe = np.maximum(n, 1.0)
        corr = shared / np.sqrt(np.outer(n_safe, n_safe))
        np.fill_diagonal(corr, 1.0)
        corr[~valid, :] = 0.0
        corr[:, ~valid] = 0.0
        np.fill_diagonal(corr, 1.0)
        for b, band in enumerate(bands):
            sig = self.sigma_per_bin(band, n_safe)
            block = np.outer(sig, sig) * corr
            cov[b::n_bands, b::n_bands] = block
        return cov


# ---------------------------------------------------------------------------
# Flux-level simulator
# ---------------------------------------------------------------------------


def _draw_redshifts(
    rng: np.random.Generator,
    n: int,
    dndz: Callable,
    z_lo: float,
    z_hi: float,
) -> NDArray[np.float64]:
    """Inverse-CDF sampling of source redshifts from dN/dz on [z_lo, z_hi]."""
    z = np.linspace(z_lo, z_hi, 4000)
    pdf = np.clip(dndz(z), 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, z)


def simulate_dithered_fluxes(
    scheme: DitherScheme,
    truth: TruthSpectrum,
    *,
    dndz: Callable | None = None,
    catalog_z: NDArray[np.float64] | None = None,
    n_total: int = 200_000,
    sigma_z0: float = 0.01,
    f_cat: float = 0.0,
    noise: NoiseModel | None = None,
    noise_scale: float = 1.0,
    z_margin: float = 0.3,
    seed: int = 42,
) -> dict:
    """Simulate stacked fluxes for every (run, z-bin, property-bin).

    Per-source realization: true redshifts are drawn from dN/dz (or taken
    from catalog_z), photo-z's add Gaussian σ_z0·(1+z) scatter plus a
    catastrophic-outlier fraction f_cat (uniform over the survey range);
    sources are assigned to dither bins by photo-z; each bin's noiseless
    flux is the mean true in-band flux of its members. Noise is drawn from
    the full shared-source covariance (Cholesky), per property bin.

    Returns {"df", "cov", "true_params", "scheme"}; "cov" maps
    prop_bin_id → covariance over that property bin's flattened
    (bin, band) vector, row order matching the df subset.
    """
    rng = np.random.default_rng(seed)
    noise = noise or NoiseModel()
    if dndz is None and catalog_z is None:
        dndz = make_dndz("cosmos_like")
    bt = scheme.bin_table()
    n_bins = len(bt)
    bands = scheme.bands
    n_bands = len(bands)
    m_props = len(scheme.property_bins)
    n_per_prop = n_total // m_props

    # sources can scatter into the scheme from just outside it
    z_lo_pool = max(1e-3, scheme.z_min - z_margin)
    z_hi_pool = scheme.z_max + z_margin
    z_fine = np.linspace(z_lo_pool, z_hi_pool, 1500)

    n_runs = len(scheme.runs)

    frames = []
    cov_per_prop: dict[int, NDArray[np.float64]] = {}
    for m, prop in enumerate(scheme.property_bins):
        if catalog_z is not None:
            pool = np.asarray(catalog_z, dtype=float)
            pool = pool[(pool >= z_lo_pool) & (pool <= z_hi_pool)]
            z_true = rng.choice(pool, size=n_per_prop, replace=True)
        else:
            z_true = _draw_redshifts(rng, n_per_prop, dndz, z_lo_pool, z_hi_pool)
        z_phot = z_true + rng.normal(0.0, sigma_z0 * (1.0 + z_true))
        if f_cat > 0:
            is_cat = rng.random(n_per_prop) < f_cat
            z_phot[is_cat] = rng.uniform(scheme.z_min, scheme.z_max, is_cat.sum())

        flux_src = {
            band: np.interp(z_true, z_fine, truth.band_flux_curve(z_fine, band, prop))
            for band in bands
        }

        # bin membership by photo-z, per run (bins within a run are disjoint)
        bin_idx_per_run = []
        for edges in scheme.runs:
            idx = np.searchsorted(edges, z_phot, side="right") - 1
            idx[(z_phot < edges[0]) | (z_phot >= edges[-1])] = -1
            bin_idx_per_run.append(idx)

        n_src = np.zeros(n_bins)
        mean_flux = np.full((n_bins, n_bands), np.nan)
        row0_of_run = np.concatenate(
            [[0], np.cumsum([len(e) - 1 for e in scheme.runs])]
        )
        for r, idx in enumerate(bin_idx_per_run):
            nb = len(scheme.runs[r]) - 1
            counts = np.bincount(idx[idx >= 0], minlength=nb)
            n_src[row0_of_run[r] : row0_of_run[r] + nb] = counts
            for b, band in enumerate(bands):
                sums = np.bincount(
                    idx[idx >= 0], weights=flux_src[band][idx >= 0], minlength=nb
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_flux[row0_of_run[r] : row0_of_run[r] + nb, b] = np.where(
                        counts > 0, sums / np.maximum(counts, 1), np.nan
                    )

        # exact shared-source counts between runs (joint 2-D histograms)
        shared = np.zeros((n_bins, n_bins))
        for r in range(n_runs):
            s0 = row0_of_run[r]
            nb_r = len(scheme.runs[r]) - 1
            shared[s0 : s0 + nb_r, s0 : s0 + nb_r] = np.diag(n_src[s0 : s0 + nb_r])
            for t in range(r + 1, n_runs):
                t0 = row0_of_run[t]
                nb_t = len(scheme.runs[t]) - 1
                both = (bin_idx_per_run[r] >= 0) & (bin_idx_per_run[t] >= 0)
                joint = np.zeros((nb_r, nb_t))
                if both.any():
                    flat = bin_idx_per_run[r][both] * nb_t + bin_idx_per_run[t][both]
                    joint = np.bincount(flat, minlength=nb_r * nb_t).reshape(nb_r, nb_t)
                shared[s0 : s0 + nb_r, t0 : t0 + nb_t] = joint
                shared[t0 : t0 + nb_t, s0 : s0 + nb_r] = joint.T

        cov = noise.covariance(n_src, shared, bands)
        cov_per_prop[m] = cov

        flux = mean_flux.copy()
        if noise_scale > 0:
            # jitter keeps Cholesky alive when runs coincide (corr → 1 exactly)
            jitter = 1e-10 * np.mean(np.diag(cov)) * np.eye(len(cov))
            chol = np.linalg.cholesky(cov + jitter)
            draw = noise_scale * (chol @ rng.standard_normal(len(cov)))
            flux = flux + draw.reshape(n_bins, n_bands)

        df_m = bt.copy()
        df_m["prop_bin_id"] = m
        for key, val in prop.items():
            df_m[key] = val
        df_m["n_sources"] = n_src.astype(int)
        for b, band in enumerate(bands):
            sig = noise.sigma_per_bin(band, np.maximum(n_src, 1.0))
            df_m[band] = flux[:, b]
            df_m[f"{band}_err"] = np.where(n_src > 0, sig, np.nan)
        frames.append(df_m)

    df = pd.concat(frames, ignore_index=True)
    true_params = {
        "A_per_prop": np.vstack([truth.amplitudes(p) for p in scheme.property_bins]),
        "C": truth.continuum_amp,
        "T_warm": truth.T_warm,
        "beta_warm": truth.beta_warm,
        "beta_mass": truth.beta_mass,
        "beta_sigma": truth.beta_sigma,
        "sigma_z0": sigma_z0,
        "f_cat": f_cat,
    }
    return {"df": df, "cov": cov_per_prop, "true_params": true_params, "scheme": scheme}


# ---------------------------------------------------------------------------
# Fisher / CRLB strategy evaluation
# ---------------------------------------------------------------------------


@dataclass
class FisherResult:
    """Cramér–Rao bounds on the feature-group amplitudes for one scheme.

    Two flavors of bound, matching solve_linear_amplitudes:
    - crlb_flux: marginal 1σ on the absolute feature peak flux Ã_g (map
      units) — detection-oriented; snr = Ã_true/crlb_flux is the feature
      detection significance.
    - crlb: 1σ on the ratio A_g = Ã_g/C by the delta method, which
      includes the continuum amplitude's uncertainty — the ratio can be
      much worse than the flux bound when C is poorly pinned (the warm
      continuum is faint in-band over most of the z range).
    """

    fisher: NDArray[np.float64]  # (G+1, G+1) full Fisher of [C, Ã_1..Ã_G]
    crlb: NDArray[np.float64]  # (G,) marginal 1σ bounds on ratio A_g
    crlb_flux: NDArray[np.float64]  # (G,) marginal 1σ bounds on Ã_g
    snr: NDArray[np.float64]  # (G,) detection SNR = Ã_true / crlb_flux
    C_err: float  # marginal 1σ on the continuum amplitude C
    corr: NDArray[np.float64]  # (G+1, G+1) param correlation matrix
    cond: float  # condition number of the whitened design
    eigvals: NDArray[np.float64]  # amplitude-block Fisher eigenvalues
    eigvecs: NDArray[np.float64]  # degeneracy directions (columns)
    n_eff: NDArray[np.float64]  # (n_bins,) expected sources per dither bin
    labels: list[str]  # parameter names ["C", "A(6.2)", ...]


def _group_labels(
    features: list[PAHFeature], feature_groups: list[list[int]]
) -> list[str]:
    return [
        "A(" + "+".join(f"{features[j][0]:g}" for j in grp) + ")"
        for grp in feature_groups
    ]


def fisher_for_scheme(
    scheme: DitherScheme,
    truth: TruthSpectrum | None = None,
    noise: NoiseModel | None = None,
    *,
    dndz: Callable | None = None,
    sigma_z0: float = 0.01,
    f_cat: float = 0.0,
    n_total: int = 200_000,
) -> FisherResult:
    """Marginal CRLB on feature amplitudes for a dither scheme — no fitting.

    Builds the photo-z-smeared design matrix X = [W | K], the shared-source
    noise covariance Σ, and inverts the Fisher matrix XᵀΣ⁻¹X. Because the
    model is linear in (C, Ã), the bounds are exact, not local — this is
    the principled figure of merit for dithering-strategy optimization.
    Evaluated for one property bin holding n_total / n_property_bins
    sources (CRLBs scale with sqrt of the property-split factor).
    """
    truth = truth or TruthSpectrum()
    noise = noise or NoiseModel()
    if dndz is None:
        dndz = make_dndz("cosmos_like")
    n_per_prop = n_total // len(scheme.property_bins)

    pz, z_grid = compute_pz_matrix(scheme, dndz, sigma_z0=sigma_z0, f_cat=f_cat)
    K = build_design_matrix(
        pz, z_grid, scheme.bands, truth.features, truth.feature_groups
    )
    W = warm_continuum_kernel(
        pz, z_grid, scheme.bands, T_w=truth.T_warm, beta_w=truth.beta_warm
    )

    frac = shared_fraction_matrix(scheme, dndz)
    n_eff = np.diag(frac) * n_per_prop
    shared = frac * n_per_prop
    cov = noise.covariance(n_eff, shared, scheme.bands)

    G = K.shape[-1]
    X = np.column_stack([W.ravel(), K.reshape(-1, G)])
    # whiten with jitter (identical runs make Σ exactly singular)
    jitter = 1e-10 * np.mean(np.diag(cov)) * np.eye(len(cov))
    L = np.linalg.cholesky(cov + jitter)
    Xw = np.linalg.solve(L, X)
    fisher = Xw.T @ Xw

    # eigen-regularized inverse: a feature that never enters any band makes
    # the Fisher exactly singular — its marginal bound should come out huge,
    # not crash the inversion
    f_eval, f_evec = np.linalg.eigh(fisher)
    floor = 1e-12 * f_eval.max()
    param_cov = f_evec @ np.diag(1.0 / np.clip(f_eval, floor, None)) @ f_evec.T
    sig = np.sqrt(np.diag(param_cov))
    corr = param_cov / np.outer(sig, sig)
    C_true = truth.continuum_amp
    A_true = truth.amplitudes(scheme.property_bins[0])
    A_tilde_true = C_true * A_true

    crlb_flux = sig[1:]
    # ratio A = Ã/C — delta method, same as solve_linear_amplitudes
    J = np.zeros((G, G + 1))
    J[:, 0] = -A_tilde_true / C_true**2
    J[:, 1:] = np.eye(G) / C_true
    crlb = np.sqrt(np.diag(J @ param_cov @ J.T))
    snr = np.where(crlb_flux > 0, A_tilde_true / crlb_flux, np.inf)

    eigvals, eigvecs = np.linalg.eigh(fisher[1:, 1:])
    labels = ["C"] + _group_labels(truth.features, truth.feature_groups)
    return FisherResult(
        fisher=fisher,
        crlb=crlb,
        crlb_flux=crlb_flux,
        snr=snr,
        C_err=float(sig[0]),
        corr=corr,
        cond=float(np.linalg.cond(Xw)),
        eigvals=eigvals,
        eigvecs=eigvecs,
        n_eff=n_eff,
        labels=labels,
    )


def sweep_strategies(
    param_grid: dict[str, Sequence],
    truth: TruthSpectrum | None = None,
    noise: NoiseModel | None = None,
    *,
    dndz: Callable | None = None,
    n_total: int = 200_000,
    z_min: float = 0.5,
    z_max: float = 3.5,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    verbose: bool = False,
) -> pd.DataFrame:
    """CRLB grid over dithering-strategy parameters — answers problem 1.

    param_grid keys (each a sequence of values to sweep):
        dz, n_stagger, sigma_z0, f_cat, n_property_bins
    Each combination builds a uniform DitherScheme and computes its
    FisherResult. Returns one row per combination with the marginal CRLB
    and SNR per feature group plus D-optimality (logdet) and conditioning.
    """
    truth = truth or TruthSpectrum()
    grid = {
        "dz": list(param_grid.get("dz", [0.15])),
        "n_stagger": list(param_grid.get("n_stagger", [3])),
        "sigma_z0": list(param_grid.get("sigma_z0", [0.01])),
        "f_cat": list(param_grid.get("f_cat", [0.0])),
        "n_property_bins": list(param_grid.get("n_property_bins", [1])),
    }
    labels = _group_labels(truth.features, truth.feature_groups)
    rows = []
    from itertools import product

    for dz, n_st, sz0, fc, n_props in product(*grid.values()):
        scheme = DitherScheme.uniform(
            z_min=z_min,
            z_max=z_max,
            dz=dz,
            n_stagger=n_st,
            property_bins=[
                {"log_M_star": 10.5, "log_sigma_sfr": 0.0} for _ in range(n_props)
            ],
            bands=bands,
        )
        fr = fisher_for_scheme(
            scheme,
            truth,
            noise,
            dndz=dndz,
            sigma_z0=sz0,
            f_cat=fc,
            n_total=n_total,
        )
        row = {
            "dz": dz,
            "n_stagger": n_st,
            "sigma_z0": sz0,
            "f_cat": fc,
            "n_property_bins": n_props,
            "n_zbins": scheme.n_zbins,
            "snr_min": float(fr.snr.min()),
            "snr_mean": float(fr.snr.mean()),
            "logdet_fisher": float(np.linalg.slogdet(fr.fisher[1:, 1:])[1]),
            "cond": fr.cond,
        }
        for lab, c, s in zip(labels, fr.crlb, fr.snr, strict=True):
            row[f"crlb {lab}"] = c
            row[f"snr {lab}"] = s
        rows.append(row)
        if verbose:
            print(
                f"dz={dz:.3f} n_stagger={n_st} sigma_z0={sz0} "
                f"-> snr_min={row['snr_min']:.1f}"
            )
    return pd.DataFrame(rows)


def injection_recovery_sweep(
    schemes: Sequence[DitherScheme],
    truth: TruthSpectrum | None = None,
    *,
    n_realizations: int = 20,
    n_total: int = 200_000,
    sigma_z0: float = 0.01,
    f_cat: float = 0.0,
    noise: NoiseModel | None = None,
    dndz: Callable | None = None,
    ridge: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Monte-Carlo confirmation of the Fisher predictions.

    For each scheme: simulate n_realizations noisy datasets, fit each with
    the matched GLS kernel, and tabulate bias / scatter / mean reported
    error per feature group next to the CRLB. Scatter ≈ CRLB and
    |bias| ≪ scatter validate the strategy ranking from sweep_strategies.
    """
    truth = truth or TruthSpectrum()
    noise = noise or NoiseModel()
    if dndz is None:
        dndz = make_dndz("cosmos_like")
    labels = _group_labels(truth.features, truth.feature_groups)
    rows = []
    for s_id, scheme in enumerate(schemes):
        fr = fisher_for_scheme(
            scheme,
            truth,
            noise,
            dndz=dndz,
            sigma_z0=sigma_z0,
            f_cat=f_cat,
            n_total=n_total,
        )
        pz, z_grid = compute_pz_matrix(scheme, dndz, sigma_z0=sigma_z0, f_cat=f_cat)
        K = build_design_matrix(
            pz, z_grid, scheme.bands, truth.features, truth.feature_groups
        )
        W = warm_continuum_kernel(
            pz, z_grid, scheme.bands, T_w=truth.T_warm, beta_w=truth.beta_warm
        )
        A_fit, A_err = [], []
        for r in range(n_realizations):
            sim = simulate_dithered_fluxes(
                scheme,
                truth,
                dndz=dndz,
                n_total=n_total,
                sigma_z0=sigma_z0,
                f_cat=f_cat,
                noise=noise,
                seed=seed + 1000 * s_id + r,
            )
            sub = sim["df"][sim["df"]["prop_bin_id"] == 0]
            F = sub[list(scheme.bands)].to_numpy()
            res = solve_linear_amplitudes(F, K, W, cov=sim["cov"][0], ridge=ridge)
            A_fit.append(res.A)
            A_err.append(res.A_err)
        A_fit = np.array(A_fit)
        A_err = np.array(A_err)
        A_true = truth.amplitudes(scheme.property_bins[0])
        for g, lab in enumerate(labels):
            rows.append(
                {
                    "scheme_id": s_id,
                    "group": lab,
                    "A_true": A_true[g],
                    "bias": float(A_fit[:, g].mean() - A_true[g]),
                    "scatter": float(A_fit[:, g].std(ddof=1)),
                    "mean_reported_err": float(A_err[:, g].mean()),
                    "crlb": float(fr.crlb[g]),
                }
            )
    return pd.DataFrame(rows)
