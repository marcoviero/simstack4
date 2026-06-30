"""
Forward-model deconvolution of PAH pseudo-spectra from dithered stacking.

Dithered stacking measures broadband MIPS 24/70 µm fluxes in many offset
fine redshift bins, so rest-frame PAH features sweep through the bandpasses
and modulate the stacked flux with z. This module inverts those
measurements into feature strengths with a linear kernel formulation:

    F_i,b = C_m · W_b(z_i; T_w, β_w) + Σ_g Ã_g,m · K_ib,g       (Ã ≡ C·A)

    K_ib,g = Σ_k p_i(z_k) · T_g,b(z_k)
    T_g,b(z) = ∫ BP_b(λ_obs) · G_g(λ_obs/(1+z)) dλ / ∫ BP_b dλ

where p_i(z) is the (photo-z smeared) redshift distribution of dither bin
i, G_g is the unit-peak Gaussian template of feature group g, and W_b is
the bandpass-integrated warm modified-blackbody continuum. With the
global shape parameters (T_w, β_w) fixed, the per-property-bin continuum
amplitude C_m and feature amplitudes Ã_g,m are linear and solve by GLS —
the same profiled-amplitude pattern as DustEvolutionModel. Feature
amplitudes A_g = Ã_g/C are peak feature-to-continuum ratios in f_ν.

Design choices that address why earlier approaches were unsatisfying:
- the continuum baseline is a *physical* warm MBB Wien tail, not a free
  polynomial in z (free polynomials absorb the 7.7 µm bump, which spans
  Δz ≈ 1.8 through MIPS 24 — the failure mode of detrend-then-fit);
- photo-z smearing enters the kernel through p_i(z), so dithers finer
  than σ_z(1+z) are correctly down-weighted instead of biasing the fit.

The companion module pah_dither.py provides the simulator, dither-scheme
abstraction, photo-z matrix builder, and Fisher/CRLB strategy evaluation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dust_evolution import _greybody_nu
from .pah_bandpass import get_bandpass

_c_um_hz = 2.998e14  # speed of light [µm·Hz]

# (center_um, relative_strength, fwhm_um) — values mirror the frozen
# reference lists pah_model.PAH_FEATURES (+ 16.4/17.0 from
# PAH_FEATURES_MINOR, included by default because MIPS 70 probes
# rest-frame 15–47 µm over z = 0.5–3.5).
PAHFeature = tuple[float, float, float]

DEFAULT_FEATURES: list[PAHFeature] = [
    (6.2, 0.1262, 0.19),  # C-C stretch
    (7.7, 0.4577, 0.70),  # C-C stretch (strongest)
    (8.6, 0.6089, 0.34),  # C-H in-plane bend
    (11.3, 0.30, 0.24),  # C-H out-of-plane bend
    (12.7, 0.5187, 0.45),  # C-H out-of-plane bend
    (16.4, 0.10, 0.20),  # C-H/C-C bend
    (17.0, 0.08, 0.30),  # C-C-C bend
]

# 7.7+8.6 and 16.4+17.0 are kernel-blended (the rest-frame kernel width
# at z=2 is ~4.7 µm, far wider than their separations) so they share a
# group amplitude; 6.2, 11.3 and 12.7 stay separable in principle and
# get their own groups — the Fisher column correlations say whether a
# given dither scheme actually separates them.
DEFAULT_GROUPS: list[list[int]] = [[0], [1, 2], [3], [4], [5, 6]]

DEFAULT_BANDS: tuple[str, ...] = ("MIPS_24", "MIPS_70")


def group_weights(
    features: list[PAHFeature], groups: list[list[int]]
) -> list[NDArray[np.float64]]:
    """Within-group feature weights, normalized so the strongest is 1.

    A single-feature group always gets weight 1 (its amplitude IS the
    feature strength); multi-feature groups keep the catalog strength
    ratios so the group amplitude scales the strongest member's peak.
    """
    weights = []
    for grp in groups:
        w = np.array([features[j][1] for j in grp], dtype=float)
        if len(grp) == 1 or w.max() <= 0:
            w = np.ones(len(grp))
        else:
            w = w / w.max()
        weights.append(w)
    return weights


def feature_band_curves(
    z_grid: NDArray[np.float64],
    band: str,
    features: list[PAHFeature] | None = None,
    feature_groups: list[list[int]] | None = None,
) -> NDArray[np.float64]:
    """Bandpass-integrated feature-group templates T_g,b(z).

    Returns (n_z, G): the mean in-band response to a unit-peak feature
    group at each redshift. This is the sharp-z building block of the
    design matrix; photo-z smearing is applied afterwards via p_i(z).
    """
    features = DEFAULT_FEATURES if features is None else features
    feature_groups = DEFAULT_GROUPS if feature_groups is None else feature_groups
    weights = group_weights(features, feature_groups)
    bp = get_bandpass(band)

    # (n_z, n_fine) rest-frame wavelengths probed by the band at each z
    lam_rest = bp.lam_fine[None, :] / (1.0 + np.asarray(z_grid)[:, None])

    curves = np.zeros((len(z_grid), len(feature_groups)))
    for g, (grp, w) in enumerate(zip(feature_groups, weights, strict=False)):
        spec = np.zeros_like(lam_rest)
        for j, wj in zip(grp, w, strict=False):
            center, _, fwhm = features[j]
            sigma = fwhm / 2.355
            spec += wj * np.exp(-0.5 * ((lam_rest - center) / sigma) ** 2)
        curves[:, g] = np.trapezoid(spec * bp.resp_fine, bp.lam_fine, axis=1) / bp.norm
    return curves


def warm_band_curve(
    z_grid: NDArray[np.float64],
    band: str,
    T_w: float = 60.0,
    beta_w: float = 1.5,
) -> NDArray[np.float64]:
    """Bandpass-integrated warm MBB continuum W_b(z), peak-normalized in f_ν.

    Returns (n_z,): the mean in-band response to the rest-frame warm
    modified blackbody (peak-normalized, same convention as
    dust_evolution._greybody_nu) at each redshift.
    """
    bp = get_bandpass(band)
    lam_rest = bp.lam_fine[None, :] / (1.0 + np.asarray(z_grid)[:, None])
    nu_rest = _c_um_hz / lam_rest
    sed = _greybody_nu(nu_rest, T_w, beta_w)
    return np.trapezoid(sed * bp.resp_fine, bp.lam_fine, axis=1) / bp.norm


def build_design_matrix(
    pz_matrix: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    bands: tuple[str, ...] = DEFAULT_BANDS,
    features: list[PAHFeature] | None = None,
    feature_groups: list[list[int]] | None = None,
) -> NDArray[np.float64]:
    """Feature kernel matrix K[i, b, g] = Σ_k p_i(z_k) T_g,b(z_k).

    pz_matrix rows are discrete probability masses over z_grid (each row
    sums to 1); they encode bin width, dN/dz weighting, photo-z smearing
    and any catastrophic-outlier pedestal. Returns (n_bins, n_bands, G).
    """
    pz = np.asarray(pz_matrix, dtype=float)
    n_groups = len(DEFAULT_GROUPS if feature_groups is None else feature_groups)
    K = np.zeros((pz.shape[0], len(bands), n_groups))
    for b, band in enumerate(bands):
        K[:, b, :] = pz @ feature_band_curves(z_grid, band, features, feature_groups)
    return K


def warm_continuum_kernel(
    pz_matrix: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    bands: tuple[str, ...] = DEFAULT_BANDS,
    T_w: float = 60.0,
    beta_w: float = 1.5,
) -> NDArray[np.float64]:
    """Continuum kernel W[i, b] = Σ_k p_i(z_k) W_b(z_k). Returns (n_bins, n_bands)."""
    pz = np.asarray(pz_matrix, dtype=float)
    W = np.zeros((pz.shape[0], len(bands)))
    for b, band in enumerate(bands):
        W[:, b] = pz @ warm_band_curve(z_grid, band, T_w, beta_w)
    return W


@dataclass
class LinearSolveResult:
    """GLS solution for one property bin: continuum + feature amplitudes."""

    C: float  # continuum amplitude C_m
    C_err: float
    A: NDArray[np.float64]  # (G,) feature/continuum peak ratios A_g = Ã_g/C
    A_err: NDArray[np.float64]
    A_cov: NDArray[np.float64]  # (G, G) covariance of A (delta method)
    theta: NDArray[np.float64]  # raw linear solution [C, Ã_1..Ã_G]
    theta_cov: NDArray[np.float64]
    chi2: float
    dof: int
    residuals: NDArray[np.float64]  # (n_valid,) whitened-units residuals F - Xθ
    mask: NDArray[np.bool_]  # valid (finite flux) entries of the flattened data


def solve_linear_amplitudes(
    F_obs: NDArray[np.float64],
    K: NDArray[np.float64],
    W: NDArray[np.float64],
    sigma: NDArray[np.float64] | None = None,
    cov: NDArray[np.float64] | None = None,
    ridge: float = 0.0,
) -> LinearSolveResult:
    """Profile out the linear amplitudes (C, Ã_g) for one property bin by GLS.

    F_obs : (n_bins, n_bands) stacked fluxes (NaN = band unavailable)
    K     : (n_bins, n_bands, G) feature kernel from build_design_matrix
    W     : (n_bins, n_bands) continuum kernel from warm_continuum_kernel
    sigma : (n_bins, n_bands) per-point errors (diagonal noise), or
    cov   : (n_flat, n_flat) full covariance of F_obs.ravel() — use this to
            propagate the shared-source correlation between staggered runs
    ridge : Tikhonov term on the feature amplitudes only (not C), in units
            of the GLS normal matrix diagonal — stabilizes blended groups

    Exactly one of sigma/cov must be given unless both are None (unit
    weights). Feature ratios A = Ã/C carry full delta-method errors.
    """
    F_flat = np.asarray(F_obs, dtype=float).ravel()
    G = K.shape[-1]
    X_full = np.column_stack([W.ravel(), K.reshape(-1, G)])
    mask = np.isfinite(F_flat)
    F_v = F_flat[mask]
    X = X_full[mask]

    if cov is not None:
        if sigma is not None:
            raise ValueError("give either sigma or cov, not both")
        cov_v = np.asarray(cov, dtype=float)[np.ix_(mask, mask)]
        L = np.linalg.cholesky(cov_v)
        Xw = np.linalg.solve(L, X)
        Fw = np.linalg.solve(L, F_v)
    elif sigma is not None:
        s = np.asarray(sigma, dtype=float).ravel()[mask]
        Xw = X / s[:, None]
        Fw = F_v / s
    else:
        Xw, Fw = X, F_v

    H = Xw.T @ Xw
    if ridge > 0.0:
        penalty = np.zeros(G + 1)
        penalty[1:] = ridge * np.diag(H)[1:]
        H = H + np.diag(penalty)
    theta_cov = np.linalg.inv(H)
    theta = theta_cov @ (Xw.T @ Fw)

    resid = Fw - Xw @ theta
    chi2 = float(resid @ resid)
    dof = int(mask.sum()) - (G + 1)

    C = float(theta[0])
    A_tilde = theta[1:]
    A = A_tilde / C
    # delta method: J[g, :] = ∂A_g/∂θ with ∂A_g/∂C = -Ã_g/C², ∂A_g/∂Ã_g = 1/C
    J = np.zeros((G, G + 1))
    J[:, 0] = -A_tilde / C**2
    J[:, 1:] = np.eye(G) / C
    A_cov = J @ theta_cov @ J.T

    return LinearSolveResult(
        C=C,
        C_err=float(np.sqrt(theta_cov[0, 0])),
        A=A,
        A_err=np.sqrt(np.diag(A_cov)),
        A_cov=A_cov,
        theta=theta,
        theta_cov=theta_cov,
        chi2=chi2,
        dof=dof,
        residuals=resid,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Forward-model fitter
# ---------------------------------------------------------------------------


@dataclass
class PAHSpectrumResult:
    """Output of PAHSpectrumModel.fit_lstsq / fit_mcmc.

    A and A_err are (M, G): feature/continuum peak ratios per property bin.
    For fit_lstsq they are free per-bin GLS estimates; for fit_mcmc they
    are the pooled-model posteriors evaluated per bin. theta_global holds
    the nonlinear/hyper parameters (see param_names).
    """

    A: NDArray[np.float64]
    A_err: NDArray[np.float64]
    C_per_bin: NDArray[np.float64]  # (M,) continuum amplitudes
    C_err_per_bin: NDArray[np.float64]
    theta_global: NDArray[np.float64]
    theta_err: NDArray[np.float64]
    param_names: list[str]
    chi2: float
    dof: int
    labels: list[str]  # feature-group labels
    per_bin: list[LinearSolveResult] | None = None  # fit_lstsq only
    sampler: object | None = None  # emcee sampler (fit_mcmc only)
    acceptance_fraction: float = 0.0

    @property
    def chi2_red(self) -> float:
        return self.chi2 / max(self.dof, 1)


def _scheme_from_df(df: "pd.DataFrame") -> "object":
    """Reconstruct a DitherScheme from a stacked-flux DataFrame.

    Needs columns run_id, z_lo, z_hi (and optionally prop_bin_id plus
    property-value columns). Lets the fitter run on real stacking output
    loaded from JSON without the original DitherScheme object.
    """
    from .pah_dither import DitherScheme

    runs = []
    for _, sub in df.drop_duplicates(["run_id", "z_lo", "z_hi"]).groupby("run_id"):
        sub = sub.sort_values("z_lo")
        edges = np.append(sub["z_lo"].to_numpy(), sub["z_hi"].to_numpy()[-1])
        runs.append(edges)
    property_bins = [{"log_M_star": 10.5, "log_sigma_sfr": 0.0}]
    if "prop_bin_id" in df.columns:
        props = df.drop_duplicates("prop_bin_id").sort_values("prop_bin_id")
        property_bins = [
            {
                key: float(row[key])
                for key in ("log_M_star", "log_sigma_sfr")
                if key in df.columns
            }
            for _, row in props.iterrows()
        ]
    return DitherScheme(runs=runs, property_bins=property_bins)


class PAHSpectrumModel:
    """Forward-model fitter for dithered-stacking pseudo-spectra.

    Twin of DustEvolutionModel: nonlinear globals are kept to a minimum
    (the warm-continuum temperature T_w and, in fit_mcmc, the pooled
    amplitude-evolution parameters), while everything linear — per-bin
    continuum amplitudes C_m and feature amplitudes — is profiled
    analytically inside the likelihood.

    Typical use
    -----------
    model = PAHSpectrumModel(sigma_z0=0.01)
    res = model.fit_lstsq(sim["df"], cov=sim["cov"])     # free amplitudes
    res = model.fit_mcmc(sim["df"], cov=sim["cov"])      # pooled evolution
    spec = model.pseudo_spectrum(sim["df"], res)          # for overlays
    """

    T_GRID = np.arange(35.0, 105.0 + 1e-9, 0.5)  # warm-T interpolation grid

    def __init__(
        self,
        features: list[PAHFeature] | None = None,
        feature_groups: list[list[int]] | None = None,
        bands: tuple[str, ...] = DEFAULT_BANDS,
        T_w_prior: tuple[float, float] = (60.0, 10.0),
        T_w_bounds: tuple[float, float] = (40.0, 100.0),
        beta_w: float = 1.5,
        sigma_z0: float = 0.01,
        f_cat: float = 0.0,
        ridge: float = 0.0,
        beta_prior_sigma: float = 1.0,
        log_a0_bounds: tuple[float, float] = (-3.0, 1.0),
        pivot_log_mass: float = 10.5,
        pivot_log_sigma_sfr: float = 0.0,
    ):
        self.features = DEFAULT_FEATURES if features is None else features
        self.feature_groups = (
            DEFAULT_GROUPS if feature_groups is None else feature_groups
        )
        self.bands = bands
        self.T_w_prior = T_w_prior
        self.T_w_bounds = T_w_bounds
        self.beta_w = beta_w
        self.sigma_z0 = sigma_z0
        self.f_cat = f_cat
        self.ridge = ridge
        self.beta_prior_sigma = beta_prior_sigma
        self.log_a0_bounds = log_a0_bounds
        self.pivot_log_mass = pivot_log_mass
        self.pivot_log_sigma_sfr = pivot_log_sigma_sfr
        self.n_groups = len(self.feature_groups)

    # -- kernel/whitening preparation ------------------------------------

    def _prepare(
        self,
        df,
        cov,
        scheme,
        dndz,
        sigma_z0,
        f_cat,
        baseline_col=None,
        baseline_cols=None,
        ssfr_col=None,
    ):
        """Build whitened per-property-bin data and kernel structures.

        ``baseline_col`` stashes one cold-baseline column per bin (fit_shared).
        ``baseline_cols`` (dict band→column) and ``ssfr_col`` additionally stash
        a per-band baseline map, the per-row ``z_mid`` and the per-row log sSFR
        used by the multi-band, sSFR-anchored :meth:`fit_evolving`.
        """
        from .pah_dither import compute_pz_matrix, make_dndz

        if scheme is None:
            scheme = _scheme_from_df(df)
        if dndz is None:
            dndz = make_dndz("cosmos_like")
        sigma_z0 = self.sigma_z0 if sigma_z0 is None else sigma_z0
        f_cat = self.f_cat if f_cat is None else f_cat

        pz, z_grid = compute_pz_matrix(scheme, dndz, sigma_z0=sigma_z0, f_cat=f_cat)
        K = build_design_matrix(
            pz, z_grid, self.bands, self.features, self.feature_groups
        )
        # warm-continuum kernel tabulated over T_GRID for fast interpolation
        W_grid = np.stack(
            [
                warm_continuum_kernel(pz, z_grid, self.bands, T_w=t, beta_w=self.beta_w)
                for t in self.T_GRID
            ]
        )  # (n_T, n_scheme_bins, n_bands)

        # Lookup: (run_id, z_lo) → row index in the scheme's pz matrix.
        # Different property bins may cover different subsets of z-bins (due to
        # quality filtering), so we must slice K/W_grid per bin rather than using
        # the global arrays directly.
        bt = scheme.bin_table()
        scheme_lookup = {
            (int(row.run_id), round(float(row.z_lo), 8)): i for i, row in bt.iterrows()
        }

        prop_ids = sorted(df["prop_bin_id"].unique()) if "prop_bin_id" in df else [0]
        bins = []
        for m in prop_ids:
            sub = (
                df[df["prop_bin_id"] == m] if "prop_bin_id" in df else df
            ).sort_values(["run_id", "z_lo"])

            # Map each sub row to its scheme row index; drop any that don't match.
            sidx = np.array(
                [
                    scheme_lookup.get(
                        (int(r["run_id"]), round(float(r["z_lo"]), 8)), -1
                    )
                    for _, r in sub.iterrows()
                ]
            )
            valid = sidx >= 0
            if not valid.all():
                sub = sub.iloc[valid].reset_index(drop=True)
                sidx = sidx[valid]

            # Per-bin kernel slices: (n_rows_m, n_bands, G) and (n_T, n_rows_m, n_bands)
            K_m = K[sidx]
            W_grid_m = W_grid[:, sidx, :]

            F = sub[list(self.bands)].to_numpy()
            mask = np.isfinite(F.ravel())
            if cov is not None:
                cov_m = cov[m] if isinstance(cov, dict) else cov
                cov_v = np.asarray(cov_m)[np.ix_(mask, mask)]
                jitter = 1e-10 * np.mean(np.diag(cov_v)) * np.eye(len(cov_v))
                L = np.linalg.cholesky(cov_v + jitter)
            else:
                sig = sub[[f"{b}_err" for b in self.bands]].to_numpy().ravel()[mask]
                L = np.diag(sig)
            G = self.n_groups
            Fw = np.linalg.solve(L, F.ravel()[mask])
            Kw = np.linalg.solve(L, K_m.reshape(-1, G)[mask])
            Ww_grid = np.stack(
                [
                    np.linalg.solve(L, W_grid_m[t].reshape(-1)[mask])
                    for t in range(len(self.T_GRID))
                ]
            )
            props = {
                "log_M_star": (
                    float(sub["log_M_star"].iloc[0])
                    if "log_M_star" in sub
                    else self.pivot_log_mass
                ),
                "log_sigma_sfr": (
                    float(sub["log_sigma_sfr"].iloc[0])
                    if "log_sigma_sfr" in sub
                    else self.pivot_log_sigma_sfr
                ),
            }
            # Per-row band errors and (optionally) the cold-baseline column, kept
            # aligned with K_m for the shared-ratio fit (fit_shared).
            Ferr = sub[[f"{b}_err" for b in self.bands]].to_numpy()
            f_cold = (
                sub[baseline_col].to_numpy()
                if (baseline_col is not None and baseline_col in sub.columns)
                else None
            )
            # Multi-band / sSFR extras for fit_evolving (aligned with K_m rows).
            z_mid = (
                sub["z_mid"].to_numpy()
                if "z_mid" in sub
                else 0.5 * (sub["z_lo"].to_numpy() + sub["z_hi"].to_numpy())
            )
            log_ssfr = (
                sub[ssfr_col].to_numpy()
                if (ssfr_col is not None and ssfr_col in sub.columns)
                else None
            )
            f_cold_by_band = None
            if baseline_cols is not None:
                f_cold_by_band = {
                    band: (sub[col].to_numpy() if col in sub.columns else None)
                    for band, col in baseline_cols.items()
                }
            bins.append(
                {
                    "m": m,
                    "F": F,
                    "Ferr": Ferr,
                    "f_cold": f_cold,
                    "f_cold_by_band": f_cold_by_band,
                    "z_mid": z_mid,
                    "log_ssfr": log_ssfr,
                    "mask": mask,
                    "Fw": Fw,
                    "Kw": Kw,
                    "Ww_grid": Ww_grid,
                    "K": K_m,
                    "sidx": sidx,
                    "props": props,
                    "n_valid": int(mask.sum()),
                }
            )
        return {"scheme": scheme, "bins": bins, "K": K, "z_grid": z_grid, "pz": pz}

    def _Ww(self, bin_data, T_w):
        """Whitened warm kernel at T_w by linear interpolation over T_GRID."""
        t = np.clip(
            (T_w - self.T_GRID[0]) / (self.T_GRID[1] - self.T_GRID[0]),
            0,
            len(self.T_GRID) - 1 - 1e-9,
        )
        i0 = int(t)
        frac = t - i0
        return (1 - frac) * bin_data["Ww_grid"][i0] + frac * bin_data["Ww_grid"][i0 + 1]

    def _labels(self) -> list[str]:
        return [
            "A(" + "+".join(f"{self.features[j][0]:g}" for j in grp) + ")"
            for grp in self.feature_groups
        ]

    # -- free-amplitude fit (MAP over T_w, GLS inside) --------------------

    def fit_lstsq(
        self,
        df,
        *,
        cov=None,
        scheme=None,
        dndz=None,
        sigma_z0: float | None = None,
        f_cat: float | None = None,
        fix_T_w: float | None = None,
    ) -> PAHSpectrumResult:
        """Fast MAP fit: free amplitudes per property bin, shared T_w.

        Profiles (C_m, Ã_g,m) by GLS at each trial T_w and minimizes the
        total chi² + T_w prior over the scalar T_w (golden-section via
        scipy). With fix_T_w the linear solve runs once.
        """
        from scipy.optimize import minimize_scalar

        prep = self._prepare(df, cov, scheme, dndz, sigma_z0, f_cat)
        bins = prep["bins"]
        G = self.n_groups

        def solve_at(T_w):
            total_chi2 = 0.0
            results = []
            for b in bins:
                Xw = np.column_stack([self._Ww(b, T_w), b["Kw"]])
                H = Xw.T @ Xw
                if self.ridge > 0:
                    pen = np.zeros(G + 1)
                    pen[1:] = self.ridge * np.diag(H)[1:]
                    H = H + np.diag(pen)
                try:
                    theta_cov = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    # Feature kernels are identically zero for this band/z range
                    # (e.g. MIPS 70 null test at z<0.8 probes rest >38 µm, no PAH).
                    # Pseudoinverse: unconstrained amplitudes map to zero with zero
                    # formal error — caller should treat SNR=nan as "not constrained".
                    theta_cov = np.linalg.pinv(
                        H, rcond=1e-8 * max(np.max(np.abs(H)), 1.0)
                    )
                theta = theta_cov @ (Xw.T @ b["Fw"])
                resid = b["Fw"] - Xw @ theta
                chi2 = float(resid @ resid)
                total_chi2 += chi2
                results.append((theta, theta_cov, chi2, resid))
            mu, sig = self.T_w_prior
            total_chi2 += ((T_w - mu) / sig) ** 2
            return total_chi2, results

        if fix_T_w is not None:
            T_best = float(fix_T_w)
            T_err = 0.0
        else:
            opt = minimize_scalar(
                lambda t: solve_at(t)[0], bounds=self.T_w_bounds, method="bounded"
            )
            T_best = float(opt.x)
            # 1σ from the local curvature of chi²(T_w)
            h = 1.0
            c0, cp, cm = (
                solve_at(T_best)[0],
                solve_at(min(T_best + h, self.T_w_bounds[1]))[0],
                solve_at(max(T_best - h, self.T_w_bounds[0]))[0],
            )
            curv = max((cp + cm - 2 * c0) / h**2, 1e-12)
            T_err = float(np.sqrt(2.0 / curv))

        chi2_prior, results = solve_at(T_best)
        A = np.zeros((len(bins), G))
        A_err = np.zeros_like(A)
        C = np.zeros(len(bins))
        C_err = np.zeros(len(bins))
        per_bin = []
        chi2 = 0.0
        dof = 0
        for i, (b, (theta, theta_cov, chi2_m, resid)) in enumerate(
            zip(bins, results, strict=True)
        ):
            C[i] = theta[0]
            C_err[i] = np.sqrt(max(float(theta_cov[0, 0]), 0.0))
            A_tilde = theta[1:]
            A_cov = np.zeros((G, G))
            if C[i] != 0:
                A[i] = A_tilde / C[i]
                J = np.zeros((G, G + 1))
                J[:, 0] = -A_tilde / C[i] ** 2
                J[:, 1:] = np.eye(G) / C[i]
                A_cov = J @ theta_cov @ J.T
                A_err[i] = np.sqrt(np.maximum(np.diag(A_cov), 0.0))
            else:
                # C=0: design matrix was rank-1 (no PAH features present in
                # this band/z-range, e.g. MIPS 70 null test at z<0.8).
                # Feature amplitudes are unconstrained; report 0 ± NaN.
                A[i] = np.zeros(G)
                A_err[i] = np.full(G, np.nan)
            chi2 += chi2_m
            dof += b["n_valid"] - (G + 1)
            per_bin.append(
                LinearSolveResult(
                    C=float(C[i]),
                    C_err=float(C_err[i]),
                    A=A[i],
                    A_err=A_err[i],
                    A_cov=A_cov,
                    theta=theta,
                    theta_cov=theta_cov,
                    chi2=chi2_m,
                    dof=b["n_valid"] - (G + 1),
                    residuals=resid,
                    mask=b["mask"],
                )
            )
        dof -= 0 if fix_T_w is not None else 1
        return PAHSpectrumResult(
            A=A,
            A_err=A_err,
            C_per_bin=C,
            C_err_per_bin=C_err,
            theta_global=np.array([T_best]),
            theta_err=np.array([T_err]),
            param_names=["T_w"],
            chi2=chi2,
            dof=dof,
            labels=self._labels(),
            per_bin=per_bin,
        )

    # -- shared-ratio fit against a cold-greybody baseline ----------------

    def fit_shared(
        self,
        df,
        *,
        baseline_col="f24_cold",
        band=None,
        cov=None,
        scheme=None,
        dndz=None,
        sigma_z0=None,
        f_cat=None,
        smooth_baseline=False,
        n_iter=200,
        tol=1e-7,
    ):
        """Shared-ratio PAH fit against a cold-greybody baseline.

        Sibling of :meth:`fit_lstsq`. Instead of a warm-MBB continuum with free
        per-group amplitudes, the continuum is the cold-greybody Wien tail
        supplied per row in ``baseline_col`` (scaled by a free per-bin ``C_m``),
        and the feature groups share one global ratio vector ``r`` (``r_0 ≡ 1``)
        with a single per-bin amplitude ``alpha_m``. Model per property bin m,
        point i::

            f_obs = C_m · (f_cold / median f_cold) + alpha_m · Σ_g r_g · K_g(z)

        Solved by alternating WLS (per-bin ``[C_m, alpha_m]`` ↔ global ``r``),
        which breaks the per-group degeneracy of the free-amplitude fit on
        source-correlated MIPS tomography. Reports ``A = alpha/C_m``, the
        PAH/continuum amplitude (NOT divided by ``median f_cold``).

        With ``smooth_baseline=True`` the baseline is first stabilized via
        :func:`smoothed_ms_baseline` (needs ``T_dust``/``log_amp``/``beta``).

        Returns a dict: ``alpha, alpha_err, C_m, C_m_err, r, r_err, A_pah,
        A_pah_err, A, labels, chi2, dof, chi2_red, n_iter, valid``.
        """
        if smooth_baseline:
            df = smoothed_ms_baseline(df, baseline_col=baseline_col)
        band = self.bands[0] if band is None else band
        bidx = list(self.bands).index(band)
        prep = self._prepare(
            df, cov, scheme, dndz, sigma_z0, f_cat, baseline_col=baseline_col
        )
        bins = prep["bins"]
        G = self.n_groups

        data = []
        for b in bins:
            if b["f_cold"] is None:
                raise ValueError(f"baseline column {baseline_col!r} not in df")
            f_obs = b["F"][:, bidx]
            f_err = b["Ferr"][:, bidx]
            f_cold = b["f_cold"]
            K_g = b["K"][:, bidx, :]  # (n_rows, G) photo-z-smeared feature kernel
            ok = (
                np.isfinite(f_obs)
                & np.isfinite(f_err)
                & np.isfinite(f_cold)
                & (f_err > 0)
                & (f_cold > 0)
                & (f_obs > 0)
            )
            if ok.sum() < 3:
                data.append(None)
                continue
            med = float(np.median(f_cold[ok]))
            data.append(
                {
                    "f_obs": f_obs[ok],
                    "w": 1.0 / f_err[ok] ** 2,
                    "f_cold_norm": f_cold[ok] / med,
                    "K": K_g[ok],
                }
            )
        valid = [i for i, d in enumerate(data) if d is not None]
        n_m = len(bins)
        if not valid:
            return None

        def _wls(D, y, w):
            H = D.T @ (w[:, None] * D)
            rhs = D.T @ (w * y)
            try:
                return np.linalg.solve(H, rhs), np.linalg.pinv(H)
            except np.linalg.LinAlgError:
                Hi = np.linalg.pinv(H)
                return Hi @ rhs, Hi

        r = np.ones(G)
        C_m = np.full(n_m, np.nan)
        alpha = np.full(n_m, np.nan)
        n_done = 0
        while n_done < n_iter:
            n_done += 1
            r_prev = r.copy()
            for i in valid:
                d = data[i]
                D = np.column_stack([d["f_cold_norm"], d["K"] @ r])
                theta, _ = _wls(D, d["f_obs"], d["w"])
                C_m[i], alpha[i] = float(theta[0]), float(theta[1])
            if G > 1:
                Ms, ys, ws = [], [], []
                for i in valid:
                    d = data[i]
                    y = d["f_obs"] - C_m[i] * d["f_cold_norm"] - alpha[i] * d["K"][:, 0]
                    Ms.append(alpha[i] * d["K"][:, 1:])
                    ys.append(y)
                    ws.append(d["w"])
                M, y, w = np.vstack(Ms), np.concatenate(ys), np.concatenate(ws)
                r[1:], _ = _wls(M, y, w)
                r[0] = 1.0
            if np.max(np.abs(r - r_prev)) < tol:
                break

        alpha_err = np.full(n_m, np.nan)
        C_m_err = np.full(n_m, np.nan)
        A_pah = np.full(n_m, np.nan)
        A_pah_err = np.full(n_m, np.nan)
        chi2 = 0.0
        ndata = 0
        for i in valid:
            d = data[i]
            D = np.column_stack([d["f_cold_norm"], d["K"] @ r])
            _, cov_i = _wls(D, d["f_obs"], d["w"])
            C_m_err[i] = np.sqrt(max(cov_i[0, 0], 0.0))
            alpha_err[i] = np.sqrt(max(cov_i[1, 1], 0.0))
            if C_m[i] != 0:
                A_pah[i] = alpha[i] / C_m[i]
                g = np.array([-alpha[i] / C_m[i] ** 2, 1.0 / C_m[i]])
                A_pah_err[i] = np.sqrt(max(g @ cov_i @ g, 0.0))
            resid = d["f_obs"] - D @ np.array([C_m[i], alpha[i]])
            chi2 += float(np.sum(resid**2 * d["w"]))
            ndata += len(d["f_obs"])
        r_err = np.zeros(G)
        if G > 1:
            Ms, ws = [], []
            for i in valid:
                d = data[i]
                Ms.append(alpha[i] * d["K"][:, 1:])
                ws.append(d["w"])
            M, w = np.vstack(Ms), np.concatenate(ws)
            cov_r = np.linalg.pinv(M.T @ (w[:, None] * M))
            r_err[1:] = np.sqrt(np.maximum(np.diag(cov_r), 0.0))

        dof = max(1, ndata - (2 * len(valid) + (G - 1)))
        # PAH/continuum ratio per (bin, group): A[m,g] = A_pah[m] * r[g]
        A = A_pah[:, None] * r[None, :]
        return {
            "alpha": alpha,
            "alpha_err": alpha_err,
            "C_m": C_m,
            "C_m_err": C_m_err,
            "r": r,
            "r_err": r_err,
            "A_pah": A_pah,
            "A_pah_err": A_pah_err,
            "A": A,
            "labels": self._labels(),
            "chi2": chi2,
            "dof": dof,
            "chi2_red": chi2 / dof,
            "n_iter": n_done,
            "valid": valid,
        }

    # -- evolving shared-ratio fit (sSFR-anchored amplitude + ratio drift) --

    def fit_evolving(
        self,
        df,
        *,
        baseline_cols=None,
        baseline_col=None,
        ssfr_col="log_ssfr",
        ssfr_relation="speagle2014",
        evolve_amp=True,
        evolve_ratios=True,
        eta_bounds=(-3.0, 3.0),
        eta_prior_sigma=None,
        cov=None,
        scheme=None,
        dndz=None,
        sigma_z0=None,
        f_cat=None,
        smooth_baseline=False,
        n_iter=200,
        tol=1e-7,
    ):
        """sSFR-anchored, multi-band PAH fit with within-bin redshift evolution.

        Extends :meth:`fit_shared` so the PAH/continuum amplitude and the shared
        feature-group ratios drift along each mass bin as the bandpass sweeps
        rest wavelength and the contributing galaxies' specific SFR changes. For
        point i (mass bin m, band b) with centered log sSFR
        ``ŝ_i = log_ssfr(z_i, M_m) − s_pivot``::

            alpha_i  = alpha_m · 10^(η_A · ŝ_i)
            r_g(ŝ_i) = r_g0 · 10^(η_g · ŝ_i)      (g ≥ 1; r_0 ≡ 1, η_0 ≡ 0)
            f_obs_ib = C_m · f_cold_norm_ib
                     + alpha_m · 10^(η_A ŝ_i) · Σ_g r_g0 · 10^(η_g ŝ_i) · K_gib

        The slopes ``η_A`` (amplitude) and ``η_g`` (per non-reference group) are
        SHARED across mass bins. Given the slopes, the per-bin ``[C_m, alpha_m]``
        and baseline ratios ``r_g0`` stay linear and solve by the same
        alternating WLS as :meth:`fit_shared`; the few slopes are fit by an outer
        optimizer. MIPS 70 — a different rest wavelength at the same z — gives the
        leverage that separates amplitude from ratio evolution, so include both
        bands (``baseline_cols`` controls which bands enter: a band needs a cold
        baseline column to participate, so passing only 24 µm gives the
        degeneracy-prone 24-only fit).

        Per-point ``log_ssfr`` is read from ``ssfr_col`` when present, else filled
        from :func:`~simstack4.dust_evolution.main_sequence_ssfr`. ``s_pivot`` is
        the median resolved log sSFR over all fitted points. ``eta_prior_sigma``
        adds a Gaussian prior (width in dex of sSFR) on every slope; when the
        sSFR lever arm is short the slopes are degenerate with the per-bin
        amplitude and run to the bounds with unphysical pivot amplitudes, so a
        prior of order unity is recommended on real data. Slope/amplitude
        uncertainties are best taken from the disjoint-fold ensemble; formal
        curvature errors are also returned.

        Returns ``fit_shared``'s keys plus ``eta_amp``, ``eta_amp_err``,
        ``eta_ratio`` (length G, ``η_0 ≡ 0``), ``eta_ratio_err``, ``s_pivot`` and
        ``bands``.
        """
        from scipy.optimize import minimize

        from .dust_evolution import main_sequence_ssfr

        # Resolve which bands participate via their cold-baseline columns.
        default_names = {"MIPS_24": "f24_cold", "MIPS_70": "f70_cold"}
        if baseline_cols is None:
            if baseline_col is not None:
                baseline_cols = {self.bands[0]: baseline_col}
            else:
                baseline_cols = {
                    b: default_names[b] for b in self.bands if b in default_names
                }
        baseline_cols = {
            b: c
            for b, c in baseline_cols.items()
            if b in self.bands and c in df.columns
        }
        if not baseline_cols:
            raise ValueError(
                "no usable baseline columns; pass baseline_cols={band: column}"
            )
        if smooth_baseline and "MIPS_24" in baseline_cols:
            df = smoothed_ms_baseline(df, baseline_col=baseline_cols["MIPS_24"])

        prep = self._prepare(
            df,
            cov,
            scheme,
            dndz,
            sigma_z0,
            f_cat,
            baseline_cols=baseline_cols,
            ssfr_col=ssfr_col,
        )
        bins = prep["bins"]
        G = self.n_groups
        n_m = len(bins)

        def _resolve_ssfr(b):
            ls = b["log_ssfr"]
            z = np.asarray(b["z_mid"], dtype=float)
            logM = b["props"]["log_M_star"]
            ms = main_sequence_ssfr(z, logM, ssfr_relation)
            if ls is None:
                return ms
            ls = np.asarray(ls, dtype=float)
            return np.where(np.isfinite(ls), ls, ms)

        # First pass: gather resolved log sSFR to set the pivot.
        all_ls = []
        for b in bins:
            ls = _resolve_ssfr(b)
            for band in baseline_cols:
                fcold = b["f_cold_by_band"][band]
                if fcold is None:
                    continue
                bidx = self.bands.index(band)
                f_obs = b["F"][:, bidx]
                f_err = b["Ferr"][:, bidx]
                ok = (
                    np.isfinite(f_obs)
                    & np.isfinite(f_err)
                    & np.isfinite(fcold)
                    & np.isfinite(ls)
                    & (f_err > 0)
                    & (fcold > 0)
                    & (f_obs > 0)
                )
                all_ls.append(ls[ok])
        all_ls = np.concatenate(all_ls) if all_ls else np.array([0.0])
        s_pivot = float(np.median(all_ls))

        # Second pass: build θ-independent band-stacked per-bin data.
        data = []
        for b in bins:
            ls = _resolve_ssfr(b)
            f_obs_l, w_l, fcn_l, K_l, shat_l = [], [], [], [], []
            for band in baseline_cols:
                fcold = b["f_cold_by_band"][band]
                if fcold is None:
                    continue
                bidx = self.bands.index(band)
                f_obs = b["F"][:, bidx]
                f_err = b["Ferr"][:, bidx]
                K_g = b["K"][:, bidx, :]
                ok = (
                    np.isfinite(f_obs)
                    & np.isfinite(f_err)
                    & np.isfinite(fcold)
                    & np.isfinite(ls)
                    & (f_err > 0)
                    & (fcold > 0)
                    & (f_obs > 0)
                )
                if ok.sum() == 0:
                    continue
                med = float(np.median(fcold[ok]))
                f_obs_l.append(f_obs[ok])
                w_l.append(1.0 / f_err[ok] ** 2)
                fcn_l.append(fcold[ok] / med)
                K_l.append(K_g[ok])
                shat_l.append(ls[ok] - s_pivot)
            if not f_obs_l or sum(len(x) for x in f_obs_l) < 3:
                data.append(None)
                continue
            data.append(
                {
                    "f_obs": np.concatenate(f_obs_l),
                    "w": np.concatenate(w_l),
                    "f_cold_norm": np.concatenate(fcn_l),
                    "K": np.vstack(K_l),
                    "shat": np.concatenate(shat_l),
                }
            )
        valid = [i for i, d in enumerate(data) if d is not None]
        if not valid:
            return None

        def _wls(D, y, w):
            H = D.T @ (w[:, None] * D)
            rhs = D.T @ (w * y)
            try:
                return np.linalg.solve(H, rhs), np.linalg.pinv(H)
            except np.linalg.LinAlgError:
                Hi = np.linalg.pinv(H)
                return Hi @ rhs, Hi

        # θ → per-group exponent e_g = η_A + η_g (η_0 ≡ 0).
        n_amp = 1 if evolve_amp else 0
        n_rat = (G - 1) if (evolve_ratios and G > 1) else 0

        def unpack(theta):
            k = 0
            eta_amp = float(theta[k]) if evolve_amp else 0.0
            k += n_amp
            eta_ratio = np.zeros(G)
            if n_rat:
                eta_ratio[1:] = theta[k : k + n_rat]
            return eta_amp, eta_ratio

        def modulated(d, e):
            # e: (G,) per-group exponent; returns (n_pts, G) modulated kernel.
            return (10.0 ** np.outer(d["shat"], e)) * d["K"]

        def solve_inner(e, iters):
            r = np.ones(G)
            C_m = np.full(n_m, np.nan)
            alpha = np.full(n_m, np.nan)
            Kmod = {i: modulated(data[i], e) for i in valid}
            for _ in range(iters):
                r_prev = r.copy()
                for i in valid:
                    d = data[i]
                    D = np.column_stack([d["f_cold_norm"], Kmod[i] @ r])
                    theta, _ = _wls(D, d["f_obs"], d["w"])
                    C_m[i], alpha[i] = float(theta[0]), float(theta[1])
                if G > 1:
                    Ms, ys, ws = [], [], []
                    for i in valid:
                        d = data[i]
                        y = (
                            d["f_obs"]
                            - C_m[i] * d["f_cold_norm"]
                            - alpha[i] * Kmod[i][:, 0]
                        )
                        Ms.append(alpha[i] * Kmod[i][:, 1:])
                        ys.append(y)
                        ws.append(d["w"])
                    M, y, w = np.vstack(Ms), np.concatenate(ys), np.concatenate(ws)
                    r[1:], _ = _wls(M, y, w)
                    r[0] = 1.0
                if np.max(np.abs(r - r_prev)) < tol:
                    break
            chi2 = 0.0
            for i in valid:
                d = data[i]
                model = C_m[i] * d["f_cold_norm"] + alpha[i] * (Kmod[i] @ r)
                chi2 += float(np.sum((d["f_obs"] - model) ** 2 * d["w"]))
            return C_m, alpha, r, Kmod, chi2

        def chi2_of(theta):
            eta_amp, eta_ratio = unpack(theta)
            e = eta_amp + eta_ratio
            chi2 = solve_inner(e, min(n_iter, 60))[4]
            if eta_prior_sigma:
                # Gaussian prior on every free slope — tames the η_A↔alpha_m
                # runaway when the sSFR lever arm is short (real data).
                chi2 += float(np.sum((np.asarray(theta) / eta_prior_sigma) ** 2))
            return chi2

        n_theta = n_amp + n_rat
        if n_theta > 0:
            theta0 = np.zeros(n_theta)
            opt = minimize(
                chi2_of,
                theta0,
                method="L-BFGS-B",
                bounds=[eta_bounds] * n_theta,
            )
            theta_best = opt.x
        else:
            theta_best = np.zeros(0)

        eta_amp, eta_ratio = unpack(theta_best)
        e_best = eta_amp + eta_ratio
        C_m, alpha, r, Kmod, chi2 = solve_inner(e_best, n_iter)

        # Per-bin formal errors (delta method on alpha/C_m), as in fit_shared.
        alpha_err = np.full(n_m, np.nan)
        C_m_err = np.full(n_m, np.nan)
        A_pah = np.full(n_m, np.nan)
        A_pah_err = np.full(n_m, np.nan)
        ndata = 0
        for i in valid:
            d = data[i]
            D = np.column_stack([d["f_cold_norm"], Kmod[i] @ r])
            _, cov_i = _wls(D, d["f_obs"], d["w"])
            C_m_err[i] = np.sqrt(max(cov_i[0, 0], 0.0))
            alpha_err[i] = np.sqrt(max(cov_i[1, 1], 0.0))
            if C_m[i] != 0:
                A_pah[i] = alpha[i] / C_m[i]
                g = np.array([-alpha[i] / C_m[i] ** 2, 1.0 / C_m[i]])
                A_pah_err[i] = np.sqrt(max(g @ cov_i @ g, 0.0))
            ndata += len(d["f_obs"])
        r_err = np.zeros(G)
        if G > 1:
            Ms, ws = [], []
            for i in valid:
                Ms.append(alpha[i] * Kmod[i][:, 1:])
                ws.append(data[i]["w"])
            M, w = np.vstack(Ms), np.concatenate(ws)
            cov_r = np.linalg.pinv(M.T @ (w[:, None] * M))
            r_err[1:] = np.sqrt(np.maximum(np.diag(cov_r), 0.0))

        # Formal slope errors from the diagonal curvature of profiled chi².
        eta_amp_err = np.nan
        eta_ratio_err = np.zeros(G)
        if n_theta > 0:
            h = 0.05
            for j in range(n_theta):
                tp = theta_best.copy()
                tp[j] += h
                tm = theta_best.copy()
                tm[j] -= h
                curv = max((chi2_of(tp) + chi2_of(tm) - 2 * chi2) / h**2, 1e-9)
                sig = float(np.sqrt(2.0 / curv))
                if evolve_amp and j == 0:
                    eta_amp_err = sig
                else:
                    g_idx = 1 + (j - n_amp)
                    eta_ratio_err[g_idx] = sig

        dof = max(1, ndata - (2 * len(valid) + (G - 1) + n_theta))
        A = A_pah[:, None] * r[None, :]
        return {
            "alpha": alpha,
            "alpha_err": alpha_err,
            "C_m": C_m,
            "C_m_err": C_m_err,
            "r": r,
            "r_err": r_err,
            "A_pah": A_pah,
            "A_pah_err": A_pah_err,
            "A": A,
            "eta_amp": eta_amp,
            "eta_amp_err": eta_amp_err,
            "eta_ratio": eta_ratio,
            "eta_ratio_err": eta_ratio_err,
            "s_pivot": s_pivot,
            "bands": tuple(baseline_cols.keys()),
            "labels": self._labels(),
            "chi2": chi2,
            "dof": dof,
            "chi2_red": chi2 / dof,
            "valid": valid,
        }

    # -- fit the Wien-slope alpha jointly, with a Gaussian prior ----------

    def fit_with_alpha(
        self,
        df,
        *,
        evolving: bool = False,
        baseline_col: str = "f24_cold",
        baseline_cols: dict | None = None,
        alpha_prior: tuple[float, float] = (2.0, 0.3),
        alpha_bounds: tuple[float, float] = (1.0, 3.0),
        alpha_ref: float = 2.0,
        baseline_recompute=None,
        **fit_kw,
    ):
        """Fit the cold-baseline Wien slope ``alpha`` jointly with the PAH model.

        The cold baseline ``f_cold`` is a greybody spliced to a power law
        ``f_ν ∝ ν^(−alpha) ∝ (1+z)^(−alpha)`` short-ward of rest ~71 µm, and the
        PAH amplitude is acutely sensitive to that slope (Δalpha≈0.5 → A_pah ×3–4).
        Rather than fix ``alpha=2``, this profiles it out: an outer
        ``minimize_scalar`` over ``alpha`` wraps the inner fit
        (:meth:`fit_shared`, or :meth:`fit_evolving` when ``evolving=True``),
        minimising ``chi² + ((alpha − mu)/sigma)²`` with a **Gaussian prior**
        ``alpha_prior = (mu, sigma)`` that keeps the slope physical (default a
        strong prior at 2). On a single band ``alpha`` is degenerate with the PAH
        amplitude and the prior dominates by design; with ≥2 bands (24+70/100) the
        longer band constrains the continuum slope and ``alpha`` becomes
        data-driven.

        By default the baseline is re-tilted in place as
        ``f_cold(alpha) = f_cold · (1+z)^(alpha_ref − alpha)`` (the exact ν^(−α)
        shape change; a sub-dominant T(z)^(α−2) term is neglected). Pass
        ``baseline_recompute(df, alpha) -> df`` to rebuild the baseline columns
        faithfully from ``greybody_model`` instead.

        Returns the inner fit's dict plus ``alpha_wien``, ``alpha_wien_err`` and
        ``alpha_prior``. (Note: the per-bin PAH amplitude is the separate
        ``alpha``/``A_pah`` key — ``alpha_wien`` is the continuum slope.)
        """
        from scipy.optimize import minimize_scalar

        mu, sigma = alpha_prior
        if evolving:
            cols = baseline_cols or {"MIPS_24": "f24_cold", "MIPS_70": "f70_cold"}
            tilt_cols = [c for c in cols.values() if c in df.columns]
        else:
            tilt_cols = [baseline_col]

        def _baseline_at(alpha):
            if baseline_recompute is not None:
                return baseline_recompute(df, alpha)
            out = df.copy()
            fac = (1.0 + out["z_mid"].to_numpy()) ** (alpha_ref - alpha)
            for c in tilt_cols:
                if c in out.columns:
                    out[c] = out[c].to_numpy() * fac
            return out

        def _run(alpha):
            d = _baseline_at(alpha)
            if evolving:
                res = self.fit_evolving(d, baseline_cols=baseline_cols, **fit_kw)
            else:
                res = self.fit_shared(d, baseline_col=baseline_col, **fit_kw)
            return res

        def _obj(alpha):
            res = _run(alpha)
            chi2 = res["chi2"] if res is not None else 1e12
            return chi2 + ((alpha - mu) / sigma) ** 2

        opt = minimize_scalar(_obj, bounds=alpha_bounds, method="bounded")
        a_best = float(opt.x)
        # 1σ from the local curvature of the penalised objective
        h = 0.05
        c0 = _obj(a_best)
        cp = _obj(min(a_best + h, alpha_bounds[1]))
        cm = _obj(max(a_best - h, alpha_bounds[0]))
        curv = max((cp + cm - 2 * c0) / h**2, 1e-9)
        a_err = float(np.sqrt(2.0 / curv))

        res = _run(a_best)
        if res is not None:
            res["alpha_wien"] = a_best
            res["alpha_wien_err"] = a_err
            res["alpha_prior"] = alpha_prior
        return res

    # -- pooled hierarchical MCMC -----------------------------------------

    def fit_mcmc(
        self,
        df,
        *,
        cov=None,
        scheme=None,
        dndz=None,
        sigma_z0: float | None = None,
        f_cat: float | None = None,
        fix_beta_mass: bool = False,
        fix_beta_sigma: bool = False,
        n_walkers: int = 32,
        n_steps: int = 1000,
        n_burn: int = 300,
        progress: bool = False,
        verbose: bool = False,
        seed: int = 0,
    ) -> PAHSpectrumResult:
        """Pooled evolution fit: amplitudes follow a power law in properties.

            log10 A_g,m = log10 A0_g + β_M·(logM*−pivot) + β_σ·(logσ−pivot)

        θ = [T_w, log10 A0_1..G, β_M?, β_σ?]; per-bin continuum amplitudes
        C_m are profiled analytically at every step (linear given A), so
        the chain stays low-dimensional — the DustEvolutionModel pattern.
        """
        import emcee

        prep = self._prepare(df, cov, scheme, dndz, sigma_z0, f_cat)
        bins = prep["bins"]
        G = self.n_groups
        x_M = np.array([b["props"]["log_M_star"] - self.pivot_log_mass for b in bins])
        x_s = np.array(
            [b["props"]["log_sigma_sfr"] - self.pivot_log_sigma_sfr for b in bins]
        )

        names = ["T_w"] + [f"logA0_{lab}" for lab in self._labels()]
        if not fix_beta_mass:
            names.append("beta_mass")
        if not fix_beta_sigma:
            names.append("beta_sigma")
        ndim = len(names)

        def unpack(theta):
            T_w = theta[0]
            log_a0 = theta[1 : 1 + G]
            k = 1 + G
            b_M = 0.0 if fix_beta_mass else theta[k]
            k += 0 if fix_beta_mass else 1
            b_s = 0.0 if fix_beta_sigma else theta[k]
            return T_w, log_a0, b_M, b_s

        def log_prob(theta):
            T_w, log_a0, b_M, b_s = unpack(theta)
            if not (self.T_w_bounds[0] < T_w < self.T_w_bounds[1]):
                return -np.inf
            if np.any(log_a0 < self.log_a0_bounds[0]) or np.any(
                log_a0 > self.log_a0_bounds[1]
            ):
                return -np.inf
            if abs(b_M) > 2.0 or abs(b_s) > 2.0:
                return -np.inf
            mu, sig = self.T_w_prior
            lp = -0.5 * ((T_w - mu) / sig) ** 2
            lp += -0.5 * (b_M / self.beta_prior_sigma) ** 2
            lp += -0.5 * (b_s / self.beta_prior_sigma) ** 2
            chi2 = 0.0
            for i, b in enumerate(bins):
                A_m = 10.0 ** (log_a0 + b_M * x_M[i] + b_s * x_s[i])
                t_w = self._Ww(b, T_w) + b["Kw"] @ A_m
                tt = float(t_w @ t_w)
                if tt <= 0:
                    return -np.inf
                C_m = float(t_w @ b["Fw"]) / tt
                resid = b["Fw"] - C_m * t_w
                chi2 += float(resid @ resid)
            return lp - 0.5 * chi2

        # initialize from the free-amplitude MAP
        init = self.fit_lstsq(
            df,
            cov=cov,
            scheme=prep["scheme"],
            dndz=dndz,
            sigma_z0=sigma_z0,
            f_cat=f_cat,
        )
        A_init = np.clip(np.nanmean(init.A, axis=0), 1e-3, 10.0)
        theta0 = [init.theta_global[0]] + list(np.log10(A_init))
        if not fix_beta_mass:
            theta0.append(0.0)
        if not fix_beta_sigma:
            theta0.append(0.0)
        theta0 = np.array(theta0)

        rng = np.random.default_rng(seed)
        p0 = theta0 + 1e-3 * rng.standard_normal((n_walkers, ndim))
        p0[:, 0] = np.clip(p0[:, 0], *np.add(self.T_w_bounds, (0.5, -0.5)))

        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        sampler.run_mcmc(p0, n_steps, progress=progress)
        chain = sampler.get_chain(discard=n_burn, flat=True)
        theta_med = np.median(chain, axis=0)
        theta_err = np.std(chain, axis=0)
        if verbose:
            for n, v, e in zip(names, theta_med, theta_err, strict=True):
                print(f"  {n} = {v:.3f} ± {e:.3f}")

        # posterior amplitudes per bin
        T_w, log_a0, b_M, b_s = unpack(theta_med)
        M = len(bins)
        A = np.zeros((M, G))
        A_err = np.zeros((M, G))
        C = np.zeros(M)
        C_err = np.zeros(M)
        chi2 = 0.0
        dof = -ndim
        # amplitude posteriors from a chain subsample
        sub_idx = np.linspace(0, len(chain) - 1, min(200, len(chain))).astype(int)
        for i, b in enumerate(bins):
            A_samp = np.zeros((len(sub_idx), G))
            C_samp = np.zeros(len(sub_idx))
            for k, j in enumerate(sub_idx):
                T_j, la0_j, bM_j, bs_j = unpack(chain[j])
                A_mj = 10.0 ** (la0_j + bM_j * x_M[i] + bs_j * x_s[i])
                t_w = self._Ww(b, T_j) + b["Kw"] @ A_mj
                C_samp[k] = float(t_w @ b["Fw"]) / float(t_w @ t_w)
                A_samp[k] = A_mj
            A[i] = A_samp.mean(axis=0)
            A_err[i] = A_samp.std(axis=0)
            C[i] = C_samp.mean()
            C_err[i] = C_samp.std()
            A_med = 10.0 ** (log_a0 + b_M * x_M[i] + b_s * x_s[i])
            t_w = self._Ww(b, T_w) + b["Kw"] @ A_med
            C_med = float(t_w @ b["Fw"]) / float(t_w @ t_w)
            resid = b["Fw"] - C_med * t_w
            chi2 += float(resid @ resid)
            dof += b["n_valid"] - 1  # C_m profiled per bin
        return PAHSpectrumResult(
            A=A,
            A_err=A_err,
            C_per_bin=C,
            C_err_per_bin=C_err,
            theta_global=theta_med,
            theta_err=theta_err,
            param_names=names,
            chi2=chi2,
            dof=dof,
            labels=self._labels(),
            sampler=sampler,
            acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
        )

    # -- diagnostics -------------------------------------------------------

    def pseudo_spectrum(
        self,
        df,
        result: PAHSpectrumResult,
        *,
        scheme=None,
        dndz=None,
        sigma_z0: float | None = None,
        f_cat: float | None = None,
    ):
        """Continuum-normalized pseudo-spectrum for overlay plots.

        Per (dither bin, band): λ_rest = λ_eff/(1+z_mid) and the excess
        ratio R = F/(C_m·W) − 1, whose model counterpart is Σ_g A_g K_g/W.
        The ratio diverges where the continuum is negligible (high z at
        24 µm), so excess_snr = (F − C_m·W)/σ_F is also provided — it is
        bounded and peaks where features carry real significance.
        """
        prep = self._prepare(df, None, scheme, dndz, sigma_z0, f_cat)
        pz, z_grid = prep["pz"], prep["z_grid"]
        T_w = float(result.theta_global[0])
        W = warm_continuum_kernel(pz, z_grid, self.bands, T_w=T_w, beta_w=self.beta_w)
        rows = []
        for i, b in enumerate(prep["bins"]):
            sub = (
                df[df["prop_bin_id"] == b["m"]] if "prop_bin_id" in df else df
            ).sort_values(["run_id", "z_lo"])
            # Apply the same scheme-row filter that _prepare used for this bin.
            sidx = b.get("sidx")
            if sidx is not None and len(sub) != len(sidx):
                bt = prep["scheme"].bin_table()
                slookup = {
                    (int(r.run_id), round(float(r.z_lo), 8)): True
                    for r in bt.itertuples()
                }
                vmask = np.array(
                    [
                        slookup.get(
                            (int(r["run_id"]), round(float(r["z_lo"]), 8)), False
                        )
                        for _, r in sub.iterrows()
                    ]
                )
                sub = sub.iloc[vmask].reset_index(drop=True)
            C_m = result.C_per_bin[i]
            for bi, band in enumerate(self.bands):
                bp = get_bandpass(band)
                # Index the global W by the per-bin scheme rows.
                W_m = W[sidx, bi] if sidx is not None else W[:, bi]
                cont = C_m * W_m
                F = sub[band].to_numpy()
                Ferr = sub[f"{band}_err"].to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = F / cont - 1.0
                    ratio_err = Ferr / cont
                    excess_snr = (F - cont) / Ferr
                rows.append(
                    pd.DataFrame(
                        {
                            "prop_bin_id": b["m"],
                            "band": band,
                            "z_mid": sub["z_mid"].to_numpy(),
                            "lam_rest": bp.lam_eff / (1.0 + sub["z_mid"].to_numpy()),
                            "ratio": ratio,
                            "ratio_err": ratio_err,
                            "excess_snr": excess_snr,
                            "n_sources": (
                                sub["n_sources"].to_numpy()
                                if "n_sources" in sub
                                else np.nan
                            ),
                        }
                    )
                )
        return pd.concat(rows, ignore_index=True).sort_values("lam_rest")


# ─────────────────────────────────────────────────────────────────────────────
# Smoothed main-sequence baseline helper (used by PAHSpectrumModel.fit_shared).
#
# On the Wien side f24_cold ∝ A·T^(3+β+α) ≈ A·T^6.8, so per-bin scatter in the
# FIR-only greybody (T, logA) is hugely amplified into the baseline. Replacing
# the per-bin params with smooth relations T(z,M*), logA(z,M*) fit to the
# well-constrained (Tier A/B) bins removes that. PAH-free by construction (the
# SEDs that produced T/logA exclude 24 µm).
# ─────────────────────────────────────────────────────────────────────────────


def _baseline_design(z, dM, quad):
    """Design matrix for the baseline relation: columns [1, z, (z^2), dM]."""
    cols = [np.ones_like(z), z]
    if quad:
        cols.append(z**2)
    cols.append(dM)
    return np.column_stack(cols)


def _baseline_fit_bic(y, z, dM, allow_quad=True):
    """OLS fit of y ~ (z[, z^2], dM); pick linear vs quadratic-in-z by BIC.

    Returns (label, coef, predict_fn, bic).
    """
    n = len(y)
    cands = [False, True] if (allow_quad and n >= 6) else [False]
    best = None
    for quad in cands:
        X = _baseline_design(z, dM, quad)
        if X.shape[1] >= n:
            continue
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((y - X @ coef) ** 2))
        k = X.shape[1]
        bic = n * np.log(max(rss, 1e-300) / n) + k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, "quadratic" if quad else "linear", coef, quad)
    if best is None:
        coef = np.array([float(np.mean(y)), 0.0, 0.0])
        return "constant", coef, (lambda zz, dd: np.full_like(zz, coef[0])), np.inf
    _, lbl, coef, quad = best
    return lbl, coef, (lambda zz, dd: _baseline_design(zz, dd, quad) @ coef), best[0]


def smoothed_ms_baseline(
    df,
    *,
    mass_pivot=10.0,
    t_clip=(15.0, 60.0),
    tier_col="tier",
    train_tiers=("A", "B"),
    baseline_col="f24_cold",
    band_um=24.0,
    verbose=False,
):
    """Replace per-bin (T_dust, log_amp) with a smooth main-sequence baseline.

    Fits ``T_dust(z, M*)`` and ``log_amp(z, M*)`` (BIC picks linear vs
    quadratic in z) on the well-constrained (Tier A/B) bins, then recomputes
    ``baseline_col`` for every row from the smoothed params via the greybody
    Wien tail. Predictors are held flat outside the training box and T is
    clipped to ``t_clip`` so the T^6.8 tail cannot run away. The original column
    is preserved as ``{baseline_col}_raw``.

    Required columns: ``z_mid``, ``log_M_star``, ``T_dust``, ``log_amp``,
    ``beta`` (and optionally ``tier_col``).
    """
    from .greybody import Greybody  # lazy: greybody imports pah_model

    out = df.copy()
    if tier_col in out.columns:
        tmask = out[tier_col].isin(train_tiers).to_numpy()
    else:
        tmask = np.ones(len(out), bool)
    train = out[tmask & np.isfinite(out["T_dust"]) & np.isfinite(out["log_amp"])]
    if len(train) < 4:
        out[f"{baseline_col}_raw"] = out[baseline_col]
        if verbose:
            print("smoothed_ms_baseline: <4 training bins; baseline unchanged")
        return out

    zt = train["z_mid"].to_numpy()
    dMt = train["log_M_star"].to_numpy() - mass_pivot
    beta0 = float(np.nanmedian(train["beta"].to_numpy()))
    lblT, _, predT, _ = _baseline_fit_bic(train["T_dust"].to_numpy(), zt, dMt)
    lblA, _, predA, _ = _baseline_fit_bic(train["log_amp"].to_numpy(), zt, dMt)

    zc = out["z_mid"].to_numpy()
    dMc = out["log_M_star"].to_numpy() - mass_pivot
    zcp = np.clip(zc, float(zt.min()), float(zt.max()))
    dcp = np.clip(dMc, float(dMt.min()), float(dMt.max()))
    T_sm = np.clip(predT(zcp, dcp), t_clip[0], t_clip[1])
    A_sm = predA(zcp, dcp)

    gb = Greybody()
    f_sm = np.array(
        [
            float(gb.greybody_model(np.array([band_um / (1.0 + z)]), a, t, beta0)[0])
            for z, a, t in zip(zc, A_sm, T_sm, strict=False)
        ]
    )
    out["T_dust_smooth"] = T_sm
    out["log_amp_smooth"] = A_sm
    out[f"{baseline_col}_raw"] = out[baseline_col]
    out[baseline_col] = f_sm
    if verbose:
        print(
            f"smoothed_ms_baseline: T_dust({lblT}), log_amp({lblA}), beta={beta0:.2f}; "
            f"{len(train)}/{len(out)} training bins"
        )
    return out
