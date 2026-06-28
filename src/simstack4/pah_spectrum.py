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

    def _prepare(self, df, cov, scheme, dndz, sigma_z0, f_cat):
        """Build whitened per-property-bin data and kernel structures."""
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
            (int(row.run_id), round(float(row.z_lo), 8)): i
            for i, row in bt.iterrows()
        }

        prop_ids = sorted(df["prop_bin_id"].unique()) if "prop_bin_id" in df else [0]
        bins = []
        for m in prop_ids:
            sub = (
                df[df["prop_bin_id"] == m] if "prop_bin_id" in df else df
            ).sort_values(["run_id", "z_lo"])

            # Map each sub row to its scheme row index; drop any that don't match.
            sidx = np.array([
                scheme_lookup.get((int(r["run_id"]), round(float(r["z_lo"]), 8)), -1)
                for _, r in sub.iterrows()
            ])
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
            bins.append(
                {
                    "m": m,
                    "F": F,
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
                    theta_cov = np.linalg.pinv(H, rcond=1e-8 * max(np.max(np.abs(H)), 1.0))
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
                vmask = np.array([
                    slookup.get((int(r["run_id"]), round(float(r["z_lo"]), 8)), False)
                    for _, r in sub.iterrows()
                ])
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
