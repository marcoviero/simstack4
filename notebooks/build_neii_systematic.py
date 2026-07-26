"""Measure the [Ne II] 12.81 um systematic on the PAH band-ratio mass slope.

Branch-12 §3d parked [Ne II] as a bracketed systematic because the quantity that
actually biases the band-ratio mass TREND is not the mean contamination fraction
(which cancels exactly -- it is pure normalisation) but its sSFR *gradient*:

    d log([Ne II] / PAH 11.3) / d log sSFR

This script measures both, from Smith et al. (2007, ApJ 656, 770) -- the SINGS
PAHFIT decompositions, which tabulate the 11.3 and 12.6 um PAH complexes AND the
[Ne II] 12.813 um line for the same apertures, plus LIR/LB as an sSFR proxy.

Run:  uv run python notebooks/build_neii_systematic.py
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

VIZIER = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/656/770/catalog"
    "&-out.max=unlimited&-out=Galaxy,Type,LTIR,LIR/LB,F11.3,e_F11.3,F12.6,e_F12.6,NeII,e_NeII"
)
# d log sSFR / d log M* across our mass bins at z~1 (branch-12 §3d: sSFR falls
# 0.31 dex over 9.9 < log M* < 11.5).
M_SSFR = -0.31
NUM = ["LTIR", "LIR/LB", "F11.3", "e_F11.3", "F12.6", "e_F12.6", "NeII", "e_NeII"]


def load() -> pd.DataFrame:
    with urllib.request.urlopen(VIZIER, timeout=60) as fh:
        text = fh.read().decode("utf-8", "replace")
    rows, hdr = [], None
    for ln in io.StringIO(text):
        if ln.startswith("#") or not ln.strip():
            continue
        p = [c.strip() for c in ln.rstrip("\n").split("\t")]
        if hdr is None:
            hdr = p
            continue
        if not p[0] or p[0].startswith("---"):
            continue
        rows.append(p)
    d = pd.DataFrame(rows, columns=hdr)
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def wls_slope(x, y):
    X = np.vstack([x - x.mean(), np.ones_like(x)]).T
    C = np.linalg.inv(X.T @ X)
    b = C @ (X.T @ y)
    r = y - X @ b
    s2 = float(r @ r) / max(1, len(x) - 2)
    return float(b[0]), float(np.sqrt(C[0, 0] * s2))


def main() -> None:
    d = load()
    print(f"Smith+2007 SINGS: {len(d)} galaxies; types {d['Type'].value_counts().to_dict()}")

    # AGN suppress PAH and boost high-ionisation lines -> H II nuclei only.
    sf = d[d["Type"].str.strip().isin(["H II", "HII"])].copy()
    sf = sf[(sf["F11.3"] > 0) & (sf["F12.6"] > 0) & (sf["NeII"] > 0) & (sf["LIR/LB"] > 0)]
    sf["r_ne"] = sf["NeII"] / sf["F11.3"]
    sf["r_pah"] = sf["F12.6"] / sf["F11.3"]
    sf["fne"] = sf["NeII"] / (sf["F11.3"] + sf["F12.6"] + sf["NeII"])
    sf["x"] = np.log10(sf["LIR/LB"])
    print(f"H II nuclei with all quantities positive: {len(sf)}"
          f"   sSFR proxy spans {sf['x'].max() - sf['x'].min():.2f} dex")

    for c, lab in [("fne", "[Ne II] fraction of the 11-13um group"),
                   ("r_ne", "[Ne II] / PAH 11.3"),
                   ("r_pah", "PAH 12.6 / PAH 11.3")]:
        v = sf[c]
        print(f"  {lab:<40} median {v.median():.3f}"
              f"  16-84% [{v.quantile(.16):.3f}, {v.quantile(.84):.3f}]")

    x, y = sf["x"].to_numpy(), np.log10(sf["r_ne"].to_numpy())
    g, ge = wls_slope(x, y)
    rng = np.random.default_rng(7)
    bs = np.array([wls_slope(x[i], y[i])[0]
                   for i in (rng.integers(0, len(x), len(x)) for _ in range(4000))])
    hi95 = float(np.percentile(bs, 97.5))
    print(f"\ngradient g = d log([Ne II]/PAH 11.3) / d log(LIR/LB)")
    print(f"  OLS {g:+.3f} +/- {ge:.3f}   bootstrap 95% [{np.percentile(bs, 2.5):+.3f}, {hi95:+.3f}]")
    print(f"  P(g > 0.5) = {(bs > 0.5).mean():.4f}  <- the branch-12 §3d bracket was 0.5-1.0")

    # Observed welded group G = PAH + [Ne II], u = [Ne II]/PAH:
    #   d log r_obs/d logM* = d log r_true/d logM* + (u/(1+u)) * g * M_SSFR
    print("\nbias on the band-ratio mass slope (fold error 0.052, measured slope -0.341):")
    for f_, ftag in [(float(sf["fne"].median()), "median f"),
                     (float(sf["fne"].quantile(.84)), "84th-pct f")]:
        u = f_ / (1 - f_)
        for gg, gtag in [(g, "measured g"), (hi95, "95% upper g")]:
            print(f"  f={f_:.3f} ({ftag:<11}) g={gg:.3f} ({gtag:<11})"
                  f" -> bias {(u / (1 + u)) * gg * M_SSFR:+.4f} dex/dex")
    print("\nCaveats, two of three conservative:")
    print("  - LIR/LB compresses relative to true sSFR (L_B has a young component),")
    print("    so the true sSFR gradient is <= the measured value.")
    print("  - SINGS apertures are nuclear and [Ne II] is more centrally concentrated")
    print("    than 11.3um PAH (Pereira-Santaella+2010), so f is an over-estimate.")
    print("  - OPEN: SINGS is z~0; our galaxies are z~1-3 main-sequence at higher sSFR.")


if __name__ == "__main__":
    main()
