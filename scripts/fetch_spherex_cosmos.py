#!/usr/bin/env python
"""Download SPHEREx spectral-image cutouts covering the COSMOS field.

Follows the IRSA recipe at
https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts/
with three changes that matter for science use:

1. All six detectors (D1-D6) are queried at once, not one bandpass, so the
   full 0.75-5 um range is retrieved in a single pass.
2. Downloads are retried with backoff and the run is resumable; the tutorial
   silently drops cutouts that hit a transient read error.
3. Extensions are written under their ORIGINAL names. The wavelength solution
   is a FITS ``-TAB`` lookup whose ``PS1_0W``/``PS2_0W`` keywords name the
   ``WCS-WAVE`` extension by string. The tutorial appends an index to every
   EXTNAME without updating those keywords, which breaks ``WCS(..., key="W")``
   downstream. When ``--mef`` renames extensions here, the PS keywords are
   rewritten to match.

A note on "bands": SPHEREx does not have ~102 separate archived bandpasses.
Each detector carries a linear variable filter, so wavelength is a function of
POSITION on the array. The archive exposes only six bandpass names (the
detectors). The ~102 spectral channels are recovered by observing a given sky
position many times at different array locations. So the spectrum at COSMOS is
built by downloading every visit and binning by wavelength -- which is what
this script sets up. Because wavelength varies across the array, a cutout of
angular size S is not monochromatic: at S=0.5 deg it spans ~2.4 channels, so
always use the per-pixel WCS-WAVE solution rather than a single scalar.

Usage
-----
    python scripts/fetch_spherex_cosmos.py --outdir $MAPSPATH/spherex/cosmos

    # dry run: report what would be fetched, download nothing
    python scripts/fetch_spherex_cosmos.py --dry-run

    # also bundle everything into one multi-extension FITS file
    python scripts/fetch_spherex_cosmos.py --mef spherex_cosmos_mef.fits
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

# astropy replaces the stdlib Logger class at import time, so the level must be
# set only AFTER astropy is imported (doing it first raises AttributeError).
import logging

logging.getLogger("astropy").setLevel(logging.ERROR)

TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_ROOT = "https://irsa.ipac.caltech.edu/"

# COSMOS field centre (Scoville+2007).
COSMOS_RA = 150.11916 * u.degree
COSMOS_DEC = 2.20583 * u.degree

# Extensions worth keeping. WCS-WAVE is mandatory: it holds the -TAB lookup
# that turns pixel position into wavelength.
DEFAULT_KEEP = ("IMAGE", "VARIANCE", "WCS-WAVE")

TRANSIENT = (
    TimeoutError,
    socket.timeout,
    urllib.error.URLError,
    urllib.error.HTTPError,
    http.client.IncompleteRead,
    http.client.HTTPException,
    OSError,
)


def query_tap(ra, dec, size, bandpasses=None, timeout=300):
    """Return an astropy Table of cutout URLs covering (ra, dec).

    Uses the TAP sync endpoint directly so pyvo is not a hard dependency.
    The cutout parameters are appended to the artifact URI server-side, exactly
    as in the IRSA tutorial.
    """
    where_band = ""
    if bandpasses:
        joined = ", ".join(f"'{b}'" for b in bandpasses)
        where_band = f"AND p.energy_bandpassname IN ({joined})"

    query = f"""
SELECT
    '{IRSA_ROOT}' || a.uri || '?center={ra.value},{dec.value}d&size={size.value}' AS uri,
    a.uri AS raw_uri,
    p.energy_bandpassname,
    p.energy_bounds_lower,
    p.energy_bounds_upper,
    p.time_bounds_lower
FROM spherex.artifact a
JOIN spherex.plane p ON a.planeid = p.planeid
WHERE 1 = CONTAINS(POINT('ICRS', {ra.value}, {dec.value}), p.poly)
    {where_band}
"""
    # Sorted client-side rather than with ORDER BY: the sync endpoint returns
    # intermittent 504s under load and a smaller query is likelier to land.
    payload = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        }
    ).encode()

    # The sync endpoint throws intermittent 504s; retry with backoff.
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(TAP_SYNC, data=payload, timeout=timeout) as resp:
                text = resp.read().decode()
            break
        except TRANSIENT as exc:
            last = exc
            print(f"  TAP query attempt {attempt + 1} failed ({exc}); retrying ...", file=sys.stderr)
            time.sleep(3 * 2**attempt)
    else:
        raise RuntimeError(f"TAP query failed after 5 attempts: {last}")

    # Pass a list of lines: astropy's fast CSV reader rejects a StringIO here.
    table = Table.read(text.splitlines(), format="ascii.csv")
    table.sort(["energy_bandpassname", "time_bounds_lower"])
    return table


def wavelength_stats(hdulist, ra, dec):
    """Central wavelength and the wavelength span across the cutout, in um.

    The -TAB spectral WCS is 2-D (wavelength depends on both array axes), so
    the span is measured on a coarse grid over the whole cutout rather than
    assumed constant.
    """
    header = hdulist["IMAGE"].header

    spatial = WCS(header)
    x, y = spatial.world_to_pixel(SkyCoord(ra=ra, dec=dec, frame="icrs"))

    spectral = WCS(header, fobj=hdulist, key="W")
    spectral.sip = None

    lam_c, band_c = spectral.pixel_to_world(x, y)

    ny, nx = hdulist["IMAGE"].data.shape
    step = max(1, min(ny, nx) // 16)
    ys, xs = np.mgrid[0:ny:step, 0:nx:step]
    lam_grid, _ = spectral.pixel_to_world(xs.ravel(), ys.ravel())
    lam_grid = lam_grid.to(u.micrometer).value

    return dict(
        central_wavelength=float(lam_c.to(u.micrometer).value),
        bandwidth=float(band_c.to(u.micrometer).value),
        wavelength_min=float(np.nanmin(lam_grid)),
        wavelength_max=float(np.nanmax(lam_grid)),
        x_pix=float(x),
        y_pix=float(y),
    )


def fetch_one(row, ra, dec, outdir, keep, retries=4, overwrite=False):
    """Download one cutout, strip to `keep`, write it, and return its metadata.

    Returns a dict on success or None if the cutout could not be retrieved.
    """
    fname = os.path.basename(row["raw_uri"].split("?")[0])
    dest = os.path.join(outdir, fname)

    if os.path.exists(dest) and not overwrite:
        try:
            with fits.open(dest) as hdulist:
                stats = wavelength_stats(hdulist, ra, dec)
            return dict(filename=fname, cached=True, **stats)
        except Exception:
            # Truncated leftover from an interrupted run; refetch it.
            os.remove(dest)

    last = None
    for attempt in range(retries):
        try:
            with fits.open(row["uri"], cache=False) as hdulist:
                stats = wavelength_stats(hdulist, ra, dec)
                out = [fits.PrimaryHDU(header=hdulist[0].header)]
                for hdu in hdulist[1:]:
                    if hdu.header.get("EXTNAME") in keep:
                        out.append(hdu.copy())
                names = [h.header.get("EXTNAME") for h in out[1:]]
                if "WCS-WAVE" not in names:
                    raise ValueError(f"no WCS-WAVE extension in {fname}")
                # Record provenance so the file stands alone.
                phdr = out[0].header
                phdr["SRCURI"] = (row["uri"][:68], "cutout request URL (truncated)")
                phdr["BANDPASS"] = (row["energy_bandpassname"], "SPHEREx detector")
                phdr["CUTRA"] = (ra.value, "cutout centre RA [deg]")
                phdr["CUTDEC"] = (dec.value, "cutout centre Dec [deg]")
                phdr["LAMCEN"] = (stats["central_wavelength"], "wavelength at centre [um]")
                fits.HDUList(out).writeto(dest, overwrite=True)
            return dict(filename=fname, cached=False, **stats)
        except TRANSIENT as exc:
            last = exc
            time.sleep(2**attempt)
        except Exception as exc:  # malformed file, unexpected structure
            last = exc
            break

    print(f"  FAILED {fname}: {type(last).__name__}: {last}", file=sys.stderr)
    return None


def build_mef(summary, outdir, mef_path):
    """Bundle the per-cutout files into one MEF with unique extension names.

    Renaming EXTNAMEs requires rewriting the PS{i}_0W keywords that point the
    -TAB wavelength lookup at its table, otherwise the spectral WCS silently
    stops resolving.
    """
    cols = fits.ColDefs(
        [
            fits.Column(name="cutout_index", format="J", array=summary["cutout_index"]),
            fits.Column(name="bandpass", format="A16", array=summary["energy_bandpassname"]),
            fits.Column(name="observation_date", format="D", array=summary["time_bounds_lower"], unit="d"),
            fits.Column(name="central_wavelength", format="D", array=summary["central_wavelength"], unit="um"),
            fits.Column(name="wavelength_min", format="D", array=summary["wavelength_min"], unit="um"),
            fits.Column(name="wavelength_max", format="D", array=summary["wavelength_max"], unit="um"),
            fits.Column(name="filename", format="A80", array=summary["filename"]),
        ]
    )
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header["EXTNAME"] = "CUTOUT_INFO"

    hdus = [fits.PrimaryHDU(), table_hdu]
    for row in summary:
        idx = row["cutout_index"]
        with fits.open(os.path.join(outdir, row["filename"])) as hdulist:
            for hdu in hdulist[1:]:
                hdu = hdu.copy()
                old = hdu.header["EXTNAME"]
                hdu.header["EXTNAME"] = f"{old}{idx}"
                for key in ("PS1_0W", "PS2_0W"):
                    if key in hdu.header and hdu.header[key] == "WCS-WAVE":
                        hdu.header[key] = f"WCS-WAVE{idx}"
                hdus.append(hdu)

    fits.HDUList(hdus).writeto(mef_path, overwrite=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", type=float, default=COSMOS_RA.value, help="centre RA [deg]")
    ap.add_argument("--dec", type=float, default=COSMOS_DEC.value, help="centre Dec [deg]")
    ap.add_argument("--size", type=float, default=0.5, help="cutout size [deg]")
    ap.add_argument(
        "--outdir",
        default=os.path.join(os.path.expandvars("$MAPSPATH"), "spherex", "cosmos"),
        help="output directory for per-cutout FITS files",
    )
    ap.add_argument("--bandpass", action="append", help="restrict to a detector, e.g. SPHEREx-D2 (repeatable)")
    ap.add_argument("--keep", default=",".join(DEFAULT_KEEP), help="comma-separated extensions to retain")
    ap.add_argument("--workers", type=int, default=10, help="parallel download threads")
    ap.add_argument("--limit", type=int, help="only fetch the first N cutouts (for testing)")
    ap.add_argument("--overwrite", action="store_true", help="refetch files that already exist")
    ap.add_argument("--mef", help="also write a combined multi-extension FITS file here")
    ap.add_argument("--dry-run", action="store_true", help="query and report, download nothing")
    args = ap.parse_args()

    ra = args.ra * u.degree
    dec = args.dec * u.degree
    size = args.size * u.degree
    keep = tuple(s.strip() for s in args.keep.split(",") if s.strip())
    if "WCS-WAVE" not in keep:
        ap.error("--keep must include WCS-WAVE; it carries the wavelength solution")

    print(f"Querying IRSA TAP at ({ra.value}, {dec.value}) deg, cutout {size.value} deg ...")
    t0 = time.time()
    results = query_tap(ra, dec, size, bandpasses=args.bandpass)
    print(f"  {len(results)} images found in {time.time() - t0:.1f} s")

    if len(results) == 0:
        print("Nothing to do.")
        return

    by_band = {}
    for row in results:
        by_band.setdefault(row["energy_bandpassname"], 0)
        by_band[row["energy_bandpassname"]] += 1
    for band in sorted(by_band):
        print(f"    {band}: {by_band[band]} visits")

    if args.limit:
        results = results[: args.limit]

    if args.dry_run:
        print(f"\nDry run: would download {len(results)} cutouts to {args.outdir}")
        print(f"Extensions kept: {', '.join(keep)}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nDownloading {len(results)} cutouts to {args.outdir} (keeping {', '.join(keep)}) ...")

    t0 = time.time()
    records = [None] * len(results)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, row, ra, dec, args.outdir, keep, overwrite=args.overwrite): i
            for i, row in enumerate(results)
        }
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            records[futures[fut]] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(results):
                print(f"  {done}/{len(results)} ({time.time() - t0:.0f} s)")

    ok = [i for i, r in enumerate(records) if r is not None]
    n_cached = sum(1 for i in ok if records[i]["cached"])
    print(
        f"\nRetrieved {len(ok)}/{len(results)} cutouts in {(time.time() - t0) / 60:.2f} min "
        f"({n_cached} already on disk, {len(results) - len(ok)} failed)"
    )

    if not ok:
        return

    summary = results[ok].copy()
    summary.remove_column("uri")
    summary.remove_column("raw_uri")
    for key in ("filename", "central_wavelength", "bandwidth", "wavelength_min", "wavelength_max", "x_pix", "y_pix"):
        summary[key] = [records[i][key] for i in ok]
    summary.sort("central_wavelength")
    summary["cutout_index"] = range(1, len(summary) + 1)

    summary_path = os.path.join(args.outdir, "spherex_cosmos_summary.ecsv")
    summary.write(summary_path, format="ascii.ecsv", overwrite=True)
    print(f"Wrote summary table: {summary_path}")

    lam = np.asarray(summary["central_wavelength"], dtype=float)
    print(f"\nWavelength coverage at field centre: {lam.min():.3f} - {lam.max():.3f} um")
    # Bin onto the native channel grid to show how filled the spectrum is.
    edges = np.geomspace(lam.min(), lam.max() * 1.001, 103)
    filled = np.count_nonzero(np.histogram(lam, bins=edges)[0])
    print(f"Distinct channels sampled: {filled}/102 (R~35-130 grid)")

    if args.mef:
        print(f"\nBuilding combined MEF -> {args.mef}")
        build_mef(summary, args.outdir, args.mef)
        print(f"Wrote {args.mef}")


if __name__ == "__main__":
    main()
