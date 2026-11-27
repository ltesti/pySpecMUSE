# SPDX-License-Identifier: LGPL-3.0-or-later
#
# This file is part of pySpecMUSE.
# Copyright (C) 2023-2026 the pySpecMUSE authors (see README)
#
# pySpecMUSE is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pySpecMUSE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pySpecMUSE (files COPYING.LESSER and COPYING). If not, see
# <https://www.gnu.org/licenses/>.

"""
vista_gaia_astrocorr.py
=======================
Astrometric registration of a VISTA source catalog onto the Gaia DR3 frame.

Workflow
--------
Stage 1 — Initial cross-match
    Load a VISTA catalog, select a cone, query Gaia DR3, match at a user-
    supplied maximum separation, compute median ΔRA / ΔDec and their σ.

Stage 2 — Corrected cross-match + radius refinement
    Apply Stage-1 offsets to VISTA coordinates, repeat the cross-match,
    then use the separation histogram to find a tighter "reliable" radius
    (the first local minimum to the right of the peak, i.e. the valley
    between true matches and the random-match background).

Stage 3 — Final offsets on original coordinates
    Re-derive ΔRA / ΔDec using the *original* VISTA coordinates and the
    reliable-match subset, so the corrections are self-consistent.

Output
------
  * ASCII table: original VISTA columns + RA_CORR, DE_CORR
    (names set by the module constants RA_CORR_COL / DE_CORR_COL)
  * Multi-panel diagnostic PDF/PNG

Usage (as a script)
-------------------
    python vista_gaia_astrocorr.py \\
        --vista   J_A+A_572_A116_catalog.dat.gz.fits \\
        --ra      160.96   --dec  -59.535 \\
        --radius  0.6      --max_sep  3.0

Usage (as a module)
-------------------
    from vista_gaia_astrocorr import run_pipeline
    result = run_pipeline(
        vista_file="...", ra_center=160.96, dec_center=-59.535,
        radius_deg=0.6,  max_sep_arcsec=3.0
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import astropy.io.ascii as aio
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.signal import argrelmin


# ── Output column names ───────────────────────────────────────────────────────
# Names of the corrected-coordinate columns written to the output table.
# Also used as defaults by CubeMUSE.correct_astrometry() (VISTA reference).
RA_CORR_COL = "RA_CORR"
DE_CORR_COL = "DE_CORR"


def _get_gaia():
    """
    Import and configure astroquery's Gaia service lazily, so that this module
    (and its matching / plotting helpers) can be imported without astroquery.

    Sets the default table so that any helper method that uses MAIN_GAIA_TABLE
    also hits DR3.  The actual query in query_gaia_dr3() uses explicit ADQL so
    it is not affected by version differences in how this attribute is handled.
    """
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1      # return all rows (some astroquery versions cap at 2000)
    return Gaia


# ═══════════════════════════════════════════════════════════════════════════════
#  Coordinate parsing helper (from CorrelationVISTA.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def src_coo(name: str, verbose: bool = False) -> tuple[float, float]:
    """
    Parse a sexagesimal source name of the form
        HHMMSSsss±DDMMSSs
    and return (RA, Dec) in decimal degrees.
    """
    rah = name[0:2];  ram = name[2:4];  ras = name[4:9]
    deg = name[9:12]; dem = name[12:14]; des = name[14:17]
    if verbose:
        print(f"{name}  ra={rah}:{ram}:{ras}  de={deg}:{dem}:{des}")
    sgd = -1 if name[9] == "-" else 1.0
    ra  = 15.0 * (float(rah) + (float(ram) + float(ras) / 60.0) / 60.0)
    dec = sgd  * (abs(float(deg)) + (float(dem) + float(des) / 60.0) / 60.0)
    return ra, dec


def add_coords_from_name(tab: Table,
                          name_col: str = "Name") -> Table:
    """
    Add RA_VISTA / DE_VISTA columns by parsing the name column with src_coo.
    Returns modified table (in-place).
    """
    n = len(tab)
    vra = np.zeros(n)
    vde = np.zeros(n)
    for i in range(n):
        vra[i], vde[i] = src_coo(str(tab[i][name_col]))
    tab["RA_VISTA"] = vra
    tab["DE_VISTA"] = vde
    return tab


# ═══════════════════════════════════════════════════════════════════════════════
#  Core pipeline helpers
# ═══════════════════════════════════════════════════════════════════════════════

def select_cone(tab: Table,
                ra_center: float, dec_center: float,
                radius_deg: float,
                ra_col: str = "RA_VISTA",
                dec_col: str = "DE_VISTA") -> Table:
    """
    Return the sub-table of rows within *radius_deg* of (ra_center, dec_center).
    """
    coo_all = SkyCoord(ra=np.array(tab[ra_col]) * u.deg,
                       dec=np.array(tab[dec_col]) * u.deg)
    ctr = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)
    sel = ctr.separation(coo_all).deg <= radius_deg
    n_sel = sel.sum()
    print(f"  Cone selection: {n_sel} / {len(tab)} VISTA sources within "
          f"{radius_deg:.3f} deg of ({ra_center:.4f}, {dec_center:+.4f})")
    return tab[sel]


GAIA_TABLES = {
    "DR1":  "gaiadr1.gaia_source",
    "DR2":  "gaiadr2.gaia_source",
    "EDR3": "gaiaedr3.gaia_source",
    "DR3":  "gaiadr3.gaia_source",
}

# columns present in gaia_source for every release from DR1 to DR3
GAIA_DEFAULT_COLUMNS = ["source_id", "ra", "dec", "ref_epoch", "phot_g_mean_mag",
                        "parallax", "pmra", "pmdec"]


def gaia_table_name(release: str = "DR3") -> str:
    """
    ESA Gaia archive table for a data release ('DR2', 'EDR3', 'DR3', ...).
    Releases not listed in GAIA_TABLES map to 'gaia<release>.gaia_source'
    (e.g. 'DR4' -> 'gaiadr4.gaia_source').  A full table name containing a
    dot (e.g. 'gaiadr3.gaia_source') is returned unchanged.
    """
    rel = str(release).strip()
    if "." in rel:
        return rel
    rel = rel.upper().replace("GAIA", "").strip()
    return GAIA_TABLES.get(rel, f"gaia{rel.lower()}.gaia_source")


def query_gaia(ra_center: float, dec_center: float, radius_deg: float,
               release: str = "DR3", columns: list[str] | None = None) -> Table:
    """
    Retrieve the Gaia sources of a given data release within a cone of
    *radius_deg* around (ra_center, dec_center), with an explicit ADQL query
    to the ESA Gaia archive (see query_gaia_dr3 for why ADQL is used).

    Note: positions are at the Gaia reference epoch (column ref_epoch,
    J2016.0 for DR3/EDR3, J2015.5 for DR2), not propagated to another epoch.
    """
    table = gaia_table_name(release)
    cols = ", ".join(columns if columns else GAIA_DEFAULT_COLUMNS)
    adql = f"""
    SELECT {cols}
    FROM {table}
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center:.8f}, {dec_center:.8f}, {radius_deg:.8f})
    ) = 1
    """
    print(f"  Querying Gaia {release} ({table}) — cone r={radius_deg * 3600:.1f} arcsec "
          f"centred ({ra_center:.5f}, {dec_center:+.5f}) …")
    import contextlib, io
    # astroquery prints the archive status page on import and the whole HTML page
    # when the archive returns an error: capture its output and raise a readable
    # message instead
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            Gaia     = _get_gaia()
            job      = Gaia.launch_job_async(adql, verbose=False)
            tab_gaia = job.get_results()
    except Exception as exc:
        down = "maintenance" in captured.getvalue().lower()
        raise RuntimeError(
            f"Gaia archive query to {table} failed ({type(exc).__name__}: {exc})."
            + ("\n  The ESA Gaia archive reports it is down for maintenance: retry later."
               if down else
               "\n  Check network access to gea.esac.esa.int and that the release/table exists.")
            + "\n  A catalog obtained otherwise can be passed to CubeMUSE.correct_astrometry"
              " with gaia_catalog=..."
        ) from exc
    if tab_gaia is None or len(tab_gaia) == 0:
        raise RuntimeError(
            f"Gaia query to {table} returned no results.  Check:\n"
            "  • network/proxy access to gea.esac.esa.int (and that the archive is not down)\n"
            "  • the release name and that the cone is inside the survey footprint"
        )
    print(f"  → {len(tab_gaia):,} Gaia {release} sources retrieved.")
    return tab_gaia


def query_gaia_dr3(ra_center: float, dec_center: float,
                   radius_deg: float) -> Table:
    """
    Retrieve Gaia DR3 sources within a cone of *radius_deg* around
    (ra_center, dec_center) using an explicit ADQL query.

    Using launch_job_async + ADQL is more reliable than query_object_async
    because it bypasses version-specific behaviour of MAIN_GAIA_TABLE and
    any default row-limit caps.  The columns returned are those used by the
    astrometric pipeline; add or remove as needed.
    """
    adql = f"""
    SELECT
        source_id, ra, dec,
        parallax, parallax_error, parallax_over_error,
        pmra, pmra_error,
        pmdec, pmdec_error,
        phot_g_mean_mag, ruwe
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center:.8f}, {dec_center:.8f}, {radius_deg:.8f})
    ) = 1
    """
    print(f"  Querying Gaia DR3 — cone r={radius_deg:.4f} deg "
          f"centred ({ra_center:.4f}, {dec_center:+.4f}) …")
    print(f"  (using launch_job_async / ADQL against gaiadr3.gaia_source)")
    Gaia     = _get_gaia()
    job      = Gaia.launch_job_async(adql, verbose=False)
    tab_gaia = job.get_results()
    if tab_gaia is None or len(tab_gaia) == 0:
        raise RuntimeError(
            "Gaia DR3 query returned no results.  Check:\n"
            "  • network/proxy access to gea.esac.esa.int\n"
            "  • ra_center / dec_center are inside the survey footprint\n"
            "  • radius_deg is not so large the query times out (try < 1 deg)"
        )
    print(f"  → {len(tab_gaia):,} Gaia DR3 sources retrieved.")
    return tab_gaia


def pair_offsets(src_ra: np.ndarray, src_dec: np.ndarray,
                 ref_ra: np.ndarray, ref_dec: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Tangent-plane offsets of each ref position relative to its src position.

    Returns
    -------
    dra  : ΔRA·cos(Dec) [arcsec], positive towards East
    ddec : ΔDec         [arcsec], positive towards North
    """
    coo_src = SkyCoord(ra=src_ra * u.deg, dec=src_dec * u.deg)
    coo_ref = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)
    dra, ddec = coo_src.spherical_offsets_to(coo_ref)
    return dra.to_value(u.arcsec), ddec.to_value(u.arcsec)


def apply_offsets(ra: np.ndarray, dec: np.ndarray,
                  dra_arcsec: float, ddec_arcsec: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift positions by a tangent-plane offset (ΔRA·cos(Dec), ΔDec) in arcsec,
    i.e. the inverse of pair_offsets().  Returns (ra, dec) in degrees.
    """
    coo = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg)
    new = coo.spherical_offsets_by(dra_arcsec * u.arcsec, ddec_arcsec * u.arcsec)
    return new.ra.deg, new.dec.deg


def cross_match(src_ra: np.ndarray, src_dec: np.ndarray,
                ref_ra: np.ndarray, ref_dec: np.ndarray,
                max_sep_arcsec: float
                ) -> tuple[np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray, np.ndarray]:
    """
    Nearest-neighbour match of (src_ra, src_dec) → (ref_ra, ref_dec).

    Returns
    -------
    idx_ref   : int array, index into ref catalog for each src source
    sep_arcsec: float array, angular separation in arcsec
    dra       : float array, ΔRA·cos(Dec) of ref relative to src [arcsec]
                (tangent-plane offset, see pair_offsets)
    ddec      : float array, ΔDec of ref relative to src            [arcsec]
    good      : bool array, True where sep ≤ max_sep_arcsec
    """
    coo_src = SkyCoord(ra=src_ra  * u.deg, dec=src_dec  * u.deg)
    coo_ref = SkyCoord(ra=ref_ra  * u.deg, dec=ref_dec  * u.deg)
    idx_ref, sep, _ = coo_src.match_to_catalog_sky(coo_ref)
    dra, ddec = pair_offsets(src_ra, src_dec, ref_ra[idx_ref], ref_dec[idx_ref])
    good = sep.arcsec <= max_sep_arcsec
    return idx_ref, sep.arcsec, dra, ddec, good


def cross_match_in_box(src_ra: np.ndarray, src_dec: np.ndarray,
                       ref_ra: np.ndarray, ref_dec: np.ndarray,
                       max_sep_arcsec: float,
                       dra_range: tuple[float, float],
                       ddec_range: tuple[float, float]
                       ) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
    """
    Like cross_match(), but for each src source only the ref sources whose
    offset (ΔRA·cos(Dec), ΔDec) [arcsec] falls inside the box
    dra_range × ddec_range are considered; among those, the one closest to the
    box centre is chosen.  Useful in crowded fields when the offset is larger
    than the typical distance between reference sources, so that the plain
    nearest neighbour is mostly a chance association.

    Sources with no candidate in the box get the plain nearest neighbour and
    good = False.  Returns the same tuple as cross_match().
    """
    dra_lo, dra_hi = sorted(dra_range)
    ddec_lo, ddec_hi = sorted(ddec_range)
    dra_c, ddec_c = 0.5 * (dra_lo + dra_hi), 0.5 * (ddec_lo + ddec_hi)

    # start from the plain nearest neighbour (used where the box is empty)
    idx_ref, sep, dra, ddec, _ = cross_match(src_ra, src_dec, ref_ra, ref_dec, max_sep_arcsec)
    good = np.zeros(len(src_ra), dtype=bool)

    # all candidate pairs within max_sep_arcsec
    coo_src = SkyCoord(ra=src_ra * u.deg, dec=src_dec * u.deg)
    coo_ref = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)
    i_src, i_ref, psep, _ = coo_ref.search_around_sky(coo_src, max_sep_arcsec * u.arcsec)
    if len(i_src) == 0:
        return idx_ref, sep, dra, ddec, good
    pdra, pddec = pair_offsets(src_ra[i_src], src_dec[i_src], ref_ra[i_ref], ref_dec[i_ref])
    inbox = (pdra >= dra_lo) & (pdra <= dra_hi) & (pddec >= ddec_lo) & (pddec <= ddec_hi)
    i_src, i_ref, psep, pdra, pddec = (a[inbox] for a in (i_src, i_ref, psep.arcsec, pdra, pddec))

    # for each src keep the in-box candidate closest to the box centre
    dist_c = np.hypot(pdra - dra_c, pddec - ddec_c)
    order = np.lexsort((dist_c, i_src))
    first = np.unique(i_src[order], return_index=True)[1]
    sel = order[first]
    s = i_src[sel]
    idx_ref[s], sep[s], dra[s], ddec[s] = i_ref[sel], psep[sel], pdra[sel], pddec[sel]
    good[s] = True
    return idx_ref, sep, dra, ddec, good


def offset_stats(dra: np.ndarray, ddec: np.ndarray,
                 mask: np.ndarray,
                 label: str = "") -> tuple[float, float, float, float]:
    """
    Compute and print median offsets and their standard deviations.

    Returns
    -------
    (dra_med, dra_std, ddec_med, ddec_std)  all in arcsec
    """
    dr = dra[mask]; dd = ddec[mask]
    dra_med,  dra_std  = float(np.median(dr)), float(np.std(dr))
    ddec_med, ddec_std = float(np.median(dd)), float(np.std(dd))
    print(f"  [{label}]  N={mask.sum():6d}  "
          f"ΔRA = {dra_med:+.4f} ± {dra_std:.4f}\"  "
          f"ΔDec = {ddec_med:+.4f} ± {ddec_std:.4f}\"")
    return dra_med, dra_std, ddec_med, ddec_std


def find_reliable_radius(sep: np.ndarray,
                         nbins: int = 100,
                         fallback_fraction: float = 0.15,
                         stop_at_empty: bool = False
                         ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine a reliable association radius from the separation histogram.

    The algorithm finds the first local minimum to the right of the histogram
    peak — the valley between true (Rayleigh-distributed) matches and the flat
    random-match background.  If no local minimum is found, falls back to the
    first bin where the count drops below *fallback_fraction* × peak.

    If *stop_at_empty* is True, the radius is capped at the first empty bin
    to the right of the peak.  Useful for sparse samples (e.g. a single MUSE
    field), where the histogram has flat runs of zeros that argrelmin, which
    only finds strict minima, does not recognise as a valley.

    Returns
    -------
    r_reliable : float, arcsec
    counts     : histogram counts
    edges      : histogram bin edges
    centers    : bin centres
    """
    counts, edges = np.histogram(sep, bins=nbins, range=(0.0, sep.max()))
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_i  = int(np.argmax(counts))

    # local minima strictly to the right of the peak
    local_min = argrelmin(counts[peak_i:], order=3)[0]
    if len(local_min) > 0:
        r_reliable = float(centers[peak_i + local_min[0]])
    else:
        threshold  = fallback_fraction * counts[peak_i]
        after_peak = np.where(counts[peak_i:] < threshold)[0]
        r_reliable = (float(centers[peak_i + after_peak[0]])
                      if len(after_peak) else float(centers[-1]))

    if stop_at_empty:
        empty = np.where(counts[peak_i:] == 0)[0]
        if len(empty):
            r_reliable = min(r_reliable, float(centers[peak_i + empty[0]]))

    return r_reliable, counts, edges, centers


# ═══════════════════════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════════

def _separation_panel(ax, sep, max_sep, r_reliable=None,
                      nbins=80, title="", color_hist="steelblue"):
    """Single-axis separation histogram."""
    rng = (0.0, max_sep)
    n, b, _ = ax.hist(sep[sep <= max_sep], bins=nbins, range=rng,
                      color=color_hist, alpha=0.7, histtype="stepfilled",
                      edgecolor="none")
    ax.hist(sep[sep <= max_sep], bins=nbins, range=rng,
            histtype="step", color="k", linewidth=0.8)
    if r_reliable is not None:
        ax.axvline(r_reliable, color="crimson", linewidth=1.5,
                   linestyle="--", label=f"reliable: {r_reliable:.2f}\"")
        ax.legend(fontsize=16)
    ax.set_xlabel("Separation (arcsec)", fontsize=18)
    ax.set_ylabel("N", fontsize=18)
    ax.set_title(title, fontsize=18)


def _offset_panel(ax, dra, ddec, good,
                  r1=None, r2=None,
                  xlim=(-6, 6), ylim=(-3, 3), title=""):
    """RA vs Dec offset scatter plot (colour-coded by quality)."""
    ax.plot(dra[good], ddec[good], ".", color="steelblue",
            alpha=0.4, markersize=2, label="matched")
    if r1 is not None:
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Delta$RA (arcsec)", fontsize=18)
    ax.set_ylabel(r"$\Delta$Dec (arcsec)", fontsize=18)
    ax.set_title(title, fontsize=18)


def _offset_hist_panel(ax, dra, ddec, good, half_range=3.0, nbins=60, title=""):
    """Overlapping ΔRA / ΔDec histograms."""
    rng = (-half_range, half_range)
    ax.hist(dra[good],  bins=nbins, range=rng,
            color="steelblue", alpha=0.5, label=r"$\Delta$RA")
    ax.hist(ddec[good], bins=nbins, range=rng,
            color="tomato",    alpha=0.5, label=r"$\Delta$Dec")
    ax.axvline(np.median(dra[good]),  color="steelblue",
               linewidth=1.5, linestyle="--")
    ax.axvline(np.median(ddec[good]), color="tomato",
               linewidth=1.5, linestyle="--")
    ax.set_xlim(*rng)
    ax.set_xlabel("Offset (arcsec)", fontsize=18)
    ax.set_ylabel("N", fontsize=18)
    ax.legend(fontsize=16)
    ax.set_title(title, fontsize=18)


def _image_panel(ax, image, wcs, layers, circle_radius_arcsec=0.9, title=""):
    """
    Greyscale image (WCS axes) with one or more layers of associated sources.

    Each layer is a dictionary with keys 'ref_ra', 'ref_dec' (reference catalog
    sources, drawn as empty circles of radius circle_radius_arcsec), 'src_ra',
    'src_dec' (associated image sources, drawn as plus symbols) [deg], and
    optionally 'ref_color', 'src_color' (default red / yellow), 'ref_label',
    'src_label' (None: not shown in the legend).  Layers are drawn in order, the last one on top.
    """
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    from astropy.visualization import simple_norm
    from astropy.wcs.utils import proj_plane_pixel_scales

    # same stretch and colormap as the plots of CubeMUSE.set_stars_for_apc
    # (aperturesplots.plotstars)
    finite = np.isfinite(image)
    norm = simple_norm(image[finite], "sqrt", percent=92.0)
    ax.imshow(np.where(finite, image, np.nan), cmap="Greys", origin="lower",
              norm=norm, interpolation="nearest")
    ny, nx = image.shape
    r_pix = circle_radius_arcsec / (np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)

    for k, lay in enumerate(layers):
        ref_color = lay.get("ref_color", "red")
        src_color = lay.get("src_color", "yellow")
        # reference sources: empty circles with a fixed angular radius
        rx, ry = wcs.world_to_pixel(SkyCoord(ra=np.asarray(lay["ref_ra"]) * u.deg,
                                             dec=np.asarray(lay["ref_dec"]) * u.deg))
        ax.add_collection(PatchCollection(
            [Circle((x, y), r_pix) for x, y in zip(rx, ry)],
            facecolor="none", edgecolor=ref_color, linewidth=0.8, zorder=2 + 2 * k))
        # image sources associated with them
        sx, sy = wcs.world_to_pixel(SkyCoord(ra=np.asarray(lay["src_ra"]) * u.deg,
                                             dec=np.asarray(lay["src_dec"]) * u.deg))
        ax.plot(sx, sy, "+", color=src_color, markersize=18, markeredgewidth=0.8,
                zorder=3 + 2 * k)
        # legend proxies (the circles are a PatchCollection)
        # (a label set to None is left out of the legend)
        ref_lbl = lay.get("ref_label", f"catalog source (r = {circle_radius_arcsec:g}\")")
        src_lbl = lay.get("src_label", "associated MUSE star")
        if ref_lbl is not None:
            ax.plot([], [], "o", mfc="none", mec=ref_color, markersize=9, label=ref_lbl)
        if src_lbl is not None:
            ax.plot([], [], "+", color=src_color, markersize=9, markeredgewidth=1.5,
                    label=src_lbl)
    # legend below the image, so that it does not hide any source
    ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1,
              frameon=False)
    ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel("RA", fontsize=18); ax.set_ylabel("Dec", fontsize=18)
    ax.set_title(title, fontsize=22)


def make_diagnostic_figure(
        # Stage 1
        sep1, dra1, ddec1, good1, max_sep,
        dra1_med, dra1_std, ddec1_med, ddec1_std,
        # Stage 2
        sep2, dra2, ddec2, good2, r_reliable,
        dra2_med, dra2_std, ddec2_med, ddec2_std,
        # Stage 3
        sep3, dra3, ddec3, good3,
        dra3_med, dra3_std, ddec3_med, ddec3_std,
        # misc
        output_path: str | Path | None = None,
        title: str = "VISTA → Gaia DR3 astrometric correction",
        image_panels: list[dict] | None = None,
        ) -> plt.Figure:
    """
    Three-row, three-column diagnostic figure.

    Row 1 — Stage 1 (initial cross-match, max_sep radius)
    Row 2 — Stage 2 (corrected coords, max_sep radius, reliable radius marked)
    Row 3 — Stage 3 (original coords, reliable radius)

    If image_panels is given, a fourth row shows the images side by side (see
    _image_panel).  Each element is a dictionary with keys 'image', 'wcs'
    (celestial WCS used to place the sources and label the axes), 'layers'
    (list of source layers, see _image_panel), and optionally
    'circle_radius_arcsec' and 'title'.
    """
    has_im = bool(image_panels)
    nrows = 4 if has_im else 3
    fig = plt.figure(figsize=(21, 33) if has_im else (21, 19))
    fig.suptitle(title, fontsize=26,
                 fontweight="bold", y=0.995 if has_im else 0.99)
    gs = gridspec.GridSpec(nrows, 3, figure=fig,
                           height_ratios=[1, 1, 1, 2.0] if has_im else None,
                           hspace=0.55 if has_im else 0.6, wspace=0.38,
                           left=0.08, right=0.97,
                           top=0.95 if has_im else 0.92,
                           bottom=0.06 if has_im else 0.05)

    offset_half = min(3.0 * max(abs(dra1_med) + 3*dra1_std,
                                abs(ddec1_med) + 3*ddec1_std), max_sep)

    rows = [
        # (sep, dra, ddec, good, r_rel, dra_m, dra_s, ddec_m, ddec_s, label)
        (sep1, dra1, ddec1, good1, None,
         dra1_med, dra1_std, ddec1_med, ddec1_std,
         "Stage 1 — initial"),
        (sep2, dra2, ddec2, good2, r_reliable,
         dra2_med, dra2_std, ddec2_med, ddec2_std,
         "Stage 2 — corrected coords"),
        (sep3, dra3, ddec3, good3, r_reliable,
         dra3_med, dra3_std, ddec3_med, ddec3_std,
         "Stage 3 — final (original coords, reliable radius)"),
    ]

    for row_i, (sep, dra, ddec, good, r_rel,
                dm_r, ds_r, dm_d, ds_d, lbl) in enumerate(rows):
        ax0 = fig.add_subplot(gs[row_i, 0])
        ax1 = fig.add_subplot(gs[row_i, 1])
        ax2 = fig.add_subplot(gs[row_i, 2])

        colors = ["steelblue", "mediumseagreen", "darkorange"]

        _separation_panel(
            ax0, sep, max_sep,
            r_reliable=r_rel,
            title=f"{lbl}\nN_matched = {good.sum():,}",
            color_hist=colors[row_i])

        _offset_panel(
            ax1, dra, ddec, good,
            xlim=(-offset_half * 1.5, offset_half * 1.5),
            ylim=(-offset_half * 1.5, offset_half * 1.5),
            title=f"ΔRA={dm_r:+.3f}\"±{ds_r:.3f}\"\n"
                  f"ΔDec={dm_d:+.3f}\"±{ds_d:.3f}\"")

        _offset_hist_panel(
            ax2, dra, ddec, good,
            half_range=offset_half * 1.5,
            title=f"ΔRA (blue) / ΔDec (red)\nmedians dashed")

    if has_im:
        gs_im = gs[3, :].subgridspec(1, len(image_panels), wspace=0.3)
        for k, pan in enumerate(image_panels):
            ax_im = fig.add_subplot(gs_im[0, k], projection=pan["wcs"])
            _image_panel(
                ax_im, pan["image"], pan["wcs"], pan["layers"],
                circle_radius_arcsec=pan.get("circle_radius_arcsec", 0.9),
                title=pan.get("title", ""))

    # tick labels at twice the default size (as all the other labels)
    tick_size = 2 * plt.rcParams["xtick.labelsize"] if isinstance(
        plt.rcParams["xtick.labelsize"], (int, float)) else 2 * plt.rcParams["font.size"]
    for ax in fig.axes:
        if hasattr(ax, "coords"):           # WCSAxes (image panels)
            for c in ax.coords:
                c.set_ticklabel(size=tick_size)
        else:
            ax.tick_params(labelsize=tick_size)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Diagnostic figure saved → {output_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
        vista_file:      str,
        ra_center:       float,
        dec_center:      float,
        radius_deg:      float,
        max_sep_arcsec:  float  = 3.0,
        ra_col:          str    = "RA_VISTA",
        dec_col:         str    = "DE_VISTA",
        parse_coords:    bool   = False,
        name_col:        str    = "Name",
        gaia_ra_col:     str    = "ra",
        gaia_dec_col:    str    = "dec",
        output_file:     str    = "vista_corrected.dat",
        plot_file:       str    = "vista_gaia_diagnostic.pdf",
        hist_nbins:      int    = 100,
        ) -> dict:
    """
    Run the full astrometric correction pipeline.

    :param vista_file: Path to the VISTA catalogue (FITS or ASCII).
    :param ra_center: Field centre right ascension in degrees (J2000).
    :param dec_center: Field centre declination in degrees (J2000).
    :param radius_deg: Cone radius for source selection in degrees.
    :param max_sep_arcsec: Initial cross-match radius in arcseconds.
    :param ra_col: VISTA right ascension column, ignored when parsing names.
    :param dec_col: VISTA declination column, ignored when parsing names.
    :param parse_coords: Parse coordinates from the source name column.
    :param name_col: Source name column used when parsing coordinates.
    :param gaia_ra_col: Gaia right ascension column.
    :param gaia_dec_col: Gaia declination column.
    :param output_file: Path for the corrected catalogue.
    :param plot_file: Path for the diagnostic figure (PDF or PNG).
    :param hist_nbins: Number of bins in the separation histogram.
    :return: Dictionary containing the corrected ``table``, final offsets
        ``dra_final_arcsec`` and ``ddec_final_arcsec``, their standard deviations
        ``dra_final_std`` and ``ddec_final_std``, the association radius
        ``r_reliable_arcsec``, and the diagnostic ``figure``.
    """

    # ── 0. Load VISTA catalog ─────────────────────────────────────────────────
    print("\n━━━ Loading VISTA catalog ━━━")
    ext = Path(vista_file).suffix.lower()
    if ext in (".fits", ".gz"):
        tab_vista_full = Table.read(vista_file, format="fits")
    else:
        tab_vista_full = Table.read(vista_file, format="ascii")
    print(f"  Loaded {len(tab_vista_full):,} sources from {vista_file}")

    if parse_coords:
        print(f"  Parsing coordinates from column '{name_col}' …")
        tab_vista_full = add_coords_from_name(tab_vista_full, name_col)
        ra_col, dec_col = "RA_VISTA", "DE_VISTA"

    # ── 1. Cone selection ─────────────────────────────────────────────────────
    print("\n━━━ Stage 0 — Cone selection ━━━")
    tab_vista = select_cone(tab_vista_full,
                             ra_center, dec_center, radius_deg,
                             ra_col=ra_col, dec_col=dec_col)
    if len(tab_vista) == 0:
        raise ValueError("No VISTA sources in the specified cone.")

    vista_ra  = np.array(tab_vista[ra_col],  dtype=float)
    vista_dec = np.array(tab_vista[dec_col], dtype=float)

    # ── 2. Query Gaia DR3 ─────────────────────────────────────────────────────
    print("\n━━━ Querying Gaia DR3 ━━━")
    tab_gaia = query_gaia_dr3(ra_center, dec_center, radius_deg)
    gaia_ra  = np.array(tab_gaia[gaia_ra_col],  dtype=float)
    gaia_dec = np.array(tab_gaia[gaia_dec_col], dtype=float)

    # ── 3. Stage 1 — initial cross-match ─────────────────────────────────────
    print("\n━━━ Stage 1 — initial cross-match ━━━")
    idx1, sep1, dra1, ddec1, good1 = cross_match(
        vista_ra, vista_dec, gaia_ra, gaia_dec, max_sep_arcsec)

    dra1_med, dra1_std, ddec1_med, ddec1_std = offset_stats(
        dra1, ddec1, good1, label="Stage 1")

    # ── 4. Stage 2 — corrected cross-match ───────────────────────────────────
    print("\n━━━ Stage 2 — corrected cross-match ━━━")
    vista_ra_s1, vista_dec_s1 = apply_offsets(vista_ra, vista_dec,
                                              dra1_med, ddec1_med)

    idx2, sep2, dra2, ddec2, good2 = cross_match(
        vista_ra_s1, vista_dec_s1, gaia_ra, gaia_dec, max_sep_arcsec)

    dra2_med, dra2_std, ddec2_med, ddec2_std = offset_stats(
        dra2, ddec2, good2, label="Stage 2 (stage-1 corrected)")

    # ── 5. Reliable radius from histogram ─────────────────────────────────────
    print("\n━━━ Stage 2b — reliable radius from histogram ━━━")
    r_reliable, hist_counts, hist_edges, hist_centers = find_reliable_radius(
        sep2[good2], nbins=hist_nbins)
    print(f"  Reliable association radius: {r_reliable:.3f} arcsec  "
          f"(from {good2.sum():,} matches, "
          f"histogram valley / fallback criterion)")

    # ── 6. Stage 3 — final offsets on original coordinates ───────────────────
    print("\n━━━ Stage 3 — final offsets (original VISTA, reliable radius) ━━━")
    idx3, sep3, dra3, ddec3, good3 = cross_match(
        vista_ra, vista_dec, gaia_ra, gaia_dec, r_reliable)

    dra3_med, dra3_std, ddec3_med, ddec3_std = offset_stats(
        dra3, ddec3, good3, label="Stage 3 — FINAL")

    print(f"\n  ══ FINAL CORRECTIONS ══")
    print(f"  ΔRA  = {dra3_med:+.5f} ± {dra3_std:.5f}  arcsec")
    print(f"  ΔDec = {ddec3_med:+.5f} ± {ddec3_std:.5f}  arcsec")
    print(f"  (N reliable matches = {good3.sum():,} / {len(tab_vista):,} VISTA sources)")

    # ── 7. Apply final offsets ────────────────────────────────────────────────
    vista_ra_corr, vista_dec_corr = apply_offsets(vista_ra, vista_dec,
                                                  dra3_med, ddec3_med)

    tab_vista[RA_CORR_COL] = vista_ra_corr
    tab_vista[DE_CORR_COL] = vista_dec_corr
    tab_vista[RA_CORR_COL].description = (
        "VISTA RA corrected to Gaia DR3 frame  [deg J2000]")
    tab_vista[DE_CORR_COL].description = (
        "VISTA Dec corrected to Gaia DR3 frame [deg J2000]")

    # ── 8. Write output table ─────────────────────────────────────────────────
    print(f"\n━━━ Writing output table ━━━")
    aio.write(tab_vista, output_file, overwrite=True)
    print(f"  → {output_file}  ({len(tab_vista):,} rows)")

    # ── 9. Diagnostic figure ──────────────────────────────────────────────────
    print("\n━━━ Producing diagnostic figure ━━━")
    fig = make_diagnostic_figure(
        sep1, dra1, ddec1, good1, max_sep_arcsec,
        dra1_med, dra1_std, ddec1_med, ddec1_std,
        sep2, dra2, ddec2, good2, r_reliable,
        dra2_med, dra2_std, ddec2_med, ddec2_std,
        sep3, dra3, ddec3, good3,
        dra3_med, dra3_std, ddec3_med, ddec3_std,
        output_path=plot_file,
    )

    print("\n✓ Pipeline complete.")
    return {
        "table":             tab_vista,
        "dra_final_arcsec":  dra3_med,
        "dra_final_std":     dra3_std,
        "ddec_final_arcsec": ddec3_med,
        "ddec_final_std":    ddec3_std,
        "r_reliable_arcsec": r_reliable,
        "figure":            fig,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Command-line interface
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vista",    required=True,
                   help="Path to the VISTA catalog (FITS or ASCII).")
    p.add_argument("--ra",       required=True, type=float,
                   help="Field centre RA (deg, J2000).")
    p.add_argument("--dec",      required=True, type=float,
                   help="Field centre Dec (deg, J2000).")
    p.add_argument("--radius",   required=True, type=float,
                   help="Cone radius for source selection (deg).")
    p.add_argument("--max_sep",  default=3.0, type=float,
                   help="Initial cross-match radius (arcsec). [default: 3.0]")
    p.add_argument("--ra_col",   default="RA_VISTA",
                   help="VISTA RA column name. [default: RA_VISTA]")
    p.add_argument("--dec_col",  default="DE_VISTA",
                   help="VISTA Dec column name. [default: DE_VISTA]")
    p.add_argument("--parse_coords", action="store_true",
                   help="Parse coordinates from source Name column (src_coo).")
    p.add_argument("--name_col", default="Name",
                   help="Source name column (used with --parse_coords). "
                        "[default: Name]")
    p.add_argument("--out",      default="vista_corrected.dat",
                   help="Output catalog filename. [default: vista_corrected.dat]")
    p.add_argument("--plot",     default="vista_gaia_diagnostic.pdf",
                   help="Output plot filename. [default: vista_gaia_diagnostic.pdf]")
    p.add_argument("--hist_nbins", default=100, type=int,
                   help="Histogram bins for radius refinement. [default: 100]")
    return p


def test_gaia_connection(ra: float = 160.96, dec: float = -59.535,
                          r_deg: float = 0.01) -> None:
    """
    Quick sanity-check: query a tiny patch and report how many sources
    came back.  Call this in a notebook cell before run_pipeline() to
    verify that the ESA TAP service is reachable and that DR3 is selected.

    Example
    -------
        from vista_gaia_astrocorr import test_gaia_connection
        test_gaia_connection()
    """
    print("Testing Gaia DR3 connection …")
    try:
        tab = query_gaia_dr3(ra, dec, r_deg)
        print(f"✓  Connection OK — {len(tab)} sources in a {r_deg:.3f}-deg cone.")
        print(f"   Columns: {tab.colnames}")
    except Exception as e:
        print(f"✗  Connection FAILED:\n   {e}")
        print("\nTroubleshooting hints:")
        print("  1. In Colab run:  !pip install astroquery --upgrade -q")
        print("  2. Check the ESA Gaia archive status: https://gea.esac.esa.int/archive/")
        print("  3. If behind a proxy, set:  Gaia.USE_NAMES_OVER_IDS = True")


if __name__ == "__main__":
    matplotlib.use("Agg")      # non-interactive backend for command-line use
    args = _build_parser().parse_args()
    run_pipeline(
        vista_file     = args.vista,
        ra_center      = args.ra,
        dec_center     = args.dec,
        radius_deg     = args.radius,
        max_sep_arcsec = args.max_sep,
        ra_col         = args.ra_col,
        dec_col        = args.dec_col,
        parse_coords   = args.parse_coords,
        name_col       = args.name_col,
        output_file    = args.out,
        plot_file      = args.plot,
        hist_nbins     = args.hist_nbins,
    )
