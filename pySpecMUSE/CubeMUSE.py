#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

import numpy as np
import multiprocessing as mp
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import os
from astropy.table import Table

from .myphotutils import get_centroids, get_stars_for_apc, runphot_ima_aps, apc_spec_single_wl, get_spec_single_wl
from .apc_plots import plot_curve_of_growth_iv, plot_apc
from .StarMUSE import StarMUSE, apc_calc_single_star
from .lines import known_lines
from .vista_gaia_astrocorr import (RA_CORR_COL, DE_CORR_COL, query_gaia, gaia_table_name,
                                   cross_match, cross_match_in_box, offset_stats,
                                   pair_offsets, apply_offsets,
                                   find_reliable_radius, make_diagnostic_figure)


class CubeMUSE(object):
    """The MUSE pointing

        The class stores the parameters for a MUSE pointing

        params:
            the class requires a dictionary as input, with the following keywords/parameters
            'code' : short code of the MUSE pointing (e.g. '6')
            'datadir' : directory for the data files and outputs
            'default_names' : set True as default, run with default names for the files
            'file' : name of input cube file (required if default_names is False)'
            'file_i_image' : name of I band image computed by the MUSE pipeline
            'file_v_image' : name of V band image computed by the MUSE pipeline
            'nproc' : maximum number of processes to spawn for multiprocessing computations

        Example:
            pars={'code' : '6',
                           'nproc' : 8,
                           'datadir' : '/Users/ltesti/Carina_test/F6/',
                           'figdir' : '/Users/ltesti/Carina_test/F6/fig/',
                           'default_names' : False,
                           'file' : 'DATA_Long6.fits',
                           'file_i_image' : 'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits',
                           'file_v_image' : 'WFM_Tr14_long_6_Johnson_V_IMAGE_FOV.fits',
                           # 'file_pos' : 'i.dat',
                           'coo_offset' : [0.0,0.0,0.0], # astrometric offset [arcsec, arcsec, pa]
                           }
        """

    def __init__(self, parameters):
        """
        set up the object: this one reads the spectrum within given boundaries
        """

        self.params = parameters

        try :
            self.pointing_code = self.params['code']
            self.datadir = self.params['datadir']
            self.default_names = self.params['default_names']
            if 'coo_offset' in self.params.keys():
                self.coo_offset = self.params['coo_offset']
            else:
                self.coo_offset = None

            self.nproc = None
            if 'nproc' in self.params.keys():
                if (self.params['nproc']):
                    self.nproc = self.params['nproc']
                    if self.nproc < 2:
                        self.nproc = None
            if 'daofind_threshold' in self.params.keys():
                self.daofind_threshold = self.params['daofind_threshold']
            else:
                self.daofind_threshold = 1.0
            if 'daofind_sigma_radius' in self.params.keys():
                self.daofind_sigma_radius = self.params['daofind_sigma_radius']
            else:
                self.daofind_sigma_radius = 2.0
            if 'saturation_magnitude' in self.params.keys():
                self.magsat = self.params['saturation_magnitude']
            else:
                self.magsat = -10.0

            self.set_parameters()
            self.set_cubewcs()

            self.has_centroids = False
            self.has_apc_stars = False
            self.has_apc_values = False
            self.has_astrometry = False

            self.set_starlis()

        except KeyError:
            raise ValueError("Cannot initiate analysis without default values for pointing_code, datadir, and default_names switch\n {}".format(self.params))

    @property
    def wcs(self):
        """Best available world coordinate system of the cube (read only)

        wcs_corrected after the astrometric correction (correct_astrometry),
        wcs_original otherwise.
        """
        if self.wcs_corrected is not None:
            return self.wcs_corrected
        return self.wcs_original

    def __setstate__(self, state):
        # objects pickled before wcs_original was introduced: rebuild it from the header
        state.pop('wcs', None)
        if 'wcs_original' not in state and 'cube_header' in state:
            state['wcs_original'] = WCS(state['cube_header'])
        state.setdefault('wcs_corrected', None)
        self.__dict__.update(state)

    def _shifted_wcs(self, dra_arcsec, ddec_arcsec):
        """Copy of wcs_original with the celestial reference point (CRVAL) shifted by the
        tangent-plane offsets dRA = dalpha*cos(Dec), dDec [arcsec], as applied to the
        stellar coordinates by correct_astrometry (SkyCoord.spherical_offsets_by)
        """
        import astropy.units as u
        new = self.wcs_original.deepcopy()
        lng, lat = new.wcs.lng, new.wcs.lat
        ref = SkyCoord(new.wcs.crval[lng] * u.deg, new.wcs.crval[lat] * u.deg)
        ref = ref.spherical_offsets_by(dra_arcsec * u.arcsec, ddec_arcsec * u.arcsec)
        crval = new.wcs.crval.copy()
        crval[lng], crval[lat] = ref.ra.deg, ref.dec.deg
        new.wcs.crval = crval
        new.wcs.set()
        return new

    def set_parameters(self):
        """Function to set/read the optional parameters

        This function sets up default parameters that can be modified if the input dictionary contains the
        appropriate keywords.

        :return: void
        """
        #
        # function to set the optional parameters
        #
        if self.default_names:
            self.file = self.datadir+'DATA_Long'+self.pointing_code+'.fits'
            self.file_i_image = self.datadir+'WFM_Tr14_long_'+self.pointing_code+'_Cousins_I_IMAGE_FOV.fits'
            self.file_v_image = self.datadir+'WFM_Tr14_long_'+self.pointing_code+'_Johnson_V_IMAGE_FOV.fits'
        else:
            try:
                self.file = self.datadir + self.params['file']
                self.file_i_image = self.datadir + self.params['file_i_image']
                self.file_v_image = self.datadir + self.params['file_v_image']
            except KeyError:
                raise ValueError(
                    "When default_names is set to False then the input parameters set has to include 'file', "
                    "'file_i_image', and 'file_v_image' keywords\n")
        if 'figdir' in self.params.keys():
            self.figdir = self.params['figdir']
        else:
            self.figdir = self.datadir

    def set_cubewcs(self,verbose=True):
        """Reads the Cube and the WCS
        """
        #
        ima_i = fits.open(self.file_i_image)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
        self.data_i = ima_i['DATA'].data
        ima_i.close()
        ima_v = fits.open(self.file_v_image)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
        self.data_v = ima_v['DATA'].data
        ima_v.close()
        #
        hdul = fits.open(self.file)
        hdul.info()
        new = hdul['DATA']
        self.cube_data = hdul['DATA'].data
        self.cube_header = hdul['DATA'].header
        hdul.close()
        # world coordinate system of the cube as delivered by the MUSE pipeline;
        # wcs_corrected is set by correct_astrometry, self.wcs (property) returns
        # wcs_corrected if available, else wcs_original
        self.wcs_original = WCS(self.cube_header)
        self.wcs_corrected = None
        # Setting the wavelength vector
        # new.header
        self.cube_cd11 = new.header['CD1_1']
        self.cube_cd12 = new.header['CD1_2']
        self.cube_crval1 = new.header['CRVAL1']
        self.cube_crpix1 = new.header['CRPIX1']
        self.cube_cunit1 = new.header['CUNIT1']
        self.cube_npix1 = np.arange(new.header['NAXIS1'])
        self.cube_cd22 = new.header['CD2_2']
        self.cube_cd21 = new.header['CD2_1']
        self.cube_crval2 = new.header['CRVAL2']
        self.cube_crpix2 = new.header['CRPIX2']
        self.cube_cunit2 = new.header['CUNIT2']
        self.cube_npix2 = np.arange(new.header['NAXIS2'])
        #
        self.cube_cdelt3 = new.header['CD3_3']
        self.cube_crval3 = new.header['CRVAL3']
        self.cube_crpix3 = new.header['CRPIX3']
        self.cube_cunit3 = new.header['CUNIT3']
        self.cube_Nwl3 = np.arange(new.header['NAXIS3'])
        #

        self.wl = (self.cube_Nwl3 - self.cube_crpix3 + 1) * self.cube_cdelt3 + self.cube_crval3
        if verbose:
            print("Cube wavelength axis definition: CDELT:{0}, CRVAL={1}, CRPIX={2}\n".format(self.cube_cdelt3, self.cube_crval3, self.cube_crpix3))

    def set_starlis(self):
        """Setup the list of StarMUSE objects

        This function creates the list of stars objects starting from the stellar positons created
        by the set_centroids() method

        :return: void
        """
        #
        if not self.has_centroids:
            self.set_centroids()
        #
        mywcs = self.wcs_original
        #
        self.stars = []
        istar = 0
        for i in range(len(self.sources_i)):
            istar += 1
            if self.stars_from_file:
                mycoo = SkyCoord((self.sources_i[i])['ALPHA_J2000'], (self.sources_i[i])['DELTA_J2000'], unit="deg")
                star_id = 'f'+str(self.pointing_code)+'_'+str((self.sources_i[i])['NUMBER'])
            else:
                mycoo = mywcs.pixel_to_world((self.sources_i[i])['x_centroid'], (self.sources_i[i])['y_centroid'], 0)
                star_id = 'f'+str(self.pointing_code)+'_'+str(istar)
            starpars = {
                'star_id' : star_id,
                'x_centroid' : (self.sources_i[i])['x_centroid'],
                'y_centroid' : (self.sources_i[i])['y_centroid'],
                'mag' : (self.sources_i[i])['mag'],
                'skycoo' : mycoo
            }
            self.stars.append(StarMUSE(starpars))
        self.nstars = istar

    def set_centroids(self):
        """Set the centroids values

        Function to read or find the stellar positions.
        Uses the get_centroids function in myphotutils.py to call the daofind algorithm

        :return: void
        """
        #
        if 'file_pos' in self.params.keys():
            self.file_pos = self.params['file_pos']
            self.sources_i = Table.read(self.datadir+self.file_pos,format='csv')
            self.sources_i.rename_column('X_IMAGE', 'x_centroid')
            self.sources_i.rename_column('Y_IMAGE', 'y_centroid')
            self.sources_i.rename_column('MAG_APER', 'mag')
            self.positions_i = np.transpose((self.sources_i['x_centroid'], self.sources_i['y_centroid']))
            self.stars_from_file = True
        else:
            self.positions_i, self.sources_i = get_centroids(self.file_i_image, thres=self.daofind_threshold, sigma_radius=self.daofind_sigma_radius)
            self.stars_from_file = False
        #
        self.has_centroids =True

    def correct_astrometry(self, reference=None, vista_table=None, ra_col=None, dec_col=None,
                           gaia_release='DR3', gaia_radius_factor=1.5, gaia_catalog=None,
                           max_sep_arcsec=5.0, hist_nbins=30, reliable_radius=None,
                           offset_range=None, magsat=None,
                           discard_faint=True, faint_fraction=0.2,
                           doplo=True, plotfile='f_astrometry.pdf', verbose=True):
        """Correct the stellar coordinates to the Gaia frame (or to a Gaia-registered VISTA catalog)

        The reference catalog is either:

            'gaia'  - Gaia sources queried from the ESA Gaia archive (astroquery), in a cone
                      centred on the mean position of the stars in the cube and with radius
                      gaia_radius_factor times the largest distance of any star from it
                      (default release DR3, selectable with gaia_release)
            'vista' - a VISTA catalog on disk, as produced by vista_gaia_astrocorr.py
                      (requires vista_table)

        The stars in the cube are cross-matched with the reference catalog and the RA/Dec
        offset to be applied to the star.skycoo_original coordinates is derived with a three-stage
        procedure (the same as in vista_gaia_astrocorr.py):

            Stage 1 - nearest-neighbour match within max_sep_arcsec (optionally restricted
                      to the offset box offset_range), median offsets
            Stage 2 - re-match after applying the Stage-1 offsets; the separation
                      histogram is used to find a "reliable" association radius
            Stage 3 - final median offsets measured on the original coordinates, using
                      only the Stage-2 pairs closer than the reliable radius (the pairs
                      are selected after the Stage-1 correction, so that true matches are
                      not lost when the offset is larger than the reliable radius)

        The corrected coordinates (ICRS) are stored in star.skycoo_corrected for every
        star, and star.skycoo then returns them instead of star.skycoo_original; the
        corrected world coordinate system of the cube is stored in self.wcs_corrected,
        and self.wcs then returns it instead of self.wcs_original; the offsets and match
        details are stored in self.astrometry.
        Offsets are tangent-plane offsets in arcsec: dRA = dalpha*cos(Dec) (positive towards
        East) and dDec (positive towards North), reference minus MUSE, applied with
        SkyCoord.spherical_offsets_by.
        Gaia positions are used at the Gaia reference epoch (J2016.0 for DR3), without
        propagating them with the proper motions to the epoch of the MUSE observations.

        :param reference: (string) 'gaia' or 'vista'. If None (default): 'vista' when
                          vista_table is given, 'gaia' otherwise
        :param vista_table: (string) name of the table written by vista_gaia_astrocorr.py
                            (ASCII, or FITS if the extension is .fits/.fit/.gz); required
                            for reference='vista'
        :param ra_col: (string) column with the reference RA [deg]
                       (default: 'RA_CORR' for VISTA, 'ra' for Gaia)
        :param dec_col: (string) column with the reference Dec [deg]
                        (default: 'DE_CORR' for VISTA, 'dec' for Gaia)
        :param gaia_release: (string) Gaia data release to query: 'DR3' (default), 'EDR3',
                             'DR2', ... or a full archive table name (e.g. 'gaiadr3.gaia_source')
        :param gaia_radius_factor: (float) radius of the Gaia query in units of the largest
                                   distance of any star from the mean position (default 1.5)
        :param gaia_catalog: (astropy Table) optional Gaia catalog already retrieved, e.g.
                             self.astrometry['ref_table'] from a previous run: if given, the
                             archive is not queried again
        :param max_sep_arcsec: (float) maximum separation [arcsec] for a valid match
        :param hist_nbins: (int) number of bins of the separation histogram used to find
                           the reliable radius (fewer than in the script, since a MUSE
                           field contains far fewer sources)
        :param reliable_radius: (float) if set [arcsec], skip the histogram analysis and
                                use this value as the reliable association radius
        :param offset_range: (tuple) optional ((dra_min, dra_max), (ddec_min, ddec_max)) in
                             arcsec (dRA = dalpha*cos(Dec), reference minus MUSE). If set, in
                             Stage 1 each star is matched only to the reference sources whose
                             offset falls inside this box (the one closest to the box centre
                             is chosen), instead of the plain nearest neighbour. Use it in
                             crowded fields when the offset is comparable to or larger than
                             the typical distance between reference sources. The box must lie
                             within max_sep_arcsec of the origin.
        :param magsat: (float) stars brighter than this daofind magnitude (saturated), and
                       stars with no measured magnitude, are not used to derive the offsets
                       (the correction is still applied to them). Default: self.magsat, the same cutoff used by set_stars_for_apc
                       (parameter 'saturation_magnitude')
        :param discard_faint: (bool) if True (default), after the Stage-3 match discard the
                              faintest matched stars before computing the final offsets
        :param faint_fraction: (float) fraction of the Stage-3 matched stars to discard,
                               the faintest by daofind magnitude (default 0.2, i.e. 20%)
        :param doplo: (bool) if True produce the diagnostic plot (the last row shows the
                      I band image in the original coordinate system with the Stage-1
                      associations, and in the corrected coordinate system with the
                      Stage-2 associations and, in green, those used for the final
                      offsets)
        :param plotfile: (string) name of the figure file (saved in figdir)
        :param verbose: (bool) print the stage-by-stage results
        :return: dictionary with the results (also stored in self.astrometry)
        """
        #
        import contextlib
        import io
        from astropy.coordinates import ICRS
        import astropy.units as u

        if reference is None:
            reference = 'vista' if vista_table is not None else 'gaia'
        reference = reference.lower()
        if reference not in ('gaia', 'vista'):
            raise ValueError("reference must be 'gaia' or 'vista', not '{0}'".format(reference))
        if reference == 'vista' and vista_table is None:
            raise ValueError("reference='vista' requires the name of the VISTA table (vista_table)")

        # original stellar coordinates in the cube (ICRS, as the Gaia/VISTA reference):
        # always start from skycoo_original, so that repeated runs do not accumulate
        # the corrections
        muse_coo = [star.skycoo_original.transform_to(ICRS()) for star in self.stars]
        muse_ra = np.array([c.ra.deg for c in muse_coo])
        muse_dec = np.array([c.dec.deg for c in muse_coo])
        # saturated stars (brighter than magsat, as in set_stars_for_apc) and stars with
        # no measured magnitude are not used for the matching
        magsat = self.magsat if magsat is None else magsat
        muse_mag = np.array([star.daofind_mag for star in self.stars], dtype=float)
        use = np.isfinite(muse_mag) & (muse_mag >= magsat)

        # the helpers print their results: silence them unless verbose
        quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with quiet:
            # reference catalog
            query_center, query_radius, ref_table = None, None, None
            if reference == 'vista':
                ra_col = RA_CORR_COL if ra_col is None else ra_col
                dec_col = DE_CORR_COL if dec_col is None else dec_col
                ref_label = 'VISTA'
                ref_title = 'VISTA ({0})'.format(os.path.basename(vista_table))
                ext = os.path.splitext(vista_table)[1].lower()
                if ext in ('.fits', '.fit', '.gz'):
                    tab_ref = Table.read(vista_table, format='fits')
                else:
                    tab_ref = Table.read(vista_table, format='ascii')
                catalog_name = vista_table
            else:
                ra_col = 'ra' if ra_col is None else ra_col
                dec_col = 'dec' if dec_col is None else dec_col
                ref_label = 'Gaia {0}'.format(gaia_release)
                ref_title = ref_label
                # cone centred on the mean position of the stars (mean of the unit vectors,
                # equivalent to the mean RA/Dec for a MUSE field, but safe at RA=0/360)
                stars_coo = SkyCoord(ra=muse_ra * u.deg, dec=muse_dec * u.deg, frame='icrs')
                xyz = stars_coo.cartesian.xyz.mean(axis=1)
                ctr = SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2], representation_type='cartesian',
                               frame='icrs')
                ctr = SkyCoord(ra=ctr.spherical.lon, dec=ctr.spherical.lat, frame='icrs')
                query_center = (float(ctr.ra.deg), float(ctr.dec.deg))
                query_radius = float(gaia_radius_factor * ctr.separation(stars_coo).max().deg)
                if gaia_catalog is None:
                    tab_ref = query_gaia(query_center[0], query_center[1], query_radius,
                                         release=gaia_release)
                else:
                    tab_ref = gaia_catalog
                    print("  Using the Gaia catalog provided ({0} sources), no archive query".format(
                        len(tab_ref)))
                ref_table = tab_ref
                catalog_name = 'Gaia {0} ({1})'.format(gaia_release, gaia_table_name(gaia_release))
            for col in (ra_col, dec_col):
                if col not in tab_ref.colnames:
                    raise ValueError("Column '{0}' not found in the {1} catalog; available columns: {2}".format(
                        col, ref_label, tab_ref.colnames))
            ref_ra = np.array(tab_ref[ra_col], dtype=float)
            ref_dec = np.array(tab_ref[dec_col], dtype=float)

            print("\n━━━ Pointing {0}: astrometry of {1} stars vs {2} {3} sources ━━━".format(
                self.pointing_code, len(muse_ra), len(ref_ra), ref_label))
            print("  Excluded from the matching: {0} saturated stars (mag < {1}), "
                  "{2} stars with no measured magnitude".format(
                      int((muse_mag < magsat).sum()), magsat, int((~np.isfinite(muse_mag)).sum())))

            # Stage 1 - initial cross-match (optionally restricted to an offset box)
            if offset_range is None:
                idx1, sep1, dra1, ddec1, good1 = cross_match(
                    muse_ra, muse_dec, ref_ra, ref_dec, max_sep_arcsec)
            else:
                (dra_lo, dra_hi), (ddec_lo, ddec_hi) = offset_range
                print("  Stage 1 restricted to dRA in [{0:+.2f}, {1:+.2f}], dDec in [{2:+.2f}, {3:+.2f}] arcsec".format(
                    min(dra_lo, dra_hi), max(dra_lo, dra_hi), min(ddec_lo, ddec_hi), max(ddec_lo, ddec_hi)))
                idx1, sep1, dra1, ddec1, good1 = cross_match_in_box(
                    muse_ra, muse_dec, ref_ra, ref_dec, max_sep_arcsec,
                    (dra_lo, dra_hi), (ddec_lo, ddec_hi))
            good1 &= use
            if good1.sum() == 0:
                raise ValueError("No {0} counterpart within {1} arcsec{3} for any star in pointing {2}".format(
                    ref_label, max_sep_arcsec, self.pointing_code,
                    "" if offset_range is None else " and inside offset_range"))
            dra1_med, dra1_std, ddec1_med, ddec1_std = offset_stats(dra1, ddec1, good1, label="Stage 1")

            # Stage 2 - cross-match after the stage-1 correction
            idx2, sep2, dra2, ddec2, good2 = cross_match(
                *apply_offsets(muse_ra, muse_dec, dra1_med, ddec1_med),
                ref_ra, ref_dec, max_sep_arcsec)
            good2 &= use
            if good2.sum() == 0:
                raise ValueError("No {0} counterpart within {1} arcsec after the Stage-1 correction".format(
                    ref_label, max_sep_arcsec))
            dra2_med, dra2_std, ddec2_med, ddec2_std = offset_stats(
                dra2, ddec2, good2, label="Stage 2 (stage-1 corrected)")

            # Stage 2b - reliable radius
            if reliable_radius is None:
                r_reliable = find_reliable_radius(sep2[good2], nbins=hist_nbins, stop_at_empty=True)[0]
                print("  Reliable association radius: {0:.3f} arcsec (from histogram)".format(r_reliable))
            else:
                r_reliable = float(reliable_radius)
                print("  Reliable association radius: {0:.3f} arcsec (user supplied)".format(r_reliable))

            # Stage 3 - final offsets on the original coordinates
            idx3 = idx2
            good3 = good2 & (sep2 <= r_reliable)
            dra3, ddec3 = pair_offsets(muse_ra, muse_dec, ref_ra[idx3], ref_dec[idx3])
            sep3 = SkyCoord(ra=muse_ra * u.deg, dec=muse_dec * u.deg).separation(
                SkyCoord(ra=ref_ra[idx3] * u.deg, dec=ref_dec[idx3] * u.deg)).arcsec
            if good3.sum() == 0:
                raise ValueError("No matches within the reliable radius ({0:.3f} arcsec): "
                                 "try setting reliable_radius or hist_nbins".format(r_reliable))
            # discard the faintest matched stars before computing the final offsets
            n_faint, mag_faint = 0, None
            faint3 = np.zeros(len(muse_ra), dtype=bool)   # matched but discarded as faint
            if discard_faint and faint_fraction > 0:
                if not 0 < faint_fraction < 1:
                    raise ValueError("faint_fraction must be between 0 and 1, not {0}".format(
                        faint_fraction))
                i_matched = np.where(good3)[0]
                n_faint = int(round(faint_fraction * len(i_matched)))
                if n_faint >= len(i_matched):
                    n_faint = len(i_matched) - 1
                if n_faint > 0:
                    # faintest = largest magnitude; stable sort for reproducible ties
                    i_faint = i_matched[np.argsort(muse_mag[i_matched], kind='stable')[-n_faint:]]
                    mag_faint = float(muse_mag[i_faint].min())
                    good3[i_faint] = False
                    faint3[i_faint] = True
                    print("  Discarded the {0} faintest of {1} matched stars ({2:.0f}%, "
                          "mag >= {3:.2f})".format(n_faint, len(i_matched),
                                                   100 * faint_fraction, mag_faint))
            dra3_med, dra3_std, ddec3_med, ddec3_std = offset_stats(
                dra3, ddec3, good3, label="Stage 3 - FINAL")
            print("  FINAL: dRA = {0:+.4f} +/- {1:.4f} arcsec, dDec = {2:+.4f} +/- {3:.4f} arcsec "
                  "(N = {4} / {5} usable stars)".format(dra3_med, dra3_std, ddec3_med, ddec3_std,
                                                             good3.sum(), int(use.sum())))

        # apply the correction to all the stars
        corr_ra, corr_dec = apply_offsets(muse_ra, muse_dec, dra3_med, ddec3_med)
        corr_coo = SkyCoord(ra=corr_ra * u.deg, dec=corr_dec * u.deg, frame='icrs')
        for i, star in enumerate(self.stars):
            star.skycoo_corrected = corr_coo[i]
        # corrected WCS: same offsets applied to the celestial reference point
        self.wcs_corrected = self._shifted_wcs(dra3_med, ddec3_med)

        # diagnostic figure
        fig = None
        if doplo:
            self.plotfile_astrometry = self.figdir + os.path.split(plotfile)[1]
            # I band images (same spatial grid as the cube) with the associations:
            #   left:  original WCS, Stage-1 pairs (catalog: red circles, MUSE: yellow +)
            #   right: corrected WCS, Stage-2 pairs (red / yellow), the Stage-3 pairs
            #          discarded because the MUSE star is too faint (yellow / yellow) and,
            #          on top, the pairs used for the final offsets (green)
            circ_r = 0.9       # radius of the catalog circles [arcsec]
            r_lbl = "{0} source (r = {1:g}\")".format(ref_label, circ_r)
            image_panels = [
                {'image': self.data_i, 'wcs': self.wcs_original.celestial,
                 'circle_radius_arcsec': circ_r,
                 'title': "Original coordinate system",
                 'layers': [
                     {'ref_ra': ref_ra[idx1[good1]], 'ref_dec': ref_dec[idx1[good1]],
                      'src_ra': muse_ra[good1], 'src_dec': muse_dec[good1],
                      'ref_label': r_lbl + ", Stage 1 match",
                      'src_label': "MUSE star, Stage 1 match"}]},
                {'image': self.data_i, 'wcs': self.wcs_corrected.celestial,
                 'circle_radius_arcsec': circ_r,
                 'title': "Corrected coordinate system",
                 'layers': [
                     {'ref_ra': ref_ra[idx2[good2 & ~faint3]],
                      'ref_dec': ref_dec[idx2[good2 & ~faint3]],
                      'src_ra': corr_ra[good2 & ~faint3], 'src_dec': corr_dec[good2 & ~faint3],
                      'ref_label': r_lbl + ", Stage 2 match",
                      'src_label': "MUSE star, Stage 2 match"},
                     {'ref_ra': ref_ra[idx2[faint3]], 'ref_dec': ref_dec[idx2[faint3]],
                      'src_ra': corr_ra[faint3], 'src_dec': corr_dec[faint3],
                      'ref_color': 'yellow', 'src_color': 'yellow',
                      'ref_label': "{0} source, discarded (MUSE star too faint)".format(ref_label),
                      'src_label': None},
                     {'ref_ra': ref_ra[idx3[good3]], 'ref_dec': ref_dec[idx3[good3]],
                      'src_ra': corr_ra[good3], 'src_dec': corr_dec[good3],
                      'ref_color': 'limegreen', 'src_color': 'limegreen',
                      'ref_label': "{0} source, used in final correction".format(ref_label),
                      'src_label': "MUSE star, used in final correction"}]},
            ]
            fig = make_diagnostic_figure(
                sep1, dra1, ddec1, good1, max_sep_arcsec,
                dra1_med, dra1_std, ddec1_med, ddec1_std,
                sep2, dra2, ddec2, good2, r_reliable,
                dra2_med, dra2_std, ddec2_med, ddec2_std,
                # row 3 separation panel: Stage-2 separations of the selected pairs
                np.where(good3, sep2, np.inf), dra3, ddec3, good3,
                dra3_med, dra3_std, ddec3_med, ddec3_std,
                output_path=self.plotfile_astrometry,
                title="Pointing {0}: MUSE → {1} astrometric correction".format(
                    self.pointing_code, ref_title),
                image_panels=image_panels)

        self.astrometry = {
            'reference': reference,     # 'gaia' or 'vista'
            'catalog': catalog_name,    # VISTA file name or Gaia release/table
            'gaia_release': gaia_release if reference == 'gaia' else None,
            'query_center_deg': query_center,   # (RA, Dec) of the Gaia cone
            'query_radius_deg': query_radius,   # radius of the Gaia cone
            'ref_table': ref_table,     # Gaia catalog used (None for VISTA)
            'dra_arcsec': dra3_med,     # dRA*cos(Dec)
            'dra_std': dra3_std,
            'ddec_arcsec': ddec3_med,
            'ddec_std': ddec3_std,
            'r_reliable_arcsec': r_reliable,
            'offset_range': offset_range,
            'magsat': magsat,
            'faint_fraction': faint_fraction if discard_faint else 0.,
            'n_discarded_faint': n_faint,      # faintest matched stars not used
            'discarded_faint': faint3,         # True for the matched stars discarded as faint
            'mag_discarded_faint': mag_faint,  # brightest magnitude among them
            'used_for_matching': use,   # False for saturated stars and stars with no magnitude
            'n_matched': int(good3.sum()),
            'ref_idx': idx3,            # index of the closest reference source for each star
            'ref_sep_arcsec': sep3,     # separation from it (original coordinates)
            'ref_sep_corr_arcsec': sep2,  # separation after the Stage-1 correction
            'matched': good3,           # True if used for the final offsets (within the
                                        # reliable radius, not among the faintest discarded)
            'figure': fig,
        }
        self.has_astrometry = True

        return self.astrometry

    def set_stars_for_apc(self, mindist=10, magperc=20,
                          doplo=True, image=None, plotfile='f_stars_for_apc.pdf',
                          verbose='True'):
        """Select ApC stars

        Function to identify the APC stars in the field.

        :param mindist: (float) minimum distance (in pixels) to consider star as isolated
        :param magperc: (float) percentile to select th ebrightest stars in the field
        :param doplo: (bool) if True prepare a diagnostic plot
        :param image: (fits file name) if set uses this as background image, otherwise the I band is used
        :param plotfile: (string) file for the figure
        :return: void
        """
        #
        if not image:
            image = self.file_i_image
        self.plotfile_apc_stars = self.figdir+os.path.split(plotfile)[1]
        self.n_apc = get_stars_for_apc(self.stars,
                                       mindist=mindist, magsat=self.magsat, magperc=magperc,
                                       doplo=doplo, image=image, plotfile=self.plotfile_apc_stars)

        self.apc_stars = [self.stars[i] for i in self.n_apc]
        if verbose:
            print("Identified {0} stars with mag>={1} for ApC in pointing {2}\n".format(len(self.n_apc), self.magsat, self.pointing_code))

        self.has_apc_stars = True

    def curve_of_growth_iv(self,radii=np.arange(1,10,0.5),skyrad=(10,15), guessrad=3., doplo_iv=True, plotfile='f_curve_of_growth_iv.pdf'):
        """
        This function computes the curve of growth in the I and V band, to help identify the best apertures for
        spectrophotometry. Optionally plots a diagnostic figure.
        If not set, it identifies the apc stars (with default parameters and no diagnostic plot)

        :param radii: (float array) array of radii for the photometry curve of growth
        :param skyrad: (tuple) inner and outer radius for the sky annulus
        :param guessrad: (float) guess at the best radius (for plotting purposes)
        :param doplo_iv: (bool) if True do the diagnostic plot
        :param plotfile: (string) name of the figure file
        :return: void
        """
        #

        if not self.has_apc_stars:
            self.set_stars_for_apc(doplo=False)
        self.cog_radii = radii
        self.cog_skyrad = skyrad

        #
        self.cog_mag_i = runphot_ima_aps(self.positions_i[self.n_apc], self.cog_radii,
                                         self.cog_skyrad[0], self.cog_skyrad[1], self.data_i)
        self.cog_mag_v = runphot_ima_aps(self.positions_i[self.n_apc], self.cog_radii,
                                         self.cog_skyrad[0], self.cog_skyrad[1], self.data_v)
        self.cog_mag_i = [a[0] for a in self.cog_mag_i]
        self.cog_mag_v = [a[0] for a in self.cog_mag_v]
        for i in range(len([self.stars[j] for j in self.n_apc])):
            self.stars[self.n_apc[i]].cog_mag_i = np.array([a[i] for a in self.cog_mag_i])
            self.stars[self.n_apc[i]].cog_mag_v = np.array([a[i] for a in self.cog_mag_v])
            self.stars[self.n_apc[i]].cog_radii = self.cog_radii

        if doplo_iv:
            self.plotfile_cog = self.figdir + os.path.split(plotfile)[1]
            plot_curve_of_growth_iv(self.cog_radii, self.cog_mag_i, self.cog_mag_v,
                                    self.positions_i[self.n_apc], self.file_i_image, self.file_v_image,
                                    guessrad=guessrad, plotfile=self.plotfile_cog)
        	

    def extract_apc_spectra(self):
        self.apc_cube = np.zeros((len(self.wl),len(self.n_apc)))

        self._run_getspecapc_proc_mp()

        for i in range(len(self.n_apc)):
            self.stars[self.n_apc[i]].apc_spec = self.apc_cube[:,i]
            self.stars[self.n_apc[i]].apc_wl = self.wl
            self.stars[self.n_apc[i]].apc_radii = self.apc_radii

    def set_apc_values(self,radii=(3.,10.),skyrad=(10,15), hw_box_median=50, sclip_median=2.,
                       apc_sclip=2., apc_fit_order=3, doplo_apc=True, plotfile='f_apc_values.pdf'):
        """Computes the ApC values as a function of wavelength

        This funcion is used to compute the aperture correction as a function of wavelength.
        If not done already, first identifies the stars to be used for apc (using default parameters),
        then computes apc(wl) for each of them, computes median apc with sigma clips and combines
        all apc stars, produces a standard diagnostic plot.

        :param radii: (tuple) the two photometry apertures radii used to compute the aperture correction
        :param skyrad: (tuple) the inner and outer radius for the sky annulus
        :param hw_box_median: (int) half width of the box used for running median
        :param sclip_median: (float) number of sigma for the sigma clipping for median computation
        :param apc_sclip: (float) number of sigma for the sigma clipping in combining different stars
        :param apc_fit_order: (int) order of the polinomial fit for the apc correction
        :param doplo_apc: (bool) if Ture prepare a diagnostic plot
        :param plotfile: (string) name of the figure file
        :return: void
        """
        #

        if not self.has_apc_stars:
             self.set_stars_for_apc()
        self.apc_radii = radii
        self.apc_skyrad = skyrad

        #
        self.extract_apc_spectra()

        self.analyse_apc_spectra(radii=radii, skyrad=skyrad, apc_fit_order=apc_fit_order, hw_box_median=hw_box_median,
                                 sclip_median=sclip_median, apc_sclip=apc_sclip, doplo_apc=doplo_apc, plotfile=plotfile)

        self.has_apc_values = True

    def analyse_apc_spectra(self, radii=(3.,10.), skyrad=(10,15), apc_fit_order=3, hw_box_median=50, sclip_median=3.0,
                       apc_sclip=2.0, sclip_niter=5, min_points_sclip=3, doplo_apc=True, plotfile='f_apc_values.pdf'):

        if not self.has_apc_stars:
             self.set_stars_for_apc()

        if not self.has_apc_values:
            self.apc_radii = radii
            self.apc_skyrad = skyrad
            self.extract_apc_spectra()
            self.has_apc_values=True

        self.apc_fit_order = apc_fit_order

        self.apc_med_30 = np.zeros((len(self.wl), len(self.n_apc)))
        self.apc_std_30 = np.zeros((len(self.wl), len(self.n_apc)))
        self.apc_mean_30 = np.zeros((len(self.wl), len(self.n_apc)))

        self._run_apc_proc_mp({'hw_box' : hw_box_median, 'sclip' : sclip_median, 'maxiter' : sclip_niter})

        for i in range(len(self.n_apc)):
            self.apc_med_30[:, i] = self.apc_stars[i].apc_med
            self.apc_mean_30[:, i] = self.apc_stars[i].apc_mean
            self.apc_std_30[:, i] = self.apc_stars[i].apc_std

        self.apc_med = np.nanmedian(self.apc_med_30, axis=1)
        self.apc_mean = np.nanmean(self.apc_mean_30, axis=1)
        self.apc_std = np.nanstd(self.apc_med_30, axis=1)
        if apc_sclip:
            for i in range(len(self.wl)):
                iter = 0
                med = self.apc_med[i]
                mean = self.apc_mean[i]
                std = self.apc_std[i]
                while (iter<sclip_niter):
                    ng = np.where((self.apc_med_30[i, :] - med)**2 < (apc_sclip * std)**2)
                    if len(ng[0]) < min_points_sclip:
                        break
                    med = np.nanmedian(self.apc_med_30[i, ng], axis=1)
                    mean = np.nanmean(self.apc_mean_30[i, ng], axis=1)
                    std = np.nanstd(self.apc_med_30[i, ng], axis=1)
                    iter += 1
                self.apc_med[i] = med[0]
                self.apc_mean[i] = mean[0]
                self.apc_std[i] = std[0]

        nn = np.where(np.isfinite(self.apc_mean) == True)
        self.apc_fit = np.poly1d(np.polyfit(self.wl[nn], self.apc_mean[nn], self.apc_fit_order))

        if doplo_apc:
            self.plotfile_apc_specra = self.figdir + os.path.split(plotfile)[1]
            plot_apc(self.wl, self.apc_cube, self.apc_med, self.apc_mean, self.apc_std, self.apc_med_30,
                     self.apc_fit, nsig=apc_sclip, plotfile=self.plotfile_apc_specra)

    def _run_getspecapc_proc_mp(self):
        """Extracts the APC spectra

        This function is used to setup the multiprocessing extraction of the spectra for the aperture
        correction stars. It calls a wrapper function in myphotutils.py that executes the call to
        the apphot function.

        :return: void
        """
        #
        #
        if self.nproc:   # run with multiprocessing
            nproc = min(len(self.wl),self.nproc)
            myargs = []
            for iwl in range(len(self.wl)):
                myargs.append([self.positions_i[self.n_apc], self.apc_radii,
                               self.apc_skyrad, self.cube_data[iwl,:,:]])

            with mp.Pool(nproc) as p:
                allwl = p.map(apc_spec_single_wl, myargs)

            for iwl in range(len(self.wl)):
                self.apc_cube[iwl, :] = allwl[iwl]
        else:       # run single process
            for iwl in range(len(self.wl)):
                self.apc_cube[iwl, :] = apc_spec_single_wl([self.positions_i[self.n_apc], self.apc_radii,
                                                              self.apc_skyrad, self.cube_data[iwl,:,:]])

    def _run_apc_proc_mp(self, my_star_method_args):
        """Execute ApC spectra processing

        This function is used to setup the multiprocessing computation of the mean, median and std
        of the spectra for the aperture correction stars. It calls a wrapper function in StarMUSE.py that executes
        the call to the utils function.

        :return:
        """
        #
        if self.nproc:   # run with multiprocessing
            nproc = min(len(self.n_apc),self.nproc)
            myargs = []
            for mystar in self.apc_stars:
                myargs.append([mystar.apc_spec, my_star_method_args])

            with mp.Pool(nproc) as p:
                allstars = p.map(apc_calc_single_star, myargs)
            for i in range(len(self.apc_stars)):
                self.apc_stars[i].apc_med = (allstars[i])[0]
                self.apc_stars[i].apc_mean = (allstars[i])[1]
                self.apc_stars[i].apc_std = (allstars[i])[2]
        else:       # run single process
            for mystar in self.apc_stars:
                result = apc_calc_single_star([mystar.apc_spec, my_star_method_args])
                mystar.apc_med = result[0]
                mystar.apc_mean = result[1]
                mystar.apc_std = result[2]

    def extract_spectra(self, radius, sky_radii, add_apc=True):
        #
        self.spec_radius = radius
        self.spec_sky_radii = sky_radii
        self.spec_add_apc = add_apc
        self.magspec = np.zeros((len(self.wl), len(self.stars)))
        self.spectra = np.zeros((len(self.wl), len(self.stars)))
        self.skies = np.zeros((len(self.wl), len(self.stars)))
        self.skies_noise = np.zeros((len(self.wl), len(self.stars)))

        if self.nproc:  # run with multiprocessing
            nproc = min(len(self.wl), self.nproc)
            myargs = []
            for iwl in range(len(self.wl)):
                myargs.append([self.positions_i, self.spec_radius,
                               self.spec_sky_radii, self.cube_data[iwl, :, :]])

            with mp.Pool(nproc) as p:
                allwl = p.map(get_spec_single_wl, myargs)

            for iwl in range(len(self.wl)):
                self.magspec[iwl, :], self.spectra[iwl, :], self.skies[iwl, :], self.skies_noise[iwl, :] = allwl[iwl]
        else:  # run single process
            for iwl in range(len(self.wl)):
                self.magspec[iwl, :], self.spectra[iwl, :], self.skies[iwl, :], self.skies_noise[iwl, :] = get_spec_single_wl([self.positions_i, self.spec_radius,
                               self.spec_sky_radii, self.cube_data[iwl, :, :]])

        for j in range(len(self.stars)):
            self.stars[j].wl = self.wl
            self.stars[j].magspec = self.magspec[:, j]
            self.stars[j].spectrum = self.spectra[:, j] * 10**(-20)
            self.stars[j].sky = self.skies[:, j] * 10**(-20)
            self.stars[j].sky_noise = self.skies_noise[:, j] * 10**(-20)
            self.stars[j].has_spectrum = True

        self.compute_rms_spectra()

        self.has_spectra = True

        if add_apc:
            self.correct_spectra_apc()

    def compute_rms_spectra(self, hw_box_median=50, sclip_median=3, sclip_niter=5):
        #
        mydicargs = {'hw_box' : hw_box_median, 'sclip' : sclip_median, 'maxiter' : sclip_niter}
        if self.nproc:  # run with multiprocessing
            nproc = min(len(self.stars), self.nproc)
            myargs = []
            for mystar in self.stars:
                myargs.append([mystar.spectrum, mydicargs])

            with mp.Pool(nproc) as p:
                allstars = p.map(apc_calc_single_star, myargs)
            for i in range(len(self.stars)):
                self.stars[i].rms_spectrum = (allstars[i])[2]
        else:  # run single process
            for mystar in self.stars:
                result = apc_calc_single_star([mystar, mydicargs])
                mystar.rms_spectrum = result[2]

    def correct_spectra_apc(self):
        #
        for j in range(len(self.stars)):
            self.stars[j].corrected_magspec = self.stars[j].magspec + self.apc_fit(self.wl)
            self.stars[j].corrected_spectrum = self.stars[j].spectrum * 10.**(-0.4*self.apc_fit(self.wl))
            self.stars[j].has_corrected_spectrum = True

    def get_all_lines(self):
        #
        # if self.nproc:  # run with multiprocessing
        #     nproc = min(len(self.stars), self.nproc)
        #     myargs = []
        #     for mystar in self.stars:
        #         myargs.append([mystar.spectrum, mydicargs])


        #     with mp.Pool(nproc) as p:
        #         allwl = p.map(get_spec_single_wl, myargs)

        #     for iwl in range(len(self.wl)):
        #         self.magspec[iwl, :], self.spectra[iwl, :], self.skies[iwl, :], self.skies_noise[iwl, :] = allwl[iwl]
        # else:  # run single process
        for mystar in self.stars:
            for line in known_lines:
                mystar.get_line(line)
