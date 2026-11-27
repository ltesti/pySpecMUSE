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

#import numpy as np
#import matplotlib.pyplot as plt
#from astropy.visualization import simple_norm
#from photutils.aperture import CircularAperture

from astropy.coordinates import SkyCoord

from .utils import running_median_spec
from .lines import line_integral, known_lines, lineplot
from .star_plots import plot_star

def apc_calc_single_star(args):
    # wrapper per calcolare in multiprocessing le apc
    # arg[0] = StarMUSE object
    # arg[1] = dictionary containing
    #    the two parameters: median filter box half_width and sigmaclip threshold (None = no sigla clip)
    #    Example: {'hw_box' : hw_box, 'sclip' : 2.0, 'maxiter' = 5}

    if 'sclip' in (args[1]).keys():
        sclip = (args[1])['sclip']
    else:
        sclip = None

    med_mean_std = running_median_spec(args[0], (args[1])['hw_box'], sclip = sclip, maxiter=(args[1])['maxiter'])
    return med_mean_std


def celestial_coord(coo):
    """Return the celestial SkyCoord of a position

    The positions computed with the 3D WCS of the cube (pixel_to_world) are lists
    [SkyCoord, SpectralCoord]: in that case only the SkyCoord is returned.
    """
    if isinstance(coo, (list, tuple)):
        coo = next(c for c in coo if isinstance(c, SkyCoord))
    return coo


class StarMUSE(object):
    """Store the position, photometry, and extracted spectrum for one source.

    Instances are created by :class:`~pySpecMUSE.CubeMUSE.CubeMUSE` during
    source detection or catalogue loading.

    :param starpars: Mapping with ``star_id``, ``x_centroid``,
        ``y_centroid``, ``mag``, and ``skycoo``. Centroids are in
        image pixels, and ``skycoo`` contains the original celestial coordinates.
    """

    def __init__(self, starpars):
        """
        set up the object: this one reads the spectrum within given boundaries
        """

        self.starpars = starpars

        try :
            self.id = self.starpars['star_id']
            self.x_centroid = self.starpars['x_centroid']
            self.y_centroid = self.starpars['y_centroid']
            self.daofind_mag = self.starpars['mag']
            # original coordinates (celestial part only) from the cube WCS or the input file
            self.skycoo_original = celestial_coord(self.starpars['skycoo'])
            # astrometrically corrected coordinates (set by CubeMUSE.correct_astrometry)
            self.skycoo_corrected = None
            # self.skycoo (property) returns skycoo_corrected if available, else skycoo_original

        except KeyError:
            raise ValueError("Cannot initiate analysis without default values for pointing_code, datadir, and default_names switch\n {}".format(self.starpars))

        # Aperture correction star attributes
        self.apc_star = False
        self.cog_mag_i = None
        self.cog_mag_v = None
        self.cog_radii = None
        self.apc_spec = None
        self.apc_wl = None
        self.apc_radii = None
        self.apc_med = None
        self.apc_mean = None
        self.apc_std = None

        # Spectrum attributes
        self.has_spectrum = False
        self.wl = None
        self.magspec = None
        self.spectrum = None
        self.rms_spectrum = None
        self.sky = None
        self.sky_noise = None
        self.corrected_magspec = None
        self.corrected_spectrum = None
        self.has_corrected_spectrum = False

        # line measurements
        self.has_lines = False
        self.lines = {}

    @property
    def skycoo(self):
        """Best available coordinates of the star (read only)

        skycoo_corrected after the astrometric correction (CubeMUSE.correct_astrometry),
        skycoo_original otherwise.
        """
        if self.skycoo_corrected is not None:
            return self.skycoo_corrected
        return self.skycoo_original

    def __setstate__(self, state):
        # objects pickled before skycoo_original was introduced stored the original
        # coordinates in 'skycoo' (possibly as a [SkyCoord, SpectralCoord] list)
        if 'skycoo' in state:
            old = state.pop('skycoo')
            state.setdefault('skycoo_original', celestial_coord(old))
        state.setdefault('skycoo_corrected', None)
        self.__dict__.update(state)

    def plot_star_summary(self, images={'image_V': None, 'image_I': None}, figure_file=None):
        """Plot a summary of this source after spectrum extraction.

        :param images: Optional mapping of ``image_V`` and ``image_I`` to
            two-dimensional image arrays.
        :param figure_file: Optional destination for the figure.
        """
        #
        plot_star(self, images=images, figure_file=figure_file)

    def get_line(self, linepars=None):
        #
        if linepars:
            if 'linename' not in linepars.keys():
                linepars['linename'] = '[OI]6300'
            if 'wlmin' not in linepars.keys():
                linepars['wlmin'] = 6297
            if 'wlmax' not in linepars.keys():
                linepars['wlmax'] = 6303
            if 'wlcont1' not in linepars.keys():
                linepars['wlcont1'] = 6250
            if 'wlcont2' not in linepars.keys():
                linepars['wlcont2'] = 6350
            if 'min_snr' not in linepars.keys():
                linepars['min_snr'] = 3.
            if 'doplot' not in linepars.keys():
                linepars['doplot'] = 'True'
            if 'plotsky' not in linepars.keys():
                linepars['plotsky'] = 'True'
            if 'conttype' not in linepars.keys():
                linepars['conttype'] = 'median'
            retdic = line_integral(self, linepars['wlmin'], linepars['wlmax'], 
                                   linepars['wlcont1'], linepars['wlcont2'], 
                                   min_snr=linepars['min_snr'], doplot=linepars['doplot'], 
                                   plotsky=linepars['plotsky'], conttype=linepars['conttype'])
            self.lines[linepars['linename']] = retdic
        else:
            pass