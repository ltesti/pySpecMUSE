#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture

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


class StarMUSE(object):
    """
        The class stores the parameters for a star (id, position, spectrum, sky, etc...)
        in the MUSE field

        params:
            position

        Example:
            starpars = {
                'xcentroid' : posline['xcentroid'],
                'ycentroid' : posline['ycentroid'],
                'mag' : posline['mag'],
                'skycoo' : self.wcs.pixel_to_world(posline['xcentroid'], posline['ycentroid'])
            }
        """

    def __init__(self, starpars):
        """
        set up the object: this one reads the spectrum within given boundaries
        """

        self.starpars = starpars

        try :
            self.id = self.starpars['star_id']
            self.xcentroid = self.starpars['xcentroid']
            self.ycentroid = self.starpars['ycentroid']
            self.daofind_mag = self.starpars['mag']
            self.skycoo = self.starpars['skycoo']

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

    def plot_star_summary(self, images={'image_V': None, 'image_I': None}, figure_file=None):
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