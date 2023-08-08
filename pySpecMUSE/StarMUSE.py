#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from astropy.io import fits
#from photutils.aperture import CircularAperture, CircularAnnulus
#import scipy.interpolate as ssi
import matplotlib.pyplot as plt
#import os
#from astropy.table import Table

from .myphotutils import get_centroids, get_stars_for_apc, runphot_ima_aps
from .apc_plots import plot_curve_of_growth_iv, plot_apc
from .utils import running_median_spec

def apc_calc_single_star(args):
    # wrapper per calcolare in multiprocessing le apc
    # arg[0] = StarMUSE object
    # arg[1] = dictionary containing
    #    the two parameters: median filter box half_width and sigmaclip threshold (None = no sigla clip)
    #    Example: {'hw_box' : hw_box, 'sclip' : 2.0}

    star = args[0]
    pars = args[1]
    star.apc_med_hw_box = pars['hw_box']
    if 'sclip' in pars.keys():
        star.apc_med_sclip = pars['sclip']
    else:
        star.apc_med_sclip = None

    star.apc_med = np.zeros(len(star.apc_wl))
    star.apc_std = np.zeros(len(star.apc_wl))
    star.apc_mean = np.zeros(len(star.apc_wl))

    star.apc_med, star.apc_mean, star.apc_std = running_median_spec(star.apc_spec,
                                                                    star.apc_med_hw_box,
                                                                    sclip = star.apc_med_sclip)

    return star


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

        self.apc_star = False