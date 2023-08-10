#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import multiprocessing as mp
# from astropy.io import fits
# from photutils.aperture import CircularAperture, CircularAnnulus
# import scipy.interpolate as ssi
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture
# import os
# from astropy.table import Table

# from .myphotutils import get_centroids, get_stars_for_apc, runphot_ima_aps
# from .apc_plots import plot_curve_of_growth_iv, plot_apc
from .utils import running_median_spec

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

    def plot_star_summary(self, images={'image_V': None, 'image_I': None}, figure_file=None):
        #
        # Unpack data from dictionary
        #        #
        # data_v = images['image_V']
        # data_i = images['image_I']
        aperture = CircularAperture([self.xcentroid, self.ycentroid], r=3)
        #
        #
        fig, ax = plt.subplot_mosaic([
            ['I', 'I', 'V', 'V'],
            ['I', 'I', 'V', 'V'],
            ['sp', 'sp', 'sp', 'sp'],
            ['lsp', 'lsp', 'lsp', 'lsp'],
            ['Ha', 'HeI', 'LiI', '[OI]'],
            ['Ha', 'HeI', 'LiI', '[OI]'],
            ['Ha', 'Hb', 'CaIRT', 'CaIRT'],
            ['Ha', 'Hb', 'CaIRT', 'CaIRT']],
            layout='constrained', figsize=(12, 18))

        plotlims = {
            'sp': (4700, 9400),
            'lsp': (4700, 9400),
            'Hb': (4800, 4920),
            'Ha': (6500, 6620),
            'HeI': (5850, 5900),
            'LiI': (6690, 6740),
            '[OI]': (6200, 6400),
            'CaIRT': (8450, 8700),
        }
        plotscale = {
            'sp': 'linear',
            'lsp': 'log',
            'Hb': 'log',
            'Ha': 'log',
            'HeI': 'log',
            'LiI': 'log',
            '[OI]': 'log',
            'CaIRT': 'log',
        }
        plotlabel = {
            'sp': 'Spectra',
            'lsp': r'Log$_{10}$(Spectra)',
            'Hb': r'H$\beta$',
            'Ha': r'H$\alpha$',
            'HeI': r'HeI',
            'LiI': r'LiI',
            '[OI]': r'[OI]$\lambda$6300]',
            'CaIRT': 'Ca IR Triplet',
        }

        for lab in ['sp', 'lsp', 'Ha', 'Hb', 'HeI', 'LiI', '[OI]', 'CaIRT']:
            nlab = np.where((self.wl >= plotlims[lab][0]) & (self.wl <= plotlims[lab][1]))
            ax[lab].plot(self.wl[nlab], self.corrected_spectrum[nlab], color='g', linestyle='solid')
            ax[lab].plot(self.wl[nlab], self.spectrum[nlab], color='b', linestyle='solid', alpha=0.3)
            ax[lab].plot(self.wl[nlab], self.sky[nlab], color='k', linestyle='solid', alpha=0.3)
            ax[lab].fill_between(self.wl[nlab],
                                 self.sky[nlab] - self.sky_noise[nlab],
                                 self.sky[nlab] + self.sky_noise[nlab],
                                 color='grey', linestyle='solid', alpha=0.2)
            ax[lab].fill_between(self.wl[nlab],
                                 self.corrected_spectrum[nlab] - self.sky_noise[nlab],
                                 self.corrected_spectrum[nlab] + self.sky_noise[nlab],
                                 color='lightgreen', linestyle='solid', alpha=0.2)
            ax[lab].fill_between(self.wl[nlab],
                                 self.corrected_spectrum[nlab] - self.rms_spectrum[nlab],
                                 self.corrected_spectrum[nlab] + self.rms_spectrum[nlab],
                                 color='r', linestyle='solid', alpha=0.2)
            #
            ax[lab].set_xlabel('Wavelength ($\AA$)')
            ax[lab].set_title(plotlabel[lab])
            ax[lab].set_xlim(plotlims[lab][0], plotlims[lab][1])
            ax[lab].set_yscale(plotscale[lab])

            for filter in ['I', 'V']:
                norm = simple_norm(images['image_' + filter], 'sqrt', percent=92.)
                ax[filter].imshow(images['image_' + filter], cmap='Greys', origin='lower', norm=norm,
                                  interpolation='nearest')
                aperture.plot(ax=ax[filter], color='orange', lw=1, alpha=0.4)
                #annulus_aperture[star_to_plot].plot(ax=ax[filter], color='cyan', lw=1, alpha=0.4)
                ax[filter].set_xlabel('X (pix)')
                ax[filter].set_ylabel('Y (pix)')
                ax[filter].set_title('Filter ' + filter)

        if figure_file:
            plt.tight_layout()
            plt.savefig(figure_file)