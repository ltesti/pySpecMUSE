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
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture

#from .StarMUSE import StarMUSE

def plot_star(Star, images={'image_V': None, 'image_I': None}, figure_file=None):
    #
    aperture = CircularAperture([Star.x_centroid, Star.y_centroid], r=3)
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
        nlab = np.where((Star.wl >= plotlims[lab][0]) & (Star.wl <= plotlims[lab][1]))
        ax[lab].plot(Star.wl[nlab], Star.corrected_spectrum[nlab], color='g', linestyle='solid')
        ax[lab].plot(Star.wl[nlab], Star.spectrum[nlab], color='b', linestyle='solid', alpha=0.3)
        ax[lab].plot(Star.wl[nlab], Star.sky[nlab], color='k', linestyle='solid', alpha=0.3)
        ax[lab].fill_between(Star.wl[nlab],
                                Star.sky[nlab] - Star.sky_noise[nlab],
                                Star.sky[nlab] + Star.sky_noise[nlab],
                                color='grey', linestyle='solid', alpha=0.2)
        ax[lab].fill_between(Star.wl[nlab],
                                Star.corrected_spectrum[nlab] - Star.sky_noise[nlab],
                                Star.corrected_spectrum[nlab] + Star.sky_noise[nlab],
                                color='lightgreen', linestyle='solid', alpha=0.2)
        ax[lab].fill_between(Star.wl[nlab],
                                Star.corrected_spectrum[nlab] - Star.rms_spectrum[nlab],
                                Star.corrected_spectrum[nlab] + Star.rms_spectrum[nlab],
                                color='r', linestyle='solid', alpha=0.2)
        #
        ax[lab].set_xlabel(r'Wavelength ($\AA$)')
        ax[lab].set_title(plotlabel[lab])
        ax[lab].set_xlim(plotlims[lab][0], plotlims[lab][1])
        ax[lab].set_yscale(plotscale[lab])

        for filter in ['I', 'V']:
            if images['image_'+filter] is None:
                print("Skipping image_" + filter + " as it is set to None\n")
            else:
                norm = simple_norm(images['image_' + filter], 'sqrt', percent=92.)
                ax[filter].imshow(images['image_' + filter], cmap='Greys', origin='lower', norm=norm,
                                    interpolation='nearest')
                aperture.plot(ax=ax[filter], color='orange', lw=1, alpha=0.4)
                ax[filter].set_xlabel('X (pix)')
                ax[filter].set_ylabel('Y (pix)')
                ax[filter].set_title('Filter ' + filter)                

    if figure_file:
        plt.tight_layout()
        plt.savefig(figure_file)