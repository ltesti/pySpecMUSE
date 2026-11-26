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
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture

def plotstars(imagefile, positions, plotall=False, allpos=None, ax=None, aprad=3., fileout=None, mytitle='Star positions'):
    #
    ima = fits.open(imagefile)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
    data = ima['DATA'].data
    #
    apertures = CircularAperture(positions, r=aprad)
    norm = simple_norm(data, 'sqrt', percent=92.)
    if not ax:
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.suptitle(os.path.basename(imagefile))
        ax.set_title(mytitle, loc='left', fontstyle='oblique', fontsize='medium')
    else:
        ax.set_title(os.path.basename(imagefile), fontsize='small')
    ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
    apertures.plot(ax=ax, color='red', lw=1.5, alpha=0.5)
    if plotall:
        all_ap = CircularAperture(allpos, r=aprad/3)
        all_ap.plot(ax=ax, color='lightgreen', lw=1.5, alpha=0.5)
    if fileout:
        plt.savefig(fileout)