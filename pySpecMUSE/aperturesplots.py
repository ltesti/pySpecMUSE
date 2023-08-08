#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture

def plotstars(imagefile, positions, ax=None, aprad=3., fileout=None, mytitle='Star positions'):
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
    if fileout and (not ax):
        plt.savefig(fileout)