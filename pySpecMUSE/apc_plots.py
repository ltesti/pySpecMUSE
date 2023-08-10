#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from .aperturesplots import plotstars

def plot_curve_of_growth_iv(radii, mag_i, mag_v, positions, file_i, file_v, guessrad=3. , plotfile=None):
    #
    # median and std of norm mags at the last point
    mag_90 = -2.5*np.log10(0.9)
    mag_60 = -2.5*np.log10(0.6)
    mag_80 = -2.5*np.log10(0.8)
    mag_70 = -2.5*np.log10(0.7)
    #
    fig, ax = plt.subplots(3, 2, figsize=(12, 18))
    #fig.subplots_adjust(hspace=0)

    plotstars(file_i, positions, ax=ax[0, 0], aprad=radii[-1], fileout=None)
    plotstars(file_v, positions, ax=ax[0, 1], aprad=radii[-1], fileout=None)

    ax[1, 0].plot(radii, mag_i)
    ax[1, 0].invert_yaxis()
    # ax[0,0].set_xlabel('Aperture radius (pix)')
    # ax[0].set_ylabel('Aperture Correction')
    ax[1, 0].set_ylabel('Mag')
    ax[1 ,0].set_title('Filter I')

    ax[1, 1].plot(radii, mag_v)
    # ax[0,1].set_xlabel('Aperture (pix)')
    ax[1, 1].set_ylabel('Mag')
    ax[1, 1].set_title('Filter V')
    # ax[0,1].set_ylim(10, 20)
    ax[1, 1].invert_yaxis()

    ax[2, 0].plot(radii, mag_i - mag_i[-1])
    ax[2, 0].plot([radii[0],radii[-1]],[mag_90,mag_90],linestyle='dotted',label='90% flux level')
    ax[2, 0].plot([radii[0],radii[-1]],[mag_80,mag_80],linestyle='dotted', label='80% flux level')
    ax[2, 0].plot([radii[0],radii[-1]],[mag_70,mag_70],linestyle='dotted', label='70% flux level')
    ax[2, 0].plot([radii[0], radii[-1]], [mag_60, mag_60], linestyle='dotted', label='60% flux level')
    ax[2, 0].plot([guessrad,guessrad], [0,2], linestyle='dashed',label='Ap radius {0:3.1f}'.format(guessrad))
    ax[2, 0].invert_yaxis()
    ax[2, 0].set_xlabel('Aperture radius (pix)')
    # ax[0].set_ylabel('Aperture Correction')
    ax[2, 0].set_ylabel('Mag')
    # ax[1, 0].set_title('Filter I')
    ax[2, 0].legend()

    ax[2, 1].plot(radii, mag_v - mag_v[-1])
    ax[2, 1].plot([radii[0], radii[-1]], [mag_90, mag_90], linestyle='dotted', label='90% flux level')
    ax[2, 1].plot([radii[0], radii[-1]], [mag_80, mag_80], linestyle='dotted', label='80% flux level')
    ax[2, 1].plot([radii[0], radii[-1]], [mag_70, mag_70], linestyle='dotted', label='70% flux level')
    ax[2, 1].plot([radii[0], radii[-1]], [mag_60, mag_60], linestyle='dotted', label='60% flux level')
    ax[2, 1].plot([guessrad,guessrad], [0,2], linestyle='dashed', label='Ap radius {0:3.1f}'.format(guessrad))
    ax[2, 1].set_xlabel('Aperture radius (pix)')
    ax[2, 1].set_ylabel('Mag')
    # ax[1, 1].set_title('Filter V')
    # ax[0,1].set_ylim(10, 20)
    ax[2, 1].invert_yaxis()
    ax[2, 1].legend()

    if plotfile:
        plt.savefig(plotfile)

def plot_apc(wl, apc_cube, apc_med, apc_mean, apc_std, apc_med_30, apc_fit, nsig=None, plotfile=None):

    fig, ax = plt.subplots(figsize=(20, 7))

    if nsig:
        nsigma = nsig
    else:
        nsigma = 1.

    fig.suptitle('Aperture correction')

    ax.plot(wl, apc_med, color='blue')
    #
    ymin = np.percentile(apc_med_30, 0.1)
    ymax = np.percentile(apc_med_30, 99.9)
    ax.set_ylim(ymin, ymax)

    ax.fill_between(wl, apc_med-nsigma*apc_std, apc_med+nsigma*apc_std, alpha=0.25, color='blue')
    #
    ax.plot(wl, apc_cube[:, 0], alpha=0.4, color='orange')
    ax.plot(wl, np.nanmedian(apc_cube[:, 0]) * np.ones(len(wl)), color='green')
    #axs[0].plot(wl, -apcorrmean * np.ones(len(wl)))
    ax.plot(wl, apc_med_30, 'o', color='red', alpha=0.3)
    ax.plot(wl, apc_mean, linestyle='solid', alpha=0.4, color='cyan')
    ax.plot(wl, apc_med, linestyle='dashed', alpha=0.4, color='cyan')
    ax.plot(wl, apc_fit(wl), linestyle='dashed', linewidth=3, color='yellow')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Aperture Correction')

    if plotfile:
        plt.savefig(plotfile)
