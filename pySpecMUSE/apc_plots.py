#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    plotstars(file_i, positions, ax=ax[0, 0], aprad=radii[-1], fileout=None)
    plotstars(file_v, positions, ax=ax[0, 1], aprad=radii[-1], fileout=None)

    ax[1, 0].plot(radii, mag_i)
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_ylabel('Mag')
    ax[1 ,0].set_title('Filter I')

    ax[1, 1].plot(radii, mag_v)
    ax[1, 1].set_ylabel('Mag')
    ax[1, 1].set_title('Filter V')
    ax[1, 1].invert_yaxis()

    ax[2, 0].plot(radii, mag_i - mag_i[-1])
    ax[2, 0].plot([radii[0],radii[-1]],[mag_90,mag_90],linestyle='dotted',label='90% flux level')
    ax[2, 0].plot([radii[0],radii[-1]],[mag_80,mag_80],linestyle='dotted', label='80% flux level')
    ax[2, 0].plot([radii[0],radii[-1]],[mag_70,mag_70],linestyle='dotted', label='70% flux level')
    ax[2, 0].plot([radii[0], radii[-1]], [mag_60, mag_60], linestyle='dotted', label='60% flux level')
    ax[2, 0].plot([guessrad,guessrad], [0,2], linestyle='dashed',label='Ap radius {0:3.1f}'.format(guessrad))
    ax[2, 0].invert_yaxis()
    ax[2, 0].set_xlabel('Aperture radius (pix)')
    ax[2, 0].set_ylabel('Mag')
    ax[2, 0].legend()

    ax[2, 1].plot(radii, mag_v - mag_v[-1])
    ax[2, 1].plot([radii[0], radii[-1]], [mag_90, mag_90], linestyle='dotted', label='90% flux level')
    ax[2, 1].plot([radii[0], radii[-1]], [mag_80, mag_80], linestyle='dotted', label='80% flux level')
    ax[2, 1].plot([radii[0], radii[-1]], [mag_70, mag_70], linestyle='dotted', label='70% flux level')
    ax[2, 1].plot([radii[0], radii[-1]], [mag_60, mag_60], linestyle='dotted', label='60% flux level')
    ax[2, 1].plot([guessrad,guessrad], [0,2], linestyle='dashed', label='Ap radius {0:3.1f}'.format(guessrad))
    ax[2, 1].set_xlabel('Aperture radius (pix)')
    ax[2, 1].set_ylabel('Mag')
    ax[2, 1].invert_yaxis()
    ax[2, 1].legend()

    if plotfile:
        plt.savefig(plotfile)

def plot_apc(wl, apc_cube, apc_med, apc_mean, apc_std, apc_med_30, apc_fit, nsig=None, plotfile=None):

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(20, 7), sharey=False, sharex=True)
    plt.subplots_adjust(right=0.7)

    if nsig:
        nsigma = nsig
    else:
        nsigma = 1.

    fig.suptitle('Aperture correction')

    #
    y1min = np.percentile(apc_fit(wl), 0.1) - (np.percentile(apc_fit(wl), 0.1)/1.3)*(np.percentile(apc_fit(wl), 0.1)/abs(np.percentile(apc_fit(wl), 0.1)))
    y1max = np.percentile(apc_fit(wl), 99.9) + (np.percentile(apc_fit(wl), 99.9)/1.3)*(np.percentile(apc_fit(wl), 99.9)/abs(np.percentile(apc_fit(wl), 99.9)))
    #y1min = np.percentile(apc_med_30, 0.1111)
    #y1max = np.percentile(apc_med_30, 99.999999)
    #y1max = 0
    #y1min = -3
    #y2max = np.percentile(apc_med_30, 99.999999) # For fields 13,
    #y2min = np.percentile(apc_med_30, 0.1111)	# For fields 13,
    y2max = 7
    y2min= -5

	#
    ax1.fill_between(wl, apc_med-nsigma*apc_std, apc_med+nsigma*apc_std, alpha=0.25, color='xkcd:macaroni and cheese', label='Minimum to maximum errors of the AC curves')
    #
    ax1.plot(wl, apc_cube[:, 0], alpha=0.4, color='xkcd:pale purple', label= 'Minimum to maximum values of the AC curves')
    ax1.plot(wl, np.nanmedian(apc_cube[:, 0]) * np.ones(len(wl)), color='xkcd:boring green', label='Median value')
    ax1.plot(wl, apc_med_30, '.', color='xkcd:magenta', markersize=0.5) # alpha=0.3,
    ax1.plot(5000, 20, '.', color='xkcd:magenta', markersize=0.5, label='Median 30 AC curves')
    ax1.plot(wl, apc_mean, linestyle='solid', alpha=0.25, color='xkcd:soft pink', label='Mean AC') 
    ax1.plot(wl, apc_med, linestyle='dashed', alpha=0.25, color='xkcd:sky blue', label='Median AC')
    ax1.plot(wl, apc_fit(wl), linestyle='dashed', linewidth=3, color='xkcd:indigo', label='Fit')
    ax1.set_xlabel(r'$Wavelength   (\AA)$')
    ax1.set_ylabel('Aperture Correction')
    ax1.set_ylim(y1min, y1max)
    ax1.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    
    #
    ax2.fill_between(wl, apc_med-nsigma*apc_std, apc_med+nsigma*apc_std, alpha=0.25, color='xkcd:macaroni and cheese', label='Minimum to maximum errors of the AC curves')
    #
    ax2.plot(wl, apc_cube[:, 0], alpha=0.4, color='xkcd:pale purple', label= 'Minimum to maximum values of the AC curves')
    ax2.plot(wl, np.nanmedian(apc_cube[:, 0]) * np.ones(len(wl)), color='xkcd:boring green', label='Median value')
    ax2.plot(wl, apc_med_30, '.', color='xkcd:magenta', markersize=0.5) # alpha=0.3,
    ax2.plot(5000, 20, '.', color='xkcd:magenta', markersize=0.5, label='Median 30 AC curves')
    ax2.plot(wl, apc_mean, linestyle='solid', alpha=0.25, color='xkcd:soft pink', label='Mean AC') 
    ax2.plot(wl, apc_med, linestyle='dashed', alpha=0.25, color='xkcd:sky blue', label='Median AC')
    ax2.plot(wl, apc_fit(wl), linestyle='dashed', linewidth=3, color='xkcd:indigo', label='Fit')
    ax2.set_xlabel(r'$Wavelength   (\AA)$')
    ax2.set_ylabel('Aperture Correction')
    ax2.set_ylim(y2min, y2max)
    ax2.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

    if plotfile:
        plt.savefig(plotfile)
