#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from .aperturesplots import plotstars

def get_centroids(imagefile, verbose=True, sigma=5.0, fwhm=4.5, thres=1., sigma_radius=2.):
    #
    ima = fits.open(imagefile)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
    data = ima['DATA'].data
    #
    mean, median, std = sigma_clipped_stats(data, sigma=sigma)
    if verbose:
        print("Image stats for {0}: mean={1}, median={2}, std={3}\n".format(imagefile, mean, median, std))
    daofind = DAOStarFinder(fwhm=fwhm, threshold=thres * std, sigma_radius=sigma_radius)
    sources = daofind(data - median)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
    #
    return np.transpose((sources['xcentroid'], sources['ycentroid'])), sources

def get_stars_for_apc(sources, mindist=10, magsat=-8, magperc=5, doplo=False, image=None, plotfile='f_stars_for_apc.pdf'):
    #
    maxx = np.max(np.array(sources['xcentroid']))
    maxy = np.max(np.array(sources['ycentroid']))
    dist2 = np.ones(len(sources)) * maxx * maxy
    for i in range(len(sources)):
        for j in range(len(sources)):
            if j != i:
                dx2 = ((sources['xcentroid'])[i] - (sources['xcentroid'])[j]) ** 2
                dy2 = ((sources['ycentroid'])[i] - (sources['ycentroid'])[j]) ** 2
                d2 = dx2 + dy2
                if d2 < dist2[i]:
                    dist2[i] = d2
    #
    dist = np.sqrt(dist2)
    #
    # select stars that have neibours farther than mindist, ara at least mindist from the edges,
    #  and havemagnitues in the specified percentiles, but below saturation
    n_apc = np.where( (dist >= mindist) &
                      (sources['xcentroid'] >= mindist) &
                      (sources['xcentroid'] <= maxx-mindist) &
                      (sources['ycentroid'] >= mindist) &
                      (sources['ycentroid'] <= maxy-mindist) &
                      (sources['mag'] <= np.percentile(sources['mag'],magperc)) &
                      (sources['mag'] >= magsat)
                      )
    #
    if doplo:
        if image:
            plotstars(image, np.transpose(((sources[n_apc])['xcentroid'], (sources[n_apc])['ycentroid'])),
                      aprad=mindist, fileout=plotfile, mytitle='Aperture Correction Stars')
        else:
            printf("ERROR: to plot the results, you need to pass also the image name image='filename'")
    #
    return n_apc