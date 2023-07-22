#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

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
    return np.transpose((sources['xcentroid'], sources['ycentroid']))
