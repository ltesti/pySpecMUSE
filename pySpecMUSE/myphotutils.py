#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats

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

def get_stars_for_apc(starlis, mindist=10, magsat=-8, magperc=5, doplo=False, image=None, plotfile=None):
    #
    maxx = np.max(np.array([s.xcentroid for s in starlis]))
    maxy = np.max(np.array([s.ycentroid for s in starlis]))
    allmags = np.array([s.daofind_mag for s in starlis])
    mindist2 = mindist*mindist
    n_apc = []
    for i in range(len(starlis)):
        s1 = starlis[i]
        isolated = True
        for s2 in starlis:
            if s2.id != s1.id:
                dx2 = (s1.xcentroid - s2.xcentroid) ** 2
                dy2 = (s1.ycentroid - s2.ycentroid) ** 2
                if (dx2+dy2) < mindist2:
                    isolated = False
        if (isolated & (s1.xcentroid >= mindist) &
            (s1.xcentroid <= maxx-mindist) &
            (s1.ycentroid >= mindist) &
            (s1.ycentroid <= maxy-mindist) &
            (s1.daofind_mag <= np.percentile(allmags, magperc)) &
            (s1.daofind_mag >= magsat)) :
                s1.apc_star = True
                n_apc.append(i)

    #
    #
    if doplo:
        if image:
            all_xy = np.transpose(([s.xcentroid for s in starlis], [s.ycentroid for s in starlis]))
            xy = np.transpose(([starlis[i].xcentroid for i in n_apc], [starlis[i].ycentroid for i in n_apc]))
            plotstars(image, xy, plotall=True, allpos=all_xy,
                      aprad=mindist, fileout=plotfile, mytitle='Aperture Correction Stars')
        else:
            print("ERROR: to plot the results, you need to pass also the image name image='filename'")
    #
    return n_apc

def do_apphot(in_ima, in_aper, in_annulus):
    """Function to perform aperture photometry in one image

    Function to perform aperture photometry in a set of apertures (and corresponding skies),
    in an image. This is use as the base to perform extraction of the stellar spectra.
    The function requires as input the image and the apertures (for phot and sky).
    This function performs default sigmaclipping on the sky background

    :param in_ima: (float array) image to perform the photometry on
    :param in_aper: (CircularAperture) class defining the circular apertures
    :param in_annulus: (CircularAnnulus) class defining the sky annuli

    :return: magnitude, counts_background_subtracted, total_bkg_in_aperture, bkg_noise
    """

    ### Perform Aperture Photometry
    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    aper_stats = ApertureStats(in_ima, in_aper, sigma_clip=None, sum_method='subpixel')
    bkg_stats = ApertureStats(in_ima, in_annulus, sigma_clip=sigclip, sum_method='subpixel')

    ### Extimate the bkg dubctracted flux
    total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
    apersum_bkgsub = aper_stats.sum - total_bkg
    #print(apersum_bkgsub)
    mag=apersum_bkgsub * 0
    for elem in range(len(apersum_bkgsub)):
        if apersum_bkgsub[elem] > 0:
            mag[elem] = -2.5 * np.log10(apersum_bkgsub[elem]) + 25
        else:
            mag[elem] = 99.999
    return mag, apersum_bkgsub, total_bkg, bkg_stats.std * aper_stats.sum_aper_area.value

def runphot_ima_aps(positions, radii, sky_r1, sky_r2, data):
    apertures = []
    for i in range(len(radii)):
        apertures.append(CircularAperture(positions, r=radii[i]))
    annulus_aperture = CircularAnnulus(positions, r_in=sky_r1, r_out=sky_r2)
    #
    phot_results = []
    for ap in apertures:
        phot_results.append(do_apphot(data, ap, annulus_aperture))
    return phot_results

def apc_spec_single_wl(args):
    # wrapper to get the apc spectra for apc_stars
    phot_for_apcorr = runphot_ima_aps(args[0], args[1],
                                      (args[2])[0], (args[2])[1],
                                      args[3])
    phot_for_apcorr = [a[0] for a in phot_for_apcorr]
    return np.array(phot_for_apcorr[1] - phot_for_apcorr[0])

def get_spec_single_wl(args):
    # wrapper to extract spectra with apc applied
    # args = list containing the following:
    #      0 = positions (stellar positions for the photometry)
    #      1 = radii (assumed to contain only one radius, in any case only the first radius is returned)
    #      2 = skyrad (list of two)
    #      3 = cube_data[iwl,:,:]  (image at the selected wavelength)
    phot_for_apcorr = runphot_ima_aps(args[0], args[1],
                                      (args[2])[0], (args[2])[1],
                                      args[3])
    return np.array(phot_for_apcorr[0])
