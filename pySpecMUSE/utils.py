#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# Compute running median for the spectrum
def running_median_spec(sp, hw_box, sclip=2.0, maxiter=2):
    #
    # initialize vectors
    npt = len(sp)
    mymed = np.zeros(npt)
    myrms = np.zeros(npt)
    mymean = np.zeros(npt)
    #
    # check that hw_box is contained within vector boundaries, if not reduce it
    if hw_box > int(npt/2.):
        hw_box = int(npt/2.)
    #
    # Compute rms vector
    for i in range(npt):
        #
        # define subvector for computation (at the edges, the subvector is
        # the first - or last - 2*hw_box pixels)
        if i<hw_box:
            a = np.copy(sp[0:2*hw_box+1])
        elif i>(npt-hw_box-1):
            a = np.copy(sp[-(2*hw_box+1):-1])
        else:
            a = np.copy(sp[i-hw_box:i+hw_box+1])
        #
        # compute median and std
        mid = np.nanmedian(a)
        mean = np.nanmean(a)
        std = np.nanstd(a)
        #
        # if sclip is defined, then proceed with the iterative clipping
        if sclip:
            iter = 0
            s2 = (sclip*std)**2
            while (iter<maxiter):
                nn = np.where((a-mid)**2 <= s2)
                mid = np.nanmedian(a[nn])
                mean = np.nanmean(a[nn])
                std = np.nanstd(a[nn])
                s2 = (sclip*std)**2
                iter+=1
            #
            # assign value
            myrms[i] = std
            mymean[i] = mean
            mymed[i] = mid
    #
    # return vector
    return [mymed, mymean, myrms]

