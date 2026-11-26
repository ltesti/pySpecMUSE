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

