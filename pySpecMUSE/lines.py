#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

known_lines = [
    { 'linename' : '[OI]6300', 'wlmin' : 6297, 'wlmax' : 6303, 'wlcont1' : 6250, 'wlcont2' : 6350,
      'min_snr' : 3., 'doplot' : False, 'plotsky' : False, 'conttype' : 'median' }
]

def line_integral(Star, wlmin, wlmax, wlcont1, wlcont2, min_snr=3., 
                  doplot=False, plotsky=False, conttype='median', docontfitplot=False):
    '''
    Function to integrate the line and subtract the adjacent continuum
    '''
   
    if docontfitplot:
        doplot=False

    mywl, myfl, mysky, myskynoise = get_vec(Star, wlcont1, wlcont2)

    nline = np.where((mywl>=wlmin) & (mywl<=wlmax))
    ncont = np.where(((mywl>=wlcont1) & (mywl<=wlmin)) |  ((mywl>=wlmax) & (mywl<=wlcont2)))
  
    if conttype == 'linear':
        fit = linear(mywl[ncont], myfl[ncont], doplot=docontfitplot)
        cont = fit[0]+fit[1]*mywl
    elif conttype == 'median':
        cont = np.zeros(len(mywl))+np.median(myfl[ncont])
    else:
        cont = np.zeros(len(mywl))+np.median(myfl[ncont])
        
    myflsub = myfl-cont
    spec_rms = np.std(myflsub)
    fline_max = np.max(myflsub[nline])
    if fline_max/spec_rms >= min_snr:
        fint = (myflsub[nline]).sum()*(mywl[1]-mywl[0])
    else:
        fint = 0.
    
    fint_noise = len(nline[0])*(mywl[1]-mywl[0])*spec_rms
    fsky_noise = (mywl[1]-mywl[0])*(myskynoise[nline]).sum()

    retdic={
        'mywl' : mywl, 
        'myfl' : myfl, 
        'mysky' :mysky, 
        'myskynoise': myskynoise,
        'cont' : cont,
        'wlmin' : wlmin,
        'wlmax' : wlmax,
        'wlcont1' : wlcont1,
        'wlcont2' : wlcont2,
        'fline_max' : fline_max,
        'spec_rms' : spec_rms,
        'fint' : fint,
        'fint_noise' : fint_noise,
        'fsky_noise' : fsky_noise

    }

    if doplot:
        lineplot(retdic, plotsky=plotsky)

    return retdic

def get_vec(Star, wl1, wl2):
    nplot = np.where((Star.wl>=wl1) & (Star.wl<=wl2))

    mywl = np.copy(Star.wl[nplot])
    myfl = np.copy(Star.corrected_spectrum[nplot])
    mysky = np.copy(Star.sky[nplot])
    myskynoise = np.copy(Star.sky_noise[nplot])
    #
    return mywl, myfl, mysky, myskynoise


def lineplot(lp, plotsky=True):
    #
    #mywl, myfl, mysky, myskynoise = get_vec(Star, wlcont1, wlcont2,)
    nline = np.where((lp['mywl']>=lp['wlmin']) & (lp['mywl']<=lp['wlmax']))
    
    if plotsky:
        plt.plot(lp['mywl'], lp['mysky'], color='grey')
    plt.fill_between(lp['mywl'], lp['myfl']-lp['spec_rms'], lp['myfl']+lp['spec_rms'], color='red', alpha=0.4)
    plt.fill_between(lp['mywl'], lp['myfl']-lp['myskynoise'], lp['myfl']+lp['myskynoise'], color='green', alpha=0.3)
    plt.plot(lp['mywl'], lp['myfl'], color='g')
    plt.plot(lp['mywl'], lp['cont'], color='cyan')
    plt.plot([lp['wlmin'],lp['wlmin']],[-2*lp['spec_rms']+np.median(lp['cont'][nline]),3.*lp['spec_rms']+np.median(lp['cont'][nline])], linestyle='--', color='r')
    plt.plot([lp['wlmax'],lp['wlmax']],[-2*lp['spec_rms']+np.median(lp['cont'][nline]),3.*lp['spec_rms']+np.median(lp['cont'][nline])], linestyle='--', color='r')


def linear(x, y, doplot=False):
        #
        n = len(x)
        sx = x.sum()
        sxx = (x*x).sum()
        sy = (y).sum()
        sxy = (x*y).sum()
        det = n*sxx-sx*sx
        a = (n*sxy-sy*sx)/det
        b = (sy*sxx-sx*sxy)/det
        if doplot:
            plt.plot(x, y, 'o', color='b')
            xx=np.linspace(np.min(x),np.max(x), 100)
            plt.plot(xx, a+b*xx, linestyle='--', color='red')
        return (a, b)