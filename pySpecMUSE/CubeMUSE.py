#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from astropy.io import fits
from astropy.wcs import WCS
# import pandas as pd
from photutils.aperture import CircularAperture, CircularAnnulus
# import scipy.interpolate as ssi
# import matplotlib.pyplot as plt
# import os
from astropy.table import Table

from .myphotutils import get_centroids, get_stars_for_apc, runphot_ima_aps, apc_spec_single_wl, do_apphot
from .apc_plots import plot_curve_of_growth_iv, plot_apc
# from .utils import running_median_spec
from .StarMUSE import StarMUSE, apc_calc_single_star


class CubeMUSE(object):
    """
        The class stores the parameters for a MUSE pointing

        params:
            the class requires a dictionary as input, with the following keywords/parameters
            'code' : short code of the MUSE pointing (e.g. '6')
            'datadir' : directory for the data files and outputs
            'default_names' : set True as default, run with default names for the files
            'file' : name of input cube file (required if default_names is False)'
            'file_i_image' : name of I band image computed by the MUSE pipeline
            'file_v_image' : name of V band image computed by the MUSE pipeline
            'nproc' : maximum number of processes to spawn for multiprocessing computations

        Example:
            pars={'code' : '6',
                  'nproc' : 8,
                  'datadir' : '/users/ltesti/Desktop/GDrive-INAF/ColabDataTesiGiuseppe/F6/',
                  'default_names' : False,
                  'file' : 'DATA_Long6.fits',
                  'file_i_image' : 'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits',
                  'file_v_image' : 'WFM_Tr14_long_6_Johnson_V_IMAGE_FOV.fits',
                  'file_pos' : 'i.dat',
                  'coo_offset' : [0.0,0.0,0.0], # astrometric offset [arcsec, arcsec, pa]
                  }
        """

    def __init__(self, parameters):
        """
        set up the object: this one reads the spectrum within given boundaries
        """

        self.params = parameters

        try :
            self.pointing_code = self.params['code']
            self.datadir = self.params['datadir']
            self.default_names = self.params['default_names']
            if 'coo_offset' in self.params.keys():
                self.coo_offset = self.params['coo_offset']
            else:
                self.coo_offset = None

            self.nproc = None
            if 'nproc' in self.params.keys():
                if (self.params['nproc']):
                    self.nproc = self.params['nproc']
                    if self.nproc < 2:
                        self.nproc = None

            self.set_parameters()
            self.set_cubewcs()

            self.has_centroids = False
            self.has_apc_stars = False
            self.has_apc_values = False

            self.set_starlis()

        except KeyError:
            raise ValueError("Cannot initiate analysis without default values for pointing_code, datadir, and default_names switch\n {}".format(self.params))

    def set_parameters(self):
        #
        # function to set the optional parameters
        #
        if self.default_names:
            self.file = self.datadir+'DATA_Long'+self.pointing_code+'.fits'
            self.file_i_image = self.datadir+'WFM_Tr14_long_'+self.pointing_code+'_Cousins_I_IMAGE_FOV.fits'
            self.file_v_image = self.datadir+'WFM_Tr14_long_'+self.pointing_code+'_Johnson_V_IMAGE_FOV.fits'
        else:
            try:
                self.file = self.datadir + self.params['file']
                self.file_i_image = self.datadir + self.params['file_i_image']
                self.file_v_image = self.datadir + self.params['file_v_image']
            except KeyError:
                raise ValueError(
                    "When default_names is set to False then the input parameters set has to include 'file', "
                    "'file_i_image', and 'file_v_image' keywords\n")
        if 'figdir' in self.params.keys():
            self.figdir = self.params['figdir']
        else:
            self.figdir = self.datadir

    def set_cubewcs(self,verbose=True):
        #
        hdul = fits.open(self.file)
        hdul.info()
        new = hdul['DATA']
        hdul.close()
        # Setting the wavelength vector
        # new.header
        self.cube_cd11 = new.header['CD1_1']
        self.cube_cd12 = new.header['CD1_2']
        self.cube_crval1 = new.header['CRVAL1']
        self.cube_crpix1 = new.header['CRPIX1']
        self.cube_cunit1 = new.header['CUNIT1']
        self.cube_npix1 = np.arange(new.header['NAXIS1'])
        self.cube_cd22 = new.header['CD2_2']
        self.cube_cd21 = new.header['CD2_1']
        self.cube_crval2 = new.header['CRVAL2']
        self.cube_crpix2 = new.header['CRPIX2']
        self.cube_cunit2 = new.header['CUNIT2']
        self.cube_npix2 = np.arange(new.header['NAXIS2'])
        #
        self.cube_cdelt3 = new.header['CD3_3']
        self.cube_crval3 = new.header['CRVAL3']
        self.cube_crpix3 = new.header['CRPIX3']
        self.cube_cunit3 = new.header['CUNIT3']
        self.cube_Nwl3 = np.arange(new.header['NAXIS3'])
        #
        self.wcs = WCS(new.header)

        self.wl = (self.cube_Nwl3 - self.cube_crpix3 + 1) * self.cube_cdelt3 + self.cube_crval3
        if verbose:
            print("Cube wavelength axis definition: CDELT:{0}, CRVAL={1}, CRPIX={2}\n".format(self.cube_cdelt3, self.cube_crval3, self.cube_crpix3))

    def set_starlis(self):
        """
        This function creates the list of stars objects starting from the stellar positons created
        by the set_centroids() method

        :return: void
        """
        #
        if not self.has_centroids:
            self.set_centroids()
        #
        self.stars = []
        istar = 0
        for i in range(len(self.sources_i)):
            istar += 1
            star_id = 'f'+str(self.pointing_code)+'_'+str(istar)
            starpars = {
                'star_id' : star_id,
                'xcentroid' : (self.sources_i[i])['xcentroid'],
                'ycentroid' : (self.sources_i[i])['ycentroid'],
                'mag' : (self.sources_i[i])['mag'],
                'skycoo' : self.wcs.pixel_to_world((self.sources_i[i])['xcentroid'], (self.sources_i[i])['ycentroid'],0)
            }
            self.stars.append(StarMUSE(starpars))
        self.nstars = istar

    def set_centroids(self):
        """
        Function to read or find the stellar positions.
        Uses the get_centroids function in myphotutils.py to call the daofind algorithm

        :return: void
        """
        #
        if 'file_pos' in self.params.keys():
            self.file_pos = self.params['file_pos']
            self.sources_i = Table.read(self.datadir+self.file_pos,format='csv')
        else:
            self.positions_i, self.sources_i = get_centroids(self.file_i_image)
        #
        self.has_centroids =True

    def set_stars_for_apc(self, mindist=10, magsat=-8, magperc=8,
                          doplo=True, image=None, plotfile='f_stars_for_apc.pdf'):
        """
        Function to identify the APC stars in the field.

        :param mindist: (float) minimum distance (in pixels) to consider star as isolated
        :param magsat: (float) minimum magnitude below which a star is considered to be saturated
        :param magperc: (float) percentile to select th ebrightest stars in the field
        :param doplo: (bool) if True prepare a diagnostic plot
        :param image: (fits file name) if set uses this as background image, otherwise the I band is used
        :param plotfile: (string) file for the figure
        :return: void
        """
        #
        if not image:
            image = self.file_i_image
        plotfile = self.figdir+plotfile
        self.n_apc = get_stars_for_apc(self.stars,
                                       mindist=mindist, magsat=magsat, magperc=magperc,
                                       doplo=doplo, image=image, plotfile=plotfile)

        self.apc_stars = [self.stars[i] for i in self.n_apc]

        self.has_apc_stars = True

    def curve_of_growth_iv(self,radii=np.arange(1,10,0.5),skyrad=(10,15), guessrad=3., doplo_iv=True, plotfile='f_curve_of_growth_iv.pdf'):
        """
        This function computes the curve of growth in the I and V band, to help identify the best apertures for
        spectrophotometry. Optionally plots a diagnostic figure.
        If not set, it identifies the apc stars (with default parameters and no diagnostic plot)

        :param radii: (float array) array of radii for the photometry curve of growth
        :param skyrad: (tuple) inner and outer radius for the sky annulus
        :param guessrad: (float) guess at the best radius (for plotting purposes)
        :param doplo_iv: (bool) if True do the diagnostic plot
        :param plotfile: (string) name of the figure file
        :return: void
        """
        #
        plotfile = self.figdir + plotfile
        if not self.has_apc_stars:
            self.set_stars_for_apc(doplo=False)
        self.cog_radii = radii
        self.cog_skyrad = skyrad

        #
        ima_i = fits.open(self.file_i_image)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
        data_i = ima_i['DATA'].data
        ima_i.close()
        ima_v = fits.open(self.file_v_image)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
        data_v = ima_v['DATA'].data
        ima_v.close()
        #
        self.cog_mag_i = runphot_ima_aps(self.positions_i[self.n_apc], self.cog_radii,
                                         self.cog_skyrad[0], self.cog_skyrad[1], data_i)
        self.cog_mag_v = runphot_ima_aps(self.positions_i[self.n_apc], self.cog_radii,
                                         self.cog_skyrad[0], self.cog_skyrad[1], data_v)
        for i in range(len([self.stars[j] for j in self.n_apc])):
            self.stars[self.n_apc[i]].cog_mag_i = np.array([a[i] for a in self.cog_mag_i])
            self.stars[self.n_apc[i]].cog_mag_v = np.array([a[i] for a in self.cog_mag_v])
            self.stars[self.n_apc[i]].cog_radii = self.cog_radii

        if doplo_iv:
            plot_curve_of_growth_iv(self.cog_radii, self.cog_mag_i, self.cog_mag_v,
                                    self.positions_i[self.n_apc], self.file_i_image, self.file_v_image,
                                    guessrad=guessrad, plotfile=plotfile)

    def set_apc_values(self,radii=(3.,10.),skyrad=(10,15), hw_box_median=30, sclip_median=2.0,
                       apc_sclip=3.0, doplo_apc=True, plotfile='f_apc_values.pdf'):
        """
        This funcion is used to compute the aperture correction as a function of wavelength.
        If not done already, first identifies the stars to be used for apc (using default parameters),
        then computes apc(wl) for each of them, computes median apc with sigma clips and combines
        all apc stars, produces a standard diagnostic plot.

        :param radii: (tuple) the two photometry apertures radii used to compute the aperture correction
        :param skyrad: (tuple) the inner and outer radius for the sky annulus
        :param hw_box_median: (int) half width of the box used for running median
        :param sclip_median: (float) number of sigma for the sigma clipping for median computation
        :param apc_sclip: (float) number of sigma for the sigma clipping in combining different stars
        :param doplo_apc: (bool) if Ture prepare a diagnostic plot
        :param plotfile: (string) name of the figure file
        :return: void
        """
        #
        plotfile = self.figdir + plotfile

        if not self.has_apc_stars:
             self.set_stars_for_apc()
        self.apc_radii = radii
        self.apc_skyrad = skyrad

        #
        #
        hdul = fits.open(self.file)
        hdul.info()
        self.cube_data = hdul['DATA'].data
        hdul.close()
        #
        self.apc_cube = np.zeros((len(self.wl),len(self.n_apc)))

        self._run_getspecapc_proc_mp()

        for i in range(len(self.n_apc)):
            self.stars[self.n_apc[i]].apc_spec = self.apc_cube[:,i]
            self.stars[self.n_apc[i]].apc_wl = self.wl
            self.stars[self.n_apc[i]].apc_radii = self.apc_radii

        self.apc_med_30 = np.zeros((len(self.wl), len(self.n_apc)))
        self.apc_std_30 = np.zeros((len(self.wl), len(self.n_apc)))
        self.apc_mean_30 = np.zeros((len(self.wl), len(self.n_apc)))

        self._run_apc_proc_mp({'hw_box' : hw_box_median, 'sclip' : sclip_median})

        for i in range(len(self.n_apc)):
            self.apc_med_30[:,i] = self.apc_stars[i].apc_med
            self.apc_mean_30[:, i] = self.apc_stars[i].apc_mean
            self.apc_std_30[:, i] = self.apc_stars[i].apc_std

        self.apc_med = np.nanmedian(self.apc_med_30, axis=1)
        self.apc_mean = np.nanmean(self.apc_mean_30, axis=1)
        self.apc_std = np.nanstd(self.apc_med_30, axis=1)
        if apc_sclip:
            a2 = apc_sclip*apc_sclip
            for i in range(len(self.wl)):
                ng = np.where((self.apc_med_30[i, :] - self.apc_med[i])**2 < a2)
                self.apc_med[i] = np.nanmedian(self.apc_med_30[i,ng], axis=1)
                self.apc_mean[i] = np.nanmedian(self.apc_mean_30[i, ng], axis=1)
                self.apc_std[i] = np.nanmedian(self.apc_std_30[i, ng], axis=1)


        self.has_apc_values = True

        if doplo_apc:
            plot_apc(self.wl, self.apc_cube, self.apc_med, self.apc_mean, self.apc_std, self.apc_med_30, nsig=apc_sclip, plotfile=plotfile)

    def _run_getspecapc_proc_mp(self):
        """
        This function is used to setup the multiprocessing extraction of the spectra for the aperture
        correction stars. It calls a wrapper function in myphotutils.py that executes the call to
        the apphot function.

        :return:
        """
        #
        #
        if self.nproc:   # run with multiprocessing
            nproc = min(len(self.wl),self.nproc)
            myargs = []
            for iwl in range(len(self.wl)):
                myargs.append([self.positions_i[self.n_apc], self.apc_radii,
                               self.apc_skyrad, self.cube_data[iwl,:,:]])
                #myargs.append([self, iwl])

            with mp.Pool(nproc) as p:
                allwl = p.map(apc_spec_single_wl, myargs)

            for iwl in range(len(self.wl)):
                self.apc_cube[iwl, :] = allwl[iwl]
        else:       # run single process
            for iwl in range(len(self.wl)):
                self.apc_cube[iwl, :] = apc_spec_single_wl([self.positions_i[self.n_apc], self.apc_radii,
                                                              self.apc_skyrad, self.cube_data[iwl,:,:]])

    def _run_apc_proc_mp(self, my_star_method_args):
        """
        This function is used to setup the multiprocessing computation of the mean, median and std
        of the spectra for the aperture correction stars. It calls a wrapper function in StarMUSE.py that executes
        the call to the utils function.

        :return:
        """
        #
        if self.nproc:   # run with multiprocessing
            nproc = min(len(self.n_apc),self.nproc)
            myargs = []
            for mystar in self.apc_stars:
                myargs.append([mystar, my_star_method_args])

            with mp.Pool(nproc) as p:
                self.apc_stars = p.map(apc_calc_single_star, myargs)
        else:       # run single process
            for mystar in self.apc_stars:
                mystar = apc_calc_single_star([mystar, my_star_method_args])

    def extract_spectra(self, add_apc=True, radius=3, sky_radius=10., sky_dannulus=5.):
        self.my_spectra = np.zeros((len(self.positions_i),len(self.wl)))
        self.my_magspec = np.copy(self.my_spectra)
        self.my_skies = np.copy(self.my_spectra)
        self.my_skies_noise = np.copy(self.my_spectra)

        if add_apc:
            radius = self.apc_radii[0]

        self.apertures_i = CircularAperture(self.positions_i, r=radius)
        self.annulus_aperture = CircularAnnulus(positions_i, sky_radius, sky_radius+sky_dannulus)

        if self.nproc:
            nproc = min(len(self.wl), self.nproc)
            myargs = [[self.cube_data[i,:,:], self.apertures_i, self.annulus_aperture] for i in range(len(self.wl))]

            with mp.Pool(nproc) as p:
                magspecout, spectraout, skiesout, skies_noiseout = p.map(do_apphot, myargs)
            for i in range(len(self.wl)):
                my_magspec[:,i] = magspecout[i]
                my_spectra[:,i] = spectraout[i]
                my_skies[:,i] = skiesout[i]
                my_skies_noise[:,i] = skies_noiseout[i]
        else:
            for i in range(len(self.wl)):
                my_magspec[:,i], my_spectra[:,i], my_skies[:,i], my_skies_noise[:,i] = do_apphot(new.data[i,:,:],apertures_i,annulus_aperture)
            
