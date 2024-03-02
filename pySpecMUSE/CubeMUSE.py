#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import os
from astropy.table import Table

from .myphotutils import get_centroids, get_stars_for_apc, runphot_ima_aps, apc_spec_single_wl, get_spec_single_wl
from .apc_plots import plot_curve_of_growth_iv, plot_apc
from .StarMUSE import StarMUSE, apc_calc_single_star
from .lines import known_lines


class CubeMUSE(object):
    """The MUSE pointing

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
                           'datadir' : '/Users/ltesti/Carina_test/F6/',
                           'figdir' : '/Users/ltesti/Carina_test/F6/fig/',
                           'default_names' : False,
                           'file' : 'DATA_Long6.fits',
                           'file_i_image' : 'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits',
                           'file_v_image' : 'WFM_Tr14_long_6_Johnson_V_IMAGE_FOV.fits',
                           # 'file_pos' : 'i.dat',
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
            if 'daofind_threshold' in self.params.keys():
                self.daofind_threshold = self.params['daofind_threshold']
            else:
                self.daofind_threshold = 1.0
            if 'daofind_sigma_radius' in self.params.keys():
                self.daofind_sigma_radius = self.params['daofind_sigma_radius']
            else:
                self.daofind_sigma_radius = 2.0
            if 'saturation_magnitude' in self.params.keys():
                self.magsat = self.params['saturation_magnitude']
            else:
                self.magsat = -8.0

            self.set_parameters()
            self.set_cubewcs()

            self.has_centroids = False
            self.has_apc_stars = False
            self.has_apc_values = False

            self.set_starlis()

        except KeyError:
            raise ValueError("Cannot initiate analysis without default values for pointing_code, datadir, and default_names switch\n {}".format(self.params))

    def set_parameters(self):
        """Function to set/read the optional parameters

        This function sets up default parameters that can be modified if the input dictionary contains the
        appropriate keywords.

        :return: void
        """
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
        """Reads the Cube and the WCS
        """
        #
        ima_i = fits.open(self.file_i_image)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
        self.data_i = ima_i['DATA'].data
        ima_i.close()
        ima_v = fits.open(self.file_v_image)  # datadir+'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits')
        self.data_v = ima_v['DATA'].data
        ima_v.close()
        #
        hdul = fits.open(self.file)
        hdul.info()
        new = hdul['DATA']
        self.cube_data = hdul['DATA'].data
        self.cube_header = hdul['DATA'].header
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

        self.wl = (self.cube_Nwl3 - self.cube_crpix3 + 1) * self.cube_cdelt3 + self.cube_crval3
        if verbose:
            print("Cube wavelength axis definition: CDELT:{0}, CRVAL={1}, CRPIX={2}\n".format(self.cube_cdelt3, self.cube_crval3, self.cube_crpix3))

    def set_starlis(self):
        """Setup the list of StarMUSE objects

        This function creates the list of stars objects starting from the stellar positons created
        by the set_centroids() method

        :return: void
        """
        #
        if not self.has_centroids:
            self.set_centroids()
        #
        mywcs = WCS(self.cube_header)
        #
        self.stars = []
        istar = 0
        for i in range(len(self.sources_i)):
            istar += 1
            if self.stars_from_file:
                mycoo = SkyCoord((self.sources_i[i])['ALPHA_J2000'], (self.sources_i[i])['DELTA_J2000'], unit="deg")
                star_id = 'f'+str(self.pointing_code)+'_'+str((self.sources_i[i])['NUMBER'])
            else:
                mycoo = mywcs.pixel_to_world((self.sources_i[i])['xcentroid'], (self.sources_i[i])['ycentroid'], 0)
                star_id = 'f'+str(self.pointing_code)+'_'+str(istar)
            starpars = {
                'star_id' : star_id,
                'xcentroid' : (self.sources_i[i])['xcentroid'],
                'ycentroid' : (self.sources_i[i])['ycentroid'],
                'mag' : (self.sources_i[i])['mag'],
                'skycoo' : mycoo
            }
            self.stars.append(StarMUSE(starpars))
        self.nstars = istar

    def set_centroids(self):
        """Set the centroids values

        Function to read or find the stellar positions.
        Uses the get_centroids function in myphotutils.py to call the daofind algorithm

        :return: void
        """
        #
        if 'file_pos' in self.params.keys():
            self.file_pos = self.params['file_pos']
            self.sources_i = Table.read(self.datadir+self.file_pos,format='csv')
            self.sources_i.rename_column('X_IMAGE', 'xcentroid')
            self.sources_i.rename_column('Y_IMAGE', 'ycentroid')
            self.sources_i.rename_column('MAG_APER', 'mag')
            self.positions_i = np.transpose((self.sources_i['xcentroid'], self.sources_i['ycentroid']))
            self.stars_from_file = True
        else:
            self.positions_i, self.sources_i = get_centroids(self.file_i_image, thres=self.daofind_threshold, sigma_radius=self.daofind_sigma_radius)
            self.stars_from_file = False
        #
        self.has_centroids =True

    def set_stars_for_apc(self, mindist=10, magperc=8,
                          doplo=True, image=None, plotfile='f_stars_for_apc.pdf'):
        """Select ApC stars

        Function to identify the APC stars in the field.

        :param mindist: (float) minimum distance (in pixels) to consider star as isolated
        :param magperc: (float) percentile to select th ebrightest stars in the field
        :param doplo: (bool) if True prepare a diagnostic plot
        :param image: (fits file name) if set uses this as background image, otherwise the I band is used
        :param plotfile: (string) file for the figure
        :return: void
        """
        #
        if not image:
            image = self.file_i_image
        self.plotfile_apc_stars = self.figdir+os.path.split(plotfile)[1]
        self.n_apc = get_stars_for_apc(self.stars,
                                       mindist=mindist, magsat=self.magsat, magperc=magperc,
                                       doplo=doplo, image=image, plotfile=self.plotfile_apc_stars)

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

        if not self.has_apc_stars:
            self.set_stars_for_apc(doplo=False)
        self.cog_radii = radii
        self.cog_skyrad = skyrad

        #
        self.cog_mag_i = runphot_ima_aps(self.positions_i[self.n_apc], self.cog_radii,
                                         self.cog_skyrad[0], self.cog_skyrad[1], self.data_i)
        self.cog_mag_v = runphot_ima_aps(self.positions_i[self.n_apc], self.cog_radii,
                                         self.cog_skyrad[0], self.cog_skyrad[1], self.data_v)
        self.cog_mag_i = [a[0] for a in self.cog_mag_i]
        self.cog_mag_v = [a[0] for a in self.cog_mag_v]
        for i in range(len([self.stars[j] for j in self.n_apc])):
            self.stars[self.n_apc[i]].cog_mag_i = np.array([a[i] for a in self.cog_mag_i])
            self.stars[self.n_apc[i]].cog_mag_v = np.array([a[i] for a in self.cog_mag_v])
            self.stars[self.n_apc[i]].cog_radii = self.cog_radii

        if doplo_iv:
            self.plotfile_cog = self.figdir + os.path.split(plotfile)[1]
            plot_curve_of_growth_iv(self.cog_radii, self.cog_mag_i, self.cog_mag_v,
                                    self.positions_i[self.n_apc], self.file_i_image, self.file_v_image,
                                    guessrad=guessrad, plotfile=self.plotfile_cog)
        	

    def extract_apc_spectra(self):
        self.apc_cube = np.zeros((len(self.wl),len(self.n_apc)))

        self._run_getspecapc_proc_mp()

        for i in range(len(self.n_apc)):
            self.stars[self.n_apc[i]].apc_spec = self.apc_cube[:,i]
            self.stars[self.n_apc[i]].apc_wl = self.wl
            self.stars[self.n_apc[i]].apc_radii = self.apc_radii

    def set_apc_values(self,radii=(3.,10.),skyrad=(10,15), hw_box_median=50, sclip_median=2.,
                       apc_sclip=2., apc_fit_order=3, doplo_apc=True, plotfile='f_apc_values.pdf'):
        """Computes the ApC values as a function of wavelength

        This funcion is used to compute the aperture correction as a function of wavelength.
        If not done already, first identifies the stars to be used for apc (using default parameters),
        then computes apc(wl) for each of them, computes median apc with sigma clips and combines
        all apc stars, produces a standard diagnostic plot.

        :param radii: (tuple) the two photometry apertures radii used to compute the aperture correction
        :param skyrad: (tuple) the inner and outer radius for the sky annulus
        :param hw_box_median: (int) half width of the box used for running median
        :param sclip_median: (float) number of sigma for the sigma clipping for median computation
        :param apc_sclip: (float) number of sigma for the sigma clipping in combining different stars
        :param apc_fit_order: (int) order of the polinomial fit for the apc correction
        :param doplo_apc: (bool) if Ture prepare a diagnostic plot
        :param plotfile: (string) name of the figure file
        :return: void
        """
        #

        if not self.has_apc_stars:
             self.set_stars_for_apc()
        self.apc_radii = radii
        self.apc_skyrad = skyrad

        #
        self.extract_apc_spectra()

        self.analyse_apc_spectra(radii=radii, skyrad=skyrad, apc_fit_order=apc_fit_order, hw_box_median=hw_box_median,
                                 sclip_median=sclip_median, apc_sclip=apc_sclip, doplo_apc=doplo_apc, plotfile=plotfile)

        self.has_apc_values = True

    def analyse_apc_spectra(self, radii=(3.,10.), skyrad=(10,15), apc_fit_order=3, hw_box_median=50, sclip_median=3.0,
                       apc_sclip=2.0, sclip_niter=5, doplo_apc=True, plotfile='f_apc_values.pdf'):

        if not self.has_apc_stars:
             self.set_stars_for_apc()

        if not self.has_apc_values:
            self.apc_radii = radii
            self.apc_skyrad = skyrad
            self.extract_apc_spectra()
            self.has_apc_values=True

        self.apc_fit_order = apc_fit_order

        self.apc_med_30 = np.zeros((len(self.wl), len(self.n_apc)))
        self.apc_std_30 = np.zeros((len(self.wl), len(self.n_apc)))
        self.apc_mean_30 = np.zeros((len(self.wl), len(self.n_apc)))

        self._run_apc_proc_mp({'hw_box' : hw_box_median, 'sclip' : sclip_median, 'maxiter' : sclip_niter})

        for i in range(len(self.n_apc)):
            self.apc_med_30[:,i] = self.apc_stars[i].apc_med
            self.apc_mean_30[:, i] = self.apc_stars[i].apc_mean
            self.apc_std_30[:, i] = self.apc_stars[i].apc_std

        self.apc_med = np.nanmedian(self.apc_med_30, axis=1)
        self.apc_mean = np.nanmean(self.apc_mean_30, axis=1)
        self.apc_std = np.nanstd(self.apc_med_30, axis=1)
        if apc_sclip:
            for i in range(len(self.wl)):
                iter = 0
                med = self.apc_med[i]
                mean = self.apc_mean[i]
                std = self.apc_std[i]
                while (iter<sclip_niter):
                    ng = np.where((self.apc_med_30[i, :] - med)**2 < (apc_sclip * std)**2)
                    med = np.nanmedian(self.apc_med_30[i, ng], axis=1)
                    mean = np.nanmean(self.apc_mean_30[i, ng], axis=1)
                    std = np.nanstd(self.apc_med_30[i, ng], axis=1)
                    iter += 1
                self.apc_med[i] = med
                self.apc_mean[i] = mean
                self.apc_std[i] = std

        self.apc_fit = np.poly1d(np.polyfit(self.wl, self.apc_mean, self.apc_fit_order))

        if doplo_apc:
            self.plotfile_apc_specra = self.figdir + os.path.split(plotfile)[1]
            plot_apc(self.wl, self.apc_cube, self.apc_med, self.apc_mean, self.apc_std, self.apc_med_30,
                     self.apc_fit, nsig=apc_sclip, plotfile=self.plotfile_apc_specra)

    def _run_getspecapc_proc_mp(self):
        """Extracts the APC spectra

        This function is used to setup the multiprocessing extraction of the spectra for the aperture
        correction stars. It calls a wrapper function in myphotutils.py that executes the call to
        the apphot function.

        :return: void
        """
        #
        #
        if self.nproc:   # run with multiprocessing
            nproc = min(len(self.wl),self.nproc)
            myargs = []
            for iwl in range(len(self.wl)):
                myargs.append([self.positions_i[self.n_apc], self.apc_radii,
                               self.apc_skyrad, self.cube_data[iwl,:,:]])

            with mp.Pool(nproc) as p:
                allwl = p.map(apc_spec_single_wl, myargs)

            for iwl in range(len(self.wl)):
                self.apc_cube[iwl, :] = allwl[iwl]
        else:       # run single process
            for iwl in range(len(self.wl)):
                self.apc_cube[iwl, :] = apc_spec_single_wl([self.positions_i[self.n_apc], self.apc_radii,
                                                              self.apc_skyrad, self.cube_data[iwl,:,:]])

    def _run_apc_proc_mp(self, my_star_method_args):
        """Execute ApC spectra processing

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
                myargs.append([mystar.apc_spec, my_star_method_args])

            with mp.Pool(nproc) as p:
                allstars = p.map(apc_calc_single_star, myargs)
            for i in range(len(self.apc_stars)):
                self.apc_stars[i].apc_med = (allstars[i])[0]
                self.apc_stars[i].apc_mean = (allstars[i])[1]
                self.apc_stars[i].apc_std = (allstars[i])[2]
        else:       # run single process
            for mystar in self.apc_stars:
                result = apc_calc_single_star([mystar.apc_spec, my_star_method_args])
                mystar.apc_med = result[0]
                mystar.apc_mean = result[1]
                mystar.apc_std = result[2]

    def extract_spectra(self, radius, sky_radii, add_apc=True):
        #
        self.spec_radius = radius
        self.spec_sky_radii = sky_radii
        self.spec_add_apc = add_apc
        self.magspec = np.zeros((len(self.wl), len(self.stars)))
        self.spectra = np.zeros((len(self.wl), len(self.stars)))
        self.skies = np.zeros((len(self.wl), len(self.stars)))
        self.skies_noise = np.zeros((len(self.wl), len(self.stars)))

        if self.nproc:  # run with multiprocessing
            nproc = min(len(self.wl), self.nproc)
            myargs = []
            for iwl in range(len(self.wl)):
                myargs.append([self.positions_i, self.spec_radius,
                               self.spec_sky_radii, self.cube_data[iwl, :, :]])

            with mp.Pool(nproc) as p:
                allwl = p.map(get_spec_single_wl, myargs)

            for iwl in range(len(self.wl)):
                self.magspec[iwl, :], self.spectra[iwl, :], self.skies[iwl, :], self.skies_noise[iwl, :] = allwl[iwl]
        else:  # run single process
            for iwl in range(len(self.wl)):
                self.magspec[iwl, :], self.spectra[iwl, :], self.skies[iwl, :], self.skies_noise[iwl, :] = get_spec_single_wl([self.positions_i, self.spec_radius,
                               self.spec_sky_radii, self.cube_data[iwl, :, :]])

        for j in range(len(self.stars)):
            self.stars[j].wl = self.wl
            self.stars[j].magspec = self.magspec[:, j]
            self.stars[j].spectrum = self.spectra[:, j] * 10**(-20)
            self.stars[j].sky = self.skies[:, j] * 10**(-20)
            self.stars[j].sky_noise = self.skies_noise[:, j] * 10**(-20)
            self.stars[j].has_spectrum = True

        self.compute_rms_spectra()

        self.has_spectra = True

        if add_apc:
            self.correct_spectra_apc()

    def compute_rms_spectra(self, hw_box_median=50, sclip_median=3, sclip_niter=5):
        #
        mydicargs = {'hw_box' : hw_box_median, 'sclip' : sclip_median, 'maxiter' : sclip_niter}
        if self.nproc:  # run with multiprocessing
            nproc = min(len(self.stars), self.nproc)
            myargs = []
            for mystar in self.stars:
                myargs.append([mystar.spectrum, mydicargs])

            with mp.Pool(nproc) as p:
                allstars = p.map(apc_calc_single_star, myargs)
            for i in range(len(self.stars)):
                self.stars[i].rms_spectrum = (allstars[i])[2]
        else:  # run single process
            for mystar in self.stars:
                result = apc_calc_single_star([mystar, mydicargs])
                mystar.rms_spectrum = result[2]

    def correct_spectra_apc(self):
        #
        for j in range(len(self.stars)):
            self.stars[j].corrected_magspec = self.stars[j].magspec + self.apc_fit(self.wl)
            self.stars[j].corrected_spectrum = self.stars[j].spectrum * 10.**(-0.4*self.apc_fit(self.wl))
            self.stars[j].has_corrected_spectrum = True

    def get_all_lines(self):
        #
        # if self.nproc:  # run with multiprocessing
        #     nproc = min(len(self.stars), self.nproc)
        #     myargs = []
        #     for mystar in self.stars:
        #         myargs.append([mystar.spectrum, mydicargs])


        #     with mp.Pool(nproc) as p:
        #         allwl = p.map(get_spec_single_wl, myargs)

        #     for iwl in range(len(self.wl)):
        #         self.magspec[iwl, :], self.spectra[iwl, :], self.skies[iwl, :], self.skies_noise[iwl, :] = allwl[iwl]
        # else:  # run single process
        for mystar in self.stars:
            for line in known_lines:
                mystar.get_line(line)
