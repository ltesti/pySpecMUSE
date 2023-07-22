#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
#import scipy.interpolate as ssi
#import matplotlib.pyplot as plt
#import os
#from astropy.table import Table

#from .utils import resamp_spec, nrefrac


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

        Example:
            pars={'code' : '6',
                  'datadir' : '/users/ltesti/Desktop/GDrive-INAF/ColabDataTesiGiuseppe/F6/',
                  'default_names' : False,
                  'file' : 'DATA_Long6.fits',
                  'file_i_image' : 'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits',
                  'file_v_image' : 'WFM_Tr14_long_6_Johnson_V_IMAGE_FOV.fits',
                  }
        """

    def __init__(self, parameters):
        """
        set up the object: this one reads the spectrum within given boundaries
        """

        self.params = parameters

        try:


        default_parameters = {'code': '6',
                              'datadir': '/users/ltesti/Desktop/GDrive-INAF/ColabDataTesiGiuseppe/F6/',
                              'default_names': False,
                              'file': 'DATA_Long6.fits',
                              'file_i_image': 'WFM_Tr14_long_6_Cousins_I_IMAGE_FOV.fits',
                              'file_v_image': 'WFM_Tr14_long_6_Johnson_V_IMAGE_FOV.fits',
                              }

        try :
            self.pointing_code = self.params['code']
            self.datadir = self.params['datadir']
            self.default_names = self.params['default_names']

            self.set_parameters()
            self.set_wavelength()

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

    def set_wavelength(self,verbose=True):
        #
        hdul = fits.open(datacube)
        hdul.info()
        new = hdul['DATA']
        # Setting the wavelength vector
        # new.header
        self.cube_cdelt = new.header['CD3_3']
        self.cube_crval = new.header['CRVAL3']
        self.cube_crpix = new.header['CRPIX3']
        self.cube_cunit = new.header['CUNIT3']
        self.cube_Nwl = np.arange(new.header['NAXIS3'])

        self.wl = (self.cube_Nwl - self.cube_crpix + 1) * self.cube_cdelt + self.cube_crval
        if verbose:
            print("Cube wavelength axis definition: CDELT:{0}, CRVAL={1}, CRPIX={2}\n".format(self.cube_cdelt, self.cube_crval, self.cube_crpix))
