Quickstart
==========

From the repository root, install the package into your active Python
environment.

.. code-block:: console

   python -m pip install .

For an editable installation with Jupyter, use
``python -m pip install -e ".[notebooks]"``. The commands on this page use
your own reduced data. See :doc:`inputs` for the expected FITS structure.

Extract spectra from one pointing
---------------------------------

Set ``data_dir`` and the three filenames to match your local files.
Construction reads the data, builds the cube WCS, and detects sources in
the I-band image unless you supply a source catalogue.

.. code-block:: python

   import os
   from pathlib import Path

   from pySpecMUSE import CubeMUSE

   data_dir = Path("/path/to/pointing")
   field = CubeMUSE({
       "code": "7",
       "datadir": str(data_dir) + os.sep,
       "default_names": False,
       "file": "WFM_Tr14_long_7_DATACUBE_FINAL.fits",
       "file_i_image": "WFM_Tr14_long_7_Cousins_I_IMAGE_FOV.fits",
       "file_v_image": "WFM_Tr14_long_7_Johnson_V_IMAGE_FOV.fits",
       "nproc": 1,
   })

   print(f"Detected {field.nstars} sources")
   field.extract_spectra([3], (10, 15), add_apc=False)
   star = field.stars[0]

   import matplotlib.pyplot as plt

   plt.plot(star.wl, star.spectrum)
   plt.xlabel(f"Wavelength ({field.cube_cunit3})")
   plt.ylabel("Extracted flux density")
   plt.show()

This uses a 3-pixel aperture and estimates the local sky in an annulus
from 10 to 15 pixels. Check that the source list contains the target and
that the aperture and annulus suit the data. The per-star outputs include
``spectrum``, ``sky``, ``sky_noise``, and ``rms_spectrum``.

Apply an aperture correction
----------------------------

Select isolated stars, inspect the I- and V-band curves of growth, and
estimate the wavelength-dependent correction before requesting it during
extraction. The calls below write diagnostic PDFs in the current working
directory.

.. code-block:: python

   field.set_stars_for_apc()
   field.curve_of_growth_iv()
   field.set_apc_values(radii=(3, 10), skyrad=(10, 15))
   field.extract_spectra([3], (10, 15), add_apc=True)
   corrected_flux = field.stars[0].corrected_spectrum

Use the same small aperture radius for the correction and extraction.
See :doc:`astrometry` if you also need registered sky coordinates.
