API reference
=============

CubeMUSE
--------

.. autoclass:: pySpecMUSE.CubeMUSE.CubeMUSE
   :members: wcs, correct_astrometry, set_stars_for_apc, curve_of_growth_iv,
             set_apc_values, extract_spectra, compute_rms_spectra

StarMUSE
--------

.. autoclass:: pySpecMUSE.StarMUSE.StarMUSE
   :members: skycoo, plot_star_summary

VISTA and Gaia registration
---------------------------

.. autofunction:: pySpecMUSE.vista_gaia_astrocorr.run_pipeline

.. autofunction:: pySpecMUSE.vista_gaia_astrocorr.query_gaia
