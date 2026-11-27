Astrometric registration
========================

:py:meth:`~pySpecMUSE.CubeMUSE.CubeMUSE.correct_astrometry` estimates
a translation for one pointing. It makes an initial match, matches again
after applying the preliminary shift, and estimates the final shift from
the selected pairs. This updates the in-memory sky coordinates and WCS.
Aperture extraction continues to use image pixel positions.

Match directly to Gaia DR3
--------------------------

With ``field`` initialized as in :doc:`quickstart`, run the following
with network access.

.. code-block:: python

   result = field.correct_astrometry(reference="gaia", gaia_release="DR3")
   print(result["n_matched"], result["dra_arcsec"], result["ddec_arcsec"])

To repeat the correction using the same Gaia results, pass
``gaia_catalog=result["ref_table"]`` in a subsequent call. Gaia
positions are used at their reference epoch (J2016.0 for DR3) without
proper-motion propagation to the observation epoch.

Match against Gaia-registered VISTA
-----------------------------------

Use :py:func:`~pySpecMUSE.vista_gaia_astrocorr.run_pipeline` to
register a VISTA catalogue, or supply an existing registered catalogue
with documented coordinate columns.

.. code-block:: python

   result = field.correct_astrometry(
       reference="vista",
       vista_table="vista_corrected.dat",
       ra_col="RA_CORR",
       dec_col="DE_CORR",
   )

Inspect the diagnostic figure and match count before using the fitted
offsets. Offsets are measured in arcseconds, in the sense reference
minus MUSE. The RA offset is the eastward tangent-plane offset
(``delta_RA * cos(Dec)``), and the Dec offset is positive northward.
The ``field.wcs`` and ``star.skycoo`` properties select the corrected
values when available. The original values remain available as
``field.wcs_original`` and ``star.skycoo_original``.
