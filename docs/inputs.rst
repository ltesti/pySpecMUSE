Input files
===========

Each pointing uses a reduced MUSE datacube and corresponding Cousins I
and Johnson V images. Supply co-registered images on the same spatial
pixel grid as the cube. All three FITS files must contain a named
``DATA`` extension. The cube is indexed as wavelength, y, x.

The cube reader uses the spatial ``CD1_1``, ``CD1_2``, ``CD2_1``, and
``CD2_2`` keywords and ``CRVALn``, ``CRPIXn``, and ``CUNITn`` for the
spatial and wavelength axes. It constructs a linear wavelength array
using ``CD3_3``. Wavelength values retain the header's unit. Per-star
flux, sky and sky-noise arrays are scaled by a fixed factor of
``1e-20``; check the input ``BUNIT`` against this convention before
interpreting the output.

Configuration
-------------

``code``
   Pointing identifier. Use a string such as ``"7"``.
``datadir``
   Input directory as a string ending with a path separator. The current
   implementation concatenates it with each filename.
``default_names``
   With ``False``, supply ``file``, ``file_i_image``, and
   ``file_v_image`` explicitly. With ``True``, the cube name is
   ``DATA_Long<code>.fits``, and the two image names follow the
   Trumpler 14 convention in :doc:`quickstart`.
``nproc``
   Optional number of processes for supported computations. Use 1 for
   serial execution.
``file_pos``
   Optional CSV filename relative to ``datadir``. Required columns are
   ``X_IMAGE``, ``Y_IMAGE``, ``MAG_APER``, ``ALPHA_J2000``,
   ``DELTA_J2000``, and ``NUMBER``. The reader uses pixel positions as
   supplied, so provide zero-based image coordinates, and give sky
   coordinates in degrees. When omitted, DAOStarFinder detects sources
   in the I image.
