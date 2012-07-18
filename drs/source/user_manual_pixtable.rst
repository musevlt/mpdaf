PixTable class
**************

This class manages input/output for MUSE pixel table files.


Attributes
==========

+----------------+----------------+--------------------------------+
|filename        | string         | Possible FITS filename.        |
+----------------+----------------+--------------------------------+
| primary_header | pyfits.CardList| Possible FITS header instance. |
+----------------+----------------+--------------------------------+
| nrows          | integer        |  Number of rows.               |
+----------------+----------------+--------------------------------+
| ncols          | integer        |  Number of columns.            |
+----------------+----------------+--------------------------------+


Tutorial
========




Reference
=========


:func:`mpdaf.drs.PixTable.copy` copies PixTable object in a new one and returns it.

:func:`mpdaf.drs.PixTable.info` prints information.

:func:`mpdaf.drs.PixTable.get_xpos` gets the xpos column.

:func:`mpdaf.drs.PixTable.get_ypos` gets the ypos column.

:func:`mpdaf.drs.PixTable.get_lambda` gets the lambda column.

:func:`mpdaf.drs.PixTable.get_data` gets the data column.

:func:`mpdaf.drs.PixTable.get_stat` gets the stat column.

:func:`mpdaf.drs.PixTable.get_dq` gets the dq column.

:func:`mpdaf.drs.PixTable.get_origin` gets the origin column.

:func:`mpdaf.drs.PixTable.write` saves the pixtable in a FITS file.

:func:`mpdaf.drs.PixTable.extract` extracts a subset of a pixtable.

:func:`mpdaf.drs.PixTable.origin2ifu` converts the origin value and returns the ifu number.

:func:`mpdaf.drs.PixTable.origin2slice` converts the origin value and returns the slice number.

:func:`mpdaf.drs.PixTable.origin2ypix` converts the origin value and returns the y coordinates.

:func:`mpdaf.drs.PixTable.origin2xoffset` converts the origin value and returns the x coordinates offset.

:func:`mpdaf.drs.PixTable.origin2xpix` converts the origin value and returns the x coordinates.

:func:`mpdaf.drs.PixTable.origin2coords` converts the origin value and returns (ifu, slice, ypix, xpix).

:func:`mpdaf.drs.PixTable.get_slices` returns slices dictionary.

:func:`mpdaf.drs.PixTable.get_keywords` returns the keyword value corresponding to a key.

:func:`mpdaf.drs.PixTable.reconstruct_det_image` reconstructs the image on the detector from the pixtable.
