# -*- coding: utf-8 -*-

"""
Compatibility module for useful functions added in Astropy but not available in
older versions.

See https://github.com/astropy/astropy/blob/master/licenses/LICENSE.rst for the
license information.

"""

from __future__ import absolute_import, division

import numpy as np
import six
import textwrap
import warnings
from astropy.io import fits
from astropy.units.format.fits import UnitScaleError
from astropy.utils import minversion
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['zscale', 'table_to_hdu', 'write_hdulist_to', 'write_fits_to',
           'ASTROPY_LT_1_1', 'ASTROPY_LT_1_2', 'ASTROPY_LT_1_3']

ASTROPY_LT_1_1 = not minversion('astropy', '1.1')
ASTROPY_LT_1_2 = not minversion('astropy', '1.2')
ASTROPY_LT_1_3 = not minversion('astropy', '1.3')


if ASTROPY_LT_1_3:
    # the 'clobber' parameter was renamed to 'overwrite' in 1.3
    def write_hdulist_to(hdulist, fileobj, overwrite=False, **kwargs):
        hdulist.writeto(fileobj, clobber=overwrite, **kwargs)

    def write_fits_to(filename, data, overwrite=False, **kwargs):
        fits.writeto(filename, data, clobber=overwrite, **kwargs)
else:
    def write_hdulist_to(hdulist, fileobj, overwrite=False, **kwargs):
        hdulist.writeto(fileobj, overwrite=overwrite, **kwargs)

    def write_fits_to(filename, data, overwrite=False, **kwargs):
        fits.writeto(filename, data, overwrite=overwrite, **kwargs)

write_hdulist_to.__doc__ = """
Wrapper function for `astropy.io.fits.HDUList.writeto`.

The aim of this function is to provide a compatible way to overwrite a file,
with ``clobber`` for Astropy < 1.3 and ``overwrite`` for Astropy >= 1.3.

Original docstring follows:
""" + textwrap.dedent(fits.HDUList.writeto.__doc__)


write_fits_to.__doc__ = """
Wrapper function for `astropy.io.fits.writeto`.

The aim of this function is to provide a compatible way to overwrite a file,
with ``clobber`` for Astropy < 1.3 and ``overwrite`` for Astropy >= 1.3.

Original docstring follows:
""" + textwrap.dedent(fits.writeto.__doc__)


if not ASTROPY_LT_1_2:
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval

    def table_to_hdu(table):
        return fits.table_to_hdu(table)

    table_to_hdu.__doc__ = fits.table_to_hdu.__doc__

    def zscale(image, nsamples=1000, contrast=0.25, max_reject=0.5,
               min_npixels=5, krej=2.5, max_iterations=5):
        interval = ZScaleInterval(nsamples=nsamples, contrast=contrast,
                                  max_reject=max_reject,
                                  min_npixels=min_npixels, krej=krej,
                                  max_iterations=max_iterations)
        return interval.get_limits(image)

else:
    def table_to_hdu(table):
        """
        Convert an astropy.table.Table object to a FITS BinTableHDU

        Parameters
        ----------
        table : astropy.table.Table
            The table to convert.

        Returns
        -------
        table_hdu : astropy.io.fits.BinTableHDU
            The FITS binary table HDU
        """
        # Avoid circular imports
        from astropy.io.fits import FITS_rec, BinTableHDU
        from astropy.io.fits.connect import is_column_keyword, REMOVE_KEYWORDS

        # Tables with mixin columns are not supported
        if table.has_mixin_columns:
            mixin_names = [name for name, col in table.columns.items()
                           if not isinstance(col, table.ColumnClass)]
            raise ValueError('cannot write table with mixin column(s) {0}'
                             .format(mixin_names))

        # Create a new HDU object
        if table.masked:
            # float column's default mask value needs to be Nan
            for column in six.itervalues(table.columns):
                fill_value = column.get_fill_value()
                if column.dtype.kind == 'f' and np.allclose(fill_value, 1e20):
                    column.set_fill_value(np.nan)

            fits_rec = FITS_rec.from_columns(np.array(table.filled()))
            table_hdu = BinTableHDU(fits_rec)
            for col in table_hdu.columns:
                # Binary FITS tables support TNULL *only* for integer data columns
                # TODO: Determine a schema for handling non-integer masked columns
                # in FITS (if at all possible)
                int_formats = ('B', 'I', 'J', 'K')
                if not (col.format in int_formats or
                        col.format.p_format in int_formats):
                    continue

                # The astype is necessary because if the string column is less
                # than one character, the fill value will be N/A by default which
                # is too long, and so no values will get masked.
                fill_value = table[col.name].get_fill_value()

                col.null = fill_value.astype(table[col.name].dtype)
        else:
            fits_rec = FITS_rec.from_columns(np.array(table.filled()))
            table_hdu = BinTableHDU(fits_rec)

        # Set units for output HDU
        for col in table_hdu.columns:
            unit = table[col.name].unit
            if unit is not None:
                try:
                    col.unit = unit.to_string(format='fits')
                except UnitScaleError:
                    scale = unit.scale
                    raise UnitScaleError(
                        "The column '{0}' could not be stored in FITS format "
                        "because it has a scale '({1})' that "
                        "is not recognized by the FITS standard. Either scale "
                        "the data or change the units.".format(col.name, str(scale)))
                except ValueError:
                    warnings.warn(
                        "The unit '{0}' could not be saved to FITS format".format(
                            unit.to_string()), AstropyUserWarning)

        for key, value in table.meta.items():
            if is_column_keyword(key.upper()) or key.upper() in REMOVE_KEYWORDS:
                warnings.warn(
                    "Meta-data keyword {0} will be ignored since it conflicts "
                    "with a FITS reserved keyword".format(key), AstropyUserWarning)

            if isinstance(value, list):
                for item in value:
                    try:
                        table_hdu.header.append((key, item))
                    except ValueError:
                        warnings.warn(
                            "Attribute `{0}` of type {1} cannot be added to "
                            "FITS Header - skipping".format(key, type(value)),
                            AstropyUserWarning)
            else:
                try:
                    table_hdu.header[key] = value
                except ValueError:
                    warnings.warn(
                        "Attribute `{0}` of type {1} cannot be added to FITS "
                        "Header - skipping".format(key, type(value)),
                        AstropyUserWarning)
        return table_hdu

    def zscale(image, nsamples=1000, contrast=0.25, max_reject=0.5,
               min_npixels=5, krej=2.5, max_iterations=5):
        """Implement IRAF zscale algorithm.

        Parameters
        ----------
        image : array_like
            Input array.
        nsamples : int, optional
            Number of points in array to sample for determining scaling factors.
            Default to 1000.
        contrast : float, optional
            Scaling factor (between 0 and 1) for determining min and max. Larger
            values increase the difference between min and max values used for
            display. Default to 0.25.
        max_reject : float, optional
            If more than ``max_reject * npixels`` pixels are rejected, then the
            returned values are the min and max of the data. Default to 0.5.
        min_npixels : int, optional
            If less than ``min_npixels`` pixels are rejected, then the
            returned values are the min and max of the data. Default to 5.
        krej : float, optional
            Number of sigma used for the rejection. Default to 2.5.
        max_iterations : int, optional
            Maximum number of iterations for the rejection. Default to 5.

        Returns
        -------
        zmin, zmax: float
            Computed min and max values.

        """

        # Sample the image
        image = np.asarray(image)
        image = image[np.isfinite(image)]
        stride = int(max(1.0, image.size / nsamples))
        samples = image[::stride][:nsamples]
        samples.sort()

        npix = len(samples)
        zmin = samples[0]
        zmax = samples[-1]

        # Fit a line to the sorted array of samples
        minpix = max(min_npixels, int(npix * max_reject))
        x = np.arange(npix)
        ngoodpix = npix
        last_ngoodpix = npix + 1

        # Bad pixels mask used in k-sigma clipping
        badpix = np.zeros(npix, dtype=bool)

        # Kernel used to dilate the bad pixels mask
        ngrow = max(1, int(npix * 0.01))
        kernel = np.ones(ngrow, dtype=bool)

        for niter in range(max_iterations):
            if ngoodpix >= last_ngoodpix or ngoodpix < minpix:
                break

            fit = np.polyfit(x, samples, deg=1, w=(~badpix).astype(int))
            fitted = np.poly1d(fit)(x)

            # Subtract fitted line from the data array
            flat = samples - fitted

            # Compute the k-sigma rejection threshold
            threshold = krej * flat[~badpix].std()

            # Detect and reject pixels further than k*sigma from the fitted line
            badpix[(flat < - threshold) | (flat > threshold)] = True

            # Convolve with a kernel of length ngrow
            badpix = np.convolve(badpix, kernel, mode='same')

            last_ngoodpix = ngoodpix
            ngoodpix = np.sum(~badpix)

        slope, intercept = fit

        if ngoodpix >= minpix:
            if contrast > 0:
                slope = slope / contrast
            center_pixel = (npix - 1) // 2
            median = np.median(samples)
            zmin = max(zmin, median - (center_pixel - 1) * slope)
            zmax = min(zmax, median + (npix - center_pixel) * slope)
        return zmin, zmax
