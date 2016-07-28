# -*- coding: utf-8 -*-

import numpy as np
import six
import warnings
from astropy.units.format.fits import UnitScaleError
from astropy.utils import minversion
from astropy.utils.exceptions import AstropyUserWarning

ASTROPY_LT_1_1 = not minversion('astropy', '1.1')
ASTROPY_LT_1_2 = not minversion('astropy', '1.2')

if not ASTROPY_LT_1_2:
    from astropy.io.fits import table_to_hdu
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
            #float column's default mask value needs to be Nan
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
