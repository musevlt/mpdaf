# -*- coding: utf-8 -*-

"""
Compatibility module for useful functions added in Astropy but not available in
older versions.

See https://github.com/astropy/astropy/blob/master/licenses/LICENSE.rst for the
license information.

"""

import textwrap
from astropy.io import fits
from astropy.utils import minversion

__all__ = ['write_hdulist_to', 'write_fits_to', 'ASTROPY_LT_1_3']

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
