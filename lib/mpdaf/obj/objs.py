"""obj.py contains generic methods used in obj package."""

from __future__ import absolute_import, division

import numbers
import numpy as np

from astropy.constants import c
from astropy.units import Quantity

__all__ = ('is_float', 'is_int', 'is_number', 'flux2mag', 'mag2flux',
           'UnitArray', 'UnitMaskedArray', 'circular_bounding_box')


def is_float(x):
    """Test if `x` is a float number."""
    return isinstance(x, numbers.Real)


def is_int(x):
    """Test if `x` is an int number."""
    return isinstance(x, numbers.Integral)


def is_number(x):
    """Test if `x` is a number."""
    return isinstance(x, numbers.Number)


def flux2mag(flux, wave):
    """Convert flux from erg.s-1.cm-2.A-1 to AB mag.

    wave is the wavelength in A

    """
    if flux > 0:
        cs = c.to('Angstrom/s').value  # speed of light in A/s
        return -48.60 - 2.5 * np.log10(wave ** 2 * flux / cs)
    else:
        return 99


def mag2flux(mag, wave):
    """Convert flux from AB mag to erg.s-1.cm-2.A-1

    wave is the wavelength in A

    """
    cs = c.to('Angstrom/s').value  # speed of light in A/s
    return 10 ** (-0.4 * (mag + 48.60)) * cs / wave ** 2


def circular_bounding_box(center, radius, shape):
    """Return Y-axis and X-axis slice objects that select a square
       image region that just encloses a circle of a specified center
       and radius.

       If the circle is partly outside of the image array, the
       returned slices are clipped at the edges of the array.

    Parameters
    ----------
    center : float, float
       The floating point array indexes of the centre of the circle,
       in the order, y,x.
    radius : float
       The radius of the circle (number of pixels).
    shape  : int, int
       The dimensions of the image array.

    Returns
    -------
    out : slice, slice
       The Y-axis and X-axis slices needed to select a square region
       of the image that just encloses the circle.

    """

    center = np.asarray(center)
    radius = np.asarray(radius)
    shape = np.asarray(shape) - 1
    imin, jmin = np.clip((center - radius + 0.5).astype(int), (0, 0), shape)
    imax, jmax = np.clip((center + radius + 0.5).astype(int), (0, 0), shape)
    return slice(imin, imax + 1), slice(jmin, jmax + 1)


def UnitArray(array, old_unit, new_unit):
    if new_unit == old_unit:
        return array
    else:
        return Quantity(array, old_unit, copy=False).to(new_unit).value


def UnitMaskedArray(mask_array, old_unit, new_unit):
    if new_unit == old_unit:
        return mask_array
    else:
        return np.ma.array(Quantity(mask_array.data, old_unit, copy=False)
                           .to(new_unit).value, mask=mask_array.mask)
