"""Copyright 2010-2016 CNRS/CRAL

This file is part of MPDAF.

MPDAF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version

MPDAF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MPDAF.  If not, see <http://www.gnu.org/licenses/>.


obj.py contains generic methods used in obj package.
"""

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
