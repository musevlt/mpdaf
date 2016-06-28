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
           'UnitArray', 'UnitMaskedArray', 'bounding_box')


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


def bounding_box(form, center, radii, posangle, shape, step):
    """Return Y-axis and X-axis slice objects that bound a rectangular
    image region that just encloses either an ellipse or a rectangle,
    where the rectangle has a specified center position, Y-axis and X-axis
    radii, and a given rotation angle relative to the image axes. The
    effective center of the region is also returned.

    If the ellipse or rectangle is partly outside of the image array,
    the returned slices are clipped at the edges of the array. The
    effective center that is returned, is the center before clipping.

    Parameters
    ----------
    form   : str
       The type of region whose rectangle image bounds are needed,
       chosen from:
       rectangle, a rotated rectangle or square.
       ellipse, a rotated ellipse or circle.
    center : float, float
       The floating point array indexes of the centre of the circle,
       in the order, y,x.
    radii : float or float,float
       The half-width and half-height of the ellipse or
       rectangle. When the posangle is zero, the width and height are
       parallel to the X and Y axes of the image array, respectively.
       More generally, the width and height are along directions that
       are posangle degrees counterclockwise of the X axis and Y axis,
       respectively. If only one number is specified, then the width
       and the height are both given that value.
    posangle : float
       The counterclockwise rotation angle of the chosen shape, in
       degrees. When posangle is 0 degrees, the width and height of
       the shape are along the X and Y axes of the image. Non-zero
       values of posangle rotate the chosen shape anti-clockwise
       of that position.
    shape  : int, int
       The dimensions of the image array.
    step   : float, float
       The per-pixel world-coordinate increments along the Y and X axes
       of the image array, or [1.0,1.0] if radii is in pixels.

    Returns
    -------
    out : clipped, unclipped, center

       The 'clipped' return value is a list of the Y-axis and
       X-axis slices to use to select all pixels within the
       rectangular region of the image that just encloses the part of
       the ellipse or rectangle that is within the image area.

       The 'unclipped' return-value is the version of 'clipped' before
       it was clipped at the edges of the image.

       The 'center' return value is the pixel index of the center of
       the region, prior to clipping.

       If the region is entirely outside the range of an axis, a
       zero-pixel slice is returned for that axis in 'clipped', with a
       start value that is 0 if the pixels were all below pixel 0, or
       shape-1 if they were all off the upper end of the range.

    """

    # If only one radius is specified, use it as both the half-width and
    # the half-height.
    if np.isscalar(radii):
        rx, ry = radii, radii
    else:
        rx, ry = radii

    # Ensure that the Y and X coordinates of the central position
    # can be used in numpy array equations.
    center = np.asarray(center)

    # Ensure that the pixel sizes are in a numpy array as well.
    step = np.asarray(step)

    # Convert the position angle to radians and precompute the sine and
    # cosine of this.
    pa = np.radians(posangle)
    sin_pa = np.sin(pa)
    cos_pa = np.cos(pa)

    # Get the bounding box of a rotated rectangle?
    if form == "rectangle":
        # Calculate the maximum of the X-axis and Y-axis distances of the
        # corners of the rotated rectangle from the rectangle's center.
        xmax = abs(rx * cos_pa) + abs(ry * sin_pa)
        ymax = abs(rx * sin_pa) + abs(ry * cos_pa)

    # Get the bounding box of a rotated ellipse?
    elif form == "ellipse":
        # We use the following parametric equations for an unrotated
        # ellipse with x and y along the image-array X and Y axes, where t
        # is a parametric angle with no physical equivalent:
        #   x(pa=0) = rx * cos(t)
        #   y(pa=0) = ry * sin(t)
        # We then rotate this anti-clockwise by pa:
        #   x(pa)  =  |cos(pa), -sin(pa)| |rx * cos(t)|
        #   y(pa)     |sin(pa),  cos(pa)| |ry * cos(t)|
        # By differentiating the resulting equations of x(pa) and y(pa) by
        # dt and setting the derivatives to zero, we obtain the following
        # values for the angle t at which x(pa) and y(pa) are maximized.
        t_xmax = np.arctan2(-ry * sin_pa, rx * cos_pa)
        t_ymax = np.arctan2(ry * cos_pa, rx * sin_pa)

        # Compute the half-width and half-height of the rectangle that
        # encloses the ellipse, by computing the X and Y values,
        # respectively, of the ellipse at the above angles.
        xmax = np.abs(rx * np.cos(t_xmax) * cos_pa -
                      ry * np.sin(t_xmax) * sin_pa)
        ymax = np.abs(rx * np.cos(t_ymax) * sin_pa +
                      ry * np.sin(t_ymax) * cos_pa)
    else:
        raise ValueError("The form argument should be 'rectangle' or 'ellipse'")

    # Put the height and width in an array, divide them by
    # the pixel sizes along the Y and X axes to convert them to pixel
    # counts, then convert them to the nearest integers.
    w = np.floor(np.abs(np.array([2 * ymax, 2 * xmax]) / step) + 0.5).astype(int)

    # Are the members of w even numbers of pixels?
    iseven = np.mod(w, 2) == 0

    # For each axis calculate the pixel index of the central pixel of
    # the region where w is odd, or the index of the first of the two
    # central pixels when w is even.
    c = np.where(iseven, np.floor(center), np.floor(center + 0.5)).astype(int)

    # Determine the indexes of the first and last pixels of the region
    # along each axis, using integer arithmetic to avoid surprises.
    first = np.where(iseven, c - w // 2 + 1, c - (w - 1) // 2)
    last = np.where(iseven, c + w // 2, c + (w - 1) // 2)

    # Calculate the effective center of the bounded region.
    center = (first + last) / 2.0

    # Compute the ideal slices that would select the bounding box.
    ideal_slices = [slice(first[0], last[0] + 1), slice(first[1], last[1] + 1)]

    # Are the selected pixels of the axes outside the bounds of the array?
    max_indexes = np.asarray(shape) - 1
    outside = np.logical_or(last < 0, first > max_indexes)

    # Clip the first and last pixels to ensure that they lie within
    # the bounds of the image.
    imin, jmin = np.clip(first, (0, 0), max_indexes)
    imax, jmax = np.clip(last, (0, 0), max_indexes)

    # Compute the corresponding slices, replacing the clipped
    # values by a zero-pixel range where the pre-clipped indexes
    # were entirely outside the array.
    clipped_slices = [slice(imin, (imax + 1) if not outside[0] else imax),
                      slice(jmin, (jmax + 1) if not outside[1] else jmax)]

    # Return the ranges as slice objects, along with the effective
    # center of the region.
    return clipped_slices, ideal_slices, center


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
