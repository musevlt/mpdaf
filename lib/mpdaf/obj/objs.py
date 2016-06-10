"""obj.py contains generic methods used in obj package."""

from __future__ import absolute_import, division

import numbers
import numpy as np

from astropy.constants import c
from astropy.units import Quantity

__all__ = ('is_float', 'is_int', 'is_number', 'flux2mag', 'mag2flux',
           'UnitArray', 'UnitMaskedArray', 'circular_bounding_box',
           'elliptical_bounding_box')


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

def elliptical_bounding_box(center, radii, posangle, shape):
    """Return Y-axis and X-axis slice objects that select a rectangular
       image region that just encloses an ellipse of a specified center
       position and specified Y-axis and X-axis radii.

       If the ellipse is partly outside of the image array, the
       returned slices are clipped at the edges of the array.

    Parameters
    ----------
    center : float, float
       The floating point array indexes of the centre of the circle,
       in the order, y,x.
    radii : float,float
       The radii of the orthogonal axes of the ellipse.  When posangle
       is zero, radius[0] is the radius along the X axis of the
       image-array, and radius[1] is the radius along the Y axis of
       the image-array.
    posangle : float
       The counterclockwise position angle of the ellipse.  When
       posangle is 0 degrees (the default), the X and Y axes of the
       ellipse lie along the X and Y axes of the image, and the radius
       values in the radii argument lie along the X and Y axes,
       respectively.
    shape  : int, int
       The dimensions of the image array.

    Returns
    -------
    out : slice, slice
       The Y-axis and X-axis slices needed to select a rectangular region
       of the image that just encloses the ellipse.

    """

    # If only one radius is specified, treat this as a cirle.
    if is_number(radii):
        return circular_bounding_box(center, radii, shape)
    else:
        rx, ry = radii

    # Ensure that the Y and X coordinates of the central position
    # can be used in numpy array equations.
    center = np.asarray(center)

    # Convert the position angle to radians and precompute the sine and
    # cosine of this.
    pa = np.radians(posangle)
    sin_pa = np.sin(pa)
    cos_pa = np.cos(pa)

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
    t_ymax = np.arctan2( ry * cos_pa, rx * sin_pa)

    # Compute the half-width and half-height of the rectangle that
    # encloses the ellipse, by computing the X and Y values,
    # respectively, of the ellipse at the above angles.
    xmax = np.abs(rx * np.cos(t_xmax) * cos_pa - ry * np.sin(t_xmax) * sin_pa)
    ymax = np.abs(rx * np.cos(t_ymax) * sin_pa + ry * np.sin(t_ymax) * cos_pa)

    # Place these values in an array.
    r = np.array([ymax, xmax])

    # Determine the index ranges along the X and Y axes of the image
    # array that enclose the extent of the ellipse centered at center.
    shape = np.asarray(shape) - 1
    imin, jmin = np.clip((center - r + 0.5).astype(int), (0, 0), shape)
    imax, jmax = np.clip((center + r + 0.5).astype(int), (0, 0), shape)

    # Turn these ranges into slice objects.
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
