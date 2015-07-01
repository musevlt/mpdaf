"""obj.py contains generic methods used in obj package."""

import numpy as np


def is_float(x):
    """Test if `x` is a float number."""
    return isinstance(x, (float, np.float32, np.float64))


def is_int(x):
    """Test if `x` is an int number."""
    return isinstance(x, (int, np.int32, np.int64))


def is_number(x):
    """Test if `x` is a number."""
    return is_int(x) or is_float(x)


def flux2mag(flux, wave):
    """ convert flux from erg.s-1.cm-2.A-1 to AB mag
    wave is the wavelength in A
    """
    if flux > 0:
        c = 2.998e18  # speed of light in A/s
        mag = -48.60 - 2.5 * np.log10(wave ** 2 * flux / c)
        return mag
    else:
        return 99


def mag2flux(mag, wave):
    """ convert flux from AB mag to erg.s-1.cm-2.A-1
    wave is the wavelength in A
    """
    c = 2.998e18  # speed of light in A/s
    flux = 10 ** (-0.4 * (mag + 48.60)) * c / wave ** 2
    return flux
