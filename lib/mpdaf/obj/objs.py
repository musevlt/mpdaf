""" obj.py contains generic methods used in obj package"""
import numpy as np


def is_float(x):
    if type(x) is float or type(x) is np.float32 or type(x) is np.float64:
        return True
    else:
        return False


def is_int(x):
    if type(x) is int or type(x) is np.int32 or type(x) is np.int64:
        return True
    else:
        return False


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
