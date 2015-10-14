# -*- coding: utf-8 -*-

import astropy.units as u
import numpy as np
from mpdaf.obj import Image, Cube, WCS, WaveCoord, Spectrum
from numpy.testing import assert_array_equal


def assert_image_equal(ima, shape=None, start=None, end=None, step=None):
    if shape is not None:
        assert_array_equal(ima.shape, shape)
    if start is not None:
        assert_array_equal(ima.get_start(), start)
    if end is not None:
        assert_array_equal(ima.get_end(), end)
    if step is not None:
        assert_array_equal(ima.get_step(), step)


def generate_image(scale=2, shape=(6, 5), unit=u.ct, wcs=None):
    wcs = wcs or WCS(crval=(0, 0), crpix=1.0, shape=shape)
    return Image(data=scale * np.ones(shape), wcs=wcs, unit=unit, copy=False)


def generate_spectrum(scale=2, unit=u.angstrom, shape=10, crpix=2.0, cdelt=3.0,
                      crval=0.5, wave=None):
    wave = wave or WaveCoord(crpix=crpix, cdelt=cdelt, crval=crval,
                             shape=shape, cunit=unit)
    data = np.arange(shape)
    data[0] = 0.5
    return Spectrum(wave=wave, copy=False, data=data)


def generate_cube(scale=2.3, uwave=u.angstrom, shape=(10, 6, 5), unit=u.ct,
                  wcs=None, wave=None):
    wcs = wcs or WCS(crval=(0, 0), crpix=1.0, shape=shape[1:])
    wave = wave or WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=shape[0],
                             cunit=uwave)
    return Cube(data=scale * np.ones(shape), wave=wave, wcs=wcs, unit=unit,
                copy=False)
