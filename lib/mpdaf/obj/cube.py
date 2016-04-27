"""cube.py manages Cube objects."""

from __future__ import absolute_import, division

import astropy.units as u
import multiprocessing
import numpy as np
import os.path
import sys
import time
import types

from astropy.io import fits as pyfits
from numpy import ma
from six.moves import range, zip
from scipy import integrate, interpolate

from .coords import WCS, WaveCoord
from .data import DataArray
from .image import Image
from .objs import is_int, UnitArray, UnitMaskedArray
from .spectrum import Spectrum
from ..tools import deprecated
from ..tools.fits import add_mpdaf_method_keywords

__all__ = ('iter_spe', 'iter_ima', 'Cube')


def iter_spe(cube, index=False):
    """An iterator for iterating over the spectra in a Cube object.

    Parameters
    ----------
    cube : `~mpdaf.obj.Cube`
       The cube that contains the spectra to be returned one after another.
    index : boolean
       If False, only return a spectrum at each iteration.
       If True, return both a spectrum and the position of that spectrum in the
       image (a tuple of image-array indexes along the axes (y,x)).

    """
    if index:
        for y, x in np.ndindex(*cube.shape[1:]):
            yield cube[:, y, x], (y, x)
    else:
        for y, x in np.ndindex(*cube.shape[1:]):
            yield cube[:, y, x]


def iter_ima(cube, index=False):
    """An iterator for iterating over the images of a Cube object.

    Parameters
    ----------
    cube : `~mpdaf.obj.Cube`
       The cube that contains the images to be returned one after another.
    index : boolean
       If False, only return an image at each iteration.
       If True, return both the image and the spectral index.

    """
    if index:
        for l in range(cube.shape[0]):
            yield cube[l, :, :], l
    else:
        for l in range(cube.shape[0]):
            yield cube[l, :, :]


def _print_multiprocessing_progress(processresult, num_tasks):
    while True:
        time.sleep(5)
        completed = processresult._index
        if completed == num_tasks:
            output = ""
            sys.stdout.write("\r\x1b[K" + output.__str__())
            sys.stdout.flush()
            break
        output = "\r Waiting for %i tasks to complete (%i%% done) ..." % (
            num_tasks - completed, completed / num_tasks * 100.0)
        sys.stdout.write("\r\x1b[K" + output.__str__())
        sys.stdout.flush()


class Cube(DataArray):

    """This class manages Cube objects.

    Parameters
    ----------
    filename : string
        Optional FITS file name. None by default.
    ext : integer or (integer,integer) or string or (string,string)
        The optional number/name of the data extension
        or the numbers/names of the data and variance extensions.
    wcs : `mpdaf.obj.WCS`
        The world coordinates of the image pixels.
    wave : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of the spectral pixels.
    unit : `astropy.units.Unit`
        The physical units of the data values. Defaults to
        u.dimensionless_unscaled.
    data : numpy.ndarray or list
        An optional array containing the values of each pixel in the
        cube (None by default). Where given, this array should be
        3 dimensional, and the python ordering of its axes should be
        (wavelength,image_y,image_x).
    var : float array
        An optional array containing the variances of each pixel in the
        cube (None by default). Where given, this array should be
        3 dimensional, and the python ordering of its axes should be
        (wavelength,image_y,image_x).
    copy : boolean
        If true (default), then the data and variance arrays are copied.
    dtype : numpy.dtype
        The type of the data (integer, float)

    Attributes
    ----------
    filename : string
        The name of the originating FITS file, if any. Otherwise None.
    primary_header : pyfits.Header
        The FITS primary header instance, if a FITS file was
        provided. Otherwise None.
    wcs : `mpdaf.obj.WCS`
        The world coordinates of the image pixels.
    wave : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of the spectral pixels.
    shape : tuple
        The dimensions of the data axes (python axis ordering (nz,ny,nx)).
    data : numpy.ma.MaskedArray
        A masked array containing the pixel values of the cube.
    data_header : pyfits.Header
        The FITS header of the DATA extension.
    unit : `astropy.units.Unit`
        The physical units of the data values.
    dtype : numpy.dtype
        The type of the data (integer, float)
    var : float array
        An optional array containing the variance, or None.

    """

    # Tell the DataArray base class that cubes require 3 dimensional
    # data arrays, image world coordinates and wavelength coordinates.

    _ndim_required = 3
    _has_wcs = True
    _has_wave = True

    def mask_region(self, center, radius, lmin=None, lmax=None, inside=True,
                    unit_center=u.deg, unit_radius=u.arcsec,
                    unit_wave=u.angstrom):
        """Mask values inside or outside a specified region.

        Parameters
        ----------
        center : (float,float)
            The center of the region.
        radius : float or (float,float)
            The radius of the region.
            If radius is a scalar, it denotes the radius of a circular region.
            If radius is a tuple, it denotes the width of a square region.
        lmin : float
            The minimum wavelength of the region.
        lmax : float
            The maximum wavelength of the region.
        inside : boolean
            If inside is True, pixels inside the described region are masked.
            If inside is False, pixels outside the described region are masked.
        unit_wave : `astropy.units.Unit`
            The units of the lmin and lmax wavelength coordinates
            (Angstroms by default). If None, the units of the lmin and lmax
            arguments are assumed to be pixels.
        unit_center : `astropy.units.Unit`
            The units of the coordinates of the center argument
            (degrees by default).  If None, the units of the center
            argument are assumed to be pixels.
        unit_radius : `astropy.units.Unit`
            The units of the radius argument (arcseconds by default).
            If None, the units are assumed to be pixels.

        """

        center = np.array(center)

        if np.isscalar(radius):
            circular = True
            radius2 = radius * radius
            radius = (radius, radius)
        else:
            circular = False

        radius = np.array(radius)

        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]
        if unit_radius is not None:
            radius = radius / self.wcs.get_step(unit=unit_radius)
            radius2 = radius[0] * radius[1]

        if lmin is None:
            lmin = 0
        elif unit_wave is not None:
            lmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)

        if lmax is None:
            lmax = self.shape[0]
        elif unit_wave is not None:
            lmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave)

        ny, nx = self.shape[1:]
        imin, jmin = np.maximum(np.minimum((center - radius + 0.5).astype(int),
                                           [ny - 1, nx - 1]), [0, 0])
        imax, jmax = np.maximum(np.minimum((center + radius + 0.5).astype(int),
                                           [ny - 1, nx - 1]), [0, 0])
        imax += 1
        jmax += 1

        mask = np.zeros((lmax - lmin, self.shape[1], self.shape[2]),
                        dtype=bool)
        if circular:
            xx = np.arange(imin, imax) - center[0]
            yy = np.arange(jmin, jmax) - center[1]
            grid = (xx[:, np.newaxis]**2 + yy[np.newaxis, :]**2) < radius2
            mask[:, imin:imax, jmin:jmax] = grid[np.newaxis, :, :]
        else:
            mask[:, imin:imax, jmin:jmax] = True

        if inside:
            self.data[lmin:lmax, :, :][mask] = ma.masked
        else:
            self.data[:lmin, :, :] = ma.masked
            self.data[lmax:, :, :] = ma.masked
            self.data[lmin:lmax, :, :][~mask] = ma.masked

    def mask_ellipse(self, center, radius, posangle, lmin=None, lmax=None,
                     pix=False, inside=True, unit_center=u.deg,
                     unit_radius=u.arcsec, unit_wave=u.angstrom):
        """Mask values inside/outside the described region. Uses an elliptical
        shape.

        Parameters
        ----------
        center : (float,float)
            Center of the explored region.
        radius : (float,float)
            Radius defined the explored region.  radius is (float,float), it
            defines an elliptical region with semi-major and semi-minor axes.
        posangle : float
            Position angle of the first axis. It is defined in degrees against
            the horizontal (q) axis of the image, counted counterclockwise.
        lmin : float
            minimum wavelength.
        lmax : float
            maximum wavelength.
        inside : boolean
            If inside is True, pixels inside the described region are masked.
        unit_wave : `astropy.units.Unit`
            Type of the wavelengths coordinates (Angstrom by default)
            If None, inputs are in pixels
        unit_center : `astropy.units.Unit`
            Type of the coordinates of the center (degrees by default)
            If None, inputs are in pixels
        unit_radius : `astropy.units.Unit`
            Radius unit (arcseconds by default)
            If None, inputs are in pixels

        """
        center = np.array(center)
        radius = np.array(radius)
        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]
        if unit_radius is not None:
            radius = radius / self.wcs.get_step(unit=unit_radius)
        if lmin is None:
            lmin = 0
        else:
            if unit_wave is not None:
                lmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)
        if lmax is None:
            lmax = self.shape[0]
        else:
            if unit_wave is not None:
                lmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave)

        maxradius = max(radius[0], radius[1])

        imin, jmin = np.maximum(np.minimum((center - maxradius + 0.5).astype(int),
                                           [self.shape[1] - 1, self.shape[2] - 1]),
                                [0, 0])
        imax, jmax = np.maximum(np.minimum((center + maxradius + 0.5).astype(int),
                                           [self.shape[1] - 1, self.shape[2] - 1]),
                                [0, 0])
        imax += 1
        jmax += 1

        cospa = np.cos(np.radians(posangle))
        sinpa = np.sin(np.radians(posangle))

        if inside:
            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1], indexing='ij')
            grid3d = np.resize(((grid[1] * cospa + grid[0] * sinpa) / radius[0]) ** 2
                               + ((grid[0] * cospa - grid[1] * sinpa)
                                  / radius[1]) ** 2 < 1,
                               (lmax - lmin, imax - imin, jmax - jmin))
            self.data[lmin:lmax, imin:imax, jmin:jmax][grid3d] = ma.masked
        if not inside:
            self.data[:lmin, :, :] = ma.masked
            self.data[lmax:, :, :] = ma.masked
            self.data[:, :imin, :] = ma.masked
            self.data[:, imax:, :] = ma.masked
            self.data[:, :, :jmin] = ma.masked
            self.data[:, :, jmax:] = ma.masked

            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1],
                               indexing='ij')

            grid3d = np.resize(((grid[1] * cospa + grid[0] * sinpa) / radius[0]) ** 2
                               + ((grid[0] * cospa - grid[1] * sinpa)
                                  / radius[1]) ** 2 > 1,
                               (lmax - lmin, imax - imin, jmax - jmin))
            self.data[lmin:lmax, imin:imax, jmin:jmax][grid3d] = ma.masked

    def __add__(self, other):
        """Add other.

        cube1 + number = cube2 (cube2[k,p,q]=cube1[k,p,q]+number)

        cube1 + cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]+cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 + image = cube2 (cube2[k,p,q]=cube1[k,p,q]+image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates in spatial
        directions must be the same.

        cube1 + spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]+spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = self._data + other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if other.ndim == 1 or other.ndim == 3:
                if self.wave is not None and other.wave is not None \
                        and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spectral direction')
            if other.ndim == 2 or other.ndim == 3:
                if self.wcs is not None and other.wcs is not None \
                        and not self.wcs.isEqual(other.wcs):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spatial directions')

            if other.ndim == 1:
                # cube1 + spectrum = cube2
                if other.shape[0] != self.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data + other.data[:, np.newaxis, np.newaxis]
                else:
                    res.data = self.data + UnitMaskedArray(
                        other.data[:, np.newaxis, np.newaxis],
                        other.unit, self.unit)
                # variance
                if other._var is not None:
                    if self._var is None:
                        if other.unit == self.unit:
                            res._var = np.ones(self.shape) * other._var[:, np.newaxis, np.newaxis]
                        else:
                            res._var = np.ones(self.shape) * \
                                UnitArray(other._var[:, np.newaxis, np.newaxis],
                                          other.unit**2, self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res._var = self._var + other._var[:, np.newaxis, np.newaxis]
                        else:
                            res._var = self._var + \
                                UnitArray(other._var[:, np.newaxis, np.newaxis],
                                          other.unit**2, self.unit**2)
                return res
            elif other.ndim == 2:
                # cube1 + image = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                if self.shape[2] != other.shape[1] \
                        or self.shape[1] != other.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data + other.data[np.newaxis, :, :]
                else:
                    res.data = self.data + UnitMaskedArray(other.data[np.newaxis, :, :],
                                                           other.unit, self.unit)
                # variance
                if other._var is not None:
                    if self._var is None:
                        if self.unit == other.unit:
                            res._var = np.ones(self.shape) * other._var[np.newaxis, :, :]
                        else:
                            res._var = np.ones(self.shape) \
                                * UnitArray(other._var[np.newaxis, :, :],
                                            other.unit**2, self.unit**2)
                    else:
                        if self.unit == other.unit:
                            res._var = self._var + other._var[np.newaxis, :, :]
                        else:
                            res._var = self._var + UnitArray(other._var[np.newaxis, :, :],
                                                             other.unit**2, self.unit**2)

                return res
            else:
                # cube1 + cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]+cube2[k,j,i])
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1] \
                        or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')

                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data + other.data
                else:
                    res.data = self.data + UnitMaskedArray(other.data.data,
                                                           other.unit, self.unit)
                # variance
                if res._var is not None:
                    if self._var is None:
                        if other.unit == self.unit:
                            res._var = other._var
                        else:
                            res._var = UnitArray(other._var, other.unit**2,
                                                 self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res._var = self._var + other._var
                        else:
                            res._var = self._var + UnitArray(other._var,
                                                             other.unit**2,
                                                             self.unit**2)
                return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract other.

        cube1 - number = cube2 (cube2[k,p,q]=cube1[k,p,q]-number)

        cube1 - cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]-cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 - image = cube2 (cube2[k,p,q]=cube1[k,p,q]-image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates
        in spatial directions must be the same.

        cube1 - spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]-spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = self._data - other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if other.ndim == 1 or other.ndim == 3:
                if self.wave is not None and other.wave is not None \
                        and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes '
                                  'with different world coordinates '
                                  'in spectral direction')
            if other.ndim == 2 or other.ndim == 3:
                if self.wcs is not None and other.wcs is not None \
                        and not self.wcs.isEqual(other.wcs):
                    raise IOError('Operation forbidden for cubes '
                                  'with different world coordinates '
                                  'in spatial directions')

            if other.ndim == 1:
                # cube1 - spectrum = cube2
                if other.shape[0] != self.shape[0]:
                    raise IOError('Operation forbidden '
                                  'for objects with different sizes')
                res = self.copy()
                # data
                if self.unit == other.unit:
                    res.data = self.data - other.data[:, np.newaxis, np.newaxis]
                else:
                    res.data = self.data - UnitMaskedArray(other.data[:, np.newaxis, np.newaxis],
                                                           other.unit, self.unit)
                # variance
                if other._var is not None:
                    if self._var is None:
                        if self.unit == other.unit:
                            res._var = np.ones(self.shape) \
                                * other._var[:, np.newaxis, np.newaxis]
                        else:
                            res._var = np.ones(self.shape) \
                                * UnitArray(other._var[:, np.newaxis, np.newaxis],
                                            other.unit**2, self.unit**2)
                    else:
                        if self.unit == other.unit:
                            res._var = self._var \
                                + other._var[:, np.newaxis, np.newaxis]
                        else:
                            res._var = self._var \
                                + UnitArray(other._var[:, np.newaxis, np.newaxis],
                                            other.unit**2, self.unit**2)
                return res
            elif other.ndim == 2:
                # cube1 - image = cube2 (cube2[k,j,i]=cube1[k,j,i]-image[j,i])
                if self.shape[2] != other.shape[1] \
                        or self.shape[1] != other.shape[0]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                if self.unit == other.unit:
                    res.data = self.data - other.data[np.newaxis, :, :]
                else:
                    res.data = self.data - UnitMaskedArray(other.data[np.newaxis, :, :],
                                                           other.unit, self.unit)
                # variance
                if other._var is not None:
                    if self._var is None:
                        if self.unit == other.unit:
                            res._var = np.ones(self.shape) * other._var[np.newaxis, :, :]
                        else:
                            res._var = np.ones(self.shape) \
                                * UnitArray(other._var[np.newaxis, :, :],
                                            other.unit**2, self.unit**2)
                    else:
                        if self.unit == other.unit:
                            res._var = self._var + other._var[np.newaxis, :, :]
                        else:
                            res._var = self._var + UnitArray(other._var[np.newaxis, :, :],
                                                             other.unit**2, self.unit**2)
                return res
            else:
                # cube1 - cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]-cube2[k,j,i])
                if self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1] \
                        or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data - other.data
                else:
                    res.data = self.data - UnitMaskedArray(other.data.data,
                                                           other.unit, self.unit)
                # variance
                if other._var is not None:
                    if self._var is None:
                        if other.unit == self.unit:
                            res._var = other._var
                        else:
                            res._var = UnitArray(other._var, other.unit**2,
                                                 self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res._var = self._var + other._var
                        else:
                            res._var = self._var + UnitArray(other._var,
                                                             other.unit**2,
                                                             self.unit**2)
                return res

    def __rsub__(self, other):
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = other - self._data
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__sub__(self)

    def __mul__(self, other):
        """Multiply by other.

        cube1 * number = cube2 (cube2[k,p,q]=cube1[k,p,q]*number)

        cube1 * cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]*cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 * image = cube2 (cube2[k,p,q]=cube1[k,p,q]*image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates
        in spatial directions must be the same.

        cube1 * spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]*spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data *= other
                if self._var is not None:
                    res._var *= other ** 2
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if other.ndim == 1 or other.ndim == 3:
                if self.wave is not None and other.wave is not None \
                        and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spectral direction')
            if other.ndim == 2 or other.ndim == 3:
                if self.wcs is not None and other.wcs is not None \
                        and not self.wcs.isEqual(other.wcs):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spatial directions')
            if other.ndim == 1:
                # cube1 * spectrum = cube2
                if other.shape[0] != self.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data * other.data[:, np.newaxis, np.newaxis]
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var[:, np.newaxis, np.newaxis] \
                        * self._data * self._data
                elif other._var is None:
                    res._var = self._var \
                        * other._data[:, np.newaxis, np.newaxis] \
                        * other._data[:, np.newaxis, np.newaxis]
                else:
                    res._var = (other._var[:, np.newaxis, np.newaxis] *
                                self._data * self._data + self._var *
                                other._data[:, np.newaxis, np.newaxis] *
                                other._data[:, np.newaxis, np.newaxis])
                # unit
                res.unit = self.unit * other.unit
                return res
            elif other.ndim == 2:
                # cube1 * image = cube2 (cube2[k,j,i]=cube1[k,j,i]*image[j,i])
                if self.shape[2] != other.shape[1] \
                        or self.shape[1] != other.shape[0]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data * other.data[np.newaxis, :, :]
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var[np.newaxis, :, :] \
                        * self._data * self._data
                elif other._var is None:
                    res._var = self._var * other._data[np.newaxis, :, :] \
                        * other._data[np.newaxis, :, :]
                else:
                    res._var = (other._var[np.newaxis, :, :] *
                                self._data * self._data +
                                self._var * other._data[np.newaxis, :, :] *
                                other._data[np.newaxis, :, :])
                # unit
                res.unit = self.unit * other.unit
                return res
            else:
                # cube1 * cube2 = cube3
                if self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1] \
                        or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data * other.data
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var * self._data * self._data
                elif other._var is None:
                    res._var = self._var * other._data * other._data
                else:
                    res._var = (other._var * self._data * self._data +
                                self._var * other._data * other._data)
                # unit
                res.unit = self.unit * other.unit
                return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """Divide by other.

        cube1 / number = cube2 (cube2[k,p,q]=cube1[k,p,q]/number)

        cube1 / cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]/cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 / image = cube2
        (cube2[k,p,q]=cube1[k,p,q]/image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates
        in spatial directions must be the same.

        cube1 / spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]/spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data /= other
                if self._var is not None:
                    res._var /= other ** 2
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if other.ndim == 1 or other.ndim == 3:
                if self.wave is not None and other.wave is not None \
                        and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spectral direction')
            if other.ndim == 2 or other.ndim == 3:
                if self.wcs is not None and other.wcs is not None \
                        and not self.wcs.isEqual(other.wcs):
                    raise ValueError('Operation forbidden for cubes '
                                     'with different world coordinates'
                                     ' in spatial directions')
            if other.ndim == 1:
                # cube1 / spectrum = cube2
                if other.shape[0] != self.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                # data
                res = self.copy()
                res.data = self.data / other.data[:, np.newaxis, np.newaxis]
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var[:, np.newaxis, np.newaxis] \
                        * self._data * self._data \
                        / (other._data[:, np.newaxis, np.newaxis] ** 4)
                elif other._var is None:
                    res._var = self._var \
                        * other._data[:, np.newaxis, np.newaxis] \
                        * other._data[:, np.newaxis, np.newaxis] \
                        / (other._data[:, np.newaxis, np.newaxis] ** 4)
                else:
                    res._var = (other._var[:, np.newaxis, np.newaxis] *
                                self._data * self._data + self._var *
                                other._data[:, np.newaxis, np.newaxis] *
                                other._data[:, np.newaxis, np.newaxis]) \
                        / (other._data[:, np.newaxis, np.newaxis] ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res
            elif other.ndim == 2:
                # cube1 / image = cube2 (cube2[k,j,i]=cube1[k,j,i]/image[j,i])
                if self.shape[2] != other.shape[1] \
                        or self.shape[1] != other.shape[0]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data / other.data[np.newaxis, :, :]
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var[np.newaxis, :, :] \
                        * self._data * self._data \
                        / (other._data[np.newaxis, :, :] ** 4)
                elif other._var is None:
                    res._var = self._var * other._data[np.newaxis, :, :] \
                        * other._data[np.newaxis, :, :] \
                        / (other._data[np.newaxis, :, :] ** 4)
                else:
                    res._var = (other._var[np.newaxis, :, :] *
                                self._data * self._data + self._var *
                                other._data[np.newaxis, :, :] *
                                other._data[np.newaxis, :, :]) \
                        / (other._data[np.newaxis, :, :] ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res
            else:
                # cube1 / cube2 = cube3
                if self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1] \
                        or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data / other.data
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var * self._data * self._data \
                        / (other._data ** 4)
                elif other._var is None:
                    res._var = self._var * other._data * other._data \
                        / (other._data ** 4)
                else:
                    res._var = (other._var * self._data * self._data +
                                self._var * other._data * other._data) \
                        / (other._data ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res

    def __rdiv__(self, other):
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = other / res._data
                if self._var is not None:
                    res._var = other ** 2 / res._var
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__div__(self)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __getitem__(self, item):
        """Return the corresponding object:
        cube[k,p,k] = value
        cube[k,:,:] = spectrum
        cube[:,p,q] = image
        cube[:,:,:] = sub-cube
        """
        obj = super(Cube, self).__getitem__(item)
        if isinstance(obj, DataArray):
            if obj.ndim == 3:
                return obj
            elif obj.ndim == 1:
                cls = Spectrum
            elif obj.ndim == 2:
                cls = Image
            return cls.new_from_obj(obj)
        else:
            return obj

    def get_lambda(self, lbda_min, lbda_max=None, unit_wave=u.angstrom):
        """Return the sub-cube corresponding to a wavelength range.

        Parameters
        ----------
        lbda_min : float
            Minimum wavelength.
        lbda_max : float
            Maximum wavelength.
        unit_wave : `astropy.units.Unit`
            wavelengths unit.
            If None, inputs are in pixels

        """
        if lbda_max is None:
            lbda_max = lbda_min
        if self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '
                             'along the spectral direction')
        else:
            if unit_wave is None:
                pix_min = max(0, int(lbda_min + 0.5))
                pix_max = min(self.shape[0], int(lbda_max + 0.5))
            else:
                pix_min = max(0, int(self.wave.pixel(lbda_min, unit=unit_wave)))
                pix_max = min(self.shape[0], int(self.wave.pixel(lbda_max, unit=unit_wave)) + 1)
            if (pix_min + 1) == pix_max:
                return self[pix_min, :, :]
            else:
                return self[pix_min:pix_max, :, :]

    def get_step(self, unit_wave=None, unit_wcs=None):
        """Return the cube steps [dlbda,dy,dx].

        Parameters
        ----------
        unit_wave : `astropy.units.Unit`
            wavelengths unit.
        unit_wcs : `astropy.units.Unit`
            world coordinates unit.

        """
        step = np.empty(3)
        step[0] = self.wave.get_step(unit_wave)
        step[1:] = self.wcs.get_step(unit_wcs)
        return step

    def get_range(self, unit_wave=None, unit_wcs=None):
        """Return the range of wavelengths, declinations and right ascensions
        in the cube.

        The minimum and maximum coordinates are returned as an array
        in the following order:

          [lbda_min, dec_min, ra_min, lbda_max, dec_max, ra_max]

        Note that when the rotation angle of the image on the sky is
        not zero, dec_min, ra_min, dec_max and ra_max are not at the
        corners of the image.

        Parameters
        ----------
        unit_wave : `astropy.units.Unit`
            The wavelengths units.
        unit_wcs : `astropy.units.Unit`
            The angular units of the returned sky coordinates.

        Returns
        -------
        out : numpy.ndarray
           The range of right ascensions declinations and wavelengths,
           arranged as [lbda_min, dec_min, ra_min, lbda_max, dec_max, ra_max].

        """

        wcs_range = self.wcs.get_range(unit_wcs)
        wave_range = self.wave.get_range(unit_wave)
        return np.array([wave_range[0], wcs_range[0], wcs_range[1],
                         wave_range[1], wcs_range[2], wcs_range[3]])

    def get_start(self, unit_wave=None, unit_wcs=None):
        """Return [lbda,y,x] corresponding to pixel (0,0,0).

        Parameters
        ----------
        unit_wave : `astropy.units.Unit`
            wavelengths unit.
        unit_wcs : `astropy.units.Unit`
            world coordinates unit.

        """
        start = np.empty(3)
        start[0] = self.wave.get_start(unit_wave)
        start[1:] = self.wcs.get_start(unit_wcs)
        return start

    def get_end(self, unit_wave=None, unit_wcs=None):
        """Return [lbda,y,x] corresponding to pixel (-1,-1,-1).

        Parameters
        ----------
        unit_wave : `astropy.units.Unit`
            wavelengths unit.
        unit_wcs : `astropy.units.Unit`
            world coordinates unit.

        """
        end = np.empty(3)
        end[0] = self.wave.get_end(unit_wave)
        end[1:] = self.wcs.get_end(unit_wcs)
        return end

    def get_rot(self, unit=u.deg):
        """Return the rotation angle.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            type of the angle coordinate
            degree by default

        """
        return self.wcs.get_rot(unit)

    def set_wcs(self, wcs=None, wave=None):
        """Set the world coordinates (spatial and/or spectral).

        Parameters
        ----------
        wcs : `mpdaf.obj.WCS`
            World coordinates.
        wave : `mpdaf.obj.WaveCoord`
            Wavelength coordinates.

        """
        if wcs is not None:
            self.wcs = wcs.copy()
            self.wcs.naxis1 = self.shape[2]
            self.wcs.naxis2 = self.shape[1]
            if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
                and (wcs.naxis1 != self.shape[2] or
                     wcs.naxis2 != self.shape[1]):
                self._logger.warning('world coordinates and data have not the '
                                     'same dimensions')
        if wave is not None:
            if wave.shape is not None and wave.shape != self.shape[0]:
                self._logger.warning('wavelength coordinates and data have '
                                     'not the same dimensions')
            self.wave = wave.copy()
            self.wave.shape = self.shape[0]

    def sum(self, axis=None, weights=None):
        """Return the sum over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
            Axis or axes along which a sum is performed:

            - The default (axis = None) is perform a sum over all the
              dimensions of the cube and returns a float.
            - axis = 0 is perform a sum over the wavelength dimension and
              returns an image.
            - axis = (1,2) is perform a sum over the (X,Y) axes and returns
              a spectrum.

            Other cases return None.
        weights : array_like, optional
            An array of weights associated with the data values. The weights
            array can either be 1-D (axis=(1,2)) or 2-D (axis=0) or of the
            same shape as the cube. If weights=None, then all data in a are
            assumed to have a weight equal to one.

            The method conserves the flux by using the algorithm
            from Jarle Brinchmann (jarle@strw.leidenuniv.nl):
            - Take into account bad pixels in the addition.
            - Normalize with the median value of weighting sum/no-weighting sum

        """
        if weights is not None:
            w = np.array(weights, dtype=np.float)
            excmsg = 'Incorrect dimensions for the weights (%s) (it must be (%s))'

            if len(w.shape) == 3:
                if not np.array_equal(w.shape, self.shape):
                    raise IOError(excmsg % (w.shape, self.shape))
            elif len(w.shape) == 2:
                if w.shape[0] != self.shape[1] or w.shape[1] != self.shape[2]:
                    raise IOError(excmsg % (w.shape, self.shape[1:]))
                else:
                    w = np.tile(w, (self.shape[0], 1, 1))
            elif len(w.shape) == 1:
                if w.shape[0] != self.shape[0]:
                    raise IOError(excmsg % (w.shape[0], self.shape[0]))
                else:
                    w = np.ones_like(self._data) * w[:, np.newaxis, np.newaxis]
            else:
                raise IOError(excmsg % (None, self.shape))

            # weights mask
            wmask = ma.masked_where(self._mask, ma.masked_where(w == 0, w))

        if axis is None:
            if weights is None:
                return self.data.sum()
            else:
                data = self.data * w
                npix = np.sum(~self.mask)
                data = ma.sum(data) / npix
                # flux conservation
                orig_data = self.data * ~wmask.mask
                orig_data = ma.sum(orig_data)
                rr = data / orig_data
                med_rr = ma.median(rr)
                if med_rr > 0:
                    data /= med_rr
                return data
        elif axis == 0:
            # return an image
            if weights is None:
                data = ma.sum(self.data, axis=0)
                if self._var is not None:
                    var = ma.sum(self.var, axis=0)
                else:
                    var = None
            else:
                data = self.data * w
                npix = np.sum(~self.mask, axis)
                data = ma.sum(data, axis) / npix
                orig_data = self.data * ~wmask.mask
                orig_data = ma.sum(orig_data, axis)
                rr = data / orig_data
                med_rr = ma.median(rr)
                if med_rr > 0:
                    data /= med_rr
                if self.var is not None:
                    var = ma.sum(self.masked_var * w, axis) / npix
                    dspec = ma.sqrt(var)
                    if med_rr > 0:
                        dspec /= med_rr
                    orig_var = self.var * ~wmask.mask
                    orig_var = ma.masked_where(self.mask,
                                               ma.masked_invalid(orig_var))
                    orig_var = ma.sum(orig_var, axis)
                    sn_orig = orig_data / ma.sqrt(orig_var)
                    sn_now = data / dspec
                    sn_ratio = ma.median(sn_orig / sn_now)
                    dspec /= sn_ratio
                    var = dspec * dspec
                    var = var.filled(np.inf)
                else:
                    var = None
            return Image.new_from_obj(self, data=data, var=var)
        elif axis == (1, 2):
            # return a spectrum
            if weights is None:
                data = ma.sum(ma.sum(self.data, axis=1), axis=1)
                if self._var is not None:
                    var = ma.sum(ma.sum(self.var, axis=1), axis=1).filled(np.inf)
                else:
                    var = None
            else:
                data = self.data * w
                npix = np.sum(np.sum(~self.mask, axis=1), axis=1)
                data = ma.sum(ma.sum(data, axis=1), axis=1) / npix
                orig_data = self.data * ~wmask.mask
                orig_data = ma.sum(ma.sum(orig_data, axis=1), axis=1)
                rr = data / orig_data
                med_rr = ma.median(rr)
                if med_rr > 0:
                    data /= med_rr
                if self._var is not None:
                    var = ma.sum(ma.sum(self.var * w, axis=1), axis=1) / npix
                    dspec = ma.sqrt(var)
                    if med_rr > 0:
                        dspec /= med_rr
                    orig_var = self._var * ~wmask.mask
                    orig_var = ma.masked_where(self.mask,
                                               ma.masked_invalid(orig_var))
                    orig_var = ma.sum(ma.sum(orig_var, axis=1), axis=1)
                    sn_orig = orig_data / ma.sqrt(orig_var)
                    sn_now = data / dspec
                    sn_ratio = ma.median(sn_orig / sn_now)
                    dspec /= sn_ratio
                    var = dspec * dspec
                    var = var.filled(np.inf)
                else:
                    var = None

            return Spectrum(wave=self.wave, unit=self.unit, data=data, var=var,
                            copy=False)
        else:
            raise ValueError('Invalid axis argument')

    def mean(self, axis=None):
        """Return the mean over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
            Axis or axes along which a mean is performed.

            - The default (axis = None) is perform a mean over all the
              dimensions of the cube and returns a float.
            - axis = 0 is perform a mean over the wavelength dimension and
              returns an image.
            - axis = (1,2) is perform a mean over the (X,Y) axes and
              returns a spectrum.

            Other cases return None.

        """
        if axis is None:
            return self.data.mean()
        elif axis == 0:
            # return an image
            data = ma.mean(self.data, axis)
            if self._var is not None:
                var = ma.sum(self.var, axis).filled(np.inf) \
                    / ma.count(self.data, axis) ** 2
            else:
                var = None
            return Image.new_from_obj(self, data=data, var=var)
        elif axis == (1, 2):
            # return a spectrum
            count = np.sum(np.sum(~self.mask, axis=1), axis=1)
            data = (ma.sum(ma.sum(self.data, axis=1), axis=1)) / count
            if self._var is not None:
                var = ma.sum(ma.sum(self.var, axis=1), axis=1).filled(np.inf) \
                    / (count**2)
            else:
                var = None
            return Spectrum.new_from_obj(self, data=data, var=var)
        else:
            raise ValueError('Invalid axis argument')

    def median(self, axis=None):
        """Return the median over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
            Axis or axes along which a median is performed.

            - The default (axis = None) is perform a median over all the
              dimensions of the cube and returns a float.
            - axis = 0 is perform a median over the wavelength dimension and
              returns an image.
            - axis = (1,2) is perform a median over the (X,Y) axes and
              returns a spectrum.

            Other cases return None.

        """
        if axis is None:
            return np.ma.median(self.data)
        elif axis == 0:
            # return an image
            data = np.ma.median(self.data, axis)
            if self._var is not None:
                var = np.ma.median(self.var, axis).filled(np.inf)
            else:
                var = None
            return Image.new_from_obj(self, data=data, var=var)
        elif axis == (1, 2):
            # return a spectrum
            data = np.ma.median(np.ma.median(self.data, axis=1), axis=1)
            if self._var is not None:
                var = np.ma.median(np.ma.median(self.var, axis=1),
                                   axis=1).filled(np.inf)
            else:
                var = None
            return Spectrum(wave=self.wave, unit=self.unit, data=data, var=var,
                            copy=False)
        else:
            raise ValueError('Invalid axis argument')

    def truncate(self, coord, mask=True, unit_wave=u.angstrom, unit_wcs=u.deg):
        """ Truncates the cube and return a sub-cube.

        Parameters
        ----------
        coord : array
            array containing the sub-cube boundaries
            [lbda_min,y_min,x_min,lbda_max,y_max,x_max]
            (output of mpdaf.obj.cube.get_range)
        mask : boolean
            if True, pixels outside [y_min,y_max] and [x_min,x_max] are masked.
        unit_wave : `astropy.units.Unit`
            wavelengths unit.  If None, inputs are in pixels
        unit_wcs : `astropy.units.Unit`
            world coordinates unit.  If None, inputs are in pixels

        """
        lmin, ymin, xmin, lmax, ymax, xmax = coord

        skycrd = [[ymin, xmin], [ymin, xmax],
                  [ymax, xmin], [ymax, xmax]]
        if unit_wcs is None:
            pixcrd = np.array(skycrd)
        else:
            pixcrd = self.wcs.sky2pix(skycrd, unit=unit_wcs)

        imin = int(np.min(pixcrd[:, 0]) + 0.5)
        if imin < 0:
            imin = 0
        imax = int(np.max(pixcrd[:, 0]) + 0.5) + 1
        if imax > self.shape[1]:
            imax = self.shape[1]

        if imin >= self.shape[1] or imax <= 0 or imin == imax:
            raise ValueError('sub-cube boundaries are outside the cube')

        jmin = int(np.min(pixcrd[:, 1]) + 0.5)
        if jmin < 0:
            jmin = 0
        jmax = int(np.max(pixcrd[:, 1]) + 0.5) + 1
        if jmax > self.shape[2]:
            jmax = self.shape[2]
        if jmin >= self.shape[2] or jmax <= 0 or jmin == jmax:
            raise ValueError('sub-cube boundaries are outside the cube')

        if unit_wave is None:
            kmin = int(lmin + 0.5)
            kmax = int(lmax + 0.5)
        else:
            kmin = max(0, self.wave.pixel(lmin, nearest=True, unit=unit_wave))
            kmax = min(self.shape[0], self.wave.pixel(lmax, nearest=True,
                                                      unit=unit_wave) + 1)

        if kmin == kmax:
            raise ValueError('Minimum and maximum wavelengths are equal')

        if kmax == kmin + 1:
            raise ValueError('Minimum and maximum wavelengths are outside'
                             ' the spectrum range')

        res = self[kmin:kmax, imin:imax, jmin:jmax]

        if mask:
            # mask outside pixels
            grid = np.meshgrid(np.arange(0, res.shape[1]),
                               np.arange(0, res.shape[2]), indexing='ij')
            shape = grid[1].shape
            pixcrd = np.array([[p, q] for p, q in zip(np.ravel(grid[0]),
                                                      np.ravel(grid[1]))])
            if unit_wcs is None:
                skycrd = pixcrd
            else:
                skycrd = np.array(res.wcs.pix2sky(pixcrd, unit=unit_wcs))
            x = skycrd[:, 1].reshape(shape)
            y = skycrd[:, 0].reshape(shape)
            test_x = np.logical_or(x < xmin, x > xmax)
            test_y = np.logical_or(y < ymin, y > ymax)
            test = np.logical_or(test_x, test_y)
            res.data[np.tile(test, [res.shape[0], 1, 1])] = ma.masked
            res.crop()

        return res

    def _rebin_mean_(self, factor):
        """Shrink the size of the cube by factor. New size must be an integer
        multiple of the original size.

        Parameters
        ----------
        factor : (integer,integer,integer)
            Factor in z, y and x.  Python notation: (nz,ny,nx)

        """
        assert np.array_equal(np.mod(self.shape, factor), [0, 0, 0])
        # new size is an integer multiple of the original size
        sh = self.shape // np.asarray(factor)
        newsh = (sh[0], factor[0], sh[1], factor[1], sh[2], factor[2])

        if self.mask is np.ma.nomask:
            n = np.prod(factor)
            self._data = self._data.reshape(newsh).sum(1).sum(2).sum(3) / n
            if self._var is not None:
                self._var = self._var.reshape(newsh).sum(1).sum(2).sum(3) / n**2
        else:
            mask_count = (~self.mask).reshape(newsh).sum(1).sum(2).sum(3)
            self._data = self.data.reshape(newsh).sum(1).sum(2).sum(3).data / mask_count
            if self._var is not None:
                self._var = self.var.reshape(newsh).sum(1).sum(2).sum(3).data / \
                    (mask_count * mask_count)
            self._mask = mask_count == 0
        self._ndim = self._data.ndim

        # coordinates
        self.wcs = self.wcs.rebin(factor[1:])
        self.wave.rebin(factor[0])

    def _rebin_mean(self, factor, margin='center', flux=False):
        """Shrink the size of the cube by factor.

        Parameters
        ----------
        factor : integer or (integer,integer,integer)
            Factor in z, y and x. Python notation: (nz,ny,nx).
        flux : boolean
            This parameters is used if new size is not an integer
            multiple of the original size.

            If Flux is False, the cube is truncated and the flux
            is not conserved.

            If Flux is True, margins are added to the cube to
            conserve the flux.
        margin : 'center' or 'origin'
            This parameters is used if new size is not an
            integer multiple of the original size.

            In 'center' case, cube is truncated/pixels are added on the
            left and on the right, on the bottom and of the top of the
            cube.

            In 'origin'case, cube is truncated/pixels are added at the end
            along each direction
        """
        if is_int(factor):
            factor = (factor, factor, factor)

        if not np.any(np.mod(self.shape, factor)):
            # new size is an integer multiple of the original size
            self._rebin_mean_(factor)
            return None
        else:
            factor = np.array(factor)
            newshape = self.shape // factor
            n = self.shape - newshape * factor

            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0] == 1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0] // 2
                    n0_right = self.shape[0] - n[0] + n0_left

            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1] == 1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1] // 2
                    n1_right = self.shape[1] - n[1] + n1_left

            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2] == 1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2] // 2
                    n2_right = self.shape[2] - n[2] + n2_left

            cub = self[n0_left:n0_right, n1_left:n1_right, n2_left:n2_right]
            cub._rebin_mean_(factor)

            if flux is False:
                self._data = cub._data
                self._var = cub._var
                self._mask = cub._mask
                self.wave = cub.wave
                self.wcs = cub.wcs
                return None
            else:
                newshape = list(cub.shape)
                wave = cub.wave
                wcs = cub.wcs

                if n0_left != 0:
                    newshape[0] += 1
                    wave.set_crpix(wave.get_crpix() + 1)
                    wave.shape = wave.shape + 1
                    l_left = 1
                else:
                    l_left = 0
                if n0_right != self.shape[0]:
                    newshape[0] += 1
                    l_right = newshape[0] - 1
                    wave.shape = wave.shape + 1
                else:
                    l_right = newshape[0]

                if n1_left != 0:
                    newshape[1] += 1
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    wcs.set_naxis2(wcs.naxis2 + 1)
                    p_left = 1
                else:
                    p_left = 0
                if n1_right != self.shape[1]:
                    newshape[1] += 1
                    wcs.set_naxis2(wcs.naxis2 + 1)
                    p_right = newshape[1] - 1
                else:
                    p_right = newshape[1]

                if n2_left != 0:
                    newshape[2] += 1
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 1)
                    q_left = 1
                else:
                    q_left = 0
                if n2_right != self.shape[2]:
                    newshape[2] += 1
                    wcs.set_naxis1(wcs.naxis1 + 1)
                    q_right = newshape[2] - 1
                else:
                    q_right = newshape[2]

                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[l_left:l_right, p_left:p_right, q_left:q_right] = cub.data
                mask[l_left:l_right, p_left:p_right, q_left:q_right] = \
                    cub.mask

                if self._var is None:
                    var = None
                else:
                    var = np.empty(newshape)
                    var[l_left:l_right, p_left:p_right, q_left:q_right] = cub._var

                F = factor[0] * factor[1] * factor[2]
                F2 = F * F

                if cub.shape[0] != newshape[0]:
                    d = self.data[n0_right:, n1_left:n1_right, n2_left:n2_right]\
                        .sum(axis=0)\
                        .reshape(cub.shape[1], factor[1], cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[-1, p_left:q_left, q_left:q_right] = d.data
                    mask[-1, p_left:q_left, q_left:q_right] = d.mask
                    if var is not None:
                        var[-1, p_left:q_left, q_left:q_right] = \
                            self.var[n0_right:, n1_left:n1_right, n2_left:n2_right]\
                            .sum(axis=0)\
                            .reshape(cub.shape[1], factor[1],
                                     cub.shape[2], factor[2])\
                            .sum(1).sum(2).data / F2
                if l_left == 1:
                    d = self.data[:n0_left, n1_left:n1_right, n2_left:n2_right]\
                        .sum(axis=0)\
                        .reshape(cub.shape[1], factor[1], cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[0, p_left:q_left, q_left:q_right] = d.data
                    mask[0, p_left:q_left, q_left:q_right] = d.mask
                    if var is not None:
                        var[0, p_left:q_left, q_left:q_right] = \
                            self.var[:n0_left, n1_left:n1_right, n2_left:n2_right]\
                            .sum(axis=0)\
                            .reshape(cub.shape[1], factor[1],
                                     cub.shape[2], factor[2])\
                            .sum(1).sum(2).data / F2
                if cub.shape[1] != newshape[1]:
                    d = self.data[n0_left:n0_right, n1_right:,
                                  n2_left:n2_right]\
                        .sum(axis=1).reshape(cub.shape[0], factor[0],
                                             cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, -1, q_left:q_right] = d.data
                    mask[l_left:l_right, -1, q_left:q_right] = d.mask
                    if var is not None:
                        var[l_left:l_right, -1, q_left:q_right] = \
                            self.var[n0_left:n0_right, n1_right:, n2_left:n2_right]\
                            .sum(axis=1)\
                            .reshape(cub.shape[0], factor[0],
                                     cub.shape[2], factor[2])\
                            .sum(1).sum(2).data / F2
                if p_left == 1:
                    d = self.data[n0_left:n0_right, :n1_left, n2_left:n2_right]\
                        .sum(axis=1).reshape(cub.shape[0], factor[0],
                                             cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, 0, q_left:q_right] = d.data
                    mask[l_left:l_right, 0, q_left:q_right] = d.mask
                    if var is not None:
                        var[l_left:l_right, 0, q_left:q_right] = \
                            self.var[n0_left:n0_right, :n1_left, n2_left:n2_right]\
                            .sum(axis=1).reshape(cub.shape[0], factor[0],
                                                 cub.shape[2], factor[2])\
                            .sum(1).sum(2).data / F2

                if cub.shape[2] != newshape[2]:
                    d = self.data[n0_left:n0_right,
                                  n1_left:n1_right:, n2_right:]\
                        .sum(axis=2)\
                        .reshape(cub.shape[0], factor[0], cub.shape[1], factor[1])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, p_left:p_right, -1] = d.data
                    mask[l_left:l_right, p_left:p_right, -1] = d.mask
                    if var is not None:
                        var[l_left:l_right, p_left:p_right, -1] = \
                            self.var[n0_left:n0_right,
                                     n1_left:n1_right:, n2_right:]\
                            .sum(axis=2).reshape(cub.shape[0], factor[0],
                                                 cub.shape[1], factor[1])\
                            .sum(1).sum(2).data / F2
                if q_left == 1:
                    d = self.data[n0_left:n0_right,
                                  n1_left:n1_right:, :n2_left]\
                        .sum(axis=2).reshape(cub.shape[0], factor[0],
                                             cub.shape[1], factor[1])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, p_left:p_right, 0] = d.data
                    mask[l_left:l_right, p_left:p_right, 0] = d.mask
                    if var is not None:
                        var[l_left:l_right, p_left:p_right, 0] = \
                            self.var[n0_left:n0_right, n1_left:n1_right:, :n2_left]\
                            .sum(axis=2)\
                            .reshape(cub.shape[0], factor[0],
                                     cub.shape[1], factor[1])\
                            .sum(1).sum(2).data / F2

                if l_left == 1 and p_left == 1 and q_left == 1:
                    data[0, 0, 0] = \
                        self.data[:n0_left, :n1_left, :n2_left].sum() / F
                    mask[0, 0, 0] = self.mask[:n0_left, :n1_left, :n2_left].any()
                    if var is not None:
                        var[0, 0, 0] = \
                            self.var[:n0_left, :n1_left, :n2_left].sum().data / F2
                if l_left == 1 and p_right == (newshape[1] - 1) \
                        and q_left == 1:
                    data[0, -1, 0] = \
                        self.data[:n0_left, n1_right:, :n2_left].sum() / F
                    mask[0, -1, 0] = \
                        self.mask[:n0_left, n1_right:, :n2_left].any()
                    if var is not None:
                        var[0, -1, 0] = \
                            self.var[:n0_left, n1_right:, :n2_left].sum().data / F2
                if l_left == 1 and p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    data[0, -1, -1] = \
                        self.data[:n0_left, n1_right:, n2_right:].sum() / F
                    mask[0, -1, -1] = \
                        self.mask[:n0_left, n1_right:, n2_right:].any()
                    if var is not None:
                        var[0, -1, -1] = \
                            self.var[:n0_left, n1_right:, n2_right:].sum().data / F2
                if l_left == 1 and p_left == 1 and \
                        q_right == (newshape[2] - 1):
                    data[0, 0, -1] = \
                        self.data[:n0_left, :n1_left, n2_right:].sum() / F
                    mask[0, 0, -1] = \
                        self.mask[:n0_left, :n1_left, n2_right:].any()
                    if var is not None:
                        var[0, 0, -1] = \
                            self.var[:n0_left, :n1_left, n2_right:].sum().data / F2
                if l_left == (newshape[0] - 1) and p_left == 1 \
                        and q_left == 1:
                    data[-1, 0, 0] = \
                        self.data[n0_right:, :n1_left, :n2_left].sum() / F
                    mask[-1, 0, 0] = \
                        self.mask[n0_right:, :n1_left, :n2_left].any()
                    if var is not None:
                        var[-1, 0, 0] = \
                            self.var[n0_right:, :n1_left, :n2_left].sum().data / F2
                if l_left == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1) and q_left == 1:
                    data[-1, -1, 0] = \
                        self.data[n0_right:, n1_right:, :n2_left].sum() / F
                    mask[-1, -1, 0] = \
                        self.mask[n0_right:, n1_right:, :n2_left].any()
                    if var is not None:
                        var[-1, -1, 0] = \
                            self.var[n0_right:, n1_right:, :n2_left].sum().data / F2
                if l_left == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    data[-1, -1, -1] = \
                        self.data[n0_right:, n1_right:, n2_right:].sum() / F
                    mask[-1, -1, -1] = \
                        self.mask[n0_right:, n1_right:, n2_right:].any()
                    if var is not None:
                        var[-1, -1, -1] = \
                            self.var[n0_right:, n1_right:, n2_right:].sum().data / F2
                if l_left == (newshape[0] - 1) and p_left == 1 \
                        and q_right == (newshape[2] - 1):
                    data[-1, 0, -1] = \
                        self.data[n0_right:, :n1_left, n2_right:].sum() / F
                    mask[-1, 0, -1] = \
                        self.mask[n0_right:, :n1_left, n2_right:].any()
                    if var is not None:
                        var[-1, 0, -1] = \
                            self.var[n0_right:, :n1_left, n2_right:].sum().data / F2

                if p_left == 1 and q_left == 1:
                    d = self.data[n0_left:n0_right, :n1_left, :n2_left]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, 0, 0] = d.data
                    mask[l_left:l_right, 0, 0] = d.mask
                    if var is not None:
                        var[l_left:l_right, 0, 0] = \
                            self.var[n0_left:n0_right, :n1_left, :n2_left]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1).data / F2
                if l_left == 1 and p_left == 1:
                    d = self.data[:n0_left, :n1_left, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[0, 0, q_left:q_right] = d.data
                    mask[0, 0, q_left:q_right] = d.mask
                    if var is not None:
                        var[0, 0, q_left:q_right] = \
                            self.var[:n0_left, :n1_left, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1).data / F2
                if l_left == 1 and q_left == 1:
                    d = self.data[:n0_left, n1_left:n1_right, :n2_left]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[0, p_left:p_right, 0] = d.data
                    mask[0, p_left:p_right, 0] = d.mask
                    if var is not None:
                        var[0, p_left:p_right, 0] = \
                            self.var[:n0_left, n1_left:n1_right, :n2_left]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1).data / F2

                if p_left == 1 and q_right == (newshape[2] - 1):
                    d = self.data[n0_left:n0_right, :n1_left, n2_right:]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, 0, -1] = d.data
                    mask[l_left:l_right, 0, -1] = d.mask
                    if var is not None:
                        var[l_left:l_right, 0, -1] = \
                            self.var[n0_left:n0_right, :n1_left, n2_right:]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1).data / F2
                if l_left == 1 and p_right == (newshape[1] - 1):
                    d = self.data[:n0_left, n1_right:, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[0, -1, q_left:q_right] = d.data
                    mask[0, -1, q_left:q_right] = d.mask
                    if var is not None:
                        var[0, -1, q_left:q_right] = \
                            self.var[:n0_left, n1_right:, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1).data / F2
                if l_left == 1 and q_right == (newshape[2] - 1):
                    d = self.data[:n0_left, n1_left:n1_right, n2_right:]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[0, p_left:p_right, -1] = d.data
                    mask[0, p_left:p_right, -1] = d.mask
                    if var is not None:
                        var[0, p_left:p_right, -1] = \
                            self.var[:n0_left, n1_left:n1_right, n2_right:]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1).data / F2

                if p_right == (newshape[1] - 1) and q_left == 1:
                    d = self.data[n0_left:n0_right, n1_right:, :n2_left]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, -1, 0] = d.data
                    mask[l_left:l_right, -1, 0] = d.mask
                    if var is not None:
                        var[l_left:l_right, -1, 0] = \
                            self.var[n0_left:n0_right, n1_right:, :n2_left]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1).data / F2
                if l_right == (newshape[0] - 1) and p_left == 1:
                    d = self.data[n0_right:, :n1_left, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[-1, 0, q_left:q_right] = d.data
                    mask[-1, 0, q_left:q_right] = d.mask
                    if var is not None:
                        var[-1, 0, q_left:q_right] = \
                            self.var[n0_right:, :n1_left, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1).data / F2
                if l_right == (newshape[0] - 1) and q_left == 1:
                    d = self.data[n0_right:, n1_left:n1_right, :n2_left]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[-1, p_left:p_right, 0] = d.data
                    mask[-1, p_left:p_right, 0] = d.mask
                    if var is not None:
                        var[-1, p_left:p_right, 0] = \
                            self.var[n0_right:, n1_left:n1_right, :n2_left]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1).data / F2

                if p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    d = self.data[n0_left:n0_right, n1_right:, n2_right:]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, -1, -1] = d.data
                    mask[l_left:l_right, -1, -1] = d.mask
                    if var is not None:
                        var[l_left:l_right, -1, -1] = \
                            self.var[n0_left:n0_right, n1_right:, n2_right:]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1).data / F2
                if l_right == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1):
                    d = self.data[n0_right:, n1_right:, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[-1, -1, q_left:q_right] = d.data
                    mask[-1, -1, q_left:q_right] = d.mask
                    if var is not None:
                        var[-1, -1, q_left:q_right] = \
                            self.var[n0_right:, n1_right:, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1).data / F2
                if l_right == (newshape[0] - 1) \
                        and q_right == (newshape[2] - 1):
                    d = self.data[n0_right:, n1_left:n1_right, n2_right:]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[-1, p_left:p_right, -1] = d.data
                    mask[-1, p_left:p_right, -1] = d.mask
                    if var is not None:
                        var[-1, p_left:p_right, -1] = \
                            self.var[n0_right:, n1_left:n1_right, n2_right:]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1).data / F2

                self.wcs = wcs
                self.wave = wave
                self._data = data
                self._mask = mask
                self._var = var
                return None

    def rebin_mean(self, factor, margin='center', flux=False, inplace=False):
        """Shrink the size of the cube by factor.

        Parameters
        ----------
        factor : integer or (integer,integer,integer)
            Factor in z, y and x. Python notation: (nz,ny,nx).
        flux : boolean
            This parameters is used if new size is
            not an integer multiple of the original size.

            If Flux is False, the cube is truncated and the flux
            is not conserved.

            If Flux is True, margins are added to the cube
            to conserve the flux.
        margin : 'center' or 'origin'
            This parameters is used if new size is not
            an integer multiple of the original size.

            In 'center' case, cube is truncated/pixels are added on the
            left and on the right, on the bottom and of the top of the
            cube.

            In 'origin'case, cube is truncated/pixels are added
            at the end along each direction
        inplace : bool
            If False, return a rebinned copy of the image (the default).
            If True, rebin the original image in-place, and return that.

        """
        if is_int(factor):
            factor = (factor, factor, factor)
        factor = np.array(factor)
        factor = np.maximum(factor, [1, 1, 1])
        factor = np.minimum(factor, self.shape)
        res = self if inplace else self.copy()
        res._rebin_mean(factor, margin, flux)
        return res

    def _med_(self, k, p, q, kfactor, pfactor, qfactor):
        return np.ma.median(self.data[k * kfactor:(k + 1) * kfactor,
                                      p * pfactor:(p + 1) * pfactor,
                                      q * qfactor:(q + 1) * qfactor])

    def _rebin_median_(self, factor):
        """Shrink the size of the cube by factor. New size must be an integer
        multiple of the original size.

        Parameter
        ---------
        factor : (integer,integer,integer)
            Factor in z, y and x.  Python notation: (nz,ny,nx)

        """
        assert np.array_equal(np.mod(self.shape, factor), [0, 0, 0])
        self.shape /= np.asarray(factor)
        # data
        grid = np.lib.index_tricks.nd_grid()
        g = grid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]]
        vfunc = np.vectorize(self._med_)
        data = vfunc(g[0], g[1], g[2], factor[0], factor[1], factor[2])
        mask = self._mask.reshape(self.shape[0], factor[0],
                                  self.shape[1], factor[1],
                                  self.shape[2], factor[2])\
            .sum(1).sum(2).sum(3)
        self._data = data
        self._mask = mask
        # variance
        self._var = None
        # coordinates
        self.wcs = self.wcs.rebin(factor[1:])
        self.wave.rebin(factor[0])

    def rebin_median(self, factor, margin='center'):
        """Shrink the size of the cube by factor.

        Parameters
        ----------
        factor : integer or (integer,integer,integer)
            Factor in z, y and x. Python notation: (nz,ny,nx).

        margin : 'center' or 'origin'
            This parameters is used if new size is not an
            integer multiple of the original size.

            In 'center' case, cube is truncated on the left and on the right,
            on the bottom and of the top of the cube.

            In 'origin'case, cube is truncated at the end along each direction

        Returns
        -------
        out : `~mpdaf.obj.Cube`
        """
        if is_int(factor):
            factor = (factor, factor, factor)
        factor = np.array(factor)
        factor = np.maximum(factor, [1, 1, 1])
        factor = np.minimum(factor, self.shape)
        if not np.any(np.mod(self.shape, factor)):
            # new size is an integer multiple of the original size
            res = self.copy()
        else:
            newshape = self.shape // factor
            n = self.shape - newshape * factor

            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0] == 1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0] // 2
                    n0_right = self.shape[0] - n[0] + n0_left
            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1] == 1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1] // 2
                    n1_right = self.shape[1] - n[1] + n1_left
            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2] == 1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2] // 2
                    n2_right = self.shape[2] - n[2] + n2_left

            res = self[n0_left:n0_right, n1_left:n1_right, n2_left:n2_right]

        res._rebin_median_(factor)
        return res

    def loop_spe_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """loops over all spectra to apply a function/method. Returns the
        resulting cube. Multiprocessing is used.

        Parameters
        ----------
        f : function or `~mpdaf.obj.Spectrum` method
            Spectrum method or function that the first argument
            is a spectrum object.
        cpu : integer
            number of CPUs. It is also possible to set
            the mpdaf.CPU global variable.
        verbose : boolean
            if True, progression is printed.
        kargs : kargs
            can be used to set function arguments.

        Returns
        -------
        out : `~mpdaf.obj.Cube` if f returns `~mpdaf.obj.Spectrum`,
        out : `~mpdaf.obj.Image` if f returns a number,
        out : np.array(dtype=object) in others cases.

        """
        from mpdaf import CPU
        if cpu is not None and cpu < multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU < multiprocessing.cpu_count():
            cpu_count = CPU
        else:
            cpu_count = multiprocessing.cpu_count() - 1

        pool = multiprocessing.Pool(processes=cpu_count)

        if isinstance(f, types.MethodType):
            f = f.__name__

        data = self._data
        var = self._var
        mask = self._mask
        header = self.wave.to_header()
        pv, qv = np.meshgrid(list(range(self.shape[1])),
                             list(range(self.shape[2])),
                             sparse=False, indexing='ij')
        pv = pv.ravel()
        qv = qv.ravel()
        if var is None:
            processlist = [((p, q), f, header, data[:, p, q], mask[:, p, q],
                            None, self.unit, kargs)
                           for p, q in zip(pv, qv)]
        else:
            processlist = [((p, q), f, header, data[:, p, q], mask[:, p, q],
                            var[:, p, q], self.unit, kargs)
                           for p, q in zip(pv, qv)]

        processresult = pool.imap_unordered(_process_spe, processlist)
        pool.close()

        if verbose:
            ntasks = len(processlist)
            self._logger.info('loop_spe_multiprocessing (%s): %i tasks', f,
                              ntasks)
            _print_multiprocessing_progress(processresult, ntasks)

        init = True
        for pos, dtype, out in processresult:
            p, q = pos
            if dtype == 'spectrum':
                # f return a Spectrum -> iterator return a cube
                header, data, mask, var, unit = out
                wave = WaveCoord(header, shape=data.shape[0])
                spe = Spectrum(wave=wave, unit=unit, data=data, var=var,
                               mask=mask, copy=False)

                if init:
                    cshape = (data.shape[0], self.shape[1], self.shape[2])
                    if self.var is None:
                        result = Cube(wcs=self.wcs.copy(), wave=wave,
                                      data=np.zeros(cshape), unit=unit)
                    else:
                        result = Cube(wcs=self.wcs.copy(), wave=wave,
                                      data=np.zeros(cshape),
                                      var=np.zeros(cshape), unit=unit)
                    init = False

                result.data_header = pyfits.Header(self.data_header)
                result.primary_header = pyfits.Header(self.primary_header)
                result[:, p, q] = spe

            else:
                if np.isscalar(out[0]):
                    # f returns a number -> iterator returns an image
                    if init:
                        result = Image(wcs=self.wcs.copy(),
                                       data=np.zeros((self.shape[1],
                                                      self.shape[2])),
                                       unit=self.unit)
                        init = False
                    result[p, q] = out[0]
                else:
                    # f returns dtype -> iterator returns an array of dtype
                    if init:
                        result = np.empty((self.shape[1], self.shape[2]),
                                          dtype=type(out[0]))
                        init = False
                    result[p, q] = out[0]

        return result

    def loop_ima_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """loops over all images to apply a function/method. Returns the
        resulting cube. Multiprocessing is used.

        Parameters
        ----------
        f : function or `~mpdaf.obj.Image` method
            Image method or function that the first argument
            is a Image object. It should return an Image object.
        cpu : integer
            number of CPUs. It is also possible to set
        verbose : boolean
            if True, progression is printed.
        kargs : kargs
            can be used to set function arguments.

        Returns
        -------
        out : `~mpdaf.obj.Cube` if f returns `~mpdaf.obj.Image`,
        out : `~mpdaf.obj.Spectrum` if f returns a number,
        out : np.array(dtype=object) in others cases.

        """
        from mpdaf import CPU
        if cpu is not None and cpu < multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU < multiprocessing.cpu_count():
            cpu_count = CPU
        else:
            cpu_count = multiprocessing.cpu_count() - 1

        pool = multiprocessing.Pool(processes=cpu_count)

        if isinstance(f, types.MethodType):
            f = f.__name__

        header = self.wcs.to_header()
        data = self._data
        mask = self._mask
        var = self._var
        if var is None:
            processlist = [(k, f, header, data[k, :, :], mask[k, :, :],
                            None, self.unit, kargs)
                           for k in range(self.shape[0])]
        else:
            processlist = [(k, f, header, data[k, :, :], mask[k, :, :],
                            var[k, :, :], self.unit, kargs)
                           for k in range(self.shape[0])]

        processresult = pool.imap_unordered(_process_ima, processlist)
        pool.close()

        if verbose:
            ntasks = len(processlist)
            self._logger.info('loop_ima_multiprocessing (%s): %i tasks', f,
                              ntasks)
            _print_multiprocessing_progress(processresult, ntasks)

        init = True
        for k, dtype, out in processresult:
            if dtype == 'image':
                # f returns an image -> iterator returns a cube
                header, data, mask, var, unit = out
                if init:
                    wcs = WCS(header, shape=data.shape)
                    cshape = (self.shape[0], data.shape[0], data.shape[1])
                    if self.var is None:
                        result = Cube(wcs=wcs, wave=self.wave.copy(),
                                      data=np.zeros(cshape), unit=unit)
                    else:
                        result = Cube(wcs=wcs, wave=self.wave.copy(),
                                      data=np.zeros(cshape),
                                      var=np.zeros(cshape), unit=unit)
                    init = False
                result._data[k, :, :] = data
                result._mask[k, :, :] = mask
                if self.var is not None:
                    result._var[k, :, :] = var
                result.data_header = pyfits.Header(self.data_header)
                result.primary_header = pyfits.Header(self.primary_header)
            elif dtype == 'spectrum':
                # f return a Spectrum -> iterator return a list of spectra
                header, data, mask, var, unit = out
                wave = WaveCoord(header, shape=data.shape[0])
                spe = Spectrum(wave=wave, unit=unit, data=data, var=var,
                               mask=mask, copy=False)
                if init:
                    result = np.empty(self.shape[0], dtype=type(spe))
                    init = False
                result[k] = spe
            else:
                if np.isscalar(out[0]):
                    # f returns a number -> iterator returns a spectrum
                    if init:
                        result = Spectrum(wave=self.wave.copy(),
                                          data=np.zeros(self.shape[0]),
                                          unit=self.unit)
                        init = False
                    result[k] = out[0]
                else:
                    # f returns dtype -> iterator returns an array of dtype
                    if init:
                        result = np.empty(self.shape[0], dtype=type(out[0]))
                        init = False
                    result[k] = out[0]
        return result

    def get_image(self, wave, is_sum=False, subtract_off=False, margin=10.,
                  fband=3., unit_wave=u.angstrom):
        """Extracts an image from the datacube.

        Parameters
        ----------
        wave : (float, float)
            (lbda1,lbda2) interval of wavelength in angstrom.
        unit_wave : `astropy.units.Unit`
            wavelengths unit (angstrom by default).
            If None, inputs are in pixels
        is_sum : boolean
            if True the sum is computes, otherwise this is the average.
        subtract_off : boolean
            If True, subtracting off nearby data.
            The method computes the subtracted flux by using the algorithm
            from Jarle Brinchmann (jarle@strw.leidenuniv.nl)::

                # if is_sum is False
                sub_flux = mean(flux[lbda1-margin-fband*(lbda2-lbda1)/2: lbda1-margin] +
                                flux[lbda2+margin: lbda2+margin+fband*(lbda2-lbda1)/2])
                # or if is_sum is True:
                sub_flux = sum(flux[lbda1-margin-fband*(lbda2-lbda1)/2: lbda1-margin] +
                                flux[lbda2+margin: lbda2+margin+fband*(lbda2-lbda1)/2]) /fband

        margin : float
            This off-band is offseted by margin wrt narrow-band limit.
        fband : float
            The size of the off-band is fband*narrow-band width.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        if unit_wave is None:
            k1, k2 = wave
            l1, l2 = self.wave.coord(wave)
        else:
            l1, l2 = wave
            k1, k2 = self.wave.pixel(wave, unit=unit_wave)
        if k1 - int(k1) > 0.01:
            k1 = int(k1) + 1
        else:
            k1 = int(k1)
        k2 = int(k2)

        if k1 < 0:
            k1 = 0
        if k2 > (self.shape[0] - 1):
            k2 = self.shape[0] - 1

        msg = 'Computing image for lbda %g-%g [%d-%d]' % (l1, l2, k1, k2 + 1)
        self._logger.debug(msg)
        if is_sum:
            ima = self[k1:k2 + 1, :, :].sum(axis=0)
        else:
            ima = self[k1:k2 + 1, :, :].mean(axis=0)

        if subtract_off:
            if unit_wave is not None:
                margin /= self.wave.get_step(unit=unit_wave)
            dl = (k2 + 1 - k1) * fband
            lbdas = np.arange(self.shape[0], dtype=float)
            is_off = np.where(((lbdas < k1 - margin) &
                               (lbdas > k1 - margin - dl / 2)) |
                              ((lbdas > k2 + margin) &
                               (lbdas < k2 + margin + dl / 2)))
            if is_sum:
                off_im = self[is_off[0], :, :].sum(axis=0) / (
                    1.0 * len(is_off[0]) * fband / dl)
            else:
                off_im = self[is_off[0], :, :].mean(axis=0)
            ima.data -= off_im.data
            if ima._var is not None:
                ima._var += off_im._var

        # add input in header
        if unit_wave is None:
            unit = 'pix'
        else:
            unit = str(unit_wave)
        if self.filename is None:
            f = ''
        else:
            f = os.path.basename(self.filename)
        add_mpdaf_method_keywords(ima.primary_header,
                                  "cube.get_image",
                                  ['cube', 'lbda1', 'lbda2', 'is_sum',
                                   'subtract_off', 'margin', 'fband'],
                                  [f, l1, l2,
                                   is_sum, subtract_off, margin, fband],
                                  ['cube',
                                   'min wavelength (%s)' % str(unit),
                                   'max wavelength (%s)' % str(unit),
                                   'sum/average',
                                   'subtracting off nearby data',
                                   'off-band margin',
                                   'off_band size'])

        return ima

    def bandpass_image(self, wavelengths, sensitivities, unit_wave=u.angstrom):

        """Given a cube of images versus wavelength and the bandpass
        filter-curve of a wide-band monochromatic instrument, extract
        an image from the cube that has the spectral response of the
        monochromatic instrument.

        For example, this can be used to create a MUSE image that has
        the same spectral characteristics as an HST image. The MUSE
        image can then be compared to the HST image without having to
        worry about any differences caused by different spectral
        sensitivities.

        Note that the bandpass of the monochromatic instrument must be
        fully encompassed by the wavelength coverage of the cube.

        For each channel n of the cube, the filter-curve is integrated
        over the width of that channel to obtain a weight, w[n]. The
        output image is then given by the following weighted mean:

        output_image = sum(w[n] * cube_image[n]) / sum(w[n])

        In practice, to accomodate masked pixels, the w[n] array is
        expanded into a cube w[n,y,x], and the weights of individual
        masked pixels in the cube are zeroed before the above equation
        is applied.

        Parameters
        ----------
        wavelengths : numpy.ndarray
            An array of the wavelengths of the filter curve,
            listed in ascending order of wavelength. The lowest
            and highest wavelengths must be within the range of
            wavelengths of the data cube. Outside the listed
            wavelengths the filter-curve is assumed to be zero.
        sensitivities : numpy.ndarray
            The relative flux sensitivities at the wavelengths
            in the wavelengths array. These sensititivies will be
            normalized, so only their relative values are important.
        unit_wave : `astropy.units.Unit`
            The units used in the array of wavelengths. The default is
            angstroms. To specify pixel units, pass None.

        Returns
        -------
        out : `~mpdaf.obj.Image`
            An image formed from the filter-weighted sum or mean
            of all of the channels in the cube.

        """

        # Where needed, convert the wavelengths and sensitivities
        # sequences into numpy arrays.

        wavelengths = np.asarray(wavelengths, dtype=float)
        sensitivities = np.asarray(sensitivities, dtype=float)

        # The sensititivities as wavelengths arrays must be one
        # dimensionsal and have the same length.

        if (wavelengths.ndim != 1 or sensitivities.ndim != 1 or
            len(wavelengths) != len(sensitivities)):
            raise ValueError('The wavelengths and sensititivies arguments'
                             ' should be 1D arrays of equal length')

        # Convert the array of wavelengths to floating point pixel indexes.

        if unit_wave is None:
            pixels = wavelengths.copy()
        else:
            pixels = self.wave.pixel(wavelengths, unit=unit_wave)

        # Obtain the range of indexes along the wavelength axis that
        # encompass the wavelengths in the cube, remembering that
        # integer values returned by wave.pixel() correspond to pixel
        # centers.

        kmin = int(np.floor(0.5 + pixels[0]))
        kmax = int(np.floor(0.5 + pixels[-1]))

        # The filter-curve must be fully encompassed by the wavelength range
        # of the cube.

        if kmin < 0 or kmax > self.shape[0] - 1:
            raise ValueError('The bandpass exceeds the wavelength coverage of'
                             ' the cube.')

        # Obtain a cubic interpolator of the bandpass curve.

        spline = interpolate.interp1d(x=pixels, y=sensitivities, kind='cubic')

        # Compute weights to give for each channel of the cube between
        # kmin and kmax by integrating the spline interpolation of the
        # bandpass curve over the wavelength range of each pixel and
        # dividing this by the sum of the weights.

        k = np.arange(kmin, kmax+1, dtype=int)
        w = np.empty((kmax+1 - kmin))

        # Integrate the bandpass over the range of each spectral pixel
        # to determine the weights of each pixel. For the moment skip
        # the first and last pixels, which need special treatment.
        # Integer pixel indexes refer to the centers of pixels,
        # so for integer pixel index k, we need to integrate from
        # k-0.5 to k+0.5.

        for k in range(kmin+1, kmax):
            w[k-kmin], err = integrate.quad(spline, k-0.5, k+0.5)

        # Start the integration of the weight of the first channel
        # from the lower limit of the bandpass.

        w[0], err = integrate.quad(spline, pixels[0], kmin + 0.5)

        # End the integration of the weight of the final channel
        # at the upper limit of the bandpass.

        w[-1], err = integrate.quad(spline, kmax - 0.5, pixels[-1])

        # Normalize the weights.

        w /= w.sum()

        # Create a sub-cube of the selected channels.

        subcube = self[kmin:kmax+1, :, :]

        # To accomodate masked pixels, create a cube of the above
        # weights, but with masked pixels given zero weight.

        if subcube._mask is ma.nomask:
            wcube = np.ones(subcube.shape)
        else:
            wcube = np.ones(subcube.shape) * ~subcube._mask
        wcube = np.ones(subcube.shape) * w[:,np.newaxis,np.newaxis]

        # Get an image which is the sum of the weights along the spectral
        # axis.

        wsum = wcube.sum(axis=0)

        # The output image is the weighted mean of the selected
        # channels. For each map pixel perform the following
        # calculation over spectral channels, k.
        #
        #  mean = sum(weights[k] * data[k]) / sum(weights[k]

        data = np.ma.sum(subcube.data * wcube, axis=0) / wsum

        # The variance of a weighted means is:
        #
        #  var = sum(weights[k]**2 * var[k]) / (sum(weights[k]))**2

        if subcube._var is not None:
            var = np.ma.sum(subcube.var * wcube**2) / wsum**2
        else:
            var = None

        return Image.new_from_obj(subcube, data=data, var=var)

    def subcube(self, center, size, lbda=None, unit_center=u.deg,
                unit_size=u.arcsec, unit_wave=u.angstrom):
        """Extracts a sub-cube around a position.

        Parameters
        ----------
        center : (float,float)
            Center (y, x) of the aperture.
        size : float
            The size to extract. It corresponds to the size along the delta
            axis and the image is square.
        lbda : (float, float) or None
            If not None, tuple giving the wavelength range.
        unit_center : `astropy.units.Unit`
            Type of the center coordinates (degrees by default)
        unit_size : `astropy.units.Unit`
            unit of the size value (arcseconds by default)
        unit_wave : `astropy.units.Unit`
            Wavelengths unit (angstrom by default)
            If None, inputs are in pixels

        Returns
        -------
        out : `~mpdaf.obj.Cube`

        """
        if size <= 0:
            return None

        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]
        else:
            center = np.array(center)
        if unit_size is not None:
            size = size / self.wcs.get_step(unit=unit_size)[0]
        radius = size / 2.

        if np.any((center - radius + 0.5).astype(int) > self.shape[1:]) or \
                np.any((center + radius + 0.5).astype(int) < 0):
            raise ValueError('Region is outside of the cube limits.')

        size = int(size + 0.5)
        i, j = (center - radius + 0.5).astype(int)
        ny, nx = self.shape[1:]
        imin, jmin = np.maximum(np.minimum([i, j], [ny - 1, nx - 1]), [0, 0])
        imax, jmax = np.maximum(np.minimum([i + size, j + size],
                                           [ny - 1, nx - 1]), [0, 0])
        i0, j0 = - np.minimum([i, j], [0, 0])

        slin = [slice(None), slice(imin, imax), slice(jmin, jmax)]
        slout = [slice(None), slice(i0, i0 + imax - imin),
                 slice(j0, j0 + jmax - jmin)]

        if lbda is not None:
            lmin, lmax = lbda
            if unit_wave is None:
                kmin = int(lmin + 0.5)
                kmax = int(lmax + 0.5)
            else:
                kmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)
                kmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave) + 1
            nk = kmax - kmin
            wave = self.wave[kmin:kmax]
            slin[0] = slice(kmin, kmax)
        else:
            nk = self.shape[0]
            wave = self.wave

        subcub = self[slin]
        var = None
        data = np.ma.empty((nk, size, size))
        data[:] = np.nan
        data[slout] = subcub.data

        if subcub.var is not None:
            var = np.empty((nk, size, size))
            var[:] = np.nan
            var[slout] = subcub.var

        wcs = subcub.wcs
        wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + j0)
        wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + i0)
        wcs.set_naxis1(size)
        wcs.set_naxis2(size)

        return Cube(wcs=wcs, wave=wave, unit=self.unit, copy=False,
                    data=np.ma.masked_invalid(data), var=var,
                    data_header=pyfits.Header(self.data_header),
                    primary_header=pyfits.Header(self.primary_header),
                    filename=self.filename)

    def subcube_circle_aperture(self, center, radius, unit_center=u.deg,
                                unit_radius=u.arcsec):
        """Extracts a sub-cube from an circle aperture of fixed radius.

        Pixels outside the circle are masked.

        Parameters
        ----------
        center : (float,float)
            Center of the aperture (y,x)
        radius : float
            Radius of the aperture. It corresponds to the radius
            along the delta axis and the image is square.
        unit_center : `astropy.units.Unit`
            Type of the center coordinates (degrees by default)
            If None, inputs are in pixels
        unit_radius : `astropy.units.Unit`
            unit of the radius value (arcseconds by default)
            If None, inputs are in pixels

        Returns
        -------
        out : `~mpdaf.obj.Cube`
        """
        subcub = self.subcube(center, radius * 2, unit_center=unit_center,
                              unit_size=unit_radius)
        if unit_center is None:
            center = np.array(center)
            center -= (subcub.get_start() - self.get_start())[1:]
        subcub.mask_region(center, radius, inside=False,
                           unit_center=unit_center, unit_radius=unit_radius)
        return subcub

    def aperture(self, center, radius, unit_center=u.deg,
                 unit_radius=u.arcsec):
        """Extracts a spectrum from an circle aperture of fixed radius.

        Parameters
        ----------
        center : (float,float)
            Center of the aperture (y,x).
        radius : float
            Radius of the aperture in arcsec.
            If None, spectrum at nearest pixel is returned
        unit_center : `astropy.units.Unit`
            Type of the center coordinates (degrees by default)
            If None, inputs are in pixels
        unit_radius : `astropy.units.Unit`
            unit of the radius value (arcseconds by default)
            If None, inputs are in pixels

        Returns
        -------
        out : `~mpdaf.obj.Spectrum`
        """
        if radius > 0:
            cub = self.subcube_circle_aperture(center, radius,
                                               unit_center=unit_center,
                                               unit_radius=unit_radius)
            spec = cub.sum(axis=(1, 2))
            self._logger.info('%d spaxels summed', cub.shape[1] * cub.shape[2])
        else:
            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
            spec = self[:, int(center[0] + 0.5), int(center[1] + 0.5)]
            self._logger.info('returning spectrum at nearest spaxel')
        return spec

    @deprecated('rebin_factor method is deprecated, use rebin_mean instead')
    def rebin_factor(self, factor, margin='center'):
        return self.rebin_mean(factor, margin)

    @deprecated('subcube_aperture method is deprecated: use '
                'subcube_circle_aperture instead')
    def subcube_aperture(self, center, radius, unit_center=u.deg,
                         unit_radius=u.angstrom):
        return self.subcube_circle_aperture(center, radius,
                                            unit_center, unit_radius)


def _process_spe(arglist):
    try:
        pos, f, header, data, mask, var, unit, kargs = arglist
        wave = WaveCoord(header, shape=data.shape[0])
        spe = Spectrum(wave=wave, unit=unit, data=data, var=var, mask=mask)

        if isinstance(f, types.FunctionType):
            out = f(spe, **kargs)
        else:
            out = getattr(spe, f)(**kargs)

        if isinstance(out, Spectrum):
            return pos, 'spectrum', [
                out.wave.to_header(), out.data.data,
                out.data.mask, out.var, out.unit]
        else:
            return pos, 'other', [out]
    except Exception as inst:
        raise type(inst)(str(inst) +
                         '\n The error occurred for the spectrum '
                         '[:,%i,%i]' % (pos[0], pos[1]))


def _process_ima(arglist):
    try:
        k, f, header, data, mask, var, unit, kargs = arglist
        wcs = WCS(header, shape=data.shape)
        obj = Image(wcs=wcs, unit=unit, data=data, var=var, mask=mask)

        if isinstance(f, types.FunctionType):
            out = f(obj, **kargs)
        else:
            out = getattr(obj, f)(**kargs)

        if isinstance(out, Image):
            # f returns an image -> iterator returns a cube
            return k, 'image', [out.wcs.to_header(), out.data.data,
                                out.data.mask, out.var, out.unit]
        elif isinstance(out, Spectrum):
            return k, 'spectrum', [out.wave.to_header(), out.data.data,
                                   out.data.mask, out.var, out.unit]
        else:
            # f returns dtype -> iterator returns an array of dtype
            return k, 'other', [out]
    except Exception as inst:
        raise type(inst)(str(inst) + '\n The error occurred '
                         'for the image [%i,:,:]' % k)
