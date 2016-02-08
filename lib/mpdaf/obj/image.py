"""image.py manages image objects."""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings

import astropy.units as u
from astropy.table import Table, Column
from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path
from scipy import interpolate, signal
from scipy import ndimage as ndi
from scipy.optimize import leastsq

from . import plt_norm, plt_zscale
from .coords import WCS, WaveCoord
from .data import DataArray, is_valid_fits_file
from .objs import is_float, is_int, UnitArray, UnitMaskedArray
from ..tools import deprecated


class Gauss2D(object):

    """This class stores 2D gaussian parameters.

    Attributes
    ----------
    center : (float,float)
        Gaussian center (y,x).
    flux : float
        Gaussian integrated flux.
    fwhm : (float,float)
        Gaussian fwhm (fhwm_y,fwhm_x).
    cont : float
        Continuum value.
    rot : float
        Rotation in degrees.
    peak : float
        Gaussian peak value.
    err_center : (float,float)
        Estimated error on Gaussian center.
    err_flux : float
        Estimated error on Gaussian integrated flux.
    err_fwhm : (float,float)
        Estimated error on Gaussian fwhm.
    err_cont : float
        Estimated error on continuum value.
    err_rot : float
        Estimated error on rotation.
    err_peak : float
        Estimated error on Gaussian peak value.
    ima : :class:`mpdaf.obj.Image`
        Gaussian image

    """

    def __init__(self, center, flux, fwhm, cont, rot, peak, err_center,
                 err_flux, err_fwhm, err_cont, err_rot, err_peak, ima=None):
        self._logger = logging.getLogger(__name__)
        self.center = center
        self.flux = flux
        self.fwhm = fwhm
        self.cont = cont
        self.rot = rot
        self.peak = peak
        self.err_center = err_center
        self.err_flux = err_flux
        self.err_fwhm = err_fwhm
        self.err_cont = err_cont
        self.err_rot = err_rot
        self.err_peak = err_peak
        self.ima = ima

    def copy(self):
        """Copy Gauss2D object in a new one and returns it."""
        return Gauss2D(self.center, self.flux, self.fwhm, self.cont,
                       self.rot, self.peak, self.err_center, self.err_flux,
                       self.err_fwhm, self.err_cont, self.err_rot,
                       self.err_peak)

    def print_param(self):
        """Print Gaussian parameters."""
        msg = 'Gaussian center = (%g,%g) (error:(%g,%g))' \
            % (self.center[0], self.center[1],
               self.err_center[0], self.err_center[1])
        self._logger.info(msg)
        msg = 'Gaussian integrated flux = %g (error:%g)' \
            % (self.flux, self.err_flux)
        self._logger.info(msg)
        msg = 'Gaussian peak value = %g (error:%g)' \
            % (self.peak, self.err_peak)
        self._logger.info(msg)
        msg = 'Gaussian fwhm = (%g,%g) (error:(%g,%g))' \
            % (self.fwhm[0], self.fwhm[1], self.err_fwhm[0], self.err_fwhm[1])
        self._logger.info(msg)
        msg = 'Rotation in degree: %g (error:%g)' % (self.rot, self.err_rot)
        self._logger.info(msg)
        msg = 'Gaussian continuum = %g (error:%g)' \
            % (self.cont, self.err_cont)
        self._logger.info(msg)


class Moffat2D(object):

    """This class stores 2D moffat parameters.

    Attributes
    ----------
    center : (float,float)
        peak center (y,x).
    flux : float
        integrated flux.
    fwhm : (float,float)
        fwhm (fhwm_y,fwhm_x).
    cont : float
        Continuum value.
    n : int
        Atmospheric scattering coefficient.
    rot : float
        Rotation in degrees.
    peak : float
        intensity peak value.
    err_center : (float,float)
        Estimated error on center.
    err_flux : float
        Estimated error on integrated flux.
    err_fwhm : (float,float)
        Estimated error on fwhm.
    err_cont : float
        Estimated error on continuum value.
    err_n : float
        Estimated error on n coefficient.
    err_rot : float
        Estimated error on rotation.
    err_peak : float
        Estimated error on peak value.
    ima : :class:`mpdaf.obj.Image`
        Moffat image

    """

    def __init__(self, center, flux, fwhm, cont, n, rot, peak, err_center,
                 err_flux, err_fwhm, err_cont, err_n, err_rot, err_peak,
                 ima=None):
        self._logger = logging.getLogger(__name__)
        self.center = center
        self.flux = flux
        self.fwhm = fwhm
        self.cont = cont
        self.rot = rot
        self.peak = peak
        self.n = n
        self.err_center = err_center
        self.err_flux = err_flux
        self.err_fwhm = err_fwhm
        self.err_cont = err_cont
        self.err_rot = err_rot
        self.err_peak = err_peak
        self.err_n = err_n
        self.ima = ima

    def copy(self):
        """Return a copy of a Moffat2D object."""
        return Moffat2D(self.center, self.flux, self.fwhm, self.cont,
                        self.n, self.rot, self.peak, self.err_center,
                        self.err_flux, self.err_fwhm, self.err_cont,
                        self.err_n, self.err_rot, self.err_peak)

    def print_param(self):
        """Print Moffat parameters."""
        msg = 'center = (%g,%g) (error:(%g,%g))' \
            % (self.center[0], self.center[1],
               self.err_center[0], self.err_center[1])
        self._logger.info(msg)
        msg = 'integrated flux = %g (error:%g)' % (self.flux, self.err_flux)
        self._logger.info(msg)
        msg = 'peak value = %g (error:%g)' % (self.peak, self.err_peak)
        self._logger.info(msg)
        msg = 'fwhm = (%g,%g) (error:(%g,%g))' \
            % (self.fwhm[0], self.fwhm[1], self.err_fwhm[0], self.err_fwhm[1])
        self._logger.info(msg)
        msg = 'n = %g (error:%g)' % (self.n, self.err_n)
        self._logger.info(msg)
        msg = 'rotation in degree: %g (error:%g)' % (self.rot, self.err_rot)
        self._logger.info(msg)
        msg = 'continuum = %g (error:%g)' % (self.cont, self.err_cont)
        self._logger.info(msg)


class Image(DataArray):

    """Manage image, optionally including a variance and a bad pixel mask.

    Parameters
    ----------
    filename : str
        Possible filename (.fits, .png or .bmp).
    ext : int or (int,int) or string or (string,string)
        Number/name of the data extension or numbers/names
        of the data and variance extensions.
    wcs : :class:`mpdaf.obj.WCS`
        World coordinates.
    unit : str
        Data unit type. u.dimensionless_unscaled by default.
    data : float array
        Array containing the pixel values of the image.  None by default.
    var : float array
        Array containing the variance. None by default.
    copy : boolean
        If true (default), then the data and variance arrays are copied.
    dtype : numpy.dtype
        Type of the data (int, float)

    Attributes
    ----------
    filename : str
        Possible FITS filename.
    primary_header : pyfits.Header
        FITS primary header instance.
    wcs : :class:`mpdaf.obj.WCS`
        World coordinates.
    shape : tuple
        Lengths of data (python notation (nz,ny,nx)).
    data : masked array numpy.ma
        Masked array containing the cube pixel values.
    data_header : pyfits.Header
        FITS data header instance.
    unit : astropy.units
        Physical units of the data values.
    dtype : numpy.dtype
        Type of the data (int, float)
    var : float array
        Array containing the variance.

    """

    _ndim_required = 2
    _has_wcs = True

    def __init__(self, filename=None, ext=None, wcs=None, data=None, var=None,
                 unit=u.dimensionless_unscaled, copy=True, dtype=float,
                 **kwargs):
        self._clicks = None
        self._selector = None

        if filename is not None and not is_valid_fits_file(filename):
            from PIL import Image as PILImage
            im = PILImage.open(filename)
            data = np.array(im.getdata(), dtype=dtype, copy=False)\
                .reshape(im.size[1], im.size[0])
            self.filename = filename
            filename = None

        super(Image, self).__init__(
            filename=filename, ext=ext, wcs=wcs, unit=unit, data=data, var=var,
            copy=copy, dtype=dtype, **kwargs)

    def resize(self):
        """Crops the image to remove any margins that are completely masked."""

        if self.data is not None:

            # How many columns and rows are there in the image?

            nrow = self.data.shape[0]
            ncol = self.data.shape[1]

            # Get the indexes of rows with at least one unmasked pixel.

            used_rows = np.where(np.ma.count_masked(self.data,1) < ncol)[0]

            # Get the indexes of columns with at least one unmasked pixel.

            used_cols = np.where(np.ma.count_masked(self.data,0) < nrow)[0]

            # Create a 2D slice that encloses all used rows and
            # columns. If there are no umasked elements, then arrange
            # to keep the first masked element, so that we are always
            # left with valid 2D array.

            if len(used_rows) > 0 and len(used_cols) > 0:
                item = (slice(min(used_rows), max(used_rows) + 1, None),
                        slice(min(used_cols), max(used_cols) + 1, None))
            else:
                item = (slice(0,1,None),slice(0,1,None))

            # Extract the above 2D slice.

            self.data = self.data[item]
            if self.var is not None:
                self.var = self.var[item]

            # Shift the reference pixel of the world coordinate information
            # to account for any change to the array indexes.

            try:
                self.wcs = self.wcs[item[0], item[1]]
            except:
                self.wcs = None
                self._logger.warning("Wcs not copied")

    def __add__(self, other):
        """Operator +.

        image1 + number = image2 (image2[p,q] = image1[p,q] + number)

        image1 + image2 = image3 (image3[p,q] = image1[p,q] + image2[p,q])

        image + cube1 = cube2 (cube2[k,p,q] = cube1[k,p,q] + image[p,q])

        Parameters
        ----------
        other : number or Image or Cube object.
            If x is Image: Dimensions and world coordinates must be the same.
            If x is Cube: The last two dimensions of the cube must be equal
            to the image dimensions.
            World coordinates in spatial directions must be the same.

        Returns
        -------
        out : :class:`mpdaf.obj.Image` or :class:`mpdaf.obj.Cube`

        """
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            try:
                # image1 + number = image2 (image2[j,i]=image1[j,i]+number)
                res = self.copy()
                res.data = self.data + other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            # coordinates
            if self.wcs is not None and other.wcs is not None \
                    and not self.wcs.isEqual(other.wcs):
                raise IOError('Operation forbidden for images '
                              'with different world coordinates')
            if other.ndim == 2:
                # image1 + image2 = image3 (image3[j,i]=image1[j,i]+image2[j,i])
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')

                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data + other.data
                else:
                    res.data = self.data + UnitMaskedArray(
                        other.data, other.unit, self.unit)
                # variance
                if res.var is not None:
                    if self.var is None:
                        if other.unit == self.unit:
                            res.var = other.var
                        else:
                            res.var = UnitArray(other.var, other.unit**2,
                                                self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res.var = self.var + other.var
                        else:
                            res.var = self.var + UnitArray(
                                other.var, other.unit**2, self.unit**2)

                return res
            else:
                # image + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                res = other.__add__(self)
                return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Operator -.

        image1 - number = image2 (image2[p,q] = image1[p,q] - number)

        image1 - image2 = image3 (image3[p,q] = image1[p,q] - image2[p,q])

        image - cube1 = cube2 (cube2[k,p,q] = image[p,q] - cube1[k,p,q])

        Parameters
        ----------
        other : number or Image or Cube object.
            x is Image: Dimensions and world coordinates must be the same.

            x is Cube: The last two dimensions of the cube must be equal
            to the image dimensions.
            World coordinates in spatial directions must be the same.

        Returns
        -------
        out : :class:`mpdaf.obj.Image` or :class:`mpdaf.obj.Cube`

        """
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            try:
                # image1 + number = image2 (image2[j,i]=image1[j,i]+number)
                res = self.copy()
                res.data = self.data - other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            # coordinates
            if self.wcs is not None and other.wcs is not None \
                    and not self.wcs.isEqual(other.wcs):
                raise IOError('Operation forbidden for images '
                              'with different world coordinates')
            if other.ndim == 2:
                # image1 - image2 = image3 (image3[j,i]=image1[j,i]-image2[j,i])
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')

                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data - other.data
                else:
                    res.data = self.data - UnitMaskedArray(other.data,
                                                           other.unit,
                                                           self.unit)
                # variance
                if other.var is not None:
                    if self.var is None:
                        if other.unit == self.unit:
                            res.var = other.var
                        else:
                            res.var = UnitArray(other.var,
                                                other.unit**2, self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res.var = self.var + other.var
                        else:
                            res.var = self.var + UnitArray(other.var,
                                                           other.unit**2,
                                                           self.unit**2)
                return res
            else:
                # image - cube1 = cube2
                if other.data is None or self.shape[0] != other.shape[1] \
                        or self.shape[1] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = other.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data[np.newaxis, :, :] - other.data
                else:
                    res.data = UnitMaskedArray(self.data[np.newaxis, :, :],
                                               self.unit, other.unit) \
                        - other.data

                # variance
                if self.var is not None:
                    if other.var is None:
                        if other.unit == self.unit:
                            res.var = self.var
                        else:
                            res.var = UnitArray(self.var, self.unit**2,
                                                other.unit**2)
                    else:
                        if other.unit == self.unit:
                            res.var = self.var + other.var
                        else:
                            res.var = other.var + UnitArray(self.var,
                                                            self.unit**2,
                                                            other.unit**2)
                return res

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res.data = other - self.data
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__sub__(self)

    def __mul__(self, other):
        """Operator \*.

        image1 \* number = image2 (image2[p,q] = image1[p,q] \* number)

        image1 \* image2 = image3 (image3[p,q] = image1[p,q] \* image2[p,q])

        image \* cube1 = cube2 (cube2[k,p,q] = image[p,q] \* cube1[k,p,q])

        image \* spectrum = cube (cube[k,p,q] = image[p,q] \* spectrum[k]

        Parameters
        ----------
        other : number or Spectrum or Image or Cube object.
            x is Image: Dimensions and world coordinates must be the same.

            x is Cube: The last two dimensions of the cube must be equal
            to the image dimensions.
            World coordinates in spatial directions must be the same.

        Returns
        -------
        out : Spectrum or Image or Cube object.
        """
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res.data *= other
                if self.var is not None:
                    res.var *= other ** 2
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if other.ndim == 1:
                # image * spectrum = cube
                if other.data is None:
                    raise IOError('Operation forbidden for empty data')
                # data
                data = self.data[np.newaxis, :, :] * \
                    other.data[:, np.newaxis, np.newaxis]
                # The shape of the resulting cube.
                shape = (other.shape[0],self.shape[0],self.shape[1])
                # variance
                if self.var is None and other.var is None:
                    var = None
                elif self.var is None:
                    var = other.var[:, np.newaxis, np.newaxis] \
                        * np.resize(self.data.data**2, shape)
                elif other.var is None:
                    var = np.resize(self.var, shape) \
                        * (other.data.data**2)[:, np.newaxis, np.newaxis]
                else:
                    var = other.var[:, np.newaxis, np.newaxis] \
                        * np.resize(self.data.data**2, shape) \
                        + np.resize(self.var, shape) \
                        * (other.data.data**2)[:, np.newaxis, np.newaxis]

                from .cube import Cube
                return Cube(wave=other.wave, wcs=self.wcs, data=data, var=var,
                            unit=self.unit * other.unit, copy=False)
            else:
                if self.wcs is not None and other.wcs is not None \
                        and not self.wcs.isEqual(other.wcs):
                    raise IOError('Operation forbidden for images '
                                  'with different world coordinates')
                if other.ndim == 2:
                    # image1 * image2 = image3 (image3[j,i]=image1[j,i]*image2[j,i])
                    if other.data is None or self.shape[0] != other.shape[0] \
                            or self.shape[1] != other.shape[1]:
                        raise IOError('Operation forbidden for images '
                                      'with different sizes')
                    res = self.copy()
                    # data
                    res.data = self.data * other.data
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * self.data.data * self.data.data
                    elif other.var is None:
                        res.var = self.var * other.data.data * other.data.data
                    else:
                        res.var = (other.var * self.data.data * self.data.data
                                   + self.var * other.data.data * other.data.data)
                    # unit
                    res.unit = self.unit * other.unit
                    return res
                else:
                    # image * cube1 = cube2
                    res = other.__mul__(self)
                    return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """Operator /.

        image1 / number = image2 (image2[p,q] = image1[p,q] / number)

        image1 / image2 = image3 (image3[p,q] = image1[p,q] / image2[p,q])

        image / cube1 = cube2 (cube2[k,p,q] = image[p,q] / cube1[k,p,q])

        Parameters
        ----------
        other : number or Image or Cube object.
            x is Image: Dimensions and world coordinates must be the same.

            x is Cube: The last two dimensions of the cube must
            be equal to the image dimensions.
            World coordinates in spatial directions must be the same.

        Returns
        -------
        out : :class:`mpdaf.obj.Image` or :class:`mpdaf.obj.Cube`

        """
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res.data /= other
                if self.var is not None:
                    res.var /= other ** 2
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            # coordinates
            if self.wcs is not None and other.wcs is not None \
                    and not self.wcs.isEqual(other.wcs):
                raise IOError('Operation forbidden for images '
                              'with different world coordinates')
            if other.ndim == 2:
                # image1 / image2 = image3 (image3[j,i]=image1[j,i]/image2[j,i])
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden '
                                  'for images with different sizes')
                res = self.copy()
                # data
                res.data = self.data / other.data
                # variance
                if self.var is None and other.var is None:
                    res.var = None
                elif self.var is None:
                    res.var = other.var * self.data.data * self.data.data \
                        / (other.data.data ** 4)
                elif other.var is None:
                    res.var = self.var * other.data.data * other.data.data \
                        / (other.data.data ** 4)
                else:
                    res.var = (other.var * self.data.data * self.data.data +
                               self.var * other.data.data * other.data.data) \
                        / (other.data.data ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res
            else:
                # image / cube1 = cube2
                if other.data is None or self.shape[0] != other.shape[1] \
                        or self.shape[1] != other.shape[2]:
                    raise ValueError('Operation forbidden for images '
                                     'with different sizes')
                # variance
                if self.var is None and other.var is None:
                    var = None
                elif self.var is None:
                    var = other.var * self.data.data[np.newaxis, :, :]\
                        * self.data.data[np.newaxis, :, :] \
                        / (other.data.data ** 4)
                elif other.var is None:
                    var = self.var[np.newaxis, :, :] \
                        * other.data.data * other.data.data \
                        / (other.data.data ** 4)
                else:
                    var = (
                        other.var * self.data.data[np.newaxis, :, :] *
                        self.data.data[np.newaxis, :, :] +
                        self.var[np.newaxis, :, :] *
                        other.data.data * other.data.data
                    ) / (other.data.data ** 4)

                from .cube import Cube
                return Cube(wave=other.wave, unit=self.unit / other.unit,
                            data=self.data[np.newaxis, :, :] / other.data,
                            var=var, copy=False)

    def __rdiv__(self, other):
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res.data = other / res.data
                if self.var is not None:
                    res.var = (self.var * other**2 +
                               other * self.data.data * self.data.data
                               ) / (self.data.data ** 4)
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__div__(self)

    def get_step(self, unit=None):
        """Return the image steps [dy, dx].

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        Returns
        -------
        out : float array
        """
        if self.wcs is not None:
            return self.wcs.get_step(unit)

    def get_range(self, unit=None):
        """Return [ [y_min,x_min], [y_max,x_max] ].

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        Returns
        -------
        out : float array
        """
        if self.wcs is not None:
            return self.wcs.get_range(unit)

    def get_start(self, unit=None):
        """Return [y,x] corresponding to pixel (0,0).

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        Returns
        -------
        out : float array
        """
        if self.wcs is not None:
            return self.wcs.get_start(unit)

    def get_end(self, unit=None):
        """Return [y,x] corresponding to pixel (-1,-1).

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        Returns
        -------
        out : float array
        """
        if self.wcs is not None:
            return self.wcs.get_end(unit)

    def get_rot(self, unit=u.deg):
        """Return the angle of rotation.

        Parameters
        ----------
        unit : astropy.units
            type of the angle coordinate, degree by default

        Returns
        -------
        out : float
        """
        if self.wcs is not None:
            return self.wcs.get_rot(unit)

    def set_wcs(self, wcs):
        """Set the world coordinates.

        Parameters
        ----------
        wcs : :class:`mpdaf.obj.WCS`
            World coordinates.
        """
        self.wcs = wcs.copy()
        self.wcs.set_naxis1(self.shape[1])
        self.wcs.set_naxis2(self.shape[0])
        if wcs.naxis1 != 0 and wcs.naxis2 != 0 and (
                wcs.naxis1 != self.shape[1] or
                wcs.naxis2 != self.shape[0]):
            self._logger.warning('world coordinates and data have not '
                                 'the same dimensions')

    def mask(self, center, radius, unit_center=u.deg, unit_radius=u.arcsec,
             inside=True):
        """Mask values inside/outside the described region.

        Parameters
        ----------
        center : (float,float)
            Center (y,x) of the explored region.
        radius : float or (float,float)
            Radius defined the explored region.
            If radius is float, it defined a circular region.
            If radius is (float,float), it defined a rectangular region.
        unit_center : astropy.units
            type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : astropy.units
            Radius unit. Arcseconds by default (use None for radius in pixels)
        inside : boolean
            If inside is True, pixels inside the described region are masked.
            If inside is False, pixels outside the described region are masked.

        """
        center = np.array(center)
        if is_int(radius) or is_float(radius):
            circular = True
            radius2 = radius * radius
            radius = (radius, radius)
        else:
            circular = False
        radius = np.array(radius)

        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]
        if unit_radius is not None:
            radius = radius / np.abs(self.wcs.get_step(unit=unit_radius))
            radius2 = radius[0] * radius[1]

        imin, jmin = np.maximum(np.minimum((center - radius + 0.5).astype(int),
                                           [self.shape[0] - 1, self.shape[1] - 1]), [0, 0])
        imax, jmax = np.maximum(np.minimum((center + radius + 0.5).astype(int),
                                           [self.shape[0] - 1, self.shape[1] - 1]), [0, 0])
        imax += 1
        jmax += 1

        if inside and not circular:
            self.data.mask[imin:imax, jmin:jmax] = 1
        elif inside and circular:
            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[imin:imax, jmin:jmax],
                              (grid[0] ** 2 + grid[1] ** 2) < radius2)
        elif not inside and circular:
            self.data.mask[0:imin, :] = 1
            self.data.mask[imax:, :] = 1
            self.data.mask[imin:imax, 0:jmin] = 1
            self.data.mask[imin:imax:, jmax:] = 1
            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[imin:imax, jmin:jmax],
                              (grid[0] ** 2 + grid[1] ** 2) > radius2)
        else:
            self.data.mask[0:imin, :] = 1
            self.data.mask[imax:, :] = 1
            self.data.mask[imin:imax, 0:jmin] = 1
            self.data.mask[imin:imax:, jmax:] = 1

    def mask_ellipse(self, center, radius, posangle, unit_center=u.deg,
                     unit_radius=u.arcsec, inside=True):
        """Mask values inside/outside the described region. Uses an elliptical
        shape.

        Parameters
        ----------
        center : (float,float)
            Center (y,x) of the explored region.
        radius : (float,float)
            Radius defined the explored region.  radius is (float,float), it
            defines an elliptical region with semi-major and semi-minor axes.
        posangle : float
            Position angle of the first axis. It is defined in degrees against
            the horizontal (q) axis of the image, counted counterclockwise.
        unit_center : astropy.units
            type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : astropy.units
            Radius unit. Arcseconds by default (use None for radius in pixels)
        inside : boolean
            If inside is True, pixels inside the described region are masked.

        """
        center = np.array(center)
        radius = np.array(radius)

        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]
        if unit_radius is not None:
            radius = radius / np.abs(self.wcs.get_step(unit=unit_radius))

        maxradius = max(radius[0], radius[1])

        imin, jmin = np.maximum(np.minimum((center - maxradius + 0.5).astype(int),
                                           [self.shape[0] - 1, self.shape[1] - 1]), [0, 0])
        imax, jmax = np.maximum(np.minimum((center + maxradius + 0.5).astype(int),
                                           [self.shape[0] - 1, self.shape[1] - 1]), [0, 0])
        imax += 1
        jmax += 1

        cospa = np.cos(np.radians(posangle))
        sinpa = np.sin(np.radians(posangle))

        if inside:
            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[imin:imax, jmin:jmax],
                              ((grid[1] * cospa + grid[0] * sinpa) / radius[0]) ** 2
                              + ((grid[0] * cospa - grid[1] * sinpa)
                                 / radius[1]) ** 2 < 1)
        if not inside:
            self.data.mask[0:imin, :] = 1
            self.data.mask[imax:, :] = 1
            self.data.mask[imin:imax, 0:jmin] = 1
            self.data.mask[imin:imax:, jmax:] = 1
            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[imin:imax, jmin:jmax],
                              ((grid[1] * cospa + grid[0] * sinpa) / radius[0]) ** 2
                              + ((grid[0] * cospa - grid[1] * sinpa)
                                 / radius[1]) ** 2 > 1)

    def mask_polygon(self, poly, unit=u.deg, inside=True):
        """Mask values inside/outside a polygonal region.

        Parameters
        ----------
        poly : (float, float)
            array of (float,float) containing a set of (p,q) or (dec,ra)
            values for the polygon vertices
        pix : astropy.units
            Type of the polygon coordinates (by default in degrees).
            Use unit=None to have polygon coordinates in pixels.
        inside : boolean
            If inside is True, pixels inside the described region are masked.

        """

        if unit is not None:  # convert DEC,RA (deg) values coming from poly into Y,X value (pixels)
            poly = np.array([[self.wcs.sky2pix((val[0], val[1]), unit=unit)[0][0],
                              self.wcs.sky2pix((val[0], val[1]), unit=unit)[0][1]] for val in poly])

        P, Q = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        b = np.dstack([P.ravel(), Q.ravel()])

        polymask = Path(poly)  # use the matplotlib method to create a path wich is the polygon we want to use
        c = polymask.contains_points(b[0])  # go through all pixels in the image to see if there are in the polygon, ouput is a boolean table

        if not inside:  # invert the boolean table to ''mask'' the outside part of the polygon, if it's False I mask the inside part
            c = ~np.array(c)

        c = c.reshape(self.shape[1], self.shape[0])  # convert the boolean table into a matrix
        c = c.T

        self.data.mask = np.logical_or(c, self.data.mask)  # combine the previous mask with the new one
        return poly

    def _truncate(self, y_min, y_max, x_min, x_max, mask=True, unit=u.deg):
        skycrd = [[y_min, x_min], [y_min, x_max],
                  [y_max, x_min], [y_max, x_max]]
        if unit is None:
            pixcrd = np.array(skycrd)
        else:
            pixcrd = self.wcs.sky2pix(skycrd, unit=unit)

        imin = int(np.min(pixcrd[:, 0]) + 0.5)
        if imin < 0:
            imin = 0
        imax = int(np.max(pixcrd[:, 0]) + 0.5) + 1
        if imax > self.shape[0]:
            imax = self.shape[0]
        jmin = int(np.min(pixcrd[:, 1]) + 0.5)
        if jmin < 0:
            jmin = 0
        jmax = int(np.max(pixcrd[:, 1]) + 0.5) + 1
        if jmax > self.shape[1]:
            jmax = self.shape[1]

        subima = self[imin:imax, jmin:jmax]
        self.data = subima.data
        if self.var is not None:
            self.var = subima.var
        self.wcs = subima.wcs

        if mask:
            # mask outside pixels
            grid = np.meshgrid(np.arange(0, self.shape[0]),
                               np.arange(0, self.shape[1]), indexing='ij')
            shape = grid[1].shape
            pixcrd = np.array([[p, q] for p, q in zip(np.ravel(grid[0]),
                                                      np.ravel(grid[1]))])
            if unit is None:
                skycrd = pixcrd
            else:
                skycrd = np.array(self.wcs.pix2sky(pixcrd, unit=unit))
            x = skycrd[:, 1].reshape(shape)
            y = skycrd[:, 0].reshape(shape)
            test_x = np.logical_or(x < x_min, x > x_max)
            test_y = np.logical_or(y < y_min, y > y_max)
            test = np.logical_or(test_x, test_y)
            self.data.mask = np.logical_or(self.data.mask, test)
            self.resize()

    def truncate(self, y_min, y_max, x_min, x_max, mask=True, unit=u.deg):
        """Return truncated image.

        Parameters
        ----------
        y_min : float
            Minimum value of y.
        y_max : float
            Maximum value of y.
        x_min : float
            Minimum value of x.
        x_max : float
            Maximum value of x.
        mask : boolean
            if True, pixels outside [dec_min,dec_max] and [ra_min,ra_max] are
            masked.
        unit : astropy.units
            Type of the coordinates x and y (degrees by default)

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        res._truncate(y_min, y_max, x_min, x_max, mask, unit)
        return res

    def subimage(self, center, size, unit_center=u.deg, unit_size=u.arcsec,
                 minsize=2.0):
        """Extract a sub-image around a given position.

        Parameters
        ----------
        center : (float,float)
            Center (dec, ra) of the aperture.
        size : float
            The size to extract. It corresponds to the size along the delta
            axis and the image is square.
        unit_center : astropy.units
            type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_size : astropy.units
            Size and minsize unit.
            Arcseconds by default (use None for size in pixels)
        minsize : float
            The minimum size of the output image.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        if size > 0:
            if not self.inside(center, unit_center):
                return None

            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
            if unit_size is not None:
                step0 = np.abs(self.wcs.get_step(unit=unit_size)[0])
                size = size / step0
                minsize = minsize / step0
            radius = np.array(size) / 2.

            imin, jmin = np.maximum(np.minimum(
                (center - radius + 0.5).astype(int),
                [self.shape[0] - 1, self.shape[1] - 1]), [0, 0])
            imax, jmax = np.minimum([imin + int(size + 0.5),
                                     jmin + int(size + 0.5)],
                                    [self.shape[0], self.shape[1]])

            if (imax - imin + 1) < minsize or (jmax - jmin + 1) < minsize:
                return None
            return self[imin:imax, jmin:jmax]
        else:
            return None

    def rotate_wcs(self, theta):
        """Rotate WCS coordinates to new orientation given by theta (in place).

        Parameters
        ----------
        theta : float
            Rotation in degree.
        """
        self.wcs.rotate(theta)

    def _rotate(self, theta, interp='no', reshape=False, order=3, pivot=None):
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        mask = np.array(1 - self.data.mask, dtype=bool)

        if pivot is None:
            center_coord = self.wcs.pix2sky([np.array(self.shape) / 2. - 0.5])
            mask_rot = ndi.rotate(mask, -theta, reshape=reshape, order=0)
            data_rot = ndi.rotate(data, -theta, reshape=reshape, order=order)

            shape = np.array(data_rot.shape)
            crpix1 = shape[1] / 2. + 0.5
            crpix2 = shape[0] / 2. + 0.5
        else:
            padX = [self.shape[1] - pivot[0], pivot[0]]
            padY = [self.shape[0] - pivot[1], pivot[1]]

            data_rot = np.pad(data, [padY, padX], 'constant')
            data_rot = ndi.rotate(data_rot, -theta, reshape=reshape, order=order)

            mask_rot = np.pad(mask, [padY, padX], 'constant')
            mask_rot = ndi.rotate(mask_rot, -theta, reshape=reshape, order=0)

            center_coord = self.wcs.pix2sky([pivot])
            shape = np.array(data_rot.shape)

            if reshape is False:
                data_rot = data_rot[padY[0]: -padY[1], padX[0]: -padX[1]]
                mask_rot = mask_rot[padY[0]: -padY[1], padX[0]: -padX[1]]
                crpix1 = shape[1] / 2. + 0.5 - padX[0]
                crpix2 = shape[0] / 2. + 0.5 - padY[0]
            else:
                crpix1 = shape[1] / 2. + 0.5
                crpix2 = shape[0] / 2. + 0.5

        mask_ma = np.ma.make_mask(1 - mask_rot)
        self.data = np.ma.array(data_rot, mask=mask_ma)

        try:
            self.wcs.set_naxis1(self.shape[1])
            self.wcs.set_naxis2(self.shape[0])
            self.wcs.set_crpix1(crpix1)
            self.wcs.set_crpix2(crpix2)
            self.wcs.set_crval1(center_coord[0][1])
            self.wcs.set_crval2(center_coord[0][0])
            self.wcs.rotate(-theta)
        except:
            self.wcs = None

        if reshape:
            self.resize()

    def rotate(self, theta, interp='no', reshape=False, order=3, pivot=None,
               unit=u.deg):
        """Rotate the image using spline interpolation

        Uses :func:`scipy.ndimage.rotate`.

        Parameters
        ----------
        theta : float
            Rotation in degrees.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.
        reshape : boolean
            if reshape is true, the output image is adapted
            so that the input image is contained completely in the output.
            Default is False.
        order : int in the range 0-5.
            The order of the spline interpolation that is used by the rotation
            Default is 3
        pivot : (double, double)
            rotation center
            if None, rotation will be done around the center of the image.
        unit : astropy.units
            type of the pivot coordinates (degrees by default)

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        if pivot is not None and unit is not None:
            pivot = self.wcs.sky2pix([np.array(pivot)], nearest=True, unit=unit)[0]
        res._rotate(theta, interp, reshape, order, pivot)
        return res

    def sum(self, axis=None):
        """Return the sum over the given axis.

        Parameters
        ----------
        axis : None, 0 or 1
            axis = None returns a float
            axis=0 or 1 returns a Spectrum object corresponding to a line or a
            column, other cases return None.

        Returns
        -------
        out : float or Image
        """
        if axis is None:
            return self.data.sum()
        elif axis == 0 or axis == 1:
            # return a spectrum
            data = np.ma.sum(self.data, axis)
            var = None
            if self.var is not None:
                var = np.sum(self.var, axis)
            if axis == 0:
                step = self.wcs.get_step()[1]
                start = self.wcs.get_start()[1]
                cunit = self.wcs.unit
            else:
                step = self.wcs.get_step()[0]
                start = self.wcs.get_start()[0]
                cunit = self.wcs.unit

            from .spectrum import Spectrum
            wave = WaveCoord(crpix=1.0, cdelt=step, crval=start,
                             cunit=cunit, shape=data.shape[0])
            return Spectrum(wave=wave, unit=self.unit, data=data, var=var,
                            copy=False)
        else:
            return None

    def norm(self, typ='flux', value=1.0):
        """Normalize in place total flux to value (default 1).

        Parameters
        ----------
        type : 'flux' | 'sum' | 'max'
            If 'flux',the flux is normalized and
            the pixel area is taken into account.

            If 'sum', the flux is normalized to the sum
            of flux independantly of pixel size.

            If 'max', the flux is normalized so that
            the maximum of intensity will be 'value'.
        value : float
            Normalized value (default 1).
        """
        if typ == 'flux':
            norm = value / (self.get_step().prod() * self.data.sum())
        elif typ == 'sum':
            norm = value / self.data.sum()
        elif typ == 'max':
            norm = value / self.data.max()
        else:
            raise ValueError('Error in type: only flux,sum,max permitted')
        self.data *= norm
        if self.var is not None:
            self.var *= (norm * norm)

    def background(self, niter=3, sigma=3.0):
        """Compute the image background with sigma-clipping.

        Returns the background value and its standard deviation.

        Parameters
        ----------
        niter : int
            Number of iterations.
        sigma : float
            Number of sigma used for the clipping.

        Returns
        -------
        out : 2-dim float array
        """
        tab = self.data.compressed()

        for n in range(niter + 1):
            tab = tab[tab <= (tab.mean() + sigma * tab.std())]
        return np.array([tab.mean(), tab.std()])

    def _struct(self, n):
        struct = np.zeros([n, n])
        for i in range(0, n):
            dist = abs(i - (n / 2))
            struct[i][dist: abs(n - dist)] = 1
        return struct

    def peak_detection(self, nstruct, niter, threshold=None):
        """Return a list of peak locations.

        Parameters
        ----------
        nstruct : int
            Size of the structuring element used for the erosion.
        niter : int
            number of iterations used for the erosion and the dilatation.
        threshold : float
            threshold value. If None, it is initialized with background value

        Returns
        -------
        out : np.array
        """
        # threshold value
        (background, std) = self.background()
        if threshold is None:
            threshold = background + 10 * std

        selec = self.data > threshold
        selec.fill_value = False
        struct = self._struct(nstruct)
        selec = ndi.binary_erosion(selec, structure=struct, iterations=niter)
        selec = ndi.binary_dilation(selec, structure=struct, iterations=niter)
        selec = ndi.binary_fill_holes(selec)
        structure = ndi.generate_binary_structure(2, 2)
        label = ndi.measurements.label(selec, structure)
        pos = ndi.measurements.center_of_mass(self.data, label[0],
                                              np.arange(label[1]) + 1)
        return np.array(pos)

    def peak(self, center=None, radius=0, unit_center=u.deg,
             unit_radius=u.angstrom, dpix=2, background=None, plot=False):
        """Find image peak location.

        Used :func:`scipy.ndimage.measurements.maximum_position` and
        :func:`scipy.ndimage.measurements.center_of_mass`.

        Parameters
        ----------
        center : (float,float)
            Center (y,x) of the explored region.
            If center is None, the full image is explored.
        radius : float or (float,float)
            Radius defined the explored region.
        unit_center : astropy.units
            type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : astropy.units
            Radius unit.
            Arcseconds by default (use None for radius in pixels)
        dpix : int
            Half size of the window (in pixels) to compute the center of
            gravity.
        background : float
            background value. If None, it is computed.
        plot : boolean
            If True, the peak center is overplotted on the image.

        Returns
        -------
        out : dictionary {'y', 'x', 'p', 'q', 'data'}
            containing the peak position and the peak intensity.

        """
        if center is None or radius == 0:
            d = self.data
            imin = 0
            jmin = 0
        else:
            if is_int(radius) or is_float(radius):
                radius = (radius, radius)

            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            if unit_radius is not None:
                radius = radius / np.abs(self.wcs.get_step(unit=unit_radius))

            imin = center[0] - radius[0]
            if imin < 0:
                imin = 0
            imax = center[0] + radius[0] + 1
            jmin = center[1] - radius[1]
            if jmin < 0:
                jmin = 0
            jmax = center[1] + radius[1] + 1

            d = self.data[imin:imax, jmin:jmax]
            if np.shape(d)[0] == 0 or np.shape(d)[1] == 0:
                raise ValueError('Coord area outside image limits')

        ic, jc = ndi.measurements.maximum_position(d)
        if dpix == 0:
            di = 0
            dj = 0
        else:
            if background is None:
                background = self.background()[0]
            di, dj = ndi.measurements.center_of_mass(
                d[max(0, ic - dpix):ic + dpix + 1,
                  max(0, jc - dpix):jc + dpix + 1] - background)
        ic = imin + max(0, ic - dpix) + di
        jc = jmin + max(0, jc - dpix) + dj
        [[dec, ra]] = self.wcs.pix2sky([[ic, jc]])
        maxv = self.data[int(round(ic)), int(round(jc))]
        if plot:
            self._ax.plot(jc, ic, 'r+')
            try:
                _str = 'center (%g,%g) radius (%g,%g) dpix %i peak: %g %g' % \
                    (center[0], center[1], radius[0], radius[1], dpix, jc, ic)
            except:
                _str = 'dpix %i peak: %g %g' % (dpix, ic, jc)
            self._ax.title(_str)

        return {'x': ra, 'y': dec, 'p': ic, 'q': jc, 'data': maxv}

    def fwhm(self, center=None, radius=0, unit_center=u.deg,
             unit_radius=u.angstrom):
        """Compute the fwhm.

        Parameters
        ----------
        center : (float,float)
            Center of the explored region.
            If center is None, the full image is explored.
        radius : float or (float,float)
            Radius defined the explored region.
        unit_center : astropy.units
            type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : astropy.units
            Radius unit.  Arcseconds by default (use None for radius in pixels)

        Returns
        -------
        out : array of float
              [fwhm_y,fwhm_x].
              fwhm is returned in unit_radius (arcseconds by default).

        """
        if center is None or radius == 0:
            sigma = self.moments(unit=unit_radius)
        else:
            if is_int(radius) or is_float(radius):
                radius = (radius, radius)

            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            if unit_radius is not None:
                radius = radius / np.abs(self.wcs.get_step(unit=unit_radius))

            imin = max(0, center[0] - radius[0])
            imax = min(center[0] + radius[0] + 1, self.shape[0])
            jmin = max(0, center[1] - radius[1])
            jmax = min(center[1] + radius[1] + 1, self.shape[1])

            sigma = self[imin:imax, jmin:jmax].moments(unit=unit_radius)

        return sigma * 2. * np.sqrt(2. * np.log(2.0))

    def ee(self, center=None, radius=0, unit_center=u.deg,
           unit_radius=u.angstrom, frac=False, cont=0):
        """Compute ensquared/encircled energy.

        Parameters
        ----------
        center : (float,float)
            Center of the explored region.
            If center is None, the full image is explored.
        radius : float or (float,float)
            Radius defined the explored region.
            If radius is float, it defined a circular region (encircled energy).
            If radius is (float,float), it defined a rectangular region (ensquared energy).
        unit_center : astropy.units
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : astropy.units
            Radius unit. Arcseconds by default (use None for radius in pixels)
        frac : boolean
            If frac is True, result is given relative to the total energy of
            the full image.
        cont : float
            Continuum value.

        Returns
        -------
        out : float
              Ensquared/encircled flux.

        """
        if center is None or radius == 0:
            if frac:
                return 1.
            else:
                return (self.data - cont).sum()
        else:
            if is_int(radius) or is_float(radius):
                circular = True
                radius2 = radius * radius
                radius = (radius, radius)
            else:
                circular = False

            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            if unit_radius is not None:
                radius = radius / np.abs(self.wcs.get_step(unit=unit_radius))
                radius2 = radius[0] * radius[1]

            imin = max(0, center[0] - radius[0])
            imax = min(center[0] + radius[0] + 1, self.shape[0])
            jmin = max(0, center[1] - radius[1])
            jmax = min(center[1] + radius[1] + 1, self.shape[1])
            ima = self[imin:imax, jmin:jmax]

            if circular:
                xaxis = np.arange(ima.shape[0], dtype=np.float) \
                    - ima.shape[0] / 2.
                yaxis = np.arange(ima.shape[1], dtype=np.float) \
                    - ima.shape[1] / 2.
                gridx = np.empty(ima.shape, dtype=np.float)
                gridy = np.empty(ima.shape, dtype=np.float)
                for j in range(ima.shape[1]):
                    gridx[:, j] = xaxis
                for i in range(ima.shape[0]):
                    gridy[i, :] = yaxis
                r2 = gridx * gridx + gridy * gridy
                ksel = np.where(r2 < radius2)
                if frac:
                    return (ima.data[ksel] - cont).sum() \
                        / (self.data - cont).sum()
                else:
                    return (ima.data[ksel] - cont).sum()
            else:
                if frac:
                    return (ima.data - cont).sum() / (self.data - cont).sum()
                else:
                    return (ima.data - cont).sum()

    def eer_curve(self, center=None, unit_center=u.deg, unit_radius=u.arcsec,
                  etot=None, cont=0):
        """Return containing enclosed energy as function of radius.

        The enclosed energy ratio (EER) shows how much light is concentrated
        within a certain radius around the image-center.


        Parameters
        ----------
        center : (float,float)
            Center of the explored region.
            If center is None, center of the image is used.
        unit_center : astropy.units
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : astropy.units
            Radius units (arcseconds by default)/
        etot : float
            Total energy used to comute the ratio.
            If etot is not set, it is computed from the full image.
        cont : float
            Continuum value.

        Returns
        -------
        out : (float array, float array)
              Radius array, EER array
        """
        if center is None:
            i = self.shape[0] / 2
            j = self.shape[1] / 2
        else:
            if unit_center is None:
                i = center[0]
                j = center[1]
            else:
                pixcrd = self.wcs.sky2pix([center[0], center[1]],
                                          nearest=True, unit=unit_center)
                i = pixcrd[0][0]
                j = pixcrd[0][1]

        nmax = min(self.shape[0] - i, self.shape[1] - j, i, j)
        if etot is None:
            etot = (self.data - cont).sum()
        if nmax <= 1:
            raise ValueError('Coord area outside image limits')
        ee = np.empty(nmax)
        for d in range(0, nmax):
            ee[d] = (self.data[i - d:i + d + 1, j - d:j + d + 1] - cont).sum() / etot

        radius = np.arange(0, nmax)
        if unit_radius is not None:
            step = self.get_step(unit=unit_radius)
            radius = radius * step

        return radius, ee

    def ee_size(self, center=None, unit_center=u.deg, etot=None, frac=0.9,
                cont=0, unit_size=u.arcsec):
        """Compute the size of the square centered on (y,x) containing the
        fraction of the energy.

        Parameters
        ----------
        center : (float,float)
            Center (y,x) of the explored region.
            If center is None, center of the image is used.
        unit : astropy.units
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        etot : float
            Total energy used to comute the ratio.
                      If etot is not set, it is computed from the full image.
        frac : float in ]0,1]
            Fraction of energy.
        cont : float
            continuum value
        unit_center : astropy.units
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_size : astropy.units
            Size unit.  Arcseconds by default (use None for sier in pixels).

        Returns
        -------
        out : float array
        """
        if center is None:
            i = self.shape[0] / 2
            j = self.shape[1] / 2
        else:
            if unit_center is None:
                i = center[0]
                j = center[1]
            else:
                pixcrd = self.wcs.sky2pix([[center[0], center[1]]],
                                          unit=unit_center)
                i = int(pixcrd[0][0] + 0.5)
                j = int(pixcrd[0][1] + 0.5)
        nmax = min(self.shape[0] - i, self.shape[1] - j, i, j)
        if etot is None:
            etot = (self.data - cont).sum()

        if nmax <= 1:
            if unit_size is None:
                return np.array([1, 1])
            else:
                return self.get_step(unit_size)
        for d in range(1, nmax):
            ee2 = (self.data[i - d:i + d + 1, j - d:j + d + 1] - cont).sum() / etot
            if ee2 > frac:
                break
        d -= 1
        ee1 = (self.data[i - d:i + d + 1, i - d:i + d + 1] - cont).sum() / etot
        d += (frac - ee1) / (ee2 - ee1)  # interpolate
        d *= 2
        if unit_size is None:
            return np.array([d, d])
        else:
            step = self.get_step(unit_size)
            return np.array([d * step[0], d * step[1]])

    def _interp(self, grid, spline=False):
        """Return the interpolated values corresponding to the grid points.

        Parameters
        ----------
        grid :
            pixel values
        spline : bool
            If False, linear interpolation (uses
            :func:`scipy.interpolate.griddata`), or if True: spline
            interpolation (uses :func:`scipy.interpolate.bisplrep` and
            :func:`scipy.interpolate.bisplev`).

        """
        ksel = np.where(self.data.mask == False)
        x = ksel[0]
        y = ksel[1]
        data = self.data.data[ksel]
        npoints = np.shape(data)[0]

        grid = np.array(grid)
        n = np.shape(grid)[0]

        if spline:
            if self.var is not None:
                weight = np.empty(n, dtype=float)
                for i in range(npoints):
                    weight[i] = 1. / np.sqrt(np.abs(self.var[x[i], y[i]]))
                np.ma.fix_invalid(weight, copy=False, fill_value=0)
            else:
                weight = None

            tck = interpolate.bisplrep(x, y, data, w=weight)
            res = interpolate.bisplev(grid[:, 0], grid[:, 1], tck)
            # res = np.zeros(n,dtype=float)
            # for i in range(n):
            #     res[i] = interpolate.bisplev(grid[i,0],grid[i,1],tck)
            return res
        else:
            # scipy 0.9 griddata
            # interpolate.interp2d segfaults when there are too many data points
            # f = interpolate.interp2d(x, y, data)
            points = np.empty((npoints, 2), dtype=float)
            points[:, 0] = ksel[0]
            points[:, 1] = ksel[1]
            res = interpolate.griddata(points, data,
                                       (grid[:, 0], grid[:, 1]),
                                       method='linear')
            # res = np.zeros(n,dtype=float)
            # for i in range(n):
            #     res[i] = interpolate.griddata(points, data, (grid[i,0],grid[i,1]), method='linear')
            return res

    def _interp_data(self, spline=False):
        """Return data array with interpolated values for masked pixels.

        Parameters
        ----------
        spline : bool
            False: bilinear interpolation (it uses
            :func:`scipy.interpolate.griddata`), True: spline interpolation (it
            uses :func:`scipy.interpolate.bisplrep` and
            :func:`scipy.interpolate.bisplev`).

        """
        if np.ma.count_masked(self.data) == 0:
            return self.data.data
        else:
            ksel = np.where(self.data.mask == True)
            data = self.data.data.__copy__()
            data[ksel] = self._interp(ksel, spline)
            return data

    def moments(self, unit=u.arcsec):
        """Return [width_y, width_x] first moments of the 2D gaussian.

        Parameters
        ----------
        unit : astropy.units
            Unit of the returned moments (arcseconds by default).
            If None, moments will be in pixels

        Returns
        -------
        out : float array

        """
        total = np.abs(self.data).sum()
        P, Q = np.indices(self.data.shape)
        # python convention: reverse x,y numpy.indices
        p = np.argmax((Q * np.abs(self.data)).sum(axis=1) / total)
        q = np.argmax((P * np.abs(self.data)).sum(axis=0) / total)
        col = self.data[int(p), :]
        width_q = np.sqrt(np.abs((np.arange(col.size) - p) * col).sum() /
                          np.abs(col).sum())
        row = self.data[:, int(q)]
        width_p = np.sqrt(np.abs((np.arange(row.size) - q) * row).sum() /
                          np.abs(row).sum())
        mom = np.array([width_p, width_q])
        if unit is not None:
            dy, dx = self.wcs.get_step(unit=unit)
            mom[0] = mom[0] * np.abs(dy)
            mom[1] = mom[1] * np.abs(dx)
        return mom

    def gauss_fit(self, pos_min=None, pos_max=None, center=None, flux=None,
                  fwhm=None, circular=False, cont=0, fit_back=True, rot=0,
                  peak=False, factor=1, weight=True, plot=False,
                  unit_center=u.deg, unit_fwhm=u.arcsec, maxiter=100,
                  verbose=True, full_output=0):
        """Perform Gaussian fit on image.

        Parameters
        ----------
        pos_min : (float,float)
            Minimum y and x values. Their unit is given by the unit_center
            parameter (degrees by default).
        pos_max : (float,float)
            Maximum y and x values. Their unit is given by the unit_center
            parameter (degrees by default).
        center : (float,float)
            Initial gaussian center (y_peak,x_peak) If None it is estimated.
            The unit is given by the unit_center parameter (degrees by
            default).
        flux : float
            Initial integrated gaussian flux or gaussian peak value if peak is
            True.  If None, peak value is estimated.
        fwhm : (float,float)
            Initial gaussian fwhm (fwhm_y,fwhm_x). If None, they are estimated.
            The unit is given by the unit_fwhm parameter (arcseconds by
            default).
        circular : boolean
            True: circular gaussian, False: elliptical gaussian
        cont : float
            continuum value, 0 by default.
        fit_back : boolean
            False: continuum value is fixed,
            True: continuum value is a fit parameter.
        rot : float
            Initial rotation in degree.
            If None, rotation is fixed to 0.
        peak : boolean
            If true, flux contains a gaussian peak value.
        factor : int
            If factor<=1, gaussian value is computed in the center of each
            pixel. If factor>1, for each pixel, gaussian value is the sum of
            the gaussian values on the factor*factor pixels divided by the
            pixel area.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        unit_center : astropy.units
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_fwhm : astropy.units
            FWHM unit. Arcseconds by default (use None for radius in pixels)
        maxiter : int
            The maximum number of iterations during the sum of square
            minimization.
        plot : boolean
            If True, the gaussian is plotted.
        verbose : boolean
            If True, the Gaussian parameters are printed at the end of the
            method.
        full_output : int
            non-zero to return a :class:`mpdaf.obj.Gauss2D` object containing
            the gauss image

        Returns
        -------
        out : :class:`mpdaf.obj.Gauss2D`

        """
        pmin, qmin = 0, 0
        pmax, qmax = self.shape

        if unit_center is None:
            if pos_min is not None:
                pmin, qmin = pos_min
            if pos_max is not None:
                pmax, qmax = pos_max
        else:
            if pos_min is not None:
                pixcrd = self.wcs.sky2pix(pos_min, unit=unit_center)
                pmin = pixcrd[0][0]
                qmin = pixcrd[0][1]
            if pos_max is not None:
                pixcrd = self.wcs.sky2pix(pos_max, unit=unit_center)
                pmax = pixcrd[0][0]
                qmax = pixcrd[0][1]
            if pmin > pmax:
                pmin, pmax = pmax, pmin
            if qmin > qmax:
                qmin, qmax = qmax, qmin

        pmin = max(0, pmin)
        qmin = max(0, qmin)
        ima = self[pmin:pmax, qmin:qmax]

        N = ima.data.count()
        if N == 0:
            raise ValueError('empty sub-image')
        data = ima.data.compressed()
        p, q = np.where(ima.data.mask == False)

        # weight
        if ima.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(ima.var[p, q]))
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(N)

        # initial gaussian peak position
        if center is None:
            center = np.array(np.unravel_index(ima.data.argmax(), ima.shape))
        elif unit_center is not None:
            center = ima.wcs.sky2pix(center, unit=unit_center)[0]
        else:
            center = np.array(center)
            center[0] -= pmin
            center[1] -= qmin

        # initial moment value
        if fwhm is None:
            width = ima.moments(unit=None)
            fwhm = width * 2. * np.sqrt(2. * np.log(2.0))
        else:
            if unit_fwhm is not None:
                fwhm = np.array(fwhm)
                fwhm = fwhm / np.abs(self.wcs.get_step(unit=unit_fwhm))
            width = np.array(fwhm) / (2. * np.sqrt(2. * np.log(2.0)))

        # initial gaussian integrated flux
        if flux is None:
            peak = ima.data.data[center[0], center[1]] - cont
        elif peak is True:
            peak = flux - cont

        flux = peak * np.sqrt(2 * np.pi * (width[0] ** 2)) \
            * np.sqrt(2 * np.pi * (width[1] ** 2))

        if circular:
            rot = None
            if not fit_back:
                # 2d gaussian function
                gaussfit = lambda v, p, q: \
                    cont + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                    * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(q - v[3]) ** 2 / (2 * v[2] ** 2))
                # inital guesses for Gaussian Fit
                v0 = [flux, center[0], width[0], center[1]]
            else:
                # 2d gaussian function
                gaussfit = lambda v, p, q: \
                    v[4] + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                    * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(q - v[3]) ** 2 / (2 * v[2] ** 2))
                # inital guesses for Gaussian Fit
                v0 = [flux, center[0], width[0], center[1], cont]
        else:
            if not fit_back:
                if rot is None:
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        cont + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-(q - v[3]) ** 2 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1], width[1]]
                else:
                    # rotation angle in rad
                    rot = np.pi * rot / 180.0
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        cont + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-((p - v[1]) * np.cos(v[5])
                                   - (q - v[3]) * np.sin(v[5])) ** 2
                                 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-((p - v[1]) * np.sin(v[5])
                                   + (q - v[3]) * np.cos(v[5])) ** 2
                                 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1], width[1], rot]
            else:
                if rot is None:
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        v[5] + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-(q - v[3]) ** 2 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1],
                          width[1], cont]
                else:
                    # r otation angle in rad
                    rot = np.pi * rot / 180.0
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        v[6] + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-((p - v[1]) * np.cos(v[5])
                                   - (q - v[3]) * np.sin(v[5])) ** 2
                                 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-((p - v[1]) * np.sin(v[5])
                                   + (q - v[3]) * np.cos(v[5])) ** 2
                                 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1],
                          width[1], rot, cont]

        # Minimize the sum of squares
        if factor > 1:
            factor = int(factor)
            deci = np.ones((factor, factor)) \
                * np.arange(factor)[:, np.newaxis] \
                / float(factor) + 1. / float(factor * 2) - 0.5
            fp = (p[:, np.newaxis] + deci.ravel()[np.newaxis, :]).ravel()
            fq = (q[:, np.newaxis] + deci.T.ravel()[np.newaxis, :]).ravel()
            pixcrd = np.array(zip(fp, fq))

            e_gauss_fit = lambda v, p, q, data, w: \
                w * (((gaussfit(v, p, q)).reshape(N, factor * factor).sum(1)
                      / factor / factor).T.ravel() - data)
            v, covar, info, mesg, success = \
                leastsq(e_gauss_fit, v0[:],
                        args=(pixcrd[:, 0], pixcrd[:, 1], data, wght),
                        maxfev=maxiter, full_output=1)
        else:
            e_gauss_fit = lambda v, p, q, data, w : \
                w * (gaussfit(v, p, q) - data)
            v, covar, info, mesg, success = \
                leastsq(e_gauss_fit, v0[:], args=(p, q, data, wght),
                        maxfev=maxiter, full_output=1)

        if success != 1:
            self._logger.info(mesg)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i]))
                            * np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = None

        # center in pixel in the input image
        v[1] += int(pmin)
        v[3] += int(qmin)

        # plot
        # ne fonctionne pas si colorbar
        if plot:
            pp = np.arange(pmin, pmax, float(pmax - pmin) / 100)
            qq = np.arange(qmin, qmax, float(qmax - qmin) / 100)
            ff = np.empty((np.shape(pp)[0], np.shape(qq)[0]))
            for i in range(np.shape(pp)[0]):
                ff[i, :] = gaussfit(v, pp[i], qq[:])
            self._ax.contour(qq, pp, ff, 5)

        # Gauss2D object in pixels
        flux = v[0]
        p_peak = v[1]
        q_peak = v[3]
        if circular:
            if fit_back:
                cont = v[4]
            p_width = np.abs(v[2])
            q_width = p_width
            rot = 0
        else:
            if fit_back:
                if rot is None:
                    cont = v[5]
                else:
                    cont = v[6]
            if rot is None:
                p_width = np.abs(v[2])
                q_width = np.abs(v[4])
                rot = 0
            else:
                if np.abs(v[2]) > np.abs(v[4]):
                    p_width = np.abs(v[2])
                    q_width = np.abs(v[4])
                    rot = (v[5] * 180.0 / np.pi) % 180
                else:
                    p_width = np.abs(v[4])
                    q_width = np.abs(v[2])
                    rot = (v[5] * 180.0 / np.pi + 90) % 180
        p_fwhm = p_width * 2 * np.sqrt(2 * np.log(2))
        q_fwhm = q_width * 2 * np.sqrt(2 * np.log(2))
        peak = flux / np.sqrt(2 * np.pi * (p_width ** 2)) \
            / np.sqrt(2 * np.pi * (q_width ** 2))
        # error
        if err is not None:
            err_flux = err[0]
            err_p_peak = err[1]
            err_q_peak = err[3]
            if circular:
                if fit_back:
                    err_cont = err[4]
                else:
                    err_cont = 0
                err_p_width = np.abs(err[2])
                err_q_width = err_p_width
                err_rot = 0
            else:
                if fit_back:
                    try:
                        err_cont = err[6]
                    except:
                        err_cont = err[5]
                else:
                    err_cont = 0

                if np.abs(v[2]) > np.abs(v[4]) or rot == 0:
                    err_p_width = np.abs(err[2])
                    err_q_width = np.abs(err[4])
                else:
                    err_p_width = np.abs(err[4])
                    err_q_width = np.abs(err[2])

                try:
                    err_rot = err[4] * 180.0 / np.pi
                except:
                    err_rot = 0
            err_p_fwhm = err_p_width * 2 * np.sqrt(2 * np.log(2))
            err_q_fwhm = err_q_width * 2 * np.sqrt(2 * np.log(2))
            err_peak = (err_flux * p_width * q_width - flux
                        * (err_p_width * q_width + err_q_width * p_width)) \
                / (2 * np.pi * p_width * p_width * q_width * q_width)
        else:
            err_flux = np.NAN
            err_p_peak = np.NAN
            err_p_width = np.NAN
            err_p_fwhm = np.NAN
            err_q_peak = np.NAN
            err_q_width = np.NAN
            err_q_fwhm = np.NAN
            err_rot = np.NAN
            err_peak = np.NAN
            err_cont = np.NAN

        if unit_center is not None:
            # Gauss2D object in degrees/arcseconds
            center = self.wcs.pix2sky([p_peak, q_peak], unit=unit_center)[0]

            err_center = np.array([err_p_peak, err_q_peak]) * np.abs(self.wcs.get_step(unit=unit_center))
        else:
            center = (p_peak, q_peak)
            err_center = (err_p_peak, err_q_peak)

        if unit_fwhm is not None:
            step = self.wcs.get_step(unit=unit_fwhm)
            fwhm = np.array([p_fwhm, q_fwhm]) * np.abs(step)
            err_fwhm = np.array([err_p_fwhm, err_q_fwhm]) * np.abs(step)
        else:
            fwhm = (p_fwhm, q_fwhm)
            err_fwhm = (err_p_fwhm, err_q_fwhm)

        gauss = Gauss2D(center, flux, fwhm, cont, rot,
                        peak, err_center, err_flux, err_fwhm,
                        err_cont, err_rot, err_peak)

        if verbose:
            gauss.print_param()
        if full_output != 0:
            ima = gauss_image(shape=self.shape, wcs=self.wcs, gauss=gauss)
            gauss.ima = ima
        return gauss

    def moffat_fit(self, pos_min=None, pos_max=None, center=None, fwhm=None,
                   flux=None, n=2.0, circular=False, cont=0, fit_back=True,
                   rot=0, peak=False, factor=1, weight=True, plot=False,
                   unit_center=u.deg, unit_fwhm=u.arcsec,
                   verbose=True, full_output=0, fit_n=True, maxiter=100):
        """Perform moffat fit on image.

        Parameters
        ----------

        pos_min : (float,float)
            Minimum y and x values. Their unit is given by the unit_center
            parameter (degrees by default).
        pos_max : (float,float)
            Maximum y and x values. Their unit is given by the unit_center
            parameter (degrees by default).
        center : (float,float)
            Initial moffat center (y_peak,x_peak). If None it is estimated.
            The unit is given by the unit_center parameter (degrees by
            default).
        flux : float
            Initial integrated gaussian flux or gaussian peak value if peak is
            True.  If None, peak value is estimated.
        fwhm : (float,float)
            Initial gaussian fwhm (fwhm_y,fwhm_x). If None, they are estimated.
            Their unit is given by the unit_fwhm parameter (arcseconds by
            default).
        n : int
            Initial atmospheric scattering coefficient.
        circular : boolean
            True: circular moffat, False: elliptical moffat
        cont : float
            continuum value, 0 by default.
        fit_back : boolean
            False: continuum value is fixed,
            True: continuum value is a fit parameter.
        rot : float
            Initial angle position in degree.
        peak : boolean
            If true, flux contains a gaussian peak value.
        factor : int
            If factor<=1, gaussian is computed in the center of each pixel.
            If factor>1, for each pixel, gaussian value is the sum of the
            gaussian values on the factor*factor pixels divided by the pixel
            area.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        plot : boolean
            If True, the gaussian is plotted.
        unit_center : astropy.units
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_fwhm : astropy.units
            FWHM unit. Arcseconds by default (use None for radius in pixels)
        full_output : int
            non-zero to return a :class:`mpdaf.obj.Moffat2D`
            object containing the moffat image
        fit_n : boolean
            False: n value is fixed,
            True: n value is a fit parameter.
        maxiter : int
            The maximum number of iterations during the sum of square
            minimization.

        Returns
        -------
        out : :class:`mpdaf.obj.Moffat2D`

        """
        if unit_center is None:
            if pos_min is None:
                pmin = 0
                qmin = 0
            else:
                pmin = pos_min[0]
                qmin = pos_min[1]
            if pos_max is None:
                pmax = self.shape[0]
                qmax = self.shape[1]
            else:
                pmax = pos_max[0]
                qmax = pos_max[1]
        else:
            if pos_min is None:
                pmin = 0
                qmin = 0
            else:
                pixcrd = self.wcs.sky2pix(pos_min, unit=unit_center)
                pmin = pixcrd[0][0]
                qmin = pixcrd[0][1]
            if pos_max is None:
                pmax = self.shape[0]
                qmax = self.shape[1]
            else:
                pixcrd = self.wcs.sky2pix(pos_max, unit=unit_center)
                pmax = pixcrd[0][0]
                qmax = pixcrd[0][1]
            if pmin > pmax:
                a = pmax
                pmax = pmin
                pmin = a
            if qmin > qmax:
                a = qmax
                qmax = qmin
                qmin = a

        pmin = max(0, pmin)
        qmin = max(0, qmin)
        ima = self[pmin:pmax, qmin:qmax]

        ksel = np.where(ima.data.mask == False)
        N = np.shape(ksel[0])[0]
        if N == 0:
            raise ValueError('empty sub-image')
        pixcrd = np.empty((np.shape(ksel[0])[0], 2))
        p = ksel[0]
        q = ksel[1]
        data = ima.data.data[ksel]

        # weight
        if ima.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(ima.var[ksel]))
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(np.shape(ksel[0])[0])

        # initial peak position
        if center is None:
            imax = data.argmax()
            center = np.array([p[imax], q[imax]])
        else:
            if unit_center is not None:
                center = ima.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
                center[0] -= pmin
                center[1] -= qmin

        # initial width value
        if fwhm is None:
            width = ima.moments(unit=None)
            fwhm = width * 2. * np.sqrt(2. * np.log(2.0))
        else:
            if unit_fwhm is not None:
                fwhm = np.array(fwhm) / np.abs(self.wcs.get_step(unit=unit_fwhm))
            else:
                fwhm = np.array(fwhm)

        a = fwhm[0] / (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        e = fwhm[0] / fwhm[1]

        # initial gaussian integrated flux
        if flux is None:
            I = ima.data.data[center[0], center[1]] - cont
        elif peak is True:
            I = flux - cont
        else:
            I = flux * (n - 1) / (np.pi * a * a * e)

        if circular:
            rot = None
            if not fit_back:
                # 2d moffat function
                if fit_n:
                    moffatfit = lambda v, p, q: \
                        cont + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                       + ((q - v[2]) / v[3]) ** 2) ** (-v[4])
                    # inital guesses
                    v0 = [I, center[0], center[1], a, n]
                else:
                    moffatfit = lambda v, p, q: \
                        cont + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                       + ((q - v[2]) / v[3]) ** 2) ** (-n)
                    # inital guesses
                    v0 = [I, center[0], center[1], a]
            else:
                # 2d moffat function
                if fit_n:
                    moffatfit = lambda v, p, q: \
                        v[5] + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                       + ((q - v[2]) / v[3]) ** 2) ** (-v[4])
                    # inital guesses
                    v0 = [I, center[0], center[1], a, n, cont]
                else:
                    moffatfit = lambda v, p, q: \
                        v[4] + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                       + ((q - v[2]) / v[3]) ** 2) ** (-n)
                    # inital guesses
                    v0 = [I, center[0], center[1], a, cont]
        else:
            if not fit_back:
                if rot is None:
                    if fit_n:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: \
                            cont + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                           + ((q - v[2]) / v[3] / v[5]) ** 2) ** (-v[4])
                        # inital guesses
                        v0 = [I, center[0], center[1], a, n, e]
                    else:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: \
                            cont + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                           + ((q - v[2]) / v[3] / v[4]) ** 2) ** (-n)
                        # inital guesses
                        v0 = [I, center[0], center[1], a, e]
                else:
                    # rotation angle in rad
                    rot = np.pi * rot / 180.0
                    if fit_n:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: cont + v[0] \
                            * (1 + (((p - v[1]) * np.cos(v[6]) - (q - v[2])
                                     * np.sin(v[6])) / v[3]) ** 2
                               + (((p - v[1]) * np.sin(v[6]) + (q - v[2])
                                   * np.cos(v[6])) / v[3] / v[5]) ** 2) ** (-v[4])
                        # inital guesses
                        v0 = [I, center[0], center[1], a, n, e, rot]
                    else:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: cont + v[0] \
                            * (1 + (((p - v[1]) * np.cos(v[5]) - (q - v[2])
                                     * np.sin(v[5])) / v[3]) ** 2
                               + (((p - v[1]) * np.sin(v[5]) + (q - v[2])
                                   * np.cos(v[5])) / v[3] / v[4]) ** 2) ** (-n)
                        # inital guesses
                        v0 = [I, center[0], center[1], a, e, rot]
            else:
                if rot is None:
                    if fit_n:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: v[6] + v[0] \
                            * (1 + ((p - v[1]) / v[3]) ** 2
                               + ((q - v[2]) / v[3] / v[5]) ** 2) ** (-v[4])
                        # inital guesses
                        v0 = [I, center[0], center[1], a, n, e, cont]
                    else:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: v[5] + v[0] \
                            * (1 + ((p - v[1]) / v[3]) ** 2
                               + ((q - v[2]) / v[3] / v[4]) ** 2) ** (-n)
                        # inital guesses
                        v0 = [I, center[0], center[1], a, e, cont]
                else:
                    # rotation angle in rad
                    rot = np.pi * rot / 180.0
                    if fit_n:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: v[7] + v[0] \
                            * (1 + (((p - v[1]) * np.cos(v[6])
                                     - (q - v[2]) * np.sin(v[6])) / v[3]) ** 2
                               + (((p - v[1]) * np.sin(v[6])
                                   + (q - v[2]) * np.cos(v[6])) / v[3] / v[5]) ** 2) ** (-v[4])
                        # inital guesses
                        v0 = [I, center[0], center[1], a, n, e, rot, cont]
                    else:
                        # 2d moffat function
                        moffatfit = lambda v, p, q: v[6] + v[0] \
                            * (1 + (((p - v[1]) * np.cos(v[5])
                                     - (q - v[2]) * np.sin(v[5])) / v[3]) ** 2
                               + (((p - v[1]) * np.sin(v[5])
                                   + (q - v[2]) * np.cos(v[5])) / v[3] / v[4]) ** 2) ** (-n)
                        # inital guesses
                        v0 = [I, center[0], center[1], a, e, rot, cont]

        # Minimize the sum of squares
        if factor > 1:
            factor = int(factor)
            deci = np.ones((factor, factor)) \
                * np.arange(factor)[:, np.newaxis] / float(factor) \
                + 1 / float(factor * 2)
            fp = (p[:, np.newaxis] + deci.ravel()[np.newaxis, :]).ravel()
            fq = (q[:, np.newaxis] + deci.T.ravel()[np.newaxis, :]).ravel()
            pixcrd = np.array(zip(fp, fq))

            e_moffat_fit = lambda v, p, q, data, w: \
                w * (((moffatfit(v, p, q)).reshape(N, factor * factor).sum(1)
                      / factor / factor).T.ravel() - data)
            v, covar, info, mesg, success = \
                leastsq(e_moffat_fit, v0[:], args=(pixcrd[:, 0], pixcrd[:, 1],
                                                   data, wght),
                        maxfev=maxiter, full_output=1)
            while np.abs(v[1] - v0[1]) > 0.1 or np.abs(v[2] - v0[2]) > 0.1 \
                    or np.abs(v[3] - v0[3]) > 0.1:
                v0 = v
                v, covar, info, mesg, success = \
                    leastsq(e_moffat_fit, v0[:],
                            args=(pixcrd[:, 0], pixcrd[:, 1],
                                  data, wght), maxfev=maxiter, full_output=1)
        else:
            e_moffat_fit = lambda v, p, q, data, w: \
                w * (moffatfit(v, p, q) - data)
            v, covar, info, mesg, success = \
                leastsq(e_moffat_fit, v0[:],
                        args=(p, q, data, wght),
                        maxfev=maxiter, full_output=1)
            while np.abs(v[1] - v0[1]) > 0.1 or np.abs(v[2] - v0[2]) > 0.1 \
                    or np.abs(v[3] - v0[3]) > 0.1:
                v0 = v
                v, covar, info, mesg, success = \
                    leastsq(e_moffat_fit, v0[:],
                            args=(p, q, data, wght),
                            maxfev=maxiter, full_output=1)

        if success != 1:
            self._logger.info(mesg)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i]))
                            * np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = np.zeros_like(v)
            err[:] = np.abs(v[:] - v0[:])

        # center in pixel in the input image
        v[1] += int(pmin)
        v[2] += int(qmin)

        # plot
        if plot:
            pp = np.arange(pmin, pmax, float(pmax - pmin) / 100)
            qq = np.arange(qmin, qmax, float(qmax - qmin) / 100)
            ff = np.empty((np.shape(pp)[0], np.shape(qq)[0]))
            for i in range(np.shape(pp)[0]):
                ff[i, :] = moffatfit(v, pp[i], qq[:])
            self._ax.contour(qq, pp, ff, 5)

        # Gauss2D object in pixels
        I = v[0]
        p_peak = v[1]
        q_peak = v[2]
        a = np.abs(v[3])
        if fit_n:
            n = v[4]
            if circular:
                e = 1
                rot = 0
                if fit_back:
                    cont = v[5]
            else:
                e = np.abs(v[5])
                if rot is None:
                    rot = 0
                    if fit_back:
                        cont = v[6]
                else:
                    rot = (v[6] * 180.0 / np.pi) % 180
                    if fit_back:
                        cont = v[7]
        else:
            if circular:
                e = 1
                rot = 0
                if fit_back:
                    cont = v[4]
            else:
                e = np.abs(v[4])
                if rot is None:
                    rot = 0
                    if fit_back:
                        cont = v[5]
                else:
                    rot = (v[5] * 180.0 / np.pi) % 180
                    if fit_back:
                        cont = v[6]

        fwhm[0] = a * (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        fwhm[1] = fwhm[0] / e

        flux = I / (n - 1) * (np.pi * a * a * e)

        if err is not None:
            err_I = err[0]
            err_p_peak = err[1]
            err_q_peak = err[2]
            err_a = err[3]
            if fit_n:
                err_n = err[4]
                err_fwhm = err_a * n
                if circular:
                    err_e = 0
                    err_rot = 0
                    err_fwhm = np.array([err_fwhm, err_fwhm])
                    if fit_back:
                        err_cont = err[5]
                    err_flux = err_I * err_n * err_a * err_a
                else:
                    err_e = err[5]
                    if err_e != 0:
                        err_fwhm = np.array([err_fwhm, err_fwhm / err_e])
                    else:
                        err_fwhm = np.array([err_fwhm, err_fwhm])
                    if rot is None:
                        err_rot = 0
                        if fit_back:
                            err_cont = err[6]
                        else:
                            err_cont = 0
                    else:
                        err_rot = err[6] * 180.0 / np.pi
                        if fit_back:
                            err_cont = err[7]
                        else:
                            err_cont = 0
                    err_flux = err_I * err_n * err_a * err_a * err_e
            else:
                err_n = 0
                err_fwhm = err_a * n
                if circular:
                    err_e = 0
                    err_rot = 0
                    err_fwhm = np.array([err_fwhm, err_fwhm])
                    if fit_back:
                        err_cont = err[4]
                    err_flux = err_I * err_n * err_a * err_a
                else:
                    err_e = err[4]
                    if err_e != 0:
                        err_fwhm = np.array([err_fwhm, err_fwhm / err_e])
                    else:
                        err_fwhm = np.array([err_fwhm, err_fwhm])
                    if rot is None:
                        err_rot = 0
                        if fit_back:
                            err_cont = err[5]
                        else:
                            err_cont = 0
                    else:
                        err_rot = err[5] * 180.0 / np.pi
                        if fit_back:
                            err_cont = err[6]
                        else:
                            err_cont = 0
                    err_flux = err_I * err_n * err_a * err_a * err_e
        else:
            err_I = np.NAN
            err_p_peak = np.NAN
            err_q_peak = np.NAN
            err_a = np.NAN
            err_n = np.NAN
            err_e = np.NAN
            err_rot = np.NAN
            err_cont = np.NAN
            err_fwhm = (np.NAN, np.NAN)
            err_flux = np.NAN

        if unit_center is None:
            center = (p_peak, q_peak)
            err_center = (err_p_peak, err_q_peak)
        else:
            # Gauss2D object in degrees/arcseconds
            center = self.wcs.pix2sky([p_peak, q_peak], unit=unit_center)[0]
            err_center = np.array([err_p_peak, err_q_peak]) * np.abs(self.wcs.get_step(unit=unit_center))

        if unit_fwhm is not None:
            step0 = np.abs(self.wcs.get_step(unit=unit_fwhm)[0])
            a = a * step0
            err_a = err_a * step0
            fwhm = fwhm * step0
            err_fwhm = err_fwhm * step0

        moffat = Moffat2D(center, flux, fwhm, cont, n,
                          rot, I, err_center, err_flux, err_fwhm,
                          err_cont, err_n, err_rot, err_I)

        if verbose:
            moffat.print_param()
        if full_output != 0:
            ima = moffat_image(shape=self.shape, wcs=self.wcs, moffat=moffat)
            moffat.ima = ima
        return moffat

    def _rebin_mean_(self, factor):
        """Shrink the size of the image by factor. New size is an integer
        multiple of the original size.

        Parameters
        ----------
        factor : (int,int)
            Factor in y and x.  Python notation: (ny,nx)

        """
        assert not np.sometrue(np.mod(self.shape[0], factor[0]))
        assert not np.sometrue(np.mod(self.shape[1], factor[1]))
        # new size is an integer multiple of the original size
        shape = (self.shape[0] / factor[0], self.shape[1] / factor[1])
        self.data = self.data.reshape(
            shape[0], factor[0], shape[1], factor[1]).sum(1).sum(2)\
            / (factor[0] * factor[1])
        if self.var is not None:
            self.var = self.var.reshape(
                shape[0], factor[0], shape[1], factor[1])\
                .sum(1).sum(2) / (
                    factor[0] * factor[1] * factor[0] * factor[1])
        self.wcs = self.wcs.rebin(factor)

    def _rebin_mean(self, factor, margin='center'):
        """Shrink the size of the image by factor.

        Parameters
        ----------
        factor : int or (int,int)
            Factor in y and x. Python notation: (ny,nx).
        margin : 'center' or 'origin'
            This parameters is used if new size is not an integer multiple of
            the original size. In 'center' case, pixels will be added on the
            left and on the right, on the bottom and of the top of the image.
            In 'origin'case, pixels will be added on (n+1) line/column.

        """
        if is_int(factor):
            factor = (factor, factor)
        if factor[0] <= 1 or factor[0] >= self.shape[0] \
                or factor[1] <= 1 or factor[1] >= self.shape[1]:
            raise ValueError('factor must be in ]1,shape[')
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])):
            # new size is an integer multiple of the original size
            self._rebin_mean_(factor)
            return None
        elif not np.sometrue(np.mod(self.shape[0], factor[0])):
            newshape1 = self.shape[1] / factor[1]
            n1 = self.shape[1] - newshape1 * factor[1]
            if margin == 'origin' or n1 == 1:
                ima = self[:, :-n1]
                ima._rebin_mean_(factor)
                newshape = (ima.shape[0], ima.shape[1] + 1)
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[:, 0:-1] = ima.data
                mask[:, 0:-1] = ima.data.mask
                d = self.data[:, -n1:].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1) \
                    / factor[0] / factor[1]
                data[:, -1] = d.data
                mask[:, -1] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[:, 0:-1] = ima.var
                    var[:, -1] = self.var[:, -n1:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1)\
                        / factor[0] / factor[0] / factor[1] / factor[1]
                wcs = ima.wcs
                wcs.naxis1 = wcs.naxis1 + 1
            else:
                n_left = n1 / 2
                n_right = self.shape[1] - n1 + n_left
                ima = self[:, n_left:n_right]
                ima._rebin_mean_(factor)
                newshape = (ima.shape[0], ima.shape[1] + 2)
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[:, 1:-1] = ima.data
                mask[:, 1:-1] = ima.data.mask
                d = self.data[:, 0:n_left].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1) \
                    / factor[0] / factor[1]
                data[:, 0] = d.data
                mask[:, 0] = d.mask
                d = self.data[:, n_right:].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1)\
                    / factor[0] / factor[1]
                data[:, -1] = d.data
                mask[:, -1] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[:, 1:-1] = ima.var
                    var[:, 0] = self.var[:, 0:n_left].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[:, -1] = self.var[:, n_right:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                wcs.set_naxis1(wcs.naxis1 + 2)
        elif not np.sometrue(np.mod(self.shape[1], factor[1])):
            newshape0 = self.shape[0] / factor[0]
            n0 = self.shape[0] - newshape0 * factor[0]
            if margin == 'origin' or n0 == 1:
                ima = self[:-n0, :]
                ima._rebin_mean_(factor)
                newshape = (ima.shape[0] + 1, ima.shape[1])
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[0:-1, :] = ima.data
                mask[0:-1, :] = ima.data.mask
                d = self.data[-n0:, :].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1) \
                    / factor[0] / factor[1]
                data[-1, :] = d.data
                mask[-1, :] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[0:-1, :] = ima.var
                    var[-1, :] = self.var[-n0:, :].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.naxis2 = wcs.naxis2 + 1
            else:
                n_left = n0 / 2
                n_right = self.shape[0] - n0 + n_left
                ima = self[n_left:n_right, :]
                ima._rebin_mean_(factor)
                newshape = (ima.shape[0] + 2, ima.shape[1])
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[1:-1, :] = ima.data
                mask[1:-1, :] = ima.data.mask
                d = self.data[0:n_left, :].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1)\
                    / factor[0] / factor[1]
                data[0, :] = d.data
                mask[0, :] = d.mask
                d = self.data[n_right:, :].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1) \
                    / factor[0] / factor[1]
                data[-1, :] = d.data
                mask[-1, :] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[1:-1, :] = ima.var
                    var[0, :] = self.var[0:n_left, :].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[-1, :] = self.var[n_right:, :].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                wcs.set_naxis2(wcs.naxis2 + 2)
        else:
            factor = np.array(factor)
            newshape = self.shape / factor
            n = self.shape - newshape * factor
            if n[0] == 1 and n[1] == 1:
                margin = 'origin'
            if margin == 'center':
                n_left = n / 2
                n_right = self.shape - n + n_left
                ima = self[n_left[0]:n_right[0], n_left[1]:n_right[1]]
                ima._rebin_mean_(factor)
                if n_left[0] != 0 and n_left[1] != 0:
                    newshape = (ima.shape[0] + 2, ima.shape[1] + 2)
                    data = np.empty(newshape)
                    mask = np.empty(newshape, dtype=bool)
                    data[1:-1, 1:-1] = ima.data
                    mask[1:-1, 1:-1] = ima.data.mask
                    data[0, 0] = self.data[0:n_left[0], 0:n_left[1]].sum() \
                        / factor[0] / factor[1]
                    mask[0, 0] = \
                        self.data.mask[0:n_left[0], 0:n_left[1]].any()
                    data[0, -1] = self.data[0:n_left[0], n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[0, -1] = \
                        self.data.mask[0:n_left[0], n_right[1]:].any()
                    data[-1, 0] = self.data[n_right[0]:, 0:n_left[1]].sum() \
                        / factor[0] / factor[1]
                    mask[-1, 0] = \
                        self.data.mask[n_right[0]:, 0:n_left[1]].any()
                    data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[-1, -1] = \
                        self.data.mask[n_right[0]:, n_right[1]:].any()
                    d = self.data[0:n_left[0], n_left[1]:n_right[1]]\
                        .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[0, 1:-1] = d.data
                    mask[0, 1:-1] = d.mask
                    d = self.data[n_right[0]:, n_left[1]:n_right[1]]\
                        .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[-1, 1:-1] = d.data
                    mask[-1, 1:-1] = d.mask
                    d = self.data[n_left[0]:n_right[0], 0:n_left[1]]\
                        .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[1:-1, 0] = d.data
                    mask[1:-1, 0] = d.mask
                    d = self.data[n_left[0]:n_right[0], n_right[1]:]\
                        .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[1:-1, -1] = d.data
                    mask[1:-1, -1] = d.mask
                    var = None
                    if self.var is not None:
                        var = np.empty(newshape)
                        var[1:-1, 1:-1] = ima.var
                        var[0, 0] = self.var[0:n_left[0], 0:n_left[1]].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0, -1] = self.var[0:n_left[0], n_right[1]:].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 0] = self.var[n_right[0]:, 0:n_left[1]].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, -1] = self.var[n_right[0]:, n_right[1]:].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0, 1:-1] = \
                            self.var[0:n_left[0], n_left[1]:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 1:-1] = \
                            self.var[n_right[0]:, n_left[1]:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[1:-1, 0] = \
                            self.var[n_left[0]:n_right[0], 0:n_left[1]]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[1:-1, -1] = \
                            self.var[n_left[0]:n_right[0], n_right[1]:]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                    wcs = ima.wcs
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 2)
                    wcs.set_naxis2(wcs.naxis2 + 2)
                elif n_left[0] == 0:
                    newshape = (ima.shape[0] + 1, ima.shape[1] + 2)
                    data = np.empty(newshape)
                    mask = np.empty(newshape, dtype=bool)
                    data[0:-1, 1:-1] = ima.data
                    mask[0:-1, 1:-1] = ima.data.mask

                    data[0, 0] = self.data[0, 0:n_left[1]].sum() \
                        / factor[0] / factor[1]
                    mask[0, 0] = self.data.mask[0, 0:n_left[1]].any()
                    data[0, -1] = self.data[0, n_right[1]:].sum()\
                        / factor[0] / factor[1]
                    mask[0, -1] = self.data.mask[0, n_right[1]:].any()
                    data[-1, 0] = self.data[n_right[0]:, 0:n_left[1]].sum()\
                        / factor[0] / factor[1]
                    mask[-1, 0] = self.data.mask[n_right[0]:, 0:n_left[1]].any()
                    data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum()\
                        / factor[0] / factor[1]
                    mask[-1, -1] = \
                        self.data.mask[n_right[0]:, n_right[1]:].any()
                    d = self.data[n_right[0]:, n_left[1]:n_right[1]]\
                        .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1)\
                        / factor[0] / factor[1]
                    data[-1, 1:-1] = d.data
                    mask[-1, 1:-1] = d.mask
                    d = self.data[0:n_right[0], 0:n_left[1]].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1)\
                        / factor[0] / factor[1]
                    data[0:-1, 0] = d.data
                    mask[0:-1, 0] = d.mask
                    d = self.data[0:n_right[0], n_right[1]:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[0:-1, -1] = d.data
                    mask[0:-1, -1] = d.mask
                    var = None
                    if self.var is not None:
                        var = np.empty(newshape)
                        var[0:-1, 1:-1] = ima.var
                        var[0, 0] = self.var[0, 0:n_left[1]].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0, -1] = self.var[0, n_right[1]:].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 0] = self.var[n_right[0]:, 0:n_left[1]].sum()\
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, -1] = self.var[n_right[0]:, n_right[1]:].sum()\
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 1:-1] = \
                            self.var[n_right[0]:, n_left[1]:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0:-1, 0] = \
                            self.var[0:n_right[0], 0:n_left[1]]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0:-1, -1] = \
                            self.var[0:n_right[0], n_right[1]:]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]

                    wcs = ima.wcs
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 2)
                    wcs.set_naxis2(wcs.naxis2 + 1)
                else:
                    newshape = (ima.shape[0] + 2, ima.shape[1] + 1)
                    data = np.empty(newshape)
                    mask = np.empty(newshape, dtype=bool)
                    data[1:-1, 0:-1] = ima.data
                    mask[1:-1, 0:-1] = ima.data.mask

                    data[0, 0] = self.data[0:n_left[0], 0].sum() \
                        / factor[0] / factor[1]
                    mask[0, 0] = self.data.mask[0:n_left[0], 0].any()
                    data[0, -1] = self.data[0:n_left[0], n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[0, -1] = \
                        self.data.mask[0:n_left[0], n_right[1]:].any()
                    data[-1, 0] = self.data[n_right[0]:, 0].sum() \
                        / factor[0] / factor[1]
                    mask[-1, 0] = self.data.mask[n_right[0]:, 0].any()
                    data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[-1, -1] = \
                        self.data.mask[n_right[0]:, n_right[1]:].any()
                    d = self.data[0:n_left[0], 0:n_right[1]].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[0, 0:-1] = d.data
                    mask[0, 0:-1] = d.mask
                    d = self.data[n_right[0]:, 0:n_right[1]].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[-1, 0:-1] = d.data
                    mask[-1, 0:-1] = d.mask
                    d = self.data[n_left[0]:n_right[0], n_right[1]:]\
                        .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[1:-1, -1] = d.data
                    mask[1:-1, -1] = d.mask

                    var = None
                    if self.var is not None:
                        var = np.empty(newshape)
                        var[1:-1, 0:-1] = ima.var
                        var[0, 0] = self.var[0:n_left[0], 0].sum() \
                            / factor[0] / factor[1]
                        var[0, -1] = \
                            self.var[0:n_left[0], n_right[1]:].sum() \
                            / factor[0] / factor[1]
                        var[-1, 0] = self.var[n_right[0]:, 0].sum() \
                            / factor[0] / factor[1]
                        var[-1, -1] = \
                            self.var[n_right[0]:, n_right[1]:].sum() \
                            / factor[0] / factor[1]
                        var[0, 0:-1] = \
                            self.var[0:n_left[0], 0:n_right[1]].sum(axis=0)\
                            .reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1]
                        var[-1, 0:-1] = self.var[n_right[0]:, 0:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1]
                        var[1:-1, -1] = \
                            self.var[n_left[0]:n_right[0], n_right[1]:]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1]
                    wcs = ima.wcs
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 1)
                    wcs.set_naxis2(wcs.naxis2 + 2)
            elif margin == 'origin':
                n_right = self.shape - n
                ima = self[0:n_right[0], 0:n_right[1]]
                ima._rebin_mean_(factor)
                newshape = (ima.shape[0] + 1, ima.shape[1] + 1)
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[0:-1, 0:-1] = ima.data
                mask[0:-1, 0:-1] = ima.data.mask
                d = self.data[n_right[0]:, 0:n_right[1]].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1) \
                    / factor[0] / factor[1]
                data[-1, 0:-1] = d.data
                mask[-1, 0:-1] = d.mask
                d = self.data[0:n_right[0], n_right[1]:].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1) \
                    / factor[0] / factor[1]
                data[0:-1, -1] = d.data
                mask[0:-1, -1] = d.mask
                data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum() \
                    / factor[0] / factor[1]
                mask[-1, -1] = self.data.mask[n_right[0]:, n_right[1]:].any()
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[0:-1, 0:-1] = ima.var
                    var[-1, 0:-1] = \
                        self.var[n_right[0]:, 0:n_right[1]].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[0:-1, -1] = \
                        self.var[0:n_right[0], n_right[1]:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[-1, -1] = self.var[n_right[0]:, n_right[1]:]\
                        .sum() / factor[0] / factor[1] / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.naxis1 = wcs.naxis1 + 1
                wcs.naxis2 = wcs.naxis2 + 1
            else:
                raise ValueError('margin must be center|origin')
        self.wcs = wcs
        self.data = np.ma.array(data, mask=mask)
        self.var = var

    def rebin_mean(self, factor, margin='center'):
        """Return an image that shrinks the size of the current image by
        factor.

        Parameters
        ----------
        factor : int or (int,int)
            Factor in y and x. Python notation: (ny,nx).
        margin : 'center' or 'origin'
            This parameters is used if new size is not an integer multiple of
            the original size.  In 'center' case, pixels will be added on the
            left and on the right, on the bottom and of the top of the image.
            In 'origin'case, pixels will be added on (n+1) line/column.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        res._rebin_mean(factor, margin)
        return res

    def _med_(self, p, q, pfactor, qfactor):
        return np.ma.median(self.data[p * pfactor:(p + 1) * pfactor,
                                      q * qfactor:(q + 1) * qfactor])

    def _rebin_median_(self, factor):
        """Shrink the size of the image by factor. New size is an integer
        multiple of the original size.

        Parameters
        ----------
        factor : (int,int)
            Factor in y and x.  Python notation: (ny,nx)
        """
        assert not np.sometrue(np.mod(self.shape[0], factor[0]))
        assert not np.sometrue(np.mod(self.shape[1], factor[1]))
        # new size is an integer multiple of the original size
        shape = (self.shape[0] / factor[0], self.shape[1] / factor[1])
        Nq, Np = np.meshgrid(xrange(shape[1]), xrange(shape[0]))
        vfunc = np.vectorize(self._med_)
        data = vfunc(Np, Nq, factor[0], factor[1])
        mask = self.data.mask.reshape(shape[0], factor[0],
                                      shape[1], factor[1])\
            .sum(1).sum(2) / factor[0] / factor[1]
        self.data = np.ma.array(data, mask=mask)
        self.var = None
        self.wcs = self.wcs.rebin(factor)

    def rebin_median(self, factor, margin='center'):
        """Shrink the size of the image by factor. Median values are used.

        Parameters
        ----------
        factor : int or (int,int)
            Factor in y and x. Python notation: (ny,nx).
        margin : 'center' or 'origin'
            This parameters is used if new size is not an integer multiple of
            the original size.  In 'center' case, image is truncated on the
            left and on the right, on the bottom and of the top of the image.
            In 'origin'case, image is truncated at the end along each
            direction.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        if is_int(factor):
            factor = (factor, factor)
        if factor[0] <= 1 or factor[0] >= self.shape[0] \
                or factor[1] <= 1 or factor[1] >= self.shape[1]:
            raise ValueError('factor must be in ]1,shape[')
            return None
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])):
            # new size is an integer multiple of the original size
            res = self.copy()
        elif not np.sometrue(np.mod(self.shape[0], factor[0])):
            newshape1 = self.shape[1] / factor[1]
            n1 = self.shape[1] - newshape1 * factor[1]
            if margin == 'origin' or n1 == 1:
                res = self[:, :-n1]
            else:
                n_left = n1 / 2
                n_right = self.shape[1] - n1 + n_left
                res = self[:, n_left:n_right]
        elif not np.sometrue(np.mod(self.shape[1], factor[1])):
            newshape0 = self.shape[0] / factor[0]
            n0 = self.shape[0] - newshape0 * factor[0]
            if margin == 'origin' or n0 == 1:
                res = self[:-n0, :]
            else:
                n_left = n0 / 2
                n_right = self.shape[0] - n0 + n_left
                res = self[n_left:n_right, :]
        else:
            factor = np.array(factor)
            newshape = self.shape / factor
            n = self.shape - newshape * factor
            if n[0] == 1 and n[1] == 1:
                margin = 'origin'
            if margin == 'center':
                n_left = n / 2
                n_right = self.shape - n + n_left
                res = self[n_left[0]:n_right[0], n_left[1]:n_right[1]]
            elif margin == 'origin':
                n_right = self.shape - n
                res = self[0:n_right[0], 0:n_right[1]]
            else:
                raise ValueError('margin must be center|origin')
        res._rebin_median_(factor)
        return res

    def _resample(self, newdim, newstart, newstep, flux=False, order=3,
                  interp='no', unit_start=u.deg, unit_step=u.arcsec):

        if is_int(newdim):
            newdim = (newdim, newdim)

        if newstart is None:
            newstart = self.wcs.get_start(unit=unit_start)
        elif is_int(newstart) or is_float(newstart):
            newstart = (newstart, newstart)

        if is_int(newstep) or is_float(newstep):
            newstep = (newstep, newstep)

        newdim = np.array(newdim)
        newstart = np.array(newstart)
        newstep = np.array(newstep)

        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        oldstep = self.wcs.get_step(unit=unit_step)
        pstep = newstep / oldstep

        # Before 2015/12/11: + 0.5 in old grid - 0.5 in new grid
        # pixNewstart = self.wcs.sky2pix(newstart, unit=unit_start)[0]
        # offset_oldpix = 0.5 + pixNewstart - 0.5 * pstep
        # poffset = offset_oldpix / pstep

        # Before 2015/09/04 (8ee439c1): + 0.5 in old grid
        # poffset = (self.wcs.sky2pix(newstart, unit=unit_start)[0] + 0.5) / pstep

        # After tests with a gaussian image and HST header, the best precision
        # is obtained without +/- 0.5. It seems this was also used before
        # 2015/08/25 (90f70a41)
        poffset = self.wcs.sky2pix(newstart, unit=unit_start)[0] / pstep

        data = ndi.affine_transform(data, pstep, poffset,
                                    output_shape=newdim, order=order)

        newmask = ndi.affine_transform(~self.data.mask, pstep, poffset,
                                       output_shape=newdim, order=0)
        mask = np.ma.make_mask(1 - newmask)

        if flux:
            data *= newstep.prod() / oldstep.prod()

        step = (newstep * unit_step).to(unit_start).value
        self.wcs = self.wcs.resample(step, start=newstart, unit=unit_start)
        self.wcs.naxis1 = newdim[1]
        self.wcs.naxis2 = newdim[0]
        self.data = np.ma.array(data, mask=mask)
        self.var = None

    def resample(self, newdim, newstart, newstep, flux=False,
                 order=3, interp='no', unit_start=u.deg, unit_step=u.arcsec):
        """Return resampled image to a new coordinate system.

        Uses :func:`scipy.ndimage.affine_transform`.

        Parameters
        ----------
        newdim : int or (int,int)
            New dimensions. Python notation: (ny,nx)
        newstart : float or (float, float)
            New positions (y,x) for the pixel (0,0).
            If None, old position is used.
        newstep : float or (float, float)
            New step (dy,dx).
        flux : boolean
            if flux is True, the flux is conserved.
        order : int
            The order of the spline interpolation, default is 3.
            The order has to be in the range 0-5.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.
        unit_start : astropy.units
            type of the newstart coordinates.  Degrees by default.
        unit_step : astropy.units
            step unit.  Arcseconds by default.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        # pb rotation
        res._resample(newdim, newstart, newstep, flux=flux, order=order,
                      interp=interp, unit_start=unit_start,
                      unit_step=unit_step)
        return res

    def _gaussian_filter(self, sigma=3, interp='no'):
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndi.gaussian_filter(data, sigma),
                                mask=self.data.mask)
        if self.var is not None:
            self.var = ndi.gaussian_filter(self.var, sigma)

    def gaussian_filter(self, sigma=3, interp='no'):
        """Return an image containing Gaussian filter applied to the current
        image.

        Uses :func:`scipy.ndimage.gaussian_filter`.

        Parameters
        ----------
        sigma : float
            Standard deviation for Gaussian kernel
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        res._gaussian_filter(sigma, interp)
        return res

    def _median_filter(self, size=3, interp='no'):
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndi.median_filter(data, size),
                                mask=self.data.mask)
        if self.var is not None:
            self.var = ndi.median_filter(self.var, size)

    def median_filter(self, size=3, interp='no'):
        """Return an image containing median filter applied to the current
        image.

        Uses :func:`scipy.ndimage.median_filter`.

        Parameters
        ----------
        size : float
            Shape that is taken from the input array, at every element
            position, to define the input to the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        res._median_filter(size, interp)
        return res

    def _maximum_filter(self, size=3, interp='no'):
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndi.maximum_filter(data, size),
                                mask=self.data.mask)

    def maximum_filter(self, size=3, interp='no'):
        """Return an image containing maximum filter applied to the current
        image.

        Uses :func:`scipy.ndimage.maximum_filter`.

        Parameters
        ----------
        size : float
            Shape that is taken from the input array, at every element
            position, to define the input to the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        res._maximum_filter(size, interp)
        return res

    def _minimum_filter(self, size=3, interp='no'):
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndi.minimum_filter(data, size),
                                mask=self.data.mask)

    def minimum_filter(self, size=3, interp='no'):
        """Return an image containing minimum filter applied to the current
        image.

        Uses :func:`scipy.ndimage.minimum_filter`.

        Parameters
        ----------
        size : float
            Shape that is taken from the input array, at every element
            position, to define the input to the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        res._minimum_filter(size, interp)
        return res

    def add(self, other):
        """Add the image other to the current image in place. The coordinate
        are taken into account.

        Parameters
        ----------
        other : Image
            Second image to add.
        """
        if not isinstance(other, Image):
            raise IOError('Operation forbidden')

        ima = other.copy()
        self_rot = self.wcs.get_rot()
        ima_rot = ima.wcs.get_rot()
        theta = 0
        if self_rot != ima_rot:
            if ima.wcs.get_cd()[0, 0] * self.wcs.get_cd()[0, 0] < 0:
                theta = 180 - self_rot + ima_rot
                ima = ima.rotate(theta, reshape=True)
            else:
                theta = -self_rot + ima_rot
                ima = ima.rotate(theta, reshape=True)

        unit = ima.wcs.unit
        self_cdelt = self.wcs.get_step(unit=unit)
        ima_cdelt = ima.wcs.get_step()

        if (self_cdelt != ima_cdelt).all():
            factor = self_cdelt / ima_cdelt
            try:
                if not np.sometrue(np.mod(self_cdelt[0],
                                          ima_cdelt[0])) \
                    and not np.sometrue(np.mod(self_cdelt[1],
                                               ima_cdelt[1])):
                    # ima.step is an integer multiple of the self.step
                    ima = ima.rebin_mean(factor)
                else:
                    raise ValueError('steps are not integer multiple')
            except:
                newdim = np.array(0.5 + ima.shape / factor, dtype=np.int)
                newstart = self.wcs.get_start(unit=unit)
                ima = ima.resample(newdim, newstart, self_cdelt, flux=True,
                                   unit_step=unit, unit_start=unit)

        # here ima and self have the same step and the same rotation

        [[k1, l1]] = self.wcs.sky2pix(ima.wcs.pix2sky(
            [[0, 0]], unit=self.wcs.unit))
        l1 = int(l1 + 0.5)
        k1 = int(k1 + 0.5)
        k2 = k1 + ima.shape[0]
        if k1 < 0:
            nk1 = -k1
            k1 = 0
        else:
            nk1 = 0

        if k2 > self.shape[0]:
            nk2 = ima.shape[0] - (k2 - self.shape[0])
            k2 = self.shape[0]
        else:
            nk2 = ima.shape[0]

        l2 = l1 + ima.shape[1]
        if l1 < 0:
            nl1 = -l1
            l1 = 0
        else:
            nl1 = 0

        if l2 > self.shape[1]:
            nl2 = ima.shape[1] - (l2 - self.shape[1])
            l2 = self.shape[1]
        else:
            nl2 = ima.shape[1]

        mask = self.data.mask.__copy__()
        self.data[k1:k2, l1:l2] += UnitMaskedArray(ima.data[nk1:nk2, nl1:nl2],
                                                   ima.unit, self.unit)
        self.data.mask = mask

    def segment(self, shape=(2, 2), minsize=20, minpts=None,
                background=20, interp='no', median=None):
        """Segment the image in a number of smaller images.

        Returns a list of images. Uses
        :func:`scipy.ndimage.generate_binary_structure`,
        :func:`scipy.ndimage.grey_dilation`,
        :func:`scipy.ndimage.measurements.label`, and
        :func:`scipy.ndimage.measurements.find_objects`.

        Parameters
        ----------
        shape : (int,int)
            Shape used for connectivity.
        minsize : int
            Minimmum size of the images.
        minpts : int
            Minimmum number of points in the object.
        background : float
            Under this value, flux is considered as background.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.
        median : (int,int) or None
            Size of the median filter

        Returns
        -------
        out : List of Image objects.

        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        structure = ndi.generate_binary_structure(shape[0], shape[1])
        if median is not None:
            data = np.ma.array(ndi.median_filter(data, median),
                               mask=self.data.mask)
        expanded = ndi.grey_dilation(data, (minsize, minsize))
        ksel = np.where(expanded < background)
        expanded[ksel] = 0

        lab = ndi.measurements.label(expanded, structure)
        slices = ndi.measurements.find_objects(lab[0])

        imalist = []
        for i in range(lab[1]):
            if minpts is not None:
                if (data[slices[i]].ravel() > background)\
                        .sum() < minpts:
                    continue
            [[starty, startx]] = \
                self.wcs.pix2sky(self.wcs.pix2sky([[slices[i][0].start,
                                                    slices[i][1].start]]))
            wcs = self.wcs.copy()
            wcs.set_crpix1(1.0)
            wcs.set_crpix2(1.0)
            wcs.set_crval1(startx)
            wcs.set_crval2(starty)
            wcs.naxis1 = self.data[slices[i]].shape[1]
            wcs.naxis2 = self.data[slices[i]].shape[0]
            if self.var is not None:
                res = Image(data=self.data[slices[i]], wcs=wcs,
                            unit=self.unit, var=self.var[slices[i]])
            else:
                res = Image(data=self.data[slices[i]], wcs=wcs,
                            unit=self.unit)
            imalist.append(res)
        return imalist

    def add_gaussian_noise(self, sigma, interp='no'):
        """Add Gaussian noise to image in place.

        Parameters
        ----------
        sigma : float
                 Standard deviation.
        interp : 'no' | 'linear' | 'spline'
                 if 'no', data median value replaced masked values.
                 if 'linear', linear interpolation of the masked values.
                 if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(np.random.normal(data, sigma),
                                mask=self.data.mask)
        if self.var is None:
            self.var = np.ones((self.shape)) * sigma * sigma
        else:
            self.var *= (sigma * sigma)

    def add_poisson_noise(self, interp='no'):
        """Add Poisson noise to image in place.

        Parameters
        ----------
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(np.random.poisson(data).astype(float),
                                mask=self.data.mask)
        if self.var is None:
            self.var = self.data.data.__copy__()
        else:
            self.var += self.data.data

    def inside(self, coord, unit=u.deg):
        """Return True if coord is inside image.

        Parameters
        ----------
        coord : (float,float)
                coordinates (y,x).
        unit : astropy units
                Type of the coordinates (degrees by default)

        Returns
        -------
        out : boolean
        """
        if unit is not None:
            pixcrd = self.wcs.sky2pix([coord[0], coord[1]], unit=unit)[0]
        else:
            pixcrd = coord
        if pixcrd[0] >= 0 and pixcrd[0] < self.shape[0] \
                and pixcrd[1] >= 0 and pixcrd[1] < self.shape[1]:
            return True
        else:
            return False

    def _fftconvolve(self, other, interp='no'):
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            if self.shape[0] != other.shape[0] \
                    or self.shape[1] != other.shape[1]:
                raise IOError('Operation forbidden for images '
                              'with different sizes')

            if interp == 'linear':
                data = self._interp_data(spline=False)
            elif interp == 'spline':
                data = self._interp_data(spline=True)
            else:
                data = np.ma.filled(self.data, np.ma.median(self.data))

            self.data = np.ma.array(signal.fftconvolve(data, other,
                                                       mode='same'),
                                    mask=self.data.mask)
            if self.var is not None:
                self.var = signal.fftconvolve(self.var, other,
                                              mode='same')
        elif other.ndim == 2:
            if other.data is None or self.shape[0] != other.shape[0] \
                    or self.shape[1] != other.shape[1]:
                raise IOError('Operation forbidden for images '
                              'with different sizes')

            if interp == 'linear':
                data = self._interp_data(spline=False)
                other_data = other._interp_data(spline=False)
            elif interp == 'spline':
                data = self._interp_data(spline=True)
                other_data = other._interp_data(spline=True)
            else:
                data = np.ma.filled(self.data, np.ma.median(self.data))
                other_data = other.data.filled(np.ma.median(other.data))

            if self.unit != other.unit:
                other_data = UnitMaskedArray(other_data, other.unit, self.unit)

            self.data = np.ma.array(signal.fftconvolve(data, other_data,
                                                       mode='same'),
                                    mask=self.data.mask)
            if self.var is not None:
                self.var = signal.fftconvolve(self.var,
                                              other_data, mode='same')
        else:
            raise IOError('Operation forbidden')

    def fftconvolve(self, other, interp='no'):
        """Return the convolution of the image with other using fft.

        Uses :func:`scipy.signal.fftconvolve`.

        Parameters
        ----------
        other : 2d-array or Image
            Second Image or 2d-array.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        res = self.copy()
        res._fftconvolve(other, interp)
        return res

    def fftconvolve_gauss(self, center=None, flux=1., fwhm=(1., 1.),
                          peak=False, rot=0., factor=1, unit_center=u.deg,
                          unit_fwhm=u.arcsec):
        """Return the convolution of the image with a 2D gaussian.

        Parameters
        ----------
        center : (float,float)
            Gaussian center (y_peak, x_peak). If None the center of the image
            is used.  The unit is given by the unit_center parameter (degrees
            by default).
        flux : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        fwhm : (float,float)
            Gaussian fwhm (fwhm_y,fwhm_x). The unit is given by the unit_fwhm
            parameter (arcseconds by default).
        peak : boolean
            If true, flux contains a gaussian peak value.
        rot : float
            Angle position in degree.
        factor : int
            If factor<=1, gaussian value is computed in the center of each
            pixel.  If factor>1, for each pixel, gaussian value is the sum of
            the gaussian values on the factor*factor pixels divided by the
            pixel area.
        unit_center : astropy.units
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_fwhm : astropy.units
            FWHM unit. Arcseconds by default (use None for radius in pixels)

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        ima = gauss_image(self.shape, wcs=self.wcs, center=center,
                          flux=flux, fwhm=fwhm, peak=peak, rot=rot,
                          factor=factor, gauss=None, unit_center=unit_center,
                          unit_fwhm=unit_fwhm, cont=0, unit=self.unit)
        ima.norm(typ='sum')
        return self.fftconvolve(ima)

    def fftconvolve_moffat(self, center=None, flux=1., a=1.0, q=1.0,
                           n=2, peak=False, rot=0., factor=1,
                           unit_center=u.deg, unit_a=u.arcsec):
        """Return the convolution of the image with a 2D moffat.

        Parameters
        ----------
        center : (float,float)
            Gaussian center (y_peak, x_peak).  If None the center of the image
            is used.  The unit is given by the unit_center parameter (degrees
            by default).
        flux : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        a : float
            Half width at half maximum of the image in the absence of
            atmospheric scattering.  1 by default.  The unit is given by the
            unit_a parameter (arcseconds by default).
        q : float
            Axis ratio, 1 by default.
        n : int
            Atmospheric scattering coefficient. 2 by default.
        rot : float
            Angle position in degree.
        factor : int
            If factor<=1, moffat value is computed in the center of each pixel.
            If factor>1, for each pixel, moffat value is the sum
            of the moffat values on the factor*factor pixels
            divided by the pixel area.
        peak : boolean
            If true, flux contains a gaussian peak value.
        unit_center : astropy.units
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_a : astropy.units
            a unit. Arcseconds by default (use None for radius in pixels)

        Returns
        -------
        out : :class:`mpdaf.obj.Image`

        """
        fwhmy = a * (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        fwhmx = fwhmy / q

        ima = moffat_image(self.shape, wcs=self.wcs, factor=factor,
                           center=center, flux=flux, fwhm=(fwhmy, fwhmx), n=n,
                           rot=rot, peak=peak, unit_center=unit_center,
                           unit_fwhm=unit_a, unit=self.unit)

        ima.norm(typ='sum')
        return self.fftconvolve(ima)

    def correlate2d(self, other, interp='no'):
        """Return the cross-correlation of the image with an array/image

        Uses :func:`scipy.signal.correlate2d`.

        Parameters
        ----------
        other : 2d-array or Image
            Second Image or 2d-array.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.

        """
        if self.data is None:
            raise ValueError('empty data array')

        if not isinstance(other, DataArray):
            if interp == 'linear':
                data = self._interp_data(spline=False)
            elif interp == 'spline':
                data = self._interp_data(spline=True)
            else:
                data = np.ma.filled(self.data, np.ma.median(self.data))

            res = self.copy()
            res.data = np.ma.array(signal.correlate2d(data, other, mode='same',
                                                      boundary='symm'),
                                   mask=res.data.mask)
            if res.var is not None:
                res.var = signal.correlate2d(res.var, other, mode='same',
                                             boundary='symm')
            return res
        elif other.ndim == 2:
            if interp == 'linear':
                data = self._interp_data(spline=False)
                other_data = other._interp_data(spline=False)
            elif interp == 'spline':
                data = self._interp_data(spline=True)
                other_data = other._interp_data(spline=True)
            else:
                data = np.ma.filled(self.data, np.ma.median(self.data))
                other_data = other.data.filled(np.ma.median(other.data))

            if self.unit != other.unit:
                other_data = UnitMaskedArray(other_data, other.unit, self.unit)
            res = self.copy()
            res.data = np.ma.array(signal.correlate2d(data,
                                                      other_data, mode='same'),
                                   mask=res.data.mask)
            if res.var is not None:
                res.var = signal.correlate2d(res.var, other_data,
                                             mode='same')
            return res
        else:
            raise IOError('Operation forbidden')

    def find_wcs_offsets(self, reffile, catfile, hsize=10, seeing=0.5,
                         plot=False):
        """Find WCS offsets with a better resolved image.

        For each object in ``catfile``, a subimage of ``hsize`` arcseconds is
        extracted. It is then convolved with a Gaussian PSF (with ``seeing``
        used for the FWHM) and resampled to the current image resolution. A 2D
        correlation is performed between the two images, and the offset is
        computed with the correlation peak.

        The ascii catalog ``catfile`` must be a text file which looks like
        this::

            # field ra dec type
            UDF-01 53.158007 -27.769189 CEN
            UDF-03 53.187819 -27.794050 VIS

        The first column correspond to the ``OBJECT`` keyword. The last column
        gives the type of source:

        - ``CEN`` are used to compute offset.
        - ``VIS`` ones are used only for the visualization.

        Parameters
        ----------
        reffile: str
            Path to a FITS image which is used to compute the offset, typically
            an HST image.
        catfile: str
            Path to an ascii file containing the sources (see above).
        hsize: int
            Size of the sub-images in arcseconds (default: 10)
        seeing: float
            Seeing in arcseconds, used for the convolution (default: 0.5)
        plot: bool
            Plot images (default: False)

        Returns
        -------
        offset, offpix: ndarray, ndarray
            Offset in arcseconds and in pixels.

        """
        info = self._logger.info
        catref = Table.read(catfile, format='ascii')
        info('%d objects found in the catalog', len(catref))

        bckg = self.background()
        info('Estimated background: %s (sigma: %s)', *bckg)
        self.data -= bckg[0]

        hdr = self.primary_header
        field = hdr['OBJECT'].split(' ')[0]
        info('Object: %s, Field: %s, AG Seeing: %s', hdr['OBJECT'], field,
             hdr.get('ESO OCS SGS AG FWHMX AVG'))

        hst = Image(reffile, copy=False)
        hst.info()

        # Remove units temporarily before it fails with HST images ...
        orig_unit = self.unit
        self.unit = u.dimensionless_unscaled
        hst.unit = u.dimensionless_unscaled

        t = catref[(catref['field'] == field) & (catref['type'] == 'CEN')]
        t.add_column(Column(np.zeros((len(t), 2)), name='offset_pix'))
        t.add_column(Column(np.zeros((len(t), 2)), name='offset_arc'))
        t.add_column(Column(np.zeros(len(t)), name='offset_peak'))
        centers = zip(t['dec'], t['ra'])
        info('%d centering sources found in catalog for %s', len(t), field)

        if plot:
            def plot_with_cross(img, ax, center, cross_color='k',
                                center_deg=True, **kwargs):
                img.plot(ax=ax, **kwargs)
                y, x = img.wcs.sky2pix(center)[0] if center_deg else center
                ax.axhline(y, color=cross_color)
                ax.axvline(x, color=cross_color)

            fig, ax = plt.subplots(figsize=(5, 5))
            self.plot(zscale=True, title='Background subtracted image')
            for center in centers:
                y, x = self.wcs.sky2pix(center)[0]
                ax.axhline(y, color='k')
                ax.axvline(x, color='k')
            plt.show()

        for k, row in enumerate(t):
            center = centers[k]
            # extract list of subimages
            info('Peak center %s -> muse pix: %s', center,
                 self.wcs.sky2pix(center, nearest=True)[0])
            zmuse = self.subimage(center, hsize)

            pa_hst = hst.get_rot()
            pa_muse = self.get_rot()

            # extract a larger window to get the convolution ok on the edge
            if np.abs(pa_hst - pa_muse) > 1.e-3:
                subima = hst.subimage(center, (hsize + 2*seeing) * 1.5)
                subima = subima.rotate(pa_hst - pa_muse)
                zhst = subima.subimage(center, hsize + 2*seeing)
            else:
                zhst = hst.subimage(center, hsize + 2*seeing)

            if plot:
                rhst = zhst.subimage(center, hsize)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                plot_with_cross(zmuse, ax1, center, title='Low-res image')
                plot_with_cross(rhst, ax2, center, title='High-res image')
                plot_with_cross(rhst, ax3, center, zscale=True,
                                title='High-res image (zscale)')

            # Run the gaussian convolution
            chst = zhst.fftconvolve_gauss(fwhm=(seeing, seeing))
            # Rebin to muse spaxel size and window size
            rhst = chst.resample(zmuse.shape, zmuse.get_start(),
                                 zmuse.get_step(u.arcsec))

            if plot:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                zhst.plot(ax=ax1, title='High-resolution image')
                chst.plot(ax=ax2, title='Convolved with the gaussian FWHM')
                rhst.plot(ax=ax3, title='Resampled')

            # compute first autocorrelation center
            auto = zmuse.correlate2d(zmuse)
            autogauss = auto.gauss_fit(unit_center=None, unit_fwhm=None,
                                       verbose=False)
            # then crosscorrelation
            xcorr = zmuse.correlate2d(rhst)
            xcorrgauss = xcorr.gauss_fit(unit_center=None, unit_fwhm=None,
                                         verbose=False)
            # compute offset
            offset = (np.array(xcorrgauss.center) - np.array(autogauss.center))
            offsetarcsec = offset * zmuse.get_step(u.arcsec)
            info('Offset pixels: %s', offset)
            info('Offset arcsec: %s', offsetarcsec)
            info('Offset Peak: %s', xcorrgauss.peak)
            row['offset_pix'] = offset
            row['offset_arc'] = offsetarcsec
            row['offset_peak'] = xcorrgauss.peak

            if plot:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                plot_with_cross(auto, ax1, autogauss.center, cross_color='w',
                                center_deg=False, title='Auto-correlation')
                plot_with_cross(xcorr, ax2, xcorrgauss.center, cross_color='w',
                                center_deg=False, title='Cross-correlation')
                plt.show()

        weight = np.tile(t['offset_peak'], (2, 1)).T
        offpix = np.average(t['offset_pix'], weights=weight, axis=0)
        offset = np.average(t['offset_arc'], weights=weight, axis=0)
        info('Mean Offset: %s arcsec %s pixels', offset, offpix)

        if plot:
            # apply offset to full MUSE image
            wcs = self.wcs.copy()
            wcs.set_crpix1(wcs.get_crpix1() + offpix[1])
            wcs.set_crpix2(wcs.get_crpix2() + offpix[0])
            offmuse = self.copy()
            offmuse.set_wcs(wcs)

            t = catref[(catref['field'] == field) &
                       ((catref['type'] == 'CEN') | (catref['type'] == 'VIS'))]

            for k, row in enumerate(t):
                center = (t[k]['dec'], t[k]['ra'])
                zhst = hst.subimage(center, hsize)
                zmuse = self.subimage(center, hsize)
                zoffmuse = offmuse.subimage(center, hsize)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                plot_with_cross(zmuse, ax1, center, cross_color='w',
                                title='Original image')
                plot_with_cross(zoffmuse, ax2, center, cross_color='w',
                                title='With offset')
                plot_with_cross(zhst, ax3, center, cross_color='w',
                                title='High-res image')
            plt.show()

        self.unit = orig_unit
        return offset, offpix

    def plot(self, title=None, scale='linear', vmin=None, vmax=None,
             zscale=False, colorbar=None, var=False, show_xlabel=True,
             show_ylabel=True, ax=None, unit=u.deg, **kwargs):
        """Plot the image.

        Parameters
        ----------
        title : str
                Figure title (None by default).
        scale : 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'
                The stretch function to use for the scaling
                (default is 'linear').
        vmin : float
                Minimum pixel value to use for the scaling.
                If None, vmin is set to min of data.
        vmax : float
                Maximum pixel value to use for the scaling.
                If None, vmax is set to max of data.
        zscale : boolean
                If true, vmin and vmax are computed
                using the IRAF zscale algorithm.
        colorbar : boolean
                If 'h'/'v', a horizontal/vertical colorbar is added.
        var : boolean
                If var is True, the inverse of variance
                is overplotted.
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        unit : astropy.units
                   type of the world coordinates (degrees by default)
        kwargs : matplotlib.artist.Artist
                kwargs can be used to set additional Artist properties.

        Returns
        -------
        out : matplotlib AxesImage
        """
        if ax is None:
            ax = plt.gca()

        xunit = yunit = 'pixel'
        xlabel = 'q (%s)' % xunit
        ylabel = 'p (%s)' % yunit

        if self.shape[1] == 1:
            # plot a column
            yaxis = np.arange(self.shape[0], dtype=np.float)
            ax.plot(yaxis, self.data)
            xlabel = 'p (%s)' % yunit
            ylabel = self.unit
        elif self.shape[0] == 1:
            # plot a line
            xaxis = np.arange(self.shape[1], dtype=np.float)
            ax.plot(xaxis, self.data)
            ylabel = self.unit
        else:
            if zscale:
                vmin, vmax = plt_zscale.zscale(self.data.filled(0))
            if scale == 'log':
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif scale == 'arcsinh':
                norm = plt_norm.ArcsinhNorm(vmin=vmin, vmax=vmax)
            elif scale == 'power':
                norm = plt_norm.PowerNorm(vmin=vmin, vmax=vmax)
            elif scale == 'sqrt':
                norm = plt_norm.SqrtNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            if var and self.var is not None:
                wght = 1.0 / self.var
                np.ma.fix_invalid(wght, copy=False, fill_value=0)

                normalpha = mpl.colors.Normalize(wght.min(), wght.max())

                img_array = plt.get_cmap('jet')(norm(self.data))
                img_array[:, :, 3] = 1 - normalpha(wght) / 2
                cax = ax.imshow(img_array, interpolation='nearest',
                                origin='lower', norm=norm, **kwargs)
            else:
                cax = ax.imshow(self.data, interpolation='nearest',
                                origin='lower', norm=norm, **kwargs)

            # create colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            if colorbar == "h":
                cax2 = divider.append_axes("top", size="5%", pad=0.2)
                cbar = plt.colorbar(cax, cax=cax2, orientation='horizontal')
                for t in cbar.ax.xaxis.get_major_ticks():
                    t.tick1On = True
                    t.tick2On = True
                    t.label1On = False
                    t.label2On = True
            elif colorbar == "v":
                cax2 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax, cax=cax2)

            self._ax = ax

        if show_xlabel:
            ax.set_xlabel(xlabel)
        if show_ylabel:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        ax.format_coord = self._format_coord
        self._unit = unit
        return cax

    def _format_coord(self, x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < self.shape[0] and row >= 0 and row < self.shape[1]:
            pixsky = self.wcs.pix2sky([row, col], unit=self._unit)
            yc = pixsky[0][0]
            xc = pixsky[0][1]
            val = self.data.data[col, row]
            return 'y= %g x=%g p=%i q=%i data=%g' % (yc, xc, row, col, val)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    def ipos(self, filename='None'):
        """Print cursor position in interactive mode.

        p and q define the nearest pixel, x and y are the position, data
        contains the image data value (data[p,q]) .

        To read cursor position, click on the left mouse button.

        To remove a cursor position, click on the left mouse button + <d>

        To quit the interactive mode, click on the right mouse button.

        At the end, clicks are saved in self.clicks as dictionary
        {'y','x','p','q','data'}.

        Parameters
        ----------
        filename : str
            If filename is not None, the cursor values are saved as a fits
            table with columns labeled 'I'|'J'|'RA'|'DEC'|'DATA'.

        """
        info = self._logger.info
        info('To read cursor position, click on the left mouse button')
        info('To remove a cursor position, click on the left mouse button + '
             '<d>')
        info('To quit the interactive mode, click on the right mouse button.')
        info('After quit, clicks are saved in self.clicks as dictionary '
             '{y,x,p,q,data}.')

        if self._clicks is None:
            from ..gui.clicks import ImageClicks
            binding_id = plt.connect('button_press_event', self._on_click)
            self._clicks = ImageClicks(binding_id, filename)

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")
        else:
            self._clicks.filename = filename

    def _on_click(self, event):
        """prints dec,ra,i,j and data corresponding to the cursor position."""
        if event.key == 'd':
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        j, i = event.xdata, event.ydata
                        self._clicks.remove(i, j)
                        self._logger.info("new selection:")
                        for i in range(len(self._clicks.x)):
                            self._clicks.iprint(i)
                    except:
                        pass
        else:
            if event.button == 1:
                if event.inaxes is not None:
                    j, i = event.xdata, event.ydata
                    try:
                        i = int(i)
                        j = int(j)
                        [[y, x]] = self.wcs.pix2sky([i, j], unit=self._unit)
                        val = self.data[i, j]
                        if len(self._clicks.x) == 0:
                            print ''
                        self._clicks.add(i, j, x, y, val)
                        self._clicks.iprint(len(self._clicks.x) - 1)
                    except:
                        pass
            else:
                self._clicks.write_fits()
                # save clicks in a dictionary {'i','j','x','y','data'}
                d = {'p': self._clicks.p, 'q': self._clicks.q,
                     'x': self._clicks.x, 'y': self._clicks.y,
                     'data': self._clicks.data}
                self.clicks = d
                # clear
                self._clicks.clear()
                self._clicks = None
                fig = plt.gcf()
                fig.canvas.stop_event_loop_default()

    def idist(self):
        """Get distance and center from 2 cursor positions (interactive mode).

        To quit the interactive mode, click on the right mouse button.
        """
        self._logger.info('Use left mouse button to define the line.')
        self._logger.info('To quit the interactive mode, click on the right '
                          'mouse button.')
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_dist,
                                               drawtype='line')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_dist(self, eclick, erelease):
        """Print distance and center between 2 cursor positions."""
        if eclick.button == 1:
            try:
                j1, i1 = int(eclick.xdata), int(eclick.ydata)
                [[y1, x1]] = self.wcs.pix2sky([i1, j1], unit=self._unit)
                j2, i2 = int(erelease.xdata), int(erelease.ydata)
                [[y2, x2]] = self.wcs.pix2sky([i2, j2], unit=self._unit)
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                self._logger.info('Center: (%g,%g)\tDistance: %g Unit:%s', xc,
                                  yc, dist, self._unit)
            except:
                pass
        else:
            self._logger.info('idist deactivated.')
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def istat(self):
        """Compute image statistics from windows defined with left mouse button
        (mean is the mean value, median the median value, std is the rms
        standard deviation, sum the sum, peak the peak value, npts is the total
        number of points).

        To quit the interactive mode, click on the right mouse button.
        """
        self._logger.info('Use left mouse button to define the box.')
        self._logger.info('To quit the interactive mode, click on the right '
                          'mouse button.')
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_stat,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_stat(self, eclick, erelease):
        """Print image statistics from windows defined by 2 cursor
        positions."""
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                d = self.data[i1:i2, j1:j2]
                mean = np.ma.mean(d)
                median = np.ma.median(np.ma.ravel(d))
                vsum = d.sum()
                std = np.ma.std(d)
                npts = d.shape[0] * d.shape[1]
                peak = d.max()
                msg = 'mean=%g\tmedian=%g\tstd=%g\tsum=%g\tpeak=%g\tnpts=%d' \
                    % (mean, median, std, vsum, peak, npts)
                self._logger.info(msg)
            except:
                pass
        else:
            self._logger.info('istat deactivated.')
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def ipeak(self):
        """Find peak location in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        self._logger.info('Use left mouse button to define the box.')
        self._logger.info('To quit the interactive mode, click on the right '
                          'mouse button.')
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_peak,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_peak(self, eclick, erelease):
        """Print image peak location in windows defined by 2 cursor
        positions."""
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                center = ((i2 + i1) / 2, (j2 + j1) / 2)
                radius = (np.abs(i2 - i1) / 2, np.abs(j2 - j1) / 2)
                peak = self.peak(center, radius, unit_center=None,
                                 unit_radius=None)
                msg = 'peak: y=%g\tx=%g\tp=%d\tq=%d\tdata=%g' \
                    % (peak['y'], peak['x'], peak['p'], peak['q'], peak['data'])
                self._logger.info(msg)
            except:
                pass
        else:
            self._logger.info('ipeak deactivated.')
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def ifwhm(self):
        """Compute fwhm in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        self._logger.info('Use left mouse button to define the box.')
        self._logger.info('To quit the interactive mode, click on the right '
                          'mouse button.')
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_fwhm,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_fwhm(self, eclick, erelease):
        """Print image peak location in windows defined'\ 'by 2 cursor
        positions."""
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                center = ((i2 + i1) / 2, (j2 + j1) / 2)
                radius = (np.abs(i2 - i1) / 2, np.abs(j2 - j1) / 2)
                fwhm = self.fwhm(center, radius, unit_center=None,
                                 unit_radius=None)
                self._logger.info('fwhm_y=%g\tfwhm_x=%g [pixels]',
                                  fwhm[0], fwhm[1])
            except:
                pass
        else:
            self._logger.info('ifwhm deactivated.')
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def iee(self):
        """Compute enclosed energy in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        self._logger.info('Use left mouse button to define the box.')
        self._logger.info('To quit the interactive mode, click on the right '
                          'mouse button.')
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_ee,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_ee(self, eclick, erelease):
        """Print image peak location in windows defined by 2 cursor
        positions."""
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                center = ((i2 + i1) / 2, (j2 + j1) / 2)
                radius = (np.abs(i2 - i1) / 2, np.abs(j2 - j1) / 2)
                ee = self.ee(center, radius, unit_center=None,
                             unit_radius=None)
                self._logger.info('ee=%g', ee)
            except:
                pass
        else:
            self._logger.info('iee deactivated.')
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def imask(self):
        """Over-plots masked values (interactive mode)."""
        try:
            try:
                self._plot_mask_id.remove()
                # plt.draw()
            except:
                pass
            xaxis = np.arange(self.shape[1], dtype=np.float)
            yaxis = np.arange(self.shape[0], dtype=np.float)

            if np.shape(xaxis)[0] == 1:
                # plot a  column
                plt.plot(yaxis, self.data.data, alpha=0.3)
            elif np.shape(yaxis)[0] == 1:
                # plot a line
                plt.plot(xaxis, self.data.data, alpha=0.3)
            else:
                mask = np.array(1 - self.data.mask, dtype=bool)
                data = np.ma.MaskedArray(self.data.data, mask=mask)
                self._plot_mask_id = \
                    plt.imshow(data, interpolation='nearest', origin='lower',
                               extent=(0, self.shape[1] - 1, 0,
                                       self.shape[0] - 1),
                               vmin=self.data.min(), vmax=self.data.max(),
                               alpha=0.9)
        except:
            pass

    def igauss_fit(self):
        """Perform Gaussian fit in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        self._logger.info('Use left mouse button to define the box.')
        self._logger.info('To quit the interactive mode, click on the right '
                          'mouse button.')
        self._logger.info('The parameters of the last gaussian are saved in '
                          'self.gauss.')
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_gauss_fit,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_gauss_fit(self, eclick, erelease):
        if eclick.button == 1:
            try:
                q1 = int(min(eclick.xdata, erelease.xdata))
                q2 = int(max(eclick.xdata, erelease.xdata))
                p1 = int(min(eclick.ydata, erelease.ydata))
                p2 = int(max(eclick.ydata, erelease.ydata))
                pos_min = self.wcs.pix2sky([p1, q1], unit=self._unit)[0]
                pos_max = self.wcs.pix2sky([p2, q2], unit=self._unit)[0]
                self.gauss = self.gauss_fit(pos_min, pos_max, plot=True,
                                            unit_center=self._unit)
                self.gauss.print_param()
            except:
                pass
        else:
            self._logger.info('igauss_fit deactivated.')
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def imoffat_fit(self):
        """Perform Moffat fit in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        self._logger.info('Use left mouse button to define the box.')
        self._logger.info('To quit the interactive mode, click on the right '
                          'mouse button.')
        self._logger.info('The parameters of the last moffat fit are saved '
                          'in self.moffat.')
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_moffat_fit,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_moffat_fit(self, eclick, erelease):
        if eclick.button == 1:
            try:
                q1 = int(min(eclick.xdata, erelease.xdata))
                q2 = int(max(eclick.xdata, erelease.xdata))
                p1 = int(min(eclick.ydata, erelease.ydata))
                p2 = int(max(eclick.ydata, erelease.ydata))
                pos_min = self.wcs.pix2sky([p1, q1], unit=self._unit)[0]
                pos_max = self.wcs.pix2sky([p2, q2], unit=self._unit)[0]
                self.moffat = self.moffat_fit(pos_min, pos_max, plot=True,
                                              unit_center=self._unit)
                self.moffat.print_param()
            except:
                pass
        else:
            self._logger.info('imoffat_fit deactivated.')
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    @deprecated('rebin_factor method is deprecated: use rebin_mean instead')
    def rebin_factor(self, factor, margin='center'):
        return self.rebin_mean(factor, margin)

    @deprecated('rebin method is deprecated: use resample instead')
    def rebin(self, newdim, newstart, newstep, flux=False,
              order=3, interp='no', unit_start=u.deg, unit_step=u.arcsec):
        return self.resample(newdim, newstart, newstep, flux,
                             order, interp, unit_start, unit_step)


def gauss_image(shape=(101, 101), wcs=WCS(), factor=1, gauss=None,
                center=None, flux=1., fwhm=(1., 1.), peak=False, rot=0.,
                cont=0, unit_center=u.deg, unit_fwhm=u.arcsec,
                unit=u.dimensionless_unscaled):
    """Create a new image from a 2D gaussian.

    Parameters
    ----------
    shape : int or (int,int)
        Lengths of the image in Y and X with python notation: (ny,nx).
        (101,101) by default. If wcs object contains dimensions, shape is
        ignored and wcs dimensions are used.
    wcs : :class:`mpdaf.obj.WCS`
        World coordinates.
    factor : int
        If factor<=1, gaussian value is computed in the center of each pixel.
        If factor>1, for each pixel, gaussian value is the sum of the gaussian
        values on the factor*factor pixels divided by the pixel area.
    gauss : :class:`mpdaf.obj.Gauss2D`
        Object that contains all Gaussian parameters. If it is present, the
        following parameters are not used.
    center : (float,float)
        Gaussian center (y_peak, x_peak). If None the center of the image is
        used. The unit is given by the unit_center parameter (degrees by
        default).
    flux : float
        Integrated gaussian flux or gaussian peak value if peak is True.
    fwhm : (float,float)
        Gaussian fwhm (fwhm_y,fwhm_x).
        The unit is given by the unit_fwhm parameter (arcseconds by default).
    peak : boolean
        If true, flux contains a gaussian peak value.
    rot : float
        Angle position in degree.
    cont : float
        Continuum value. 0 by default.
    unit_center : astropy.units
        type of the center and position coordinates.
        Degrees by default (use None for coordinates in pixels).
    unit_fwhm : astropy.units
        FWHM unit.  Arcseconds by default (use None for radius in pixels)

    Returns
    -------
    out : :class:`mpdaf.obj.Image`

    """
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)

    if wcs.naxis1 == 1. and wcs.naxis2 == 1.:
        wcs.naxis1 = shape[1]
        wcs.naxis2 = shape[0]
    else:
        if wcs.naxis1 != 0. or wcs.naxis2 != 0.:
            shape[1] = wcs.naxis1
            shape[0] = wcs.naxis2

    if gauss is not None:
        center = gauss.center
        flux = gauss.flux
        fwhm = gauss.fwhm
        peak = False
        rot = gauss.rot
        cont = gauss.cont

    if center is None:
        center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])
    else:
        if unit_center is not None:
            center = wcs.sky2pix(center, unit=unit_center)[0]

    if unit_fwhm is not None:
        fwhm = np.array(fwhm) / np.abs(wcs.get_step(unit=unit_fwhm))

    data = np.empty(shape=shape, dtype=float)

    if fwhm[1] == 0 or fwhm[0] == 0:
        raise ValueError('fwhm equal to 0')
    p_width = fwhm[0] / 2.0 / np.sqrt(2 * np.log(2))
    q_width = fwhm[1] / 2.0 / np.sqrt(2 * np.log(2))

    # rotation angle in rad
    theta = np.pi * rot / 180.0

    if peak is True:
        I = flux * np.sqrt(2 * np.pi * (p_width ** 2)) \
            * np.sqrt(2 * np.pi * (q_width ** 2))
    else:
        I = flux

    gauss = lambda p, q: I * (1 / np.sqrt(2 * np.pi * (p_width ** 2))) \
        * np.exp(-((p - center[0]) * np.cos(theta)
                   - (q - center[1]) * np.sin(theta)) ** 2
                 / (2 * p_width ** 2)) \
        * (1 / np.sqrt(2 * np.pi * (q_width ** 2))) \
        * np.exp(-((p - center[0]) * np.sin(theta)
                   + (q - center[1]) * np.cos(theta)) ** 2
                 / (2 * q_width ** 2))

    if factor > 1:
        if rot == 0:
            from scipy import special

            X, Y = np.meshgrid(xrange(shape[0]), xrange(shape[1]))
            pixcrd_min = np.array(zip(X.ravel(), Y.ravel())) - 0.5
            # pixsky_min = wcs.pix2sky(pixcrd)
            xmin = (pixcrd_min[:, 1] - center[1]) / np.sqrt(2.0) / q_width
            ymin = (pixcrd_min[:, 0] - center[0]) / np.sqrt(2.0) / p_width

            pixcrd_max = np.array(zip(X.ravel(), Y.ravel())) + 0.5
            # pixsky_max = wcs.pix2sky(pixcrd)
            xmax = (pixcrd_max[:, 1] - center[1]) / np.sqrt(2.0) / q_width
            ymax = (pixcrd_max[:, 0] - center[0]) / np.sqrt(2.0) / p_width

            dx = pixcrd_max[:, 1] - pixcrd_min[:, 1]
            dy = pixcrd_max[:, 0] - pixcrd_min[:, 0]
            data = I * 0.25 / dx / dy \
                * (special.erf(xmax) - special.erf(xmin)) \
                * (special.erf(ymax) - special.erf(ymin))
            data = np.reshape(data, (shape[1], shape[0])).T
        else:
            X, Y = np.meshgrid(xrange(shape[0] * factor),
                               xrange(shape[1] * factor))
            factor = float(factor)
            pixcrd = np.array(zip(X.ravel() / factor, Y.ravel() / factor))
            # pixsky = wcs.pix2sky(pixcrd)
            data = gauss(pixcrd[:, 0], pixcrd[:, 1])
            data = (data.reshape(shape[1], factor, shape[0], factor)
                    .sum(1).sum(2) / factor / factor).T
    else:
        X, Y = np.meshgrid(xrange(shape[0]), xrange(shape[1]))
        pixcrd = np.array(zip(X.ravel(), Y.ravel()))
        # data = gauss(pixcrd[:,1],pixcrd[:,0])
        data = gauss(pixcrd[:, 0], pixcrd[:, 1])
        data = np.reshape(data, (shape[1], shape[0])).T

    return Image(data=data + cont, wcs=wcs, unit=unit, copy=False, dtype=None)


def moffat_image(shape=(101, 101), wcs=WCS(), factor=1, moffat=None,
                 center=None, flux=1., fwhm=(1., 1.), peak=False, n=2,
                 rot=0., cont=0, unit_center=u.deg, unit_fwhm=u.arcsec,
                 unit=u.dimensionless_unscaled):
    """Create a new image from a 2D Moffat function.

    Parameters
    ----------
    shape : int or (int,int)
        Lengths of the image in Y and X with python notation: (ny,nx).
        (101,101) by default. If wcs object contains dimensions, shape is
        ignored and wcs dimensions are used.
    wcs : :class:`mpdaf.obj.WCS`
        World coordinates.
    factor : int
        If factor<=1, moffat value is computed in the center of each pixel.
        If factor>1, for each pixel, moffat value is the sum
        of the moffat values on the factor*factor pixels divided
        by the pixel area.
    moffat : :class:`mpdaf.obj.Moffat2D`
        object that contains all moffat parameters.
        If it is present, following parameters are not used.
    center : (float,float)
        Peak center (x_peak, y_peak). The unit is genven byt the parameter
        unit_center (degrees by default). If None the center of the image is
        used.
    flux : float
        Integrated gaussian flux or gaussian peak value
                  if peak is True.
    fwhm : (float,float)
        Gaussian fwhm (fwhm_y,fwhm_x).
        The unit is given by the parameter unit_fwhm (arcseconds by default)
    peak : boolean
        If true, flux contains a gaussian peak value.
    n : int
        Atmospheric scattering coefficient. 2 by default.
    rot : float
        Angle position in degree.
    cont : float
        Continuum value. 0 by default.
    unit_center : astropy.units
        type of the center and position coordinates.
        Degrees by default (use None for coordinates in pixels).
    unit_fwhm : astropy.units
        FWHM unit. Arcseconds by default (use None for radius in pixels)

    Returns
    -------
    out : :class:`mpdaf.obj.Image`

    """
    n = float(n)
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)

    if wcs.naxis1 == 1. and wcs.naxis2 == 1.:
        wcs.naxis1 = shape[1]
        wcs.naxis2 = shape[0]
    else:
        if wcs.naxis1 != 0. or wcs.naxis2 != 0.:
            shape[1] = wcs.naxis1
            shape[0] = wcs.naxis2

    if moffat is not None:
        center = moffat.center
        flux = moffat.flux
        fwhm = moffat.fwhm
        peak = False
        n = moffat.n
        rot = moffat.rot
        cont = moffat.cont

    fwhm = np.array(fwhm)
    a = fwhm[0] / (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
    e = fwhm[0] / fwhm[1]

    if unit_fwhm is not None:
        a = a / np.abs(wcs.get_step(unit=unit_fwhm)[0])

    if peak:
        I = flux
    else:
        I = flux * (n - 1) / (np.pi * a * a * e)

    if center is None:
        center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])
    else:
        if unit_center is not None:
            center = wcs.sky2pix(center, unit=unit_center)[0]

    data = np.empty(shape=shape, dtype=float)

    # rotation angle in rad
    theta = np.pi * rot / 180.0

    moffat = lambda p, q: \
        I * (1 + (((p - center[0]) * np.cos(theta)
                   - (q - center[1]) * np.sin(theta)) / a) ** 2
             + (((p - center[0]) * np.sin(theta)
                 + (q - center[1]) * np.cos(theta)) / a / e) ** 2) ** (-n)

    if factor > 1:
        X, Y = np.meshgrid(xrange(shape[0] * factor),
                           xrange(shape[1] * factor))
        factor = float(factor)
        pixcrd = np.array(zip(X.ravel() / factor, Y.ravel() / factor))
        data = moffat(pixcrd[:, 0], pixcrd[:, 1])
        data = (data.reshape(shape[1], factor, shape[0], factor)
                .sum(1).sum(2) / factor / factor).T
    else:
        X, Y = np.meshgrid(xrange(shape[0]), xrange(shape[1]))
        pixcrd = np.array(zip(X.ravel(), Y.ravel()))
        data = moffat(pixcrd[:, 0], pixcrd[:, 1])
        data = np.reshape(data, (shape[1], shape[0])).T

    return Image(data=data + cont, wcs=wcs, unit=unit, copy=False, dtype=None)


def make_image(x, y, z, steps, deg=True, limits=None, spline=False, order=3,
               smooth=0, unit=u.dimensionless_unscaled):
    """Interpolate z(x,y) and returns an image.

    Parameters
    ----------
    x : float array
        Coordinate array corresponding to the declinaison.
    y : float array
        Coordinate array corresponding to the right ascension.
    z : float array
        Input data.
    steps : (float,float)
        Steps of the output image (dy,dRx).
    deg : boolean
        If True, world coordinates are in decimal degrees
        (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg').
        If False (by default), world coordinates are linear
        (CTYPE1=CTYPE2='LINEAR').
    limits : (float,float,float,float)
        Limits of the image (y_min,x_min,y_max,x_max).
        If None, minum and maximum values of x,y arrays are used.
    spline : boolean
        False: bilinear interpolation (uses :func:`scipy.interpolate.griddata`)
        True: spline interpolation (uses :func:`scipy.interpolate.bisplrep` and
        :func:`scipy.interpolate.bisplev`).
    order : int
        Polynomial order for spline interpolation (default 3)
    smooth : float
        Smoothing parameter for spline interpolation (default 0: no smoothing)

    Returns
    -------
    out : :class:`mpdaf.obj.Image`

    """
    if limits is None:
        x1 = x.min()
        x2 = x.max()
        y1 = y.min()
        y2 = y.max()
    else:
        x1, x2, y1, y2 = limits
    dx, dy = steps
    nx = int((x2 - x1) / dx + 1.5)
    ny = int((y2 - y1) / dy + 1.5)

    wcs = WCS(crpix=(1, 1), crval=(x1, y1),
              cdelt=(dx, dy), deg=deg, shape=(nx, ny))

    xi = np.arange(nx) * dx + x1
    yi = np.arange(ny) * dy + y1

    Y, X = np.meshgrid(y, x)

    if spline:
        tck = interpolate.bisplrep(X, Y, z, s=smooth, kx=order, ky=order)
        data = interpolate.bisplev(xi, yi, tck)
    else:
        n = np.shape(x)[0] * np.shape(y)[0]
        points = np.empty((n, 2), dtype=float)
        points[:, 0] = X.ravel()[:]
        points[:, 1] = Y.ravel()[:]
        Yi, Xi = np.meshgrid(yi, xi)
        data = interpolate.griddata(points, z.ravel(),
                                    (Xi, Yi), method='linear')

    return Image(data=data, wcs=wcs, unit=unit, copy=False, dtype=None)


def composite_image(ImaColList, mode='lin', cuts=(10, 90),
                    bar=False, interp='no'):
    """Build composite image from a list of image and colors.

    Parameters
    ----------
    ImaColList : list of tuple (Image,float,float)
        List of images and colors [(Image, hue, saturation)].
    mode : 'lin' or 'sqrt'
        Intensity mode. Use 'lin' for linear and 'sqrt' for root square.
    cuts : (float,float)
        Minimum and maximum in percent.
    bar : boolean
        If bar is True a color bar image is created.
    interp : 'no' | 'linear' | 'spline'
        if 'no', data median value replaced masked values.
        if 'linear', linear interpolation of the masked values.
        if 'spline', spline interpolation of the masked values.

    Returns
    -------
    out : Returns a PIL RGB image (or 2 PIL images if bar is True).

    """
    from PIL import Image as PILima
    from PIL import ImageColor
    from PIL import ImageChops

    # compute statistic of intensity and derive cuts
    first = True
    for ImaCol in ImaColList:
        ima, col, sat = ImaCol

        if interp == 'linear':
            data = ima._interp_data(spline=False)
        elif interp == 'spline':
            data = ima._interp_data(spline=True)
        else:
            data = np.ma.filled(ima.data, np.ma.median(ima.data))

        if mode == 'lin':
            f = data
        elif mode == 'sqrt':
            f = np.sqrt(np.clip(data, 0, 1.e99))
        else:
            raise ValueError('Wrong cut mode')
        if first:
            d = f.ravel()
            first = False
        else:
            d = np.concatenate([d, f.ravel()])
    d.sort()
    k1, k2 = cuts
    d1 = d[max(int(0.01 * k1 * len(d) + 0.5), 0)]
    d2 = d[min(int(0.01 * k2 * len(d) + 0.5), len(d) - 1)]

    # first image
    ima, col, sat = ImaColList[0]
    p1 = PILima.new('RGB', (ima.shape[0], ima.shape[1]))
    if interp == 'linear':
        data = ima._interp_data(spline=False)
    elif interp == 'spline':
        data = ima._interp_data(spline=True)
    else:
        data = np.ma.filled(ima.data, np.ma.median(ima.data))
    if mode == 'lin':
        f = data
    elif mode == 'sqrt':
        f = np.sqrt(np.clip(data, 0, 1.e99))
    lum = np.clip((f - d1) * 100 / (d2 - d1), 0, 100)
    for i in range(ima.shape[0]):
        for j in range(ima.shape[1]):
            p1.putpixel((i, j), ImageColor.getrgb(
                'hsl(%d,%d%%,%d%%)' % (int(col), int(sat), int(lum[i, j]))))

    for ImaCol in ImaColList[1:]:
        ima, col, sat = ImaCol
        p2 = PILima.new('RGB', (ima.shape[0], ima.shape[1]))
        if interp == 'linear':
            data = ima._interp_data(spline=False)
        elif interp == 'spline':
            data = ima._interp_data(spline=True)
        else:
            data = np.ma.filled(ima.data, np.ma.median(ima.data))
        if mode == 'lin':
            f = data
        elif mode == 'sqrt':
            f = np.sqrt(np.clip(data, 0, 1.e99))
        lum = np.clip((f - d1) * 100 / (d2 - d1), 0, 100)
        for i in range(ima.shape[0]):
            for j in range(ima.shape[1]):
                p2.putpixel((i, j), ImageColor.getrgb(
                    'hsl(%d,%d%%,%d%%)' % (int(col), int(sat), int(lum[i, j]))))
        p1 = ImageChops.add(p1, p2)

    if bar:
        nxb = ima.shape[0]
        nyb = 50
        dx = nxb / len(ImaColList)
        p3 = PILima.new('RGB', (nxb, nyb))
        i1 = 0
        for ImaCol in ImaColList:
            ima, col, sat = ImaCol
            for i in range(i1, i1 + dx):
                for j in range(nyb):
                    p3.putpixel((i, j), ImageColor.getrgb(
                        'hsl(%d,%d%%,%d%%)' % (int(col), int(sat), 50)))
            i1 += dx

    if bar:
        return p1, p3
    else:
        return p1


def mask_image(shape=(101, 101), wcs=WCS(), objects=[],
               unit=u.dimensionless_unscaled):
    """Create a new image from a table of apertures.

    ra(deg), dec(deg) and radius(arcsec).

    Parameters
    ----------
    shape : int or (int,int)
        Lengths of the image in Y and X with python notation: (ny,nx).
        (101,101) by default. If wcs object contains dimensions, shape is
        ignored and wcs dimensions are used.
    wcs : :class:`mpdaf.obj.WCS`
        World coordinates.
    objects : list of (float, float, float)
        (y, x, size) describes an aperture on the sky, defined by a center
        (y, x) in degrees, and size (radius) in arcsec.

    Returns
    -------
    out : :class:`mpdaf.obj.Image`

    """
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)
    if wcs.naxis1 == 1. and wcs.naxis2 == 1.:
        wcs.naxis1 = shape[1]
        wcs.naxis2 = shape[0]
    else:
        if wcs.naxis1 != 0. or wcs.naxis2 != 0.:
            shape[1] = wcs.naxis1
            shape[0] = wcs.naxis2
    data = np.zeros(shape)
    for y, x, r in objects:
        center = wcs.sky2pix([y, x], unit=u.deg)[0]
        r = np.array(r) / np.abs(wcs.get_step(unit=u.arcsec))
        r2 = r[0] * r[1]
        imin = max(0, center[0] - r[0])
        imax = min(center[0] + r[0] + 1, shape[0])
        jmin = max(0, center[1] - r[1])
        jmax = min(center[1] + r[1] + 1, shape[1])
        grid = np.meshgrid(np.arange(imin, imax) - center[0],
                           np.arange(jmin, jmax) - center[1], indexing='ij')
        data[imin:imax, jmin:jmax] = np.array(
            (grid[0] ** 2 + grid[1] ** 2) < r2, dtype=int)
    return Image(data=data, wcs=wcs, unit=unit, copy=False, dtype=None)
