"""cube.py manages Cube objects."""

import datetime
import logging
import multiprocessing
import numpy as np
import sys
import time
import types
import warnings

import astropy.units as u
from astropy.io import fits as pyfits

from .coords import WCS, WaveCoord
from .data import DataArray
from .image import Image
from .objs import is_float, is_int, UnitArray, UnitMaskedArray
from .spectrum import Spectrum

__all__ = ['iter_spe', 'iter_ima', 'Cube', 'CubeDisk']


class iter_spe(object):

    def __init__(self, cube, index=False):
        self.cube = cube
        self.p = cube.shape[1]
        self.q = cube.shape[2]
        self.index = index

    def next(self):
        """Returns the next spectrum."""
        if self.q == 0:
            self.p -= 1
            self.q = self.cube.shape[2]
        self.q -= 1
        if self.p == 0:
            raise StopIteration
        if self.index is False:
            return self.cube[:, self.p - 1, self.q]
        else:
            return (self.cube[:, self.p - 1, self.q], (self.p - 1, self.q))

    def __iter__(self):
        """Returns the iterator itself."""
        return self


class iter_ima(object):

    def __init__(self, cube, index=False):
        self.cube = cube
        self.k = cube.shape[0]
        self.index = index

    def next(self):
        """Returns the next image."""
        if self.k == 0:
            raise StopIteration
        self.k -= 1
        if self.index is False:
            return self.cube[self.k, :, :]
        else:
            return (self.cube[self.k, :, :], self.k)

    def __iter__(self):
        """Returns the iterator itself."""
        return self


class Cube(DataArray):

    """This class manages Cube objects.

    Parameters
    ----------
    filename : string
                Possible FITS file name. None by default.
    ext      : integer or (integer,integer) or string or (string,string)
                Number/name of the data extension
                or numbers/names of the data and variance extensions.
    notnoise : boolean
                True if the noise Variance cube is not read (if it exists).
                Use notnoise=True to create cube without variance extension.
    shape    : integer or (integer,integer,integer)
                Lengths of data in Z, Y and X. Python notation is used
                (nz,ny,nx). If data is not None, its shape is used instead.
    wcs      : :class:`mpdaf.obj.WCS`
                World coordinates.
    wave     : :class:`mpdaf.obj.WaveCoord`
                Wavelength coordinates.
    unit     : string
                Possible data unit type. None by default.
    data     : float array
                Array containing the pixel values of the cube. None by
                default.
    var      : float array
                Array containing the variance. None by default.

    Attributes
    ----------
    filename       : string
                     Possible FITS filename.
    unit           : string
                     Possible data unit type
    primary_header : pyfits.Header
                     FITS primary header instance.
    data_header    : pyfits.Header
                     FITS data header instance.
    data           : masked array numpy.ma
                     Array containing the cube pixel values.
    shape          : array of 3 integers
                     Lengths of data in Z and Y and X
                     (python notation (nz,ny,nx)).
    var            : float array
                     Array containing the variance.
    wcs            : :class:`mpdaf.obj.WCS`
                     World coordinates.
    wave           : :class:`mpdaf.obj.WaveCoord`
                     Wavelength coordinates
    ima            : dict{string,:class:`mpdaf.obj.Image`}
                     dictionary of images
    """

    _ndim = 3
    _has_wcs = True
    _has_wave = True

    def __init__(self, filename=None, ext=None, notnoise=False, wcs=None,
                 wave=None, unit=u.count, data=None, var=None,
                 shape=None, ima=True, copy=True, dtype=float):
        super(Cube, self).__init__(
            filename=filename, ext=ext, notnoise=notnoise, wcs=wcs, wave=wave,
            unit=unit, data=data, var=var, copy=copy, dtype=dtype, shape=shape)
        self.cube = True
        self.ima = {}

        if filename is not None and ima:
            hdulist = pyfits.open(filename)
            for hdu in hdulist:
                try:
                    hdr = hdu.header
                    if hdr['NAXIS'] == 2 and hdr['XTENSION'] == 'IMAGE':
                        self.ima[hdr.get('EXTNAME')] = Image(
                            filename, ext=hdr.get('EXTNAME'), notnoise=True)
                except:
                    pass
            hdulist.close()

    def copy(self):
        """Returns a new copy of a Cube object."""
        obj = super(Cube, self).copy()
        for key, ima in self.ima:
            obj.ima[key] = ima.copy()
        return obj

    def info(self):
        """Prints information."""
        super(Cube, self).info()
        if len(self.ima) > 0:
            d = {'class': self.__class__.__name__, 'method': 'info'}
            self.logger.info('.ima: %s', ', '.join(self.ima.keys()), extra=d)

    def get_data_hdu(self, name='DATA', savemask='dq'):
        """ Returns astropy.io.fits.ImageHDU corresponding to the DATA extension

        Parameters
        ----------
        name     : string
                   Extension name.
                   DATA by default
        savemask : string
                   If 'dq', the mask array is saved in DQ extension.
                   If 'nan', masked data are replaced by nan in DATA extension.
                   If 'none', masked array is not saved.

        Returns
        -------
        out : astropy.io.fits.ImageHDU
        """
        if self.data.dtype == np.float64:
            self.data = self.data.astype(np.float32)

        # world coordinates
        hdr = self.wcs.to_cube_header(self.wave)

        # create scube DATA extension
        if savemask == 'nan':
            data = self.data.filled(fill_value=np.nan)
        else:
            data = self.data.data
        imahdu = pyfits.ImageHDU(name=name, data=data, header=hdr)

        for card in self.data_header.cards:
            to_copy = (card.keyword[0:2] not in ('CD', 'PC') and
                       card.keyword not in imahdu.header)
            if to_copy:
                try:
                    card.verify('fix')
                    imahdu.header[card.keyword] = (card.value, card.comment)
                except:
                    try:
                        if isinstance(card.value, str):
                            n = 80 - len(card.keyword) - 14
                            s = card.value[0:n]
                            imahdu.header['hierarch %s' % card.keyword] = \
                                (s, card.comment)
                        else:
                            imahdu.header['hierarch %s' % card.keyword] = \
                                (card.value, card.comment)
                    except:
                        d = {'class': 'Cube', 'method': 'write'}
                        self.logger.warning("%s not copied in data header",
                                            card.keyword, extra=d)

        if self.unit is not None:
            imahdu.header['BUNIT'] = ("{}".format(self.unit), 'data unit type')

        return imahdu

    def get_stat_hdu(self, name='STAT', header=None):
        """ Returns astropy.io.fits.ImageHDU corresponding to the STAT extension

        Parameters
        ----------
        name     : string
                   Extension name.
                   STAT by default

        Returns
        -------
        out : astropy.io.fits.ImageHDU
        """
        if self.var is None:
            return None

        d = {'class': 'Cube', 'method': 'write'}

        if self.var.dtype == np.float64:
            self.var = self.var.astype(np.float32)

        # world coordinates
        if header is None:
            header = self.wcs.to_cube_header(self.wave)

        imahdu = pyfits.ImageHDU(name=name, data=self.var, header=header)

        if header is None:
            for card in self.data_header.cards:
                to_copy = (card.keyword[0:2] not in ('CD', 'PC')
                           and card.keyword not in imahdu.header)
                if to_copy:
                    try:
                        card.verify('fix')
                        imahdu.header[card.keyword] = (card.value, card.comment)
                    except:
                        try:
                            if isinstance(card.value, str):
                                n = 80 - len(card.keyword) - 14
                                s = card.value[0:n]
                                imahdu.header['hierarch %s' % card.keyword] = \
                                    (s, card.comment)
                            else:
                                imahdu.header['hierarch %s' % card.keyword] = \
                                    (card.value, card.comment)
                        except:
                            self.logger.warning("%s not copied in data header",
                                                card.keyword, extra=d)

        if self.unit is not None:
            imahdu.header['BUNIT'] = ("{}".format(self.unit**2), 'data unit type')

        return imahdu

    def write(self, filename, savemask='dq'):
        """Saves the cube in a FITS file.

        Parameters
        ----------
        filename : string
                The FITS filename.
        savemask : string
                If 'dq', the mask array is saved in DQ extension
                If 'nan', masked data are replaced by nan in DATA extension.
                If 'none', masked array is not saved.
        """
        # create primary header
        warnings.simplefilter("ignore")
        prihdu = pyfits.PrimaryHDU()
        for card in self.primary_header.cards:
            try:
                card.verify('fix')
                prihdu.header[card.keyword] = (card.value, card.comment)
            except:
                try:
                    if isinstance(card.value, str):
                        n = 80 - len(card.keyword) - 14
                        s = card.value[0:n]
                        prihdu.header['hierarch %s' % card.keyword] = \
                            (s, card.comment)
                    else:
                        prihdu.header['hierarch %s' % card.keyword] = \
                            (card.value, card.comment)
                except:
                    d = {'class': 'Cube', 'method': 'write'}
                    self.logger.warning("%s not copied in primary header",
                                        card.keyword, extra=d)

        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        hdulist = [prihdu]
        warnings.simplefilter("default")

        # create cube DATA extension
        datahdu = self.get_data_hdu('DATA', savemask)
        hdulist.append(datahdu)

        # create spectrum STAT extension
        if self.var is not None:
            stathdu = self.get_stat_hdu('STAT', datahdu.header)
            hdulist.append(stathdu)

        # create DQ extension
        if savemask == 'dq' and np.ma.count_masked(self.data) != 0:
            dqhdu = pyfits.ImageHDU(name='DQ', data=np.uint8(self.data.mask))
            hdulist.append(dqhdu)

        # save to disk
        hdu = pyfits.HDUList(hdulist)
        warnings.simplefilter("ignore")
        hdu.writeto(filename, clobber=True, output_verify='silentfix')
        warnings.simplefilter("default")

        self.filename = filename

    def __le__(self, item):
        """Masks data array where greater than a given value.

        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item)
        return result

    def __lt__(self, item):
        """Masks data array where greater or equal than a given value.

        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item)
        return result

    def __ge__(self, item):
        """Masks data array where less than a given value.

        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item)
        return result

    def __gt__(self, item):
        """Masks data array where less or equal than a given value.

        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item)
        return result

    def resize(self):
        """Resizes the cube to have a minimum number of masked values."""
        if self.data is not None:
            ksel = np.where(~self.data.mask)
            item = (slice(ksel[0][0], ksel[0][-1] + 1, None),
                    slice(ksel[1][0], ksel[1][-1] + 1, None),
                    slice(ksel[2][0], ksel[2][-1] + 1, None))

            self.data = self.data[item]
            if is_int(item[0]):
                if is_int(item[1]):
                    self.data = self.data[np.newaxis, np.newaxis, :]
                elif is_int(item[2]):
                    self.data = self.data[np.newaxis, :, np.newaxis]
                else:
                    self.data = self.data[np.newaxis, :, :]
            elif is_int(item[1]):
                if is_int(item[2]):
                    self.data = self.data[:, np.newaxis, np.newaxis]
                else:
                    self.data = self.data[:, np.newaxis, :]
            elif is_int(item[2]):
                self.data = self.data[:, :, np.newaxis]

            if self.var is not None:
                self.var = self.var[item]

            try:
                self.wcs = self.wcs[item[1], item[2]]
            except:
                self.wcs = None
                d = {'class': 'Cube', 'method': 'resize'}
                self.logger.warning("wcs not copied: wcs attribute is None",
                                    extra=d)

            try:
                self.wave = self.wave[item[0]]
            except:
                self.wave = None
                d = {'class': 'Cube', 'method': 'resize'}
                self.logger.warning("wavelength solution not copied: "
                                    "wave attribute is None", extra=d)

    def unmask(self):
        """Unmasks the cube (just invalid data (nan,inf) are masked)."""
        self.data.mask = False
        self.data = np.ma.masked_invalid(self.data)

    def mask(self, center, radius, lmin=None, lmax=None, inside=True,
             unit_center=u.deg, unit_radius=u.arcsec, unit_wave=u.angstrom):
        """Masks values inside/outside the described region.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
        radius : float or (float,float)
                 Radius defined the explored region.
                 If radius is float, it defined a circular region.
                 If radius is (float,float), it defined a rectangular region.
        lmin   : float
                 minimum wavelength.
        lmax   : float
                 maximum wavelength.
        inside : boolean
                 If inside is True, pixels inside the described region are masked.
                 If inside is False, pixels outside the described region are masked.
        unit_wave : astropy.units
                    Type of the wavelengths coordinates (Angstrom by default)
                    If None, inputs are in pixels
        unit_center : astropy.units
                      Type of the coordinates of the center (degrees by default)
                      If None, inputs are in pixels
        unit_radius : astropy.units
                      Radius unit (arcseconds by default)
                      If None, inputs are in pixels
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

        imin, jmin = np.maximum(np.minimum((center - radius + 0.5).astype(int),
                                           [self.shape[1] - 1, self.shape[2] - 1]),
                                [0, 0])
        imax, jmax = np.maximum(np.minimum((center + radius + 0.5).astype(int),
                                           [self.shape[1] - 1, self.shape[2] - 1]),
                                [0, 0])
        imax += 1
        jmax += 1

        if inside and not circular:
            self.data.mask[lmin:lmax, imin:imax, jmin:jmax] = 1
        elif inside and circular:
            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1],
                               indexing='ij')
            grid3d = np.resize((grid[0] ** 2 + grid[1] ** 2) < radius2,
                               (lmax - lmin, imax - imin, jmax - jmin))
            self.data.mask[lmin:lmax, imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[lmin:lmax, imin:imax, jmin:jmax],
                              grid3d)
        elif not inside and circular:
            self.data.mask[:lmin, :, :] = 1
            self.data.mask[lmax:, :, :] = 1
            self.data.mask[:, :imin, :] = 1
            self.data.mask[:, imax:, :] = 1
            self.data.mask[:, :, :jmin] = 1
            self.data.mask[:, :, jmax:] = 1
            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1], indexing='ij')
            grid3d = np.resize((grid[0] ** 2 + grid[1] ** 2) > radius2,
                               (lmax - lmin, imax - imin, jmax - jmin))
            self.data.mask[lmin:lmax, imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[lmin:lmax, imin:imax, jmin:jmax],
                              grid3d)
        else:
            self.data.mask[:lmin, :, :] = 1
            self.data.mask[lmax:, :, :] = 1
            self.data.mask[:, :imin, :] = 1
            self.data.mask[:, imax:, :] = 1
            self.data.mask[:, :, :jmin] = 1
            self.data.mask[:, :, jmax:] = 1

    def mask_ellipse(self, center, radius, posangle, lmin=None, lmax=None, pix=False, inside=True,
                     unit_center=u.deg, unit_radius=u.arcsec, unit_wave=u.angstrom):
        """Masks values inside/outside the described region. Uses an elliptical
        shape.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
        radius : (float,float)
                 Radius defined the explored region.
                 radius is (float,float), it defines an elliptical region
                 with semi-major and semi-minor axes.
        posangle : float
                 Position angle of the first axis.
                 It is defined in degrees against the horizontal (q) axis
                 of the image, counted counterclockwise.
        lmin   : float
                 minimum wavelength.
        lmax   : float
                 maximum wavelength.
        inside : boolean
                 If inside is True, pixels inside the described region are masked.
        unit_wave : astropy.units
                    Type of the wavelengths coordinates (Angstrom by default)
                    If None, inputs are in pixels
        unit_center : astropy.units
                      Type of the coordinates of the center (degrees by default)
                      If None, inputs are in pixels
        unit_radius : astropy.units
                      Radius unit (arcseconds by default)
                      If None, inputs are in pixels
        """
        center = np.array(center)
        radius = np.array(radius)
        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]
        if unit_radius is not None:
            radius = radius / np.abs(self.wcs.get_step(unit=unit_radius))
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
            self.data.mask[lmin:lmax, imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[lmin:lmax, imin:imax, jmin:jmax], grid3d)
        if not inside:
            self.data.mask[:lmin, :, :] = 1
            self.data.mask[lmax:, :, :] = 1
            self.data.mask[:, :imin, :] = 1
            self.data.mask[:, imax:, :] = 1
            self.data.mask[:, :, :jmin] = 1
            self.data.mask[:, :, jmax:] = 1

            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1], indexing='ij')

            grid3d = np.resize(((grid[1] * cospa + grid[0] * sinpa) / radius[0]) ** 2
                               + ((grid[0] * cospa - grid[1] * sinpa)
                                  / radius[1]) ** 2 > 1,
                               (lmax - lmin, imax - imin, jmax - jmin))
            self.data.mask[lmin:lmax, imin:imax, jmin:jmax] = \
                np.logical_or(self.data.mask[lmin:lmax, imin:imax, jmin:jmax], grid3d)

    def mask_variance(self, threshold):
        """Masks pixels with a variance upper than threshold value.

        Parameters
        ----------
        threshold : float
                    Threshold value.
        """
        if self.var is None:
            raise ValueError('Operation forbidden'
                             ' without variance extension.')
        else:
            ksel = np.where(self.var > threshold)
            self.data[ksel] = np.ma.masked

    def mask_selection(self, ksel):
        """Masks pixels corresponding to the selection.

        Parameters
        ----------
        ksel : output of np.where
               elements depending on a condition
        """
        self.data[ksel] = np.ma.masked

    def __add__(self, other):
        """Adds other.

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
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                typ=1
        except:
            try:
                if other.image:
                    typ=2
            except:
                try:
                    if other.cube:
                        typ=3
                except:
                    typ=0

        if typ==0:
            try:
                res = self.copy()
                res.data = self.data + other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if typ==1 or typ==3:
                if self.wave is not None and other.wave is not None \
                and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spectral direction')
            if typ==2 or typ==3:
                if self.wcs is not None and other.wcs is not None \
                and not self.wcs.isEqual(other.wcs):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spatial directions')

            if typ==1:
                # cube1 + spectrum = cube2
                if other.data is None or other.shape != self.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data + other.data[:, np.newaxis, np.newaxis]
                else:
                    res.data = self.data + UnitMaskedArray(other.data[:, np.newaxis, np.newaxis],other.unit,self.unit)
                # variance
                if other.var is not None:
                    if self.var is None:
                        if other.unit == self.unit:
                            res.var = np.ones(self.shape) * other.var[:, np.newaxis, np.newaxis]
                        else:
                            res.var = np.ones(self.shape) \
                            * UnitArray(other.var[:, np.newaxis, np.newaxis],
                                              other.unit**2,
                                              self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res.var = self.var + other.var[:, np.newaxis, np.newaxis]
                        else:
                            res.var = self.var \
                            + UnitArray(other.var[:, np.newaxis, np.newaxis],
                                              other.unit**2, self.unit**2)
                return res
            elif typ==2:
                # cube1 + image = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                if other.data is None or self.shape[2] != other.shape[1] \
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
                if other.var is not None:
                    if self.var is None:
                        if self.unit == other.unit:
                            res.var = np.ones(self.shape) * other.var[np.newaxis, :, :]
                        else:
                            res.var = np.ones(self.shape) \
                            * UnitArray(other.var[np.newaxis, :, :],
                                              other.unit**2, self.unit**2)
                    else:
                        if self.unit==other.unit:
                            res.var = self.var + other.var[np.newaxis, :, :]
                        else:
                            res.var = self.var + UnitArray(other.var[np.newaxis, :, :],
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
                            res.var = self.var + UnitArray(other.var,
                                                                 other.unit**2,
                                                                 self.unit**2)
                return res

    def __radd__(self, other):
        return self.__add__(other)


    def __sub__(self, other):
        """Subtracts other.

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
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                typ=1
        except:
            try:
                if other.image:
                    typ=2
            except:
                try:
                    if other.cube:
                        typ=3
                except:
                    typ=0

        if typ==0:
            try:
                res = self.copy()
                res.data = self.data - other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if typ==1 or typ==3:
                if self.wave is not None and other.wave is not None \
                and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes '
                                  'with different world coordinates '
                                  'in spectral direction')
            if typ==2 or typ==3:
                if self.wcs is not None and other.wcs is not None \
                and not self.wcs.isEqual(other.wcs):
                    raise IOError('Operation forbidden for cubes '
                                  'with different world coordinates '
                                  'in spatial directions')

            if typ==1:
                # cube1 - spectrum = cube2
                if other.data is None or other.shape != self.shape[0]:
                    raise IOError('Operation forbidden '
                                  'for objects with different sizes')
                res = self.copy()
                # data
                if self.unit==other.unit:
                    res.data = self.data - other.data[:, np.newaxis, np.newaxis]
                else:
                    res.data = self.data - UnitMaskedArray(other.data[:, np.newaxis, np.newaxis],
                                                           other.unit, self.unit)
                # variance
                if other.var is not None:
                    if self.var is None:
                        if self.unit==other.unit:
                            res.var = np.ones(self.shape) \
                            * other.var[:, np.newaxis, np.newaxis]
                        else:
                            res.var = np.ones(self.shape) \
                            * UnitArray(other.var[:, np.newaxis, np.newaxis],
                                              other.unit**2, self.unit**2)
                    else:
                        if self.unit==other.unit:
                            res.var = self.var \
                            + other.var[:, np.newaxis, np.newaxis]
                        else:
                            res.var = self.var \
                            + UnitArray(other.var[:, np.newaxis, np.newaxis],
                                              other.unit**2, self.unit**2)
                return res
            elif typ==2:
                # cube1 - image = cube2 (cube2[k,j,i]=cube1[k,j,i]-image[j,i])
                if other.data is None or self.shape[2] != other.shape[1] \
                or self.shape[1] != other.shape[0]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                if self.unit==other.unit:
                    res.data = self.data - other.data[np.newaxis, :, :]
                else:
                    res.data = self.data - UnitMaskedArray(other.data[np.newaxis, :, :],
                                                           other.unit, self.unit)
                # variance
                if other.var is not None:
                    if self.var is None:
                        if self.unit==other.unit:
                            res.var = np.ones(self.shape)* other.var[np.newaxis, :, :]
                        else:
                            res.var = np.ones(self.shape) \
                            * UnitArray(other.var[np.newaxis, :, :],
                                              other.unit**2, self.unit**2)
                    else:
                        if self.unit==other.unit:
                            res.var = self.var + other.var[np.newaxis, :, :]
                        else:
                            res.var = self.var + UnitArray(other.var[np.newaxis, :, :],
                                                                 other.unit**2, self.unit**2)
                return res
            else:
                # cube1 - cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]-cube2[k,j,i])
                if other.data is None or self.shape[0] != other.shape[0] \
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
                if other.var is not None:
                    if self.var is None:
                        if other.unit == self.unit:
                            res.var = other.var
                        else:
                            res.var = UnitArray(other.var,
                                                      other.unit**2,
                                                      self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res.var = self.var + other.var
                        else:
                            res.var = self.var + UnitArray(other.var,
                                                                 other.unit**2,
                                                                 self.unit**2)
                return res

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                typ=1
        except:
            try:
                if other.image:
                    typ=2
            except:
                try:
                    if other.cube:
                        typ=3
                except:
                    typ=0

        if typ==0:
            try:
                res = self.copy()
                res.data = other - self.data
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__sub__(self)

    def __mul__(self, other):
        """Multiplies by other.

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
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                typ=1
        except:
            try:
                if other.image:
                    typ=2
            except:
                try:
                    if other.cube:
                        typ=3
                except:
                    typ=0

        if typ==0:
            try:
                res = self.copy()
                res.data *= other
                if self.var is not None:
                    res.var *= other ** 2
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if typ==1 or typ==3:
                if self.wave is not None and other.wave is not None \
                and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spectral direction')
            if typ==2 or typ==3:
                if self.wcs is not None and other.wcs is not None \
                and not self.wcs.isEqual(other.wcs):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spatial directions')
            if typ==1:
                # cube1 * spectrum = cube2
                if other.data is None or other.shape != self.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data * other.data[:, np.newaxis, np.newaxis]
                # variance
                if self.var is None and other.var is None:
                    res.var = None
                elif self.var is None:
                    res.var = other.var[:, np.newaxis, np.newaxis] \
                            * self.data.data * self.data.data
                elif other.var is None:
                    res.var = self.var \
                            * other.data.data[:, np.newaxis, np.newaxis] \
                            * other.data.data[:, np.newaxis, np.newaxis]
                else:
                    res.var = (other.var[:, np.newaxis, np.newaxis]
                                * self.data.data * self.data.data + self.var
                                * other.data.data[:, np.newaxis, np.newaxis]
                                * other.data.data[:, np.newaxis, np.newaxis])
                # unit
                res.unit = self.unit*other.unit
                return res
            elif typ==2:
                # cube1 * image = cube2 (cube2[k,j,i]=cube1[k,j,i]*image[j,i])
                if other.data is None or self.shape[2] != other.shape[1] \
                or self.shape[1] != other.shape[0]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data * other.data[np.newaxis, :, :]
                # variance
                if self.var is None and other.var is None:
                    res.var = None
                elif self.var is None:
                    res.var = other.var[np.newaxis, :, :] \
                            * self.data.data * self.data.data
                elif other.var is None:
                    res.var = self.var * other.data.data[np.newaxis, :, :] \
                            * other.data.data[np.newaxis, :, :]
                else:
                    res.var = (other.var[np.newaxis, :, :]
                                * self.data.data * self.data.data
                                + self.var * other.data.data[np.newaxis, :, :]
                                * other.data.data[np.newaxis, :, :])
                # unit
                res.unit = self.unit*other.unit
                return res
            else:
                # cube1 * cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]*cube2[k,j,i])
                if other.data is None or self.shape[0] != other.shape[0] \
                or self.shape[1] != other.shape[1] \
                or self.shape[2] != other.shape[2]:
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

    def __rmul__(self, other):
        return self.__mul__(other)


    def __div__(self, other):
        """Divides by other.

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
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                typ=1
        except:
            try:
                if other.image:
                    typ=2
            except:
                try:
                    if other.cube:
                        typ=3
                except:
                    typ=0

        if typ==0:
            try:
                res = self.copy()
                res.data /= other
                if self.var is not None:
                    res.var /= other ** 2
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            if typ==1 or typ==3:
                if self.wave is not None and other.wave is not None \
                and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spectral direction')
            if typ==2 or typ==3:
                if self.wcs is not None and other.wcs is not None \
                and not self.wcs.isEqual(other.wcs):
                    raise ValueError('Operation forbidden for cubes '
                                     'with different world coordinates'
                                     ' in spatial directions')
            if typ==1:
                # cube1 / spectrum = cube2
                if other.data is None or other.shape != self.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                # data
                res =self.copy()
                res.data = self.data / other.data[:, np.newaxis, np.newaxis]
                # variance
                if self.var is None and other.var is None:
                    res.var = None
                elif self.var is None:
                    res.var = other.var[:, np.newaxis, np.newaxis] \
                            * self.data.data * self.data.data \
                            / (other.data.data[:, np.newaxis, np.newaxis] ** 4)
                elif other.var is None:
                    res.var = self.var \
                            * other.data.data[:, np.newaxis, np.newaxis] \
                            * other.data.data[:, np.newaxis, np.newaxis] \
                            / (other.data.data[:, np.newaxis, np.newaxis] ** 4)
                else:
                    res.var = (other.var[:, np.newaxis, np.newaxis]
                               * self.data.data * self.data.data + self.var
                               * other.data.data[:, np.newaxis,
                                                     np.newaxis]
                               * other.data.data[:, np.newaxis,
                                                     np.newaxis]) \
                            / (other.data.data[:, np.newaxis,
                                               np.newaxis] ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res
            elif typ==2:
                # cube1 / image = cube2 (cube2[k,j,i]=cube1[k,j,i]/image[j,i])
                if other.data is None or self.shape[2] != other.shape[1] \
                or self.shape[1] != other.shape[0]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                res = self.copy()
                # data
                res.data = self.data / other.data[np.newaxis, :, :]
                # variance
                if self.var is None and other.var is None:
                    res.var = None
                elif self.var is None:
                    res.var = other.var[np.newaxis, :, :] \
                            * self.data.data * self.data.data \
                            / (other.data.data[np.newaxis, :, :] ** 4)
                elif other.var is None:
                    res.var = self.var * other.data.data[np.newaxis, :, :] \
                            * other.data.data[np.newaxis, :, :] \
                            / (other.data.data[np.newaxis, :, :] ** 4)
                else:
                    res.var = (other.var[np.newaxis, :, :]
                                * self.data.data * self.data.data + self.var
                                * other.data.data[np.newaxis, :, :]
                                * other.data.data[np.newaxis, :, :]) \
                            / (other.data.data[np.newaxis, :, :] ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res
            else:
                # cube1 / cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]/cube2[k,j,i])
                if other.data is None or self.shape[0] != other.shape[0] \
                or self.shape[1] != other.shape[1] \
                or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
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
                    res.var = (other.var * self.data.data * self.data.data
                                   + self.var * other.data.data * other.data.data) \
                            / (other.data.data ** 4)
                # unit
                res.unit = self.unit/other.unit
                return res

    def __rdiv__(self, other):
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                typ=1
        except:
            try:
                if other.image:
                    typ=2
            except:
                try:
                    if other.cube:
                        typ=3
                except:
                    typ=0

        if typ==0:
            try:
                res = self.copy()
                res.data = other / res.data
                if self.var is not None:
                    res.var = other ** 2 / res.var
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__div__(self)

#     def __pow__(self, other):
#         """Computes the power exponent."""
#         if self.data is None:
#             raise ValueError('empty data array')
#         res = self.copy()
#         if is_float(other) or is_int(other):
#             res.data = (self.data ** other) * (self.fscale ** (other - 1))
#             res.var = None
#         else:
#             raise ValueError('Operation forbidden')
#         return res


    def _sqrt(self):
        """Computes the positive square-root of data extension.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if self.var is not None:
            self.var = 3 * self.var / self.data.data ** 4
        self.data = np.ma.sqrt(self.data)
        self.unit /= np.sqrt(self.unit.scale)

    def sqrt(self):
        """Returns a cube containing the positive square-root
        of data extension.
        """
        res = self.copy()
        res._sqrt()
        return res

    def _abs(self):
        """Computes the absolute value of data extension."""
        if self.data is None:
            raise ValueError('empty data array')
        self.data = np.ma.abs(self.data)
        self.var = None

    def abs(self):
        """Returns a cube containing the absolute value of data extension."""
        res = self.copy()
        res._abs()
        return res

    def __getitem__(self, item):
        """Returns the corresponding object:
        cube[k,p,k] = value
        cube[k,:,:] = spectrum
        cube[:,p,q] = image
        cube[:,:,:] = sub-cube
        """
        if isinstance(item, tuple) and len(item) == 3:
            data = self.data[item]
            if is_int(item[0]):
                if is_int(item[1]) and is_int(item[2]):
                    # return a float
                    return data
                else:
                    # return an image
                    if is_int(item[1]):
                        shape = (1, data.shape[0])
                    elif is_int(item[2]):
                        shape = (data.shape[0], 1)
                    else:
                        shape = data.shape
                    var = None
                    if self.var is not None:
                        var = self.var[item]
                    try:
                        wcs = self.wcs[item[1], item[2]]
                    except:
                        wcs = None
                    res = Image(shape=shape, wcs=wcs, unit=self.unit)
                    res.data = data
                    res.var = var
                    res.filename = self.filename
                    return res
            elif is_int(item[1]) and is_int(item[2]):
                # return a spectrum
                shape = data.shape[0]
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Spectrum(shape=shape, wave=wave, unit=self.unit)
                res.data = data
                res.var = var
                res.filename = self.filename
                return res
            else:
                # return a cube
                if is_int(item[1]):
                    shape = (data.shape[0], 1, data.shape[1])
                elif is_int(item[2]):
                    shape = (data.shape[0], data.shape[1], 1)
                else:
                    shape = data.shape
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[1], item[2]]
                except:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Cube(shape=shape, wcs=wcs, wave=wave, unit=self.unit)
                res.data_header = pyfits.Header(self.data_header)
                res.primary_header = pyfits.Header(self.primary_header)
                res.data = data
                res.var = var
                res.filename = self.filename
                return res
        else:
            raise ValueError('Operation forbidden')

    def get_lambda(self, lbda_min, lbda_max=None, unit_wave=u.angstrom):
        """Returns the sub-cube corresponding to a wavelength range.

        Parameters
        ----------
        lbda_min  : float
                    Minimum wavelength.
        lbda_max  : float
                    Maximum wavelength.
        unit_wave : astropy.units
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
                pix_min = max(0, int(lbda_min+0.5))
                pix_max = min(self.shape[0], int(lbda_max+0.5))
            else:
                pix_min = max(0, int(self.wave.pixel(lbda_min, unit=unit_wave)))
                pix_max = min(self.shape[0], int(self.wave.pixel(lbda_max, unit=unit_wave)) + 1)
            if (pix_min + 1) == pix_max:
                return self.data[pix_min, :, :]
            else:
                return self[pix_min:pix_max, :, :]

    def get_step(self, unit_wave=None, unit_wcs=None):
        """Returns the cube steps [dlbda,dy,dx].

        Parameters
        ----------
        unit_wave : astropy.units
                    wavelengths unit.
        unit_wcs  : astropy.units
                    world coordinates unit.
        """
        step = np.empty(3)
        step[0] = self.wave.get_step(unit_wave)
        step[1:] = self.wcs.get_step(unit_wcs)

    def get_range(self, unit_wave=None, unit_wcs=None):
        """Returns [ [lbda_min,y_min,x_min], [lbda_max,y_max,x_max] ].

        Parameters
        ----------
        unit_wave : astropy.units
                    wavelengths unit.
        unit_wcs  : astropy.units
                    world coordinates unit.
        """
        r = np.empty((2, 3))
        r[:, 0] = self.wave.get_range(unit_wave)
        r[:, 1:] = self.wcs.get_range(unit_wcs)
        return r

    def get_start(self, unit_wave=None, unit_wcs=None):
        """Returns [lbda,y,x] corresponding to pixel (0,0,0).

        Parameters
        ----------
        unit_wave : astropy.units
                    wavelengths unit.
        unit_wcs  : astropy.units
                    world coordinates unit.
        """
        start = np.empty(3)
        start[0] = self.wave.get_start(unit_wave)
        start[1:] = self.wcs.get_start(unit_wcs)
        return start

    def get_end(self, unit_wave=None, unit_wcs=None):
        """Returns [lbda,y,x] corresponding to pixel (-1,-1,-1).

        Parameters
        ----------
        unit_wave : astropy.units
                    wavelengths unit.
        unit_wcs  : astropy.units
                    world coordinates unit.
        """
        end = np.empty(3)
        end[0] = self.wave.get_end(unit_wave)
        end[1:] = self.wcs.get_end(unit_wcs)
        return end

    def get_rot(self, unit=u.deg):
        """Returns the rotation angle.

        Parameters
        ----------
        unit : astropy.units
               type of the angle coordinate
               degree by default
        """
        return self.wcs.get_rot(unit)

#     def get_np_data(self):
#         """Returns numpy masked array containing the flux multiplied by scaling
#         factor."""
#         return self.data * self.fscale

    def __setitem__(self, key, other):
        """Sets the corresponding part of data."""
        # self.data[key] = value

        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                typ=1
        except:
            try:
                if other.image:
                    typ=2
            except:
                try:
                    if other.cube:
                        typ=3
                except:
                    typ=0


        if typ==0:
            try:
                self.data[key] = other
            except:
                        raise IOError('Operation forbidden')
        else:
            if typ==1 or typ==3:
                if self.wave is not None and other.wave is not None \
                and not self.wave.isEqual(other.wave):
                    raise IOError('Operation forbidden for cubes with '
                                  'different world coordinates '
                                  'in spectral direction')
            if typ==2 or typ==3:
                if self.wcs is not None and other.wcs is not None \
                and not self.wcs.isEqual(other.wcs):
                    raise ValueError('Operation forbidden for cubes '
                                     'with different world coordinates'
                                     ' in spatial directions')
            if self.unit == other.unit:
                self.data[key] = other.data
            else:
                self.data[key] = UnitMaskedArray(other.data,
                                                 other.unit, self.unit)


    def set_wcs(self, wcs=None, wave=None):
        """Sets the world coordinates (spatial and/or spectral).

        Parameters
        ----------
        wcs : :class:`mpdaf.obj.WCS`
              World coordinates.
        wave : :class:`mpdaf.obj.WaveCoord`
               Wavelength coordinates.
        """
        if wcs is not None:
            self.wcs = wcs.copy()
            self.wcs.naxis1 = self.shape[2]
            self.wcs.naxis2 = self.shape[1]
            if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
                and (wcs.naxis1 != self.shape[2]
                     or wcs.naxis2 != self.shape[1]):
                d = {'class': 'Cube', 'method': 'set_wcs'}
                self.logger.warning('world coordinates and data have not the '
                                    'same dimensions', extra=d)
        if wave is not None:
            if wave.shape is not None and wave.shape != self.shape[0]:
                d = {'class': 'Cube', 'method': 'set_wcs'}
                self.logger.warning('wavelength coordinates and data have not '
                                    'the same dimensions', extra=d)
            self.wave = wave.copy()
            self.wave.shape = self.shape[0]

    def set_var(self, var):
        """Sets the variance array.

        Parameters
        ----------
        var : float array
              Input variance array. If None, variance is set with zeros.
        """
        if var is None:
            self.var = np.zeros((self.shape[0], self.shape[1], self.shape[2]))
        else:
            if self.shape[0] == np.shape(var)[0] \
                    and self.shape[1] == np.shape(var)[1] \
                    and self.shape[2] == np.shape(var)[2]:
                self.var = var
            else:
                raise ValueError('var and data have not the same dimensions.')

    def sum(self, axis=None, weights=None):
        """Returns the sum over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
               Axis or axes along which a sum is performed.

               The default (axis = None) is perform a sum over all
               the dimensions of the cube and returns a float.

               axis = 0  is perform a sum over the wavelength dimension
               and returns an image.

               axis = (1,2) is perform a sum over the (X,Y) axes and
               returns a spectrum.

               Other cases return None.
        weights : array_like, optional
                  An array of weights associated with the data values.
                  The weights array can either be 1-D (axis=(1,2))
                  or 2-D (axis=0) or of the same shape as the cube.
                  If weights=None, then all data in a are assumed to have a weight equal to one.

               The method conserves the flux by using the algorithm
               from Jarle Brinchmann (jarle@strw.leidenuniv.nl):
               - Take into account bad pixels in the addition.
               - Normalize with the median value of weighting sum/no-weighting sum
        """
        if weights is not None:
            w = np.array(weights, dtype=np.float)
            if len(w.shape) == 3:
                if (w.shape[0] != self.shape[0] or \
                        w.shape[1] != self.shape[1] or \
                        w.shape[2] != self.shape[2]):
                    raise IOError('Incorrect dimensions for the weights (%i,%i,%i) (it must be (%i,%i,%i)) '\
                                  % (w.shape[0], w.shape[1], w.shape[2],
                                     self.shape[0], self.shape[1], self.shape[2]))
            elif len(w.shape) == 2:
                if w.shape[0] != self.shape[1] or \
                   w.shape[1] != self.shape[2]:
                    raise IOError('Incorrect dimensions for the weights (%i,%i) (it must be (%i,%i)) '\
                                  % (w.shape[0], w.shape[1], self.shape[1], self.shape[2]))
                else:
                    w = np.tile(w, (self.shape[0], 1, 1))
            elif len(w.shape) == 1:
                if w.shape[0] != self.shape[0]:
                    raise IOError('Incorrect dimensions for the weights (%i) (it must be (%i))' \
                                  % (w.shape[0], self.shape[0]))
                else:
                    w = np.ones_like(self.data.data) * w[:, np.newaxis, np.newaxis]
            else:
                raise IOError('Incorrect dimensions for the weights (it must be (%i,%i,%i)) '\
                              % (self.shape[0], self.shape[1], self.shape[2]))

            # weights mask
            wmask = np.ma.masked_where(self.data.mask, np.ma.masked_where(w == 0, w))

        if axis is None:
            if weights is None:
                return self.data.sum()
            else:
                data = self.data * w
                npix = np.sum(~self.data.mask)
                data = np.ma.sum(data) / npix
                # flux conservation
                orig_data = self.data * ~wmask.mask
                orig_data = np.ma.sum(orig_data)
                rr = data / orig_data
                med_rr = np.ma.median(rr)
                if med_rr > 0:
                    data /= med_rr
                return data
        elif axis == 0:
            # return an image
            if weights is None:
                data = np.ma.sum(self.data, 0)
                if self.var is not None:
                    var = np.ma.masked_where(self.data.mask, np.ma.masked_invalid(self.var))
                    var = np.ma.sum(var, 0).filled(np.NaN)
                else:
                    var = None
            else:
                data = self.data * w
                npix = np.sum(~self.data.mask, axis)
                data = np.ma.sum(data, axis) / npix
                orig_data = self.data * ~wmask.mask
                orig_data = np.ma.sum(orig_data, axis)
                rr = data / orig_data
                med_rr = np.ma.median(rr)
                if med_rr > 0:
                    data /= med_rr
                if self.var is not None:
                    var = self.var * w
                    var = np.ma.masked_where(self.data.mask, np.ma.masked_invalid(var))
                    var = np.ma.sum(var, axis) / npix
                    dspec = np.ma.sqrt(var)
                    if med_rr > 0:
                        dspec /= med_rr
                    orig_var = self.var * ~wmask.mask
                    orig_var = np.ma.masked_where(self.data.mask, np.ma.masked_invalid(orig_var))
                    orig_var = np.ma.sum(orig_var, axis)
                    sn_orig = orig_data / np.ma.sqrt(orig_var)
                    sn_now = data / dspec
                    sn_ratio = np.ma.median(sn_orig / sn_now)
                    dspec /= sn_ratio
                    var = dspec * dspec
                    var = var.filled(np.NaN)
                else:
                    var = None
            res = Image(shape=data.shape, wcs=self.wcs, unit=self.unit)
            res.data = data
            res.var = var
            return res
        elif axis == tuple([1, 2]):
            # return a spectrum
            if weights is None:
                data = np.ma.sum(np.ma.sum(self.data, axis=1), axis=1)
                if self.var is not None:
                    var = np.ma.masked_where(self.data.mask, np.ma.masked_invalid(self.var))
                    var = np.ma.sum(np.ma.sum(var, axis=1), axis=1).filled(np.NaN)
                else:
                    var = None
            else:
                data = self.data * w
                npix = np.sum(np.sum(~self.data.mask, axis=1), axis=1)
                data = np.ma.sum(np.ma.sum(data, axis=1), axis=1) / npix
                orig_data = self.data * ~wmask.mask
                orig_data = np.ma.sum(np.ma.sum(orig_data, axis=1), axis=1)
                rr = data / orig_data
                med_rr = np.ma.median(rr)
                if med_rr > 0:
                    data /= med_rr
                if self.var is not None:
                    var = self.var * w
                    var = np.ma.masked_where(self.data.mask, np.ma.masked_invalid(var))
                    var = np.ma.sum(np.ma.sum(var, axis=1), axis=1) / npix
                    dspec = np.ma.sqrt(var)
                    if med_rr > 0:
                        dspec /= med_rr
                    orig_var = self.var * ~wmask.mask
                    orig_var = np.ma.masked_where(self.data.mask, np.ma.masked_invalid(orig_var))
                    orig_var = np.ma.sum(np.ma.sum(orig_var, axis=1), axis=1)
                    sn_orig = orig_data / np.ma.sqrt(orig_var)
                    sn_now = data / dspec
                    sn_ratio = np.ma.median(sn_orig / sn_now)
                    dspec /= sn_ratio
                    var = dspec * dspec
                    var = var.filled(np.NaN)
                else:
                    var = None

            res = Spectrum(shape=data.shape[0], wave=self.wave, unit=self.unit)
            res.data = data
            res.var = var
            return res
        else:
            return None

    def mean(self, axis=None):
        """Returns the mean over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
               Axis or axes along which a mean is performed.

               The default (axis = None) is perform a mean over all the
               dimensions of the cube and returns a float.

               axis = 0  is perform a mean over the wavelength dimension
               and returns an image.

               axis = (1,2) is perform a mean over the (X,Y) axes and
               returns a spectrum.

               Other cases return None.
        """
        if axis is None:
            return self.data.mean()
        elif axis == 0:
            # return an image
            data = np.ma.mean(self.data, axis)
            if self.var is not None:
                var = np.ma.sum(np.ma.masked_where(self.data.mask, self.var), axis).filled(np.NaN) \
                    / np.sum(~self.data.mask, axis)**2
                #var = np.ma.mean(np.ma.masked_invalid(self.var), axis).filled(np.NaN)
            else:
                var = None
            res = Image(shape=data.shape, wcs=self.wcs, unit=self.unit)
            res.data = data
            res.var = var
            return res
        elif axis == tuple([1, 2]):
            # return a spectrum
            data = np.ma.sum(np.ma.sum(self.data, axis=1), axis=1) / \
                np.sum(np.sum(~self.data.mask, axis=1), axis=1)
            if self.var is not None:
                var = np.ma.sum(np.ma.sum(np.ma.masked_where(self.data.mask,
                                                             self.var),
                                          axis=1), axis=1).filled(np.NaN) \
                    / np.sum(np.sum(~self.data.mask, axis=1), axis=1)**2
            else:
                var = None
            res = Spectrum(notnoise=True, shape=data.shape[0],
                           wave=self.wave, unit=self.unit)
            res.data = data
            res.var = var
            return res
        else:
            return None

    def median(self, axis=None):
        """Returns the median over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
               Axis or axes along which a mean is performed.

               The default (axis = None) is perform a mean over all the
               dimensions of the cube and returns a float.

               axis = 0  is perform a mean over the wavelength dimension
               and returns an image.

               axis = (1,2) is perform a mean over the (X,Y) axes and
               returns a spectrum.

               Other cases return None.
        """
        if axis is None:
            return self.data.median()
        elif axis == 0:
            # return an image
            data = np.ma.median(self.data, axis)
            if self.var is not None:
                var = np.ma.masked_where(self.data.mask,
                                         np.ma.masked_invalid(self.var))
                var = np.ma.median(var, axis).filled(np.NaN)
            else:
                var = None
            res = Image(shape=data.shape, wcs=self.wcs, unit=self.unit)
            res.data = data
            res.var = var
            return res
        elif axis == tuple([1, 2]):
            # return a spectrum
            data = np.ma.median(np.ma.median(self.data, axis=1), axis=1)
            if self.var is not None:
                var = np.ma.masked_where(self.data.mask,
                                         np.ma.masked_invalid(self.var))
                var = np.ma.median(np.ma.median(var, axis=1), axis=1).filled(np.NaN)
            else:
                var = None
            res = Spectrum(notnoise=True, shape=data.shape[0],
                           wave=self.wave, unit=self.unit)
            res.data = data
            res.var = var
            return res
        else:
            return None

    def truncate(self, coord, mask=True, unit_wave=u.angstrom, unit_wcs=u.deg):
        """ Truncates the cube and return a sub-cube.

        Parameters
        ----------
        coord : array
                array containing the sub-cube boundaries
                [[lbda_min,y_min,x_min], [lbda_max,y_max,x_max]]
                (output of `mpdaf.obj.cube.get_range`)
        mask  : boolean
                if True, pixels outside [y_min,y_max]
                and [x_min,x_max] are masked.
        unit_wave : astropy.units
                    wavelengths unit.
                    If None, inputs are in pixels
        unit_wcs  : astropy.units
                    world coordinates unit.
                    If None, inputs are in pixels
        """
        lmin = coord[0][0]
        y_min = coord[0][1]
        x_min = coord[0][2]
        lmax = coord[1][0]
        y_max = coord[1][1]
        x_max = coord[1][2]

        skycrd = [[y_min, x_min], [y_min, x_max],
                  [y_max, x_min], [y_max, x_max]]
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
            kmin = int(lmin+0.5)
            kmax= int(lmax+0.5)
        else:
            kmin = max(0, self.wave.pixel(lmin, nearest=True, unit=unit_wave))
            kmax = min(self.shape[0], self.wave.pixel(lmax, nearest=True, unit=unit_wave) + 1)

        if kmin == kmax:
            raise ValueError('Minimum and maximum wavelengths are equal')

        if kmax == kmin + 1:
            raise ValueError('Minimum and maximum wavelengths are outside'
                             ' the spectrum range')

        data = self.data[kmin:kmax, imin:imax, jmin:jmax]
        shape = data.shape

        if self.var is not None:
            var = self.var[kmin:kmax, imin:imax, jmin:jmax]
        else:
            var = None
        try:
            wcs = self.wcs[imin:imax, jmin:jmax]
        except:
            wcs = None
        try:
            wave = self.wave[kmin:kmax]
        except:
            wave = None

        res = Cube(shape=shape, wcs=wcs, wave=wave, unit=self.unit)
        res.data_header = pyfits.Header(self.data_header)
        res.primary_header = pyfits.Header(self.primary_header)
        res.data = data
        res.var = var

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
            test_x = np.logical_or(x < x_min, x > x_max)
            test_y = np.logical_or(y < y_min, y > y_max)
            test = np.logical_or(test_x, test_y)
            res.data.mask = np.logical_or(res.data.mask,
                                          np.tile(test, [res.shape[0], 1, 1]))
            res.resize()

        return res

    def _rebin_mean_(self, factor):
        """Shrinks the size of the cube by factor. New size must be an integer
        multiple of the original size.

        Parameters
        ----------
        factor : (integer,integer,integer)
                 Factor in z, y and x.
                 Python notation: (nz,ny,nx)
        """
        assert np.array_equal(np.mod(self.shape, factor), [0, 0, 0])
        shape = self.shape / np.asarray(factor)
        self.data = self.data.reshape(
            shape[0], factor[0], shape[1], factor[1], shape[2], factor[2]
        ).sum(1).sum(2).sum(3)
        self.data /= np.prod(factor)

        if self.var is not None:
            self.var = self.var.reshape(self.shape[0], factor[0],
                                        self.shape[1], factor[1],
                                        self.shape[2], factor[2])\
                .sum(1).sum(2).sum(3)
            self.var /= np.prod(factor) ** 2
        # coordinates
        self.wcs = self.wcs.rebin(factor[1:])
        self.wave.rebin(factor[0])

    def _rebin_mean(self, factor, margin='center', flux=False):
        """Shrinks the size of the cube by factor.

        Parameters
        ----------
        factor : integer or (integer,integer,integer)
                 Factor in z, y and x. Python notation: (nz,ny,nx).
        flux   : boolean
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
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])) \
                and not np.sometrue(np.mod(self.shape[2], factor[2])):
            # new size is an integer multiple of the original size
            self._rebin_mean_(factor)
            return None
        else:
            factor = np.array(factor)
            newshape = self.shape / factor
            n = self.shape - newshape * factor

            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0] == 1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0] / 2
                    n0_right = self.shape[0] - n[0] + n0_left
            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1] == 1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1] / 2
                    n1_right = self.shape[1] - n[1] + n1_left
            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2] == 1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2] / 2
                    n2_right = self.shape[2] - n[2] + n2_left

            cub = self[n0_left:n0_right, n1_left:n1_right, n2_left:n2_right]
            cub._rebin_mean_(factor)

            if flux is False:
                self.data = cub.data
                self.var = cub.var
                self.wave = cub.wave
                self.wcs = cub.wcs
                return None
            else:
                newshape = list(cub.shape)
                wave = cub.wave
                wcs = cub.wcs

                if n0_left != 0:
                    newshape[0] += 1
                    wave.set_crpix(wave.get_crpix()+1)
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
                    cub.data.mask

                if self.var is None:
                    var = None
                else:
                    var = np.empty(newshape)
                    var[l_left:l_right, p_left:p_right, q_left:q_right] = cub.var

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
                            .sum(1).sum(2) / F2
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
                            .sum(1).sum(2) / F2
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
                            .sum(1).sum(2) / F2
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
                            .sum(1).sum(2) / F2

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
                            .sum(1).sum(2) / F2
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
                            .sum(1).sum(2) / F2

                if l_left == 1 and p_left == 1 and q_left == 1:
                    data[0, 0, 0] = \
                        self.data[:n0_left, :n1_left, :n2_left].sum() / F
                    mask[0, 0, 0] = self.mask[:n0_left, :n1_left, :n2_left].any()
                    if var is not None:
                        var[0, 0, 0] = \
                            self.var[:n0_left, :n1_left, :n2_left].sum() / F2
                if l_left == 1 and p_right == (newshape[1] - 1) \
                        and q_left == 1:
                    data[0, -1, 0] = \
                        self.data[:n0_left, n1_right:, :n2_left].sum() / F
                    mask[0, -1, 0] = \
                        self.mask[:n0_left, n1_right:, :n2_left].any()
                    if var is not None:
                        var[0, -1, 0] = \
                            self.var[:n0_left, n1_right:, :n2_left].sum() / F2
                if l_left == 1 and p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    data[0, -1, -1] = \
                        self.data[:n0_left, n1_right:, n2_right:].sum() / F
                    mask[0, -1, -1] = \
                        self.mask[:n0_left, n1_right:, n2_right:].any()
                    if var is not None:
                        var[0, -1, -1] = \
                            self.var[:n0_left, n1_right:, n2_right:].sum() / F2
                if l_left == 1 and p_left == 1 and \
                        q_right == (newshape[2] - 1):
                    data[0, 0, -1] = \
                        self.data[:n0_left, :n1_left, n2_right:].sum() / F
                    mask[0, 0, -1] = \
                        self.mask[:n0_left, :n1_left, n2_right:].any()
                    if var is not None:
                        var[0, 0, -1] = \
                            self.var[:n0_left, :n1_left, n2_right:].sum() / F2
                if l_left == (newshape[0] - 1) and p_left == 1 \
                        and q_left == 1:
                    data[-1, 0, 0] = \
                        self.data[n0_right:, :n1_left, :n2_left].sum() / F
                    mask[-1, 0, 0] = \
                        self.mask[n0_right:, :n1_left, :n2_left].any()
                    if var is not None:
                        var[-1, 0, 0] = \
                            self.var[n0_right:, :n1_left, :n2_left].sum() / F2
                if l_left == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1) and q_left == 1:
                    data[-1, -1, 0] = \
                        self.data[n0_right:, n1_right:, :n2_left].sum() / F
                    mask[-1, -1, 0] = \
                        self.mask[n0_right:, n1_right:, :n2_left].any()
                    if var is not None:
                        var[-1, -1, 0] = \
                            self.var[n0_right:, n1_right:, :n2_left].sum() / F2
                if l_left == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    data[-1, -1, -1] = \
                        self.data[n0_right:, n1_right:, n2_right:].sum() / F
                    mask[-1, -1, -1] = \
                        self.mask[n0_right:, n1_right:, n2_right:].any()
                    if var is not None:
                        var[-1, -1, -1] = \
                            self.var[n0_right:, n1_right:, n2_right:].sum() / F2
                if l_left == (newshape[0] - 1) and p_left == 1 \
                        and q_right == (newshape[2] - 1):
                    data[-1, 0, -1] = \
                        self.data[n0_right:, :n1_left, n2_right:].sum() / F
                    mask[-1, 0, -1] = \
                        self.mask[n0_right:, :n1_left, n2_right:].any()
                    if var is not None:
                        var[-1, 0, -1] = \
                            self.var[n0_right:, :n1_left, n2_right:].sum() / F2

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
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
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
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
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
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

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
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
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
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
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
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

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
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
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
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
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
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

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
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
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
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
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
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

                self.wcs = wcs
                self.wave = wave
                self.data = np.ma.array(data, mask=mask)
                self.var = var
                return None

    def rebin_mean(self, factor, margin='center', flux=False):
        """Shrinks the size of the cube by factor.

        Parameters
        ----------
        factor : integer or (integer,integer,integer)
                 Factor in z, y and x. Python notation: (nz,ny,nx).
        flux   : boolean
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
        """
        if is_int(factor):
            factor = (factor, factor, factor)
        factor = np.array(factor)
        factor = np.maximum(factor, [1, 1, 1])
        factor = np.minimum(factor, self.shape)
        res = self.copy()
        res._rebin_mean(factor, margin, flux)
        return res

    def _med_(self, k, p, q, kfactor, pfactor, qfactor):
        return np.ma.median(self.data[k * kfactor:(k + 1) * kfactor,
                                      p * pfactor:(p + 1) * pfactor,
                                      q * qfactor:(q + 1) * qfactor])

    def _rebin_median_(self, factor):
        """Shrinks the size of the cube by factor. New size must be an integer
        multiple of the original size.

        Parameter
        ---------
        factor : (integer,integer,integer)
                 Factor in z, y and x.
                Python notation: (nz,ny,nx)
        """
        assert np.array_equal(np.mod(self.shape, factor), [0, 0, 0])
        self.shape /= np.asarray(factor)
        # data
        grid = np.lib.index_tricks.nd_grid()
        g = grid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]]
        vfunc = np.vectorize(self._med_)
        data = vfunc(g[0], g[1], g[2], factor[0], factor[1], factor[2])
        mask = self.data.mask.reshape(self.shape[0], factor[0],
                                      self.shape[1], factor[1],
                                      self.shape[2], factor[2])\
            .sum(1).sum(2).sum(3)
        self.data = np.ma.array(data, mask=mask)
        # variance
        self.var = None
        # coordinates
        self.wcs = self.wcs.rebin(factor[1:])
        self.wave.rebin(factor[0])

    def rebin_median(self, factor, margin='center'):
        """Shrinks the size of the cube by factor.

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
        out : :class:`mpdaf.obj.Cube`
        """
        if is_int(factor):
            factor = (factor, factor, factor)
        factor = np.array(factor)
        factor = np.maximum(factor, [1, 1, 1])
        factor = np.minimum(factor, self.shape)
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])) \
                and not np.sometrue(np.mod(self.shape[2], factor[2])):
            # new size is an integer multiple of the original size
            res = self.copy()
        else:
            newshape = self.shape / factor
            n = self.shape - newshape * factor

            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0] == 1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0] / 2
                    n0_right = self.shape[0] - n[0] + n0_left
            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1] == 1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1] / 2
                    n1_right = self.shape[1] - n[1] + n1_left
            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2] == 1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2] / 2
                    n2_right = self.shape[2] - n[2] + n2_left

            res = self[n0_left:n0_right, n1_left:n1_right, n2_left:n2_right]

        res._rebin_median_(factor)
        return res

    def loop_spe_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """loops over all spectra to apply a function/method. Returns the
        resulting cube. Multiprocessing is used.

        Parameters
        ----------
        f       : function or :class:`mpdaf.obj.Spectrum` method
                  Spectrum method or function that the first argument
                  is a spectrum object.
        cpu     : integer
                  number of CPUs. It is also possible to set
                  the mpdaf.CPU global variable.
        verbose : boolean
                  if True, progression is printed.
        kargs   : kargs
                  can be used to set function arguments.

        Returns
        -------
        out : :class:`mpdaf.obj.Cube` if f returns :class:`mpdaf.obj.Spectrum`,
        out : :class:`mpdaf.obj.Image` if f returns a number,
        out : np.array(dtype=object) in others cases.
        """
        d = {'class': 'Cube', 'method': 'loop_spe_multiprocessing'}
        from mpdaf import CPU
        if cpu is not None and cpu < multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU < multiprocessing.cpu_count():
            cpu_count = CPU
        else:
            cpu_count = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=cpu_count)
        processlist = list()

        if isinstance(f, types.MethodType):
            f = f.__name__

        for sp, pos in iter_spe(self, index=True):
            header = sp.wave.to_header()
            processlist.append([pos, f, header,
                                sp.data.data, sp.data.mask, sp.var,
                                sp.unit, kargs])
        num_tasks = len(processlist)

        processresult = pool.imap_unordered(_process_spe, processlist)
        pool.close()

        if verbose:
            msg = "loop_spe_multiprocessing (%s): %i tasks" % (f, num_tasks)
            self.logger.info(msg, extra=d)

            while (True):
                time.sleep(5)
                completed = processresult._index
                if completed == num_tasks:
                    output = ""
                    sys.stdout.write("\r\x1b[K" + output.__str__())
                    sys.stdout.flush()
                    break
                output = ("\r Waiting for %i tasks to complete (%i%% done) ..."
                          % (num_tasks - completed, float(completed)
                             / float(num_tasks) * 100.0))
                sys.stdout.write("\r\x1b[K" + output.__str__())
                sys.stdout.flush()

        init = True
        for pos, dtype, out in processresult:
            p, q = pos
            if dtype == 'spectrum':
                # f return a Spectrum -> iterator return a cube
                header, data, mask, var, unit = out
                wave = WaveCoord(header, shape=data.shape[0])
                spe = Spectrum(shape=data.shape[0], wave=wave, unit=unit,
                               data=data, var=var)
                spe.data.mask = mask

                cshape = (data.shape[0], self.shape[1], self.shape[2])
                if init:
                    if self.var is None:
                        result = Cube(wcs=self.wcs.copy(), wave=wave,
                                      data=np.zeros(shape=cshape),
                                      unit=unit)
                    else:
                        result = Cube(wcs=self.wcs.copy(), wave=wave,
                                      data=np.zeros(shape=cshape),
                                      var=np.zeros(shape=cshape),
                                      unit=unit)
                    init = False

                result.data_header = pyfits.Header(self.data_header)
                result.primary_header = pyfits.Header(self.primary_header)
                result[:, p, q] = spe

            else:
                if is_float(out[0]) or is_int(out[0]):
                    # f returns a number -> iterator returns an image
                    if init:
                        result = Image(wcs=self.wcs.copy(),
                                       data=np.zeros(shape=(self.shape[1],
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
        f       : function or :class:`mpdaf.obj.Image` method
                  Image method or function that the first argument
                  is a Image object. It should return an Image object.
        cpu     : integer
                  number of CPUs. It is also possible to set
        verbose : boolean
                  if True, progression is printed.
        kargs   : kargs
                  can be used to set function arguments.

        Returns
        -------
        out : :class:`mpdaf.obj.Cube` if f returns :class:`mpdaf.obj.Image`,
        out : :class:`mpdaf.obj.Spectrum` if f returns a number,
        out : np.array(dtype=object) in others cases.
        """
        d = {'class': 'Cube', 'method': 'loop_ima_multiprocessing'}
        from mpdaf import CPU
        if cpu is not None and cpu < multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU < multiprocessing.cpu_count():
            cpu_count = CPU
        else:
            cpu_count = multiprocessing.cpu_count() - 1

        pool = multiprocessing.Pool(processes=cpu_count)
        processlist = list()

        if isinstance(f, types.MethodType):
            f = f.__name__

        for ima, k in iter_ima(self, index=True):
            header = ima.wcs.to_header()
            processlist.append([k, f, header, ima.data.data, ima.data.mask,
                                ima.var, ima.unit, kargs])
        num_tasks = len(processlist)

        processresult = pool.imap_unordered(_process_ima, processlist)
        pool.close()

        if verbose:
            msg = "loop_ima_multiprocessing (%s): %i tasks" % (f, num_tasks)
            self.logger.info(msg, extra=d)

            while (True):
                time.sleep(5)
                completed = processresult._index
                if completed == num_tasks:
                    output = ""
                    sys.stdout.write("\r\x1b[K" + output.__str__())
                    sys.stdout.flush()
                    break
                output = "\r Waiting for %i tasks to complete '\
                '(%i%% done) ..." % (num_tasks - completed,
                                     float(completed) / float(num_tasks)
                                     * 100.0)
                sys.stdout.write("\r\x1b[K" + output.__str__())
                sys.stdout.flush()

        init = True
        for k, dtype, out in processresult:
            if dtype == 'image':
                # f returns an image -> iterator returns a cube
                data = out[1]
                mask = out[2]
                var = out[3]
                if init:
                    header = out[0]
                    unit = out[4]

                    wcs = WCS(header, shape=data.shape)
                    cshape = (self.shape[0], data.shape[0], data.shape[1])
                    if self.var is None:
                        result = Cube(wcs=wcs, wave=self.wave.copy(),
                                      data=np.zeros(shape=cshape),
                                      unit=unit)
                    else:
                        result = Cube(wcs=wcs, wave=self.wave.copy(),
                                      data=np.zeros(shape=cshape),
                                      var=np.zeros(shape=cshape),
                                      unit=unit)
                    init = False
                result.data.data[k, :, :] = data
                result.data.mask[k, :, :] = mask
                if self.var is not None:
                    result.var[k, :, :] = var
                result.data_header = pyfits.Header(self.data_header)
                result.primary_header = pyfits.Header(self.primary_header)
            elif dtype == 'spectrum':
                # f return a Spectrum -> iterator return a list of spectra
                header, data, mask, var, unit = out
                wave = WaveCoord(header, shape=data.shape[0])
                spe = Spectrum(shape=data.shape[0], wave=wave, unit=unit,
                               data=data, var=var)
                spe.data.mask = mask
                if init:
                    result = np.empty(self.shape[0], dtype=type(spe))
                    init = False
                result[k] = spe
            else:
                if is_float(out[0]) or is_int(out[0]):
                    # f returns a number -> iterator returns a spectrum
                    if init:
                        result = Spectrum(wave=self.wave.copy(),
                                          data=np.zeros(shape=self.shape[0]),
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
        wave         : (float, float)
                       (lbda1,lbda2) interval of wavelength in angstrom.
        unit_wave    : astropy.units
                       wavelengths unit (angstrom by default).
                       If None, inputs are in pixels
        is_sum       : boolean
                       if True the sum is computes, otherwise this is the
                       average.
        subtract_off : boolean
                       If True, subtracting off nearby data.
                       The method computes the subtracted flux by using the algorithm
                       from Jarle Brinchmann (jarle@strw.leidenuniv.nl):
                       if is_sum is False
                       sub_flux = mean(flux[lbda1-margin-fband*(lbda2-lbda1)/2: lbda1-margin] +
                                       flux[lbda2+margin: lbda2+margin+fband*(lbda2-lbda1)/2])
                       or if is_sum is True:
                       sub_flux = sum(flux[lbda1-margin-fband*(lbda2-lbda1)/2: lbda1-margin] +
                                      flux[lbda2+margin: lbda2+margin+fband*(lbda2-lbda1)/2]) /fband
        margin       : float
                       This off-band is offseted by margin wrt narrow-band
                       limit.
        fband        : float
                       The size of the off-band is fband*narrow-band width.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`
        """
        d = {'class': 'Cube', 'method': 'get_image'}
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

        if k1<0:
            k1=0
        if k2>(self.shape[0]-1):
            k2=self.shape[0]-1

        msg = 'Computing image for lbda %g-%g [%d-%d]' % (l1, l2, k1, k2+1)
        self.logger.info(msg, extra=d)
        if is_sum:
            ima = self[k1:k2 + 1, :, :].sum(axis=0)
        else:
            ima = self[k1:k2 + 1, :, :].mean(axis=0)

        if subtract_off:
            if unit_wave is not None:
                margin /= self.wave.get_step()
            dl = (k2 +1 - k1) * fband
            lbdas = np.arange(self.shape[0], dtype=float)
            is_off = np.where(((lbdas < k1 - margin) &
                           (lbdas > k1 - margin - dl / 2)) |
                          ((lbdas > k2 + margin) &
                           (lbdas < k2 + margin + dl / 2)))
            if is_sum:
                off_im = self[is_off[0], :, :].sum(axis=0) / (1.0*len(is_off[0]) * fband / dl)
            else:
                off_im = self[is_off[0], :, :].mean(axis=0)
            ima.data -= off_im.data
            if ima.var is not None:
                ima.var += off_im.var

        return ima

    def subcube(self, center, size, lbda=None, unit_center=u.deg,
                unit_size=u.arcsec, unit_wave=u.angstrom):
        """Extracts a sub-cube

        Parameters
        ----------
        center      : (float,float)
                      Center (dec, ra) of the aperture.
        size        : float
                      The size to extract.
                      It corresponds to the size along the delta axis and the image is square.
        lbda        : (float, float) or None
                      If not None, tuple giving the wavelength range.
        unit_center : astropy.units
                      Type of the center coordinates (degrees by default)
        unit_size   : astropy.units
                      unit of the size value (arcseconds by default)
        unit_wave   : astropy.units
                      Wavelengths unit (angstrom by default)
                      If None, inputs are in pixels

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`
        """
        if size > 0 :
            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
            if unit_size is not None:
                size = size / np.abs(self.wcs.get_step(unit=unit_size)[0])
            radius = size / 2.

            imin, jmin = np.maximum(np.minimum(
                (center - radius + 0.5).astype(int),
                [self.shape[1] - 1, self.shape[2] - 1]), [0, 0])
            imax, jmax = np.minimum([imin + int(size+0.5), jmin + int(size+0.5)],
                                    [self.shape[1], self.shape[2]])

            if lbda is None:
                data = self.data[:, imin:imax, jmin:jmax].copy()
                if self.var is not None:
                    var = self.var[:, imin:imax, jmin:jmax].copy()
                else:
                    var = None
                cub = Cube(wcs=self.wcs[imin:imax, jmin:jmax], wave=self.wave,
                       unit=self.unit, data=data, var=var)
                cub.data_header = pyfits.Header(self.data_header)
                cub.primary_header = pyfits.Header(self.primary_header)
                return cub
            else:
                lmin, lmax = lbda
                if unit_wave is None:
                    kmin = int(lmin + 0.5)
                    kmax = int(lmax + 0.5)
                else:
                    kmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)
                    kmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave)+1
                data = self.data[kmin:kmax, imin:imax, jmin:jmax].copy()
                if self.var is not None:
                    var = self.var[kmin:kmax, imin:imax, jmin:jmax].copy()
                else:
                    var = None
                cub = Cube(wcs=self.wcs[imin:imax, jmin:jmax],
                           wave=self.wave[kmin:kmax],
                           unit=self.unit, data=data, var=var)
                cub.data_header = pyfits.Header(self.data_header)
                cub.primary_header = pyfits.Header(self.primary_header)
                return cub
        else:
            return None

    def subcube_circle_aperture(self, center, radius, unit_center=u.deg,
                         unit_radius=u.angstrom):
        """Extracts a sub-cube from an circle aperture of fixed radius.
        Pixels outside the circle are masked.

        Parameters
        ----------
        center      : (float,float)
                      Center (dec,ra) of the aperture.
        radius      : float
                      Radius of the aperture.
                      It corresponds to the radius along the delta axis and the image is square.
        unit_center : astropy.units
                      Type of the center coordinates (degrees by default)
                      If None, inputs are in pixels
        unit_radius : astropy.uunits
                      unit of the radius value (arcseconds by default)
                      If None, inputs are in pixels

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`
        """
        if radius > 0:
            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
            if unit_radius is not None:
                radius = radius / np.abs(self.wcs.get_step(unit=unit_radius)[0])

            radius2 = radius * radius
            imin, jmin = np.maximum(np.minimum(
                (center - radius + 0.5).astype(int),
                [self.shape[1] - 1, self.shape[2] - 1]), [0, 0])
            imax, jmax = np.minimum([imin + int(2 * radius + 0.5),
                                     jmin + int(2 * radius + 0.5)],
                                    [self.shape[1], self.shape[2]])

            grid = np.meshgrid(np.arange(imin, imax) - center[0],
                               np.arange(jmin, jmax) - center[1],
                               indexing='ij')
            grid3d = np.resize((grid[0] ** 2 + grid[1] ** 2) > radius2,
                               (self.shape[0], imax - imin, jmax - jmin))

            data = self.data[:, imin:imax, jmin:jmax].copy()
            data.mask[:, :, :] = np.logical_or(
                self.data.mask[:, imin:imax, jmin:jmax], grid3d)

            if self.var is not None:
                var = self.var[:, imin:imax, jmin:jmax].copy()
            else:
                var = None
            cub = Cube(wcs=self.wcs[imin:imax, jmin:jmax], wave=self.wave,
                       unit=self.unit, data=data, var=var)
            cub.data.mask = data.mask
            cub.data_header = pyfits.Header(self.data_header)
            cub.primary_header = pyfits.Header(self.primary_header)
            return cub
        else:
            return None

    def aperture(self, center, radius, unit_center=u.deg,
                 unit_radius=u.angstrom):
        """Extracts a spectrum from an circle aperture of fixed radius.

        Parameters
        ----------
        center      : (float,float)
                      Center (dec,ra) of the aperture.
        radius      : float
                      Radius of the aperture in arcsec.
                      If None, spectrum at nearest pixel is returned
        unit_center : astropy.units
                      Type of the center coordinates (degrees by default)
                      If None, inputs are in pixels
        unit_radius : astropu_units
                      unit of the radius value (arcseconds by default)
                      If None, inputs are in pixels

        Returns
        -------
        out : :class:`mpdaf.obj.Spectrum`
        """
        d = {'class': 'Cube', 'method': 'aperture'}
        if radius > 0:
            cub = self.subcube_circle_aperture(center, radius,
                                        unit_center=unit_center,
                                        unit_radius=unit_radius)
            msg = '%d spaxels summed' % (cub.shape[1] * cub.shape[2])
            spec = cub.sum(axis=(1, 2))
            self.logger.info(msg, extra=d)
        else:
            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
            spec = self[:, int(center[0] + 0.5), int(center[1] + 0.5)]
            msg = 'returning spectrum at nearest spaxel'
            self.logger.info(msg, extra=d)
        return spec


def _process_spe(arglist):
    try:
        pos, f, header, data, mask, var, \
            unit, kargs = arglist
        wave = WaveCoord(header, shape=data.shape[0])
        spe = Spectrum(shape=data.shape[0], wave=wave, unit=unit, data=data,
                       var=var)
        spe.data.mask = mask

        if isinstance(f, types.FunctionType):
            out = f(spe, **kargs)
        else:
            out = getattr(spe, f)(**kargs)

        try:
            if out.spectrum:
                return pos, 'spectrum', [
                    out.wave.to_header(), out.data.data,
                    out.data.mask, out.var, out.unit]
        except:
            # f returns dtype -> iterator returns an array of dtype
            return pos, 'other', [out]

    except Exception as inst:
        raise type(inst), str(inst) + \
            '\n The error occurred for the spectrum '\
            '[:,%i,%i]' % (pos[0], pos[1])


def _process_ima(arglist):
    try:
        k, f, header, data, mask, var, unit, kargs = arglist
        wcs = WCS(header, shape=data.shape)
        obj = Image(shape=data.shape, wcs=wcs, unit=unit, data=data,
                    var=var)
        obj.data.mask = mask

        if isinstance(f, types.FunctionType):
            out = f(obj, **kargs)
        else:
            out = getattr(obj, f)(**kargs)

        del obj
        del wcs

        try:
            if out.image:
                # f returns an image -> iterator returns a cube
                return k, 'image', [
                    out.wcs.to_header(), out.data.data, out.data.mask,
                    out.var, out.unit]
        except:
            try:
                if out.spectrum:
                    return k, 'spectrum', [
                        out.wave.to_header(), out.data.data,
                        out.data.mask, out.var, out.unit]
            except:
                # f returns dtype -> iterator returns an array of dtype
                return k, 'other', [out]

    except Exception as inst:
        raise type(inst), str(inst) + '\n The error occurred '\
            'for the image [%i,:,:]' % k


    def rebin_factor(self, factor, margin='center'):
        raise DeprecationWarning('Using rebin_factor method is deprecated: Please use rebin_mean instead')
        return self.rebin_mean(factor, margin)

    def subcube_aperture(self, center, radius, unit_center=u.deg,
                         unit_radius=u.angstrom):
        raise DeprecationWarning('Using subcube_aperture method is deprecated: Please use subcube_circle_aperture instead')
        return self.subcube_circle_aperture(center, radius,
                                            unit_center, unit_radius)




class CubeDisk(DataArray):

    """Sometimes, MPDAF users may want to open fairly large datacubes (> 4 Gb
    or so). This can be difficult to handle with limited RAM. This class
    provides a way to open datacube fits files with memory mapping. The methods
    of the class can extract a spectrum, an image or a smaller datacube from
    the larger one.

    Parameters
    ----------
    filename : string
               Possible FITS filename.
    ext      : integer or (integer,integer) or string or (string,string)
               Number/name of the data extension or numbers/names
               of the data and variance extensions.
    notnoise : bool
               True if the noise Variance cube is not read (if it exists).
               Use notnoise=True to create cube without variance extension.

    Attributes
    ----------
    filename       : string
                     Fits file
    data           : int or string
                     Data extension
    unit           : string
                     Possible data unit type
    primary_header : pyfits.Header
                     Possible FITS primary header instance.
    data_header    : pyfits.Header
                     Possible FITS data header instance.
    shape          : array of 3 integers)
                     Lengths of data in Z and Y and X
                     (python notation (nz,ny,nx)).
    var            : int or string
                     Variance extension (-1 if any).
    wcs            : :class:`mpdaf.obj.WCS`
                     World coordinates.
    wave           : :class:`mpdaf.obj.WaveCoord`)
                     Wavelength coordinates
    ima            : dict{string, :class:`mpdaf.obj.Image`}
                     dictionary of images
    """

    def __init__(self, filename=None, ext=None, notnoise=False, ima=True):
        """Creates a CubeDisk object.

        Parameters
        ----------
        filename : string
                   Possible FITS filename.
        ext      : integer or (integer,integer) or string or (string,string)
                   Number/name of the data extension or numbers/names
                   of the data and variance extensions.
        notnoise : bool
                   True if the noise Variance cube is not read (if it exists).
                   Use notnoise=True to create cube without variance extension.
        """
        self.logger = logging.getLogger('mpdaf corelib')
        self.filename = filename
        self.ima = {}
        if filename is not None:
            f = pyfits.open(filename, memmap=True)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1,
                # we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 3:
                    raise IOError('Wrong dimension number: not a cube')
                self.unit = u.Unit(hdr.get('BUNIT', 'count'))
                self.primary_header = pyfits.Header()
                self.data_header = hdr
                self.shape = np.array([hdr['NAXIS3'], hdr['NAXIS2'],
                                       hdr['NAXIS1']])
                self.data = 0
                self.var = -1
                self.fscale = hdr.get('FSCALE', 1.0)
                # WCS object from data header
                try:
                    self.wcs = WCS(hdr)
                except:
                    d = {'class': 'CubeDisk', 'method': '__init__'}
                    self.logger.warning("wcs not copied", extra=d)
                    self.wcs = None
                # Wavelength coordinates
                if 'CRPIX3' not in hdr or 'CRVAL3' not in hdr:
                    self.wave = None
                else:
                    self.wave = WaveCoord(hdr)
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = 'DATA'
                else:
                    if isinstance(ext, int) or isinstance(ext, str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = n
                if h['NAXIS'] != 3:
                    raise IOError('Wrong dimension number in DATA extension')
                self.unit = u.Unit(h.get('BUNIT', 'count'))
                self.primary_header = hdr
                self.data_header = h
                self.shape = np.array([h['NAXIS3'], h['NAXIS2'], h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(h)  # WCS object from data header
                except:
                    d = {'class': 'CubeDisk', 'method': '__init__'}
                    self.logger.warning("wcs not copied", extra=d)
                    self.wcs = None
                # Wavelength coordinates
                if 'CRPIX3' not in h or 'CRVAL3' not in h:
                    self.wave = None
                else:
                    self.wave = WaveCoord(hdr)
                self.var = -1
                if not notnoise:
                    try:
                        if ext is None:
                            fstat = 'STAT'
                        else:
                            n = ext[1]
                            fstat = n
                        if f[fstat].header['NAXIS'] != 3:
                            raise IOError('Wrong dimension number '
                                          'in variance extension')
                        if f[fstat].header['NAXIS1'] != self.shape[2] and \
                                f[fstat].header['NAXIS2'] != self.shape[1] and \
                                f[fstat].header['NAXIS3'] != self.shape[0]:
                            raise IOError('Number of points '
                                          'in STAT not equal to DATA')
                        self.var = fstat
                    except:
                        self.var = -1
                if ima:
                    for i in range(len(f)):
                        try:
                            hdr = f[i].header
                            if hdr['NAXIS'] != 2:
                                raise IOError(' not an image')
                            self.ima[hdr.get('EXTNAME')] = \
                                Image(filename=filename,
                                      ext=hdr.get('EXTNAME'), notnoise=True)
                        except:
                            pass
            # DQ
            f.close()

    def __getitem__(self, item):
        """Returns the corresponding object:

        cube[k,p,k] = value

        cube[k,:,:] = spectrum

        cube[:,p,q] = image

        cube[:,:,:] = sub-cube
        """
        if isinstance(item, tuple) and len(item) == 3:
            f = pyfits.open(self.filename, memmap=True)
            data = f[self.data].data[item]
            if self.var != -1:
                var = f[self.var].data[item]
            else:
                var = None
            f.close()
            if is_int(item[0]):
                if is_int(item[1]) and is_int(item[2]):
                    # return a float
                    return data
                else:
                    # return an image
                    if is_int(item[1]):
                        shape = (1, data.shape[0])
                    elif is_int(item[2]):
                        shape = (data.shape[0], 1)
                    else:
                        shape = data.shape
                    try:
                        wcs = self.wcs[item[1], item[2]]
                    except:
                        wcs = None
                    res = Image(shape=shape, wcs=wcs, unit=self.unit,
                                data=data, var=var)
                    return res
            elif is_int(item[1]) and is_int(item[2]):
                # return a spectrum
                shape = data.shape[0]
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Spectrum(shape=shape, wave=wave, unit=self.unit,
                               data=data, var=var)
                return res
            else:
                # return a cube
                if is_int(item[1]):
                    shape = (data.shape[0], 1, data.shape[1])
                elif is_int(item[2]):
                    shape = (data.shape[0], data.shape[1], 1)
                else:
                    shape = data.shape
                try:
                    wcs = self.wcs[item[1], item[2]]
                except:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Cube(shape=shape, wcs=wcs, wave=wave, unit=self.unit,
                           data=data, var=var)
                res.data_header = pyfits.Header(self.data_header)
                res.primary_header = pyfits.Header(self.primary_header)
                return res
        else:
            raise ValueError('Operation forbidden')

    def truncate(self, coord, mask=True, unit_wave=u.angstrom, unit_wcs=u.deg):
        """ Truncates the cube and return a sub-cube.

        Parameters
        ----------
        coord     : array
                    array containing the sub-cube boundaries
                    [[lbda_min,y_min,x_min], [lbda_max,y_max,x_max]]
                    (output of `mpdaf.obj.cube.get_range`)
        mask      : boolean
                    if True, pixels outside [y_min,y_max]
                    and [x_min,x_max] are masked.
        unit_wave : astropy.units
                    wavelengths unit.
                    If None, inputs are in pixels
        unit_wcs  : astropy.units
                    world coordinates unit.
                    If None, inputs are in pixels
        """
        lmin, y_min, x_min = coord[0]
        lmax, y_max, x_max = coord[1]
        skycrd = [[y_min, x_min], [y_min, x_max], [y_max, x_min],
                  [y_max, x_max]]
        if unit_wcs is None:
            pixcrd = skycrd
        else:
            pixcrd = self.wcs.sky2pix(skycrd, unit=unit_wcs)

        imin = int(np.min(pixcrd[:, 0]))
        if imin < 0:
            imin = 0
        imax = int(np.max(pixcrd[:, 0])) + 1
        if imax > self.shape[1]:
            imax = self.shape[1]
        if imin >= self.shape[1] or imax <= 0 or imin == imax:
            raise ValueError('sub-cube boundaries are outside the cube')

        jmin = int(np.min(pixcrd[:, 1]))
        if jmin < 0:
            jmin = 0
        jmax = int(np.max(pixcrd[:, 1])) + 1
        if jmax > self.shape[2]:
            jmax = self.shape[2]
        if jmin >= self.shape[2] or jmax <= 0 or jmin == jmax:
            raise ValueError('sub-cube boundaries are outside the cube')

        if unit_wave is None:
            kmin = int(lmin+0.5)
            kmax= int(lmax+0.5)
        else:
            kmin = max(0, self.wave.pixel(lmin, nearest=True, unit=unit_wave))
            kmax = min(self.shape[0], self.wave.pixel(lmax, nearest=True, unit=unit_wave) + 1)

        if kmin == kmax:
            raise ValueError('Minimum and maximum wavelengths are equal')

        if kmax == kmin + 1:
            raise ValueError('Minimum and maximum wavelengths '
                             'are outside the spectrum range')

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
                skycrd = np.array(res.wcs.pix2sky(pixcrd), unit=unit_wcs)
            x = skycrd[:, 1].reshape(shape)
            y = skycrd[:, 0].reshape(shape)
            test_x = np.logical_or(x <= x_min, x > x_max)
            test_y = np.logical_or(y <= y_min, y > y_max)
            test = np.logical_or(test_x, test_y)
            res.data.mask = np.logical_or(res.data.mask,
                                          np.tile(test, [res.shape[0], 1, 1]))
            res.resize()

        return res

    def get_white_image(self):
        """Performs a sum over the wavelength dimension and returns an
        image."""
        f = pyfits.open(self.filename, memmap=True)
        loop = True
        k = self.shape[0]
        while loop:
            try:
                data = np.sum(f[self.data].data[0:k], axis=0)
                loop = False
            except:
                k = k / 2
        kmin = k
        kmax = 2 * k
        while kmax < self.shape[0]:
            data += np.sum(f[self.data].data[kmin:kmax], axis=0)
            kmin = kmax
            kmax += k

        if self.var != -1:
            kmin = 0
            kmax = k
            var = np.zeros((self.shape[1], self.shape[2]))
            while kmax < self.shape[0]:
                var += np.sum(f[self.var].data[kmin:kmax], axis=0)
                kmin = kmax
                kmax += k
        else:
            var = None

        f.close()

        return Image(shape=data.shape, wcs=self.wcs, unit=self.unit,
                     data=data, var=var)
