"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2012-2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>
Copyright (c) 2016-2018 Roland Bacon <roland.bacon@univ-lyon1.fr>
Copyright (c)      2018 Yannick Roehlly <yannick.roehlly@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import astropy.units as u
import multiprocessing
import numpy as np
import os.path
import sys
import time
import types
import warnings

from astropy.io import fits
from matplotlib.path import Path
from numpy import ma
from scipy import integrate, interpolate, signal, ndimage as ndi

from .arithmetic import ArithmeticMixin
from .data import DataArray
from .image import Image
from .objs import bounding_box, is_number
from .spectrum import Spectrum
from ..tools import add_mpdaf_method_keywords, MpdafWarning

__all__ = ('iter_spe', 'iter_ima', 'Cube')


def iter_spe(cube, index=False):
    """An iterator over the spectra of successive image pixels in a Cube

    Each call to the iterator as the spectrum of one pixel of the
    image. The first spectrum to be returned, is the spectrum of image
    pixel 0,0. Thereafter the X-axis pixel index is incremented by one
    at each call (modulus the length of the X-axis), and the Y-axis
    pixel index is incremented by one each time that the X-axis index
    wraps back to zero.

    The return value of iter_spe() is a python generator that can be
    used in loops, such as in the following example::

      from mpdaf.obj import iter_spe
      for sp,(y,x) in iter_spe(mycube, index=True):
          print("Peak flux in pixel [%d,%d] = %g" % (y, x, sp.data.max()))

    Parameters
    ----------
    cube : `~mpdaf.obj.Cube`
       The cube that contains the spectra to be returned.
    index : bool
       If False, return just a spectrum at each iteration.
       If True, return both a spectrum and the pixel index
       of that spectrum in the image (a tuple of image-array
       indexes along the axes (y,x)).

    Returns
    -------
    out : generator
       A python generator object, suitable for using in a loop.

    """
    if index:
        for y, x in np.ndindex(*cube.shape[1:]):
            yield cube[:, y, x], (y, x)
    else:
        for y, x in np.ndindex(*cube.shape[1:]):
            yield cube[:, y, x]


def iter_ima(cube, index=False):
    """An iterator over the images of successive spectral pixels in a Cube

    Each call to the iterator returns the image of the next spectral
    pixel of the cube. The first image to be returned, is the image of
    spectral pixel 0, followed, on the next call, by the image of
    spectral pixel 1 etc.

    The return value of iter_ima() is a python generator that can be
    used in loops, such as in the following example::

      from mpdaf.obj import iter_ima
      for im,channel in iter_ima(mycube, index=True):
          print("Total flux in channel %d = %g" % (channel, im.data.sum()))

    Parameters
    ----------
    cube : `~mpdaf.obj.Cube`
       The cube that contains the images to be returned.
    index : bool
       If False, return just an image at each iteration.
       If True, return both an image and the index of that image along
       the wavelength axis of the cube.

    Returns
    -------
    out : generator
       A python generator object, suitable for using in a loop.

    """
    if index:
        for l in range(cube.shape[0]):
            yield cube[l, :, :], l
    else:
        for l in range(cube.shape[0]):
            yield cube[l, :, :]


class _MultiprocessReporter(object):
    """ A class that is used by loop_ima_multiprocessing and
    loop_spe_multiprocessing to make periodic completion reports to
    the terminal while tasks are being performed by external processes.
    """

    def __init__(self, ntask, interval=5.0):
        """Prepare to report the progress of a multi-process job.

        Parameters
        ----------
        ntask : int
           The number of tasks to be completed.
        interval : float
           The interval between reports (seconds).
        """

        # Record the configuration parameters.
        self.interval = interval
        self.ntask = ntask

        # Record the approximate start time of the multiple processes.
        self.start_time = time.time()

        # The number of reports so far.
        self.reports = 0

        # The number of tasks completed so far.
        self.completed = 0

        # Calculate the time of the first report.
        self._update_report_time()

    def countdown(self):
        """Obtain the remaining time before the next report is due."""
        return self.report_time - time.time()

    def note_completed_task(self):
        """Tell the reporter that another task has been completed."""
        self.completed += 1

        # Once all tasks have been completed, if any reports have been
        # made, erase the progress line.
        if self.completed == self.ntask and self.reports > 0:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()

    def report_if_needed(self):
        """Report the progress of the tasks if the time for the next report
        has been reached."""

        # Have we reached the time for the next report?
        now = time.time()
        if now >= self.report_time and self.completed < self.ntask:

            # Count the number of reports made, and calculate the time of
            # of the next report.
            self.reports += 1
            self._update_report_time()

            # Inform the user of the progress through the task list.
            output = "\r Completed %d of %d tasks (%d%%) in %.1fs" % (
                self.completed, self.ntask,
                self.completed * 100.0 / self.ntask, now - self.start_time
            )
            sys.stdout.write("\r\x1b[K" + output.__str__())
            sys.stdout.flush()

    def _update_report_time(self):
        """Calculate the time at which the next report should be made."""
        self.report_time = self.start_time + (self.reports + 1) * self.interval


class Cube(ArithmeticMixin, DataArray):

    """This class manages Cube objects, which contain images at multiple
    wavelengths. The images are stored in 3D numpy.ndarray arrays. The
    axes of these arrays are, image wavelength, image y-axis, image
    x-axis. For MUSE images, the y-axis is typically aligned with the
    declination axis on the sky, but in general the y and x axes are
    not aligned with either declination or right-ascension. The actual
    orientation can be queried via the get_rot() method.

    The 3D cube of images is contained in a property of the cube
    called .data. There is also a .var member. When variances are
    available, .var is a 3D array that contains the variances of each
    pixel in the .data array. When variances have not been provided,
    the .var member is given the value, None.

    The .data and the .var are either numpy masked arrays or numpy
    ndarrays. When they are masked arrays, their shared mask can also
    be accessed separately via the .mask member, which holds a 3D
    array of bool elements. When .data and .var are normal
    numpy.ndarrays, the .mask member is given the value
    numpy.ma.nomask.

    When a new Cube object is created, the data, variance and
    mask arrays can either be specified as arguments, or the name
    of a FITS file can be provided to load them from.

    Parameters
    ----------
    filename : str
        An optional FITS file name from which to load the cube.
        None by default. This argument is ignored if the data
        argument is not None.
    ext : int or (int,int) or str or (str,str)
        The optional number/name of the data extension
        or the numbers/names of the data and variance extensions.
    wcs : `mpdaf.obj.WCS`
        The world coordinates of the image pixels.
    wave : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of the spectral pixels.
    unit : str or `astropy.units.Unit`
        The physical units of the data values. Defaults to
        `astropy.units.dimensionless_unscaled`.
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
    copy : bool
        If true (default), then the data and variance arrays are copied.
    dtype : numpy.dtype
        The type of the data (int, float)

    Attributes
    ----------
    filename : str
        The name of the originating FITS file, if any. Otherwise None.
    primary_header : `astropy.io.fits.Header`
        The FITS primary header instance, if a FITS file was provided.
    data_header : `astropy.io.fits.Header`
        The FITS header of the DATA extension.
    wcs : `mpdaf.obj.WCS`
        The world coordinates of the image pixels.
    wave : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of the spectral pixels.
    unit : `astropy.units.Unit`
        The physical units of the data values.
    dtype : numpy.dtype
        The type of the data (int, float)

    """

    # Tell the DataArray base class that cubes require 3 dimensional
    # data arrays, image world coordinates and wavelength coordinates.

    _ndim_required = 3
    _has_wcs = True
    _has_wave = True

    def mask_region(self, center, radius, lmin=None, lmax=None, inside=True,
                    unit_center=u.deg, unit_radius=u.arcsec,
                    unit_wave=u.angstrom, posangle=0.0):
        """Mask values inside or outside a circular or rectangular region.

        Parameters
        ----------
        center : (float,float)
            Center (y,x) of the region, where y,x are usually celestial
            coordinates along the Y and X axes of the image, but are
            interpretted as Y,X array-indexes if unit_center is changed
            to None.
        radius : float or (float,float)
            The radius of a circular region, or the half-width and
            half-height of a rectangular region, respectively.
        lmin : float
            The minimum wavelength of the region.
        lmax : float
            The maximum wavelength of the region.
        inside : bool
            If inside is True, pixels inside the region are masked.
            If inside is False, pixels outside the region are masked.
        unit_wave : `astropy.units.Unit`
            The units of the lmin and lmax wavelength coordinates
            (Angstroms by default). If None, the units of the lmin and
            lmax arguments are assumed to be pixels.
        unit_center : `astropy.units.Unit`
            The units of the coordinates of the center argument
            (degrees by default).  If None, the units of the center
            argument are assumed to be pixels.
        unit_radius : `astropy.units.Unit`
            The units of the radius argument (arcseconds by default).
            If None, the units are assumed to be pixels.
        posangle : float
            When the region is rectangular, this is the counter-clockwise
            rotation angle of the rectangle in degrees. When posangle is
            0.0 (the default), the X and Y axes of the rectangle are along
            the X and Y axes of the image.

        """

        # If the radius argument is a scalar value, this requests
        # that a circular region be masked. Delegate this to mask_ellipse().
        if np.isscalar(radius):
            return self.mask_ellipse(center=center, radius=radius,
                                     lmin=lmin, lmax=lmax, inside=inside,
                                     posangle=0.0, unit_center=unit_center,
                                     unit_radius=unit_radius,
                                     unit_wave=unit_wave)

        # Convert the central position to a floating-point pixel index.
        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]
        else:
            center = np.asarray(center)

        # Get the image pixel sizes in the units of the radius argument.
        if unit_radius is None:
            step = np.array([1.0, 1.0])     # Pixel counts
        else:
            step = self.wcs.get_step(unit=unit_radius)

        # Treat rotated rectangles as polygons.
        if not np.isclose(posangle, 0.0):
            c = np.cos(np.radians(posangle))
            s = np.sin(np.radians(posangle))
            hw = radius[0]
            hh = radius[1]
            poly = np.array([[-hw * s - hh * c, -hw * c + hh * s],
                             [-hw * s + hh * c, -hw * c - hh * s],
                             [+hw * s + hh * c, +hw * c - hh * s],
                             [+hw * s - hh * c, +hw * c + hh * s]]) / step + center
            return self.mask_polygon(poly, lmin, lmax, unit_poly=None,
                                     unit_wave=unit_wave, inside=inside)

        # Get the minimum wavelength in the specified units.
        if lmin is None:
            lmin = 0
        elif unit_wave is not None:
            lmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)

        # Get the maximum wavelength in the specified units.
        if lmax is None:
            lmax = self.shape[0]
        elif unit_wave is not None:
            lmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave)

        # Get Y-axis and X-axis slice objects that bound the rectangular area.
        [sy, sx], _, _ = bounding_box(
            form="rectangle", center=center, radii=radius,
            shape=self.shape[1:], step=step)

        # Mask pixels inside the region.
        if inside:
            self.data[lmin:lmax, sy, sx] = ma.masked

        # Mask pixels outside the region.
        else:
            self.data[:lmin, :, :] = ma.masked
            self.data[lmax:, :, :] = ma.masked
            self.data[lmin:lmax, 0:sy.start, :] = np.ma.masked
            self.data[lmin:lmax, sy.stop:, :] = np.ma.masked
            self.data[lmin:lmax, sy, 0:sx.start] = np.ma.masked
            self.data[lmin:lmax, sy, sx.stop:] = np.ma.masked

    def mask_ellipse(self, center, radius, posangle, lmin=None, lmax=None,
                     inside=True, unit_center=u.deg,
                     unit_radius=u.arcsec, unit_wave=u.angstrom):
        """Mask values inside or outside an elliptical region.

        Parameters
        ----------
        center : (float,float)
            Center (y,x) of the region, where y,x are usually celestial
            coordinates along the Y and X axes of the image, but are
            interpretted as Y,X array-indexes if unit_center is changed
            to None.
        radius : (float,float)
            The radii of the two orthogonal axes of the ellipse.
            When posangle is zero, radius[0] is the radius along
            the X axis of the image-array, and radius[1] is
            the radius along the Y axis of the image-array.
        posangle : float
            The counter-clockwise position angle of the ellipse in
            degrees. When posangle is zero, the X and Y axes of the
            ellipse are along the X and Y axes of the image.
        lmin : float
            The minimum wavelength of the region.
        lmax : float
            The maximum wavelength of the region.
        inside : bool
            If inside is True, pixels inside the region are masked.
            If inside is False, pixels outside the region are masked.
        unit_wave : `astropy.units.Unit`
            The units of the lmin and lmax wavelength coordinates
            (Angstroms by default). If None, the units of the lmin and
            lmax arguments are assumed to be pixels.
        unit_center : `astropy.units.Unit`
            The units of the coordinates of the center argument
            (degrees by default).  If None, the units of the center
            argument are assumed to be pixels.
        unit_radius : `astropy.units.Unit`
            The units of the radius argument (arcseconds by default).
            If None, the units are assumed to be pixels.

        """

        # Convert the central position to floating-point pixel indexes.
        center = np.array(center)
        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]

        # Get the pixel sizes in the units of the radius argument.
        if unit_radius is None:
            step = np.array([1.0, 1.0])     # Pixel counts
        else:
            step = self.wcs.get_step(unit=unit_radius)

        # Get the two radii in the form of a numpy array.
        if np.isscalar(radius):
            radii = np.array([radius, radius])
        else:
            radii = np.asarray(radius)

        # Get the minimum wavelength in the specified units.
        if lmin is None:
            lmin = 0
        elif unit_wave is not None:
            lmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)

        # Get the maximum wavelength in the specified units.
        if lmax is None:
            lmax = self.shape[0]
        elif unit_wave is not None:
            lmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave)

        # Obtain Y and X axis slice objects that select the rectangular
        # region that just encloses the rotated ellipse.
        [sy, sx], _, center = bounding_box(
            form="ellipse", center=center, radii=radii,
            shape=self.shape[1:], posangle=posangle, step=step)

        # Precompute the sine and cosine of the position angle.
        cospa = np.cos(np.radians(posangle))
        sinpa = np.sin(np.radians(posangle))

        # When the position angle is zero, such that the
        # xe and ye axes of the ellipse are along the X and Y axes
        # of the image-array, the equation of the ellipse is:
        #
        #   (xe / rx)**2 + (ye / ry)**2 = 1
        #
        # Before we can use this equation with the rotated ellipse, we
        # have to rotate the pixel coordinates clockwise by the
        # counterclockwise position angle of the ellipse to align the
        # rotated axes of the ellipse along the image X and Y axes:
        #
        #   xp  =  | cos(pa),  sin(pa)| |x|
        #   yp     |-sin(pa),  cos(pa)| |y|
        #
        # The value of k returned by the following equation will then
        # be < 1 for pixels inside the ellipse, == 1 for pixels on the
        # ellipse and > 1 for pixels outside the ellipse.
        #
        #   k = (xp / rx)**2 + (yp / ry)**2
        x, y = np.meshgrid(
            (np.arange(sx.start, sx.stop) - center[1]) * step[1],
            (np.arange(sy.start, sy.stop) - center[0]) * step[0])
        ksel = (((x * cospa + y * sinpa) / radii[0]) ** 2 +
                ((y * cospa - x * sinpa) / radii[1]) ** 2)

        if inside:
            grid3d = np.resize(ksel < 1, (lmax - lmin, ) + ksel.shape)
            self.data[lmin:lmax, sy, sx][grid3d] = ma.masked
        else:
            self.data[:lmin, :, :] = ma.masked
            self.data[lmax:, :, :] = ma.masked
            self.data[lmin:lmax, :sy.start, :] = ma.masked
            self.data[lmin:lmax, sy.stop:, :] = ma.masked
            self.data[lmin:lmax, :, :sx.start] = ma.masked
            self.data[lmin:lmax, :, sx.stop:] = ma.masked

            grid3d = np.resize(ksel > 1, (lmax - lmin, ) + ksel.shape)
            self.data[lmin:lmax, sy, sx][grid3d] = ma.masked

    def mask_polygon(self, poly, lmin=None, lmax=None,
                     unit_poly=u.deg, unit_wave=u.angstrom, inside=True):
        """Mask values inside or outside a polygonal region.

        Parameters
        ----------
        poly : (float, float)
            An array of (float,float) containing a set of (p,q) or (dec,ra)
            values for the polygon vertices.
        lmin : float
            The minimum wavelength of the region.
        lmax : float
            The maximum wavelength of the region.
        unit_poly : `astropy.units.Unit`
            The units of the polygon coordinates (degrees by default).
            Use unit_poly=None for polygon coordinates in pixels.
        unit_wave : `astropy.units.Unit`
            The units of the wavelengths lmin and lmax (angstrom by default).
        inside : bool
            If inside is True, pixels inside the polygonal region are masked.
            If inside is False, pixels outside the polygonal region are masked.

        """

        # Convert DEC,RA (deg) values coming from poly into Y,X value (pixels)
        if unit_poly is not None:
            poly = np.array([
                [self.wcs.sky2pix((val[0], val[1]), unit=unit_poly)[0][0],
                 self.wcs.sky2pix((val[0], val[1]), unit=unit_poly)[0][1]]
                for val in poly])

        P, Q = np.meshgrid(list(range(self.shape[1])),
                           list(range(self.shape[2])))
        b = np.dstack([P.ravel(), Q.ravel()])

        # Use a matplotlib method to create a path, which is the polygon we
        # want to use.
        polymask = Path(poly)

        # Go through all pixels in the image to see if they are within the
        # polygon. The ouput is a boolean table.
        c = polymask.contains_points(b[0])

        # Invert the boolean table to mask pixels outside the polygon?
        if not inside:
            c = ~np.array(c)

        # Convert the boolean table into a matrix.
        c = c.reshape(self.shape[2], self.shape[1])
        c = c.T

        # Convert the minimum wavelength to a spectral pixel index.
        if lmin is None:
            lmin = 0
        elif unit_wave is not None:
            lmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)

        # Convert the maximum wavelength to a spectral pixel index.
        if lmax is None:
            lmax = self.shape[0]
        elif unit_wave is not None:
            lmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave)

        # Combine the previous mask with the new one between lmin and lmax.
        self._mask[lmin:lmax, :, :] |= c

        # When masking pixels outside the region, mask all pixels
        # outside the specified wavelength range.
        if not inside:
            self.data[:lmin, :, :] = ma.masked
            self.data[lmax:, :, :] = ma.masked

        return poly

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

    def select_lambda(self, lbda_min, lbda_max=None, unit_wave=u.angstrom):
        """Return the image or sub-cube corresponding to a wavelength range.

        Parameters
        ----------
        lbda_min : float
            The minimum wavelength to be selected.
        lbda_max : float
            The maximum wavelength to be selected, or None
            to just select one image close to lbda_min.
        unit_wave : `astropy.units.Unit`
            The wavelength units of lbda_min and lbda_max.
            The value, None, can be used to indicate that
            lbda_min and lbda_max are in pixel-index units.
            The default unit is angstrom.

        Returns
        -------
        out : `mpdaf.obj.Cube` or `mpdaf.obj.Image`
            If more than one spectral channel is selected, then
            a Cube object is returned that contains just the images
            of those channels. If a single channel is selected, then
            an Image object is returned, containing just the image
            of that channel.
        """

        # Select just the image that is closest to lbda_min?
        if lbda_max is None:
            lbda_max = lbda_min

        # If the wavelength limits are in pixels, round them to
        # the nearest integer values.
        if unit_wave is None:
            pix_min = max(0, int(lbda_min + 0.5))
            pix_max = min(self.shape[0], int(lbda_max + 0.5))

        # If wavelengths have been specified, then we need wavelength
        # world-coordinates to convert them to pixel indexes.
        elif self.wave is None:
            raise ValueError('Operation impossible without world coordinates '
                             'along the spectral direction')

        # Convert wavelength limits to the nearest pixels.
        else:
            pix_min = max(0, int(self.wave.pixel(lbda_min, unit=unit_wave)))
            pix_max = min(self.shape[0],
                          int(self.wave.pixel(lbda_max, unit=unit_wave)) + 1)

        # When just one channel is selected, return an Image. When
        # multiple channels are selected, return a sub-cube.
        if (pix_min + 1) == pix_max:
            return self[pix_min, :, :]
        else:
            return self[pix_min:pix_max, :, :]

    def get_step(self, unit_wave=None, unit_wcs=None):
        """Return the cube steps [dlbda,dy,dx].

        Parameters
        ----------
        unit_wave : `astropy.units.Unit`
            The wavelength units of the returned wavelength
            step.
        unit_wcs : `astropy.units.Unit`
            The angular units of the returned spatial
            world-coordinate steps.

        Returns
        -------
        out : [dlbda, dy, dx]
            Where, dlbda is the size of pixels along the
            wavelength axis, and dy and dx are the sizes
            of pixels along the Y and X axes of
            the image, respectively.

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
        """Return [lbda,y,x] at the center of pixel (0,0,0).

        Parameters
        ----------
        unit_wave : `astropy.units.Unit`
            The wavelenth units to use for the starting wavelength.
            The default value, None, selects the intrinsic wavelength
            units of the cube.
        unit_wcs : `astropy.units.Unit`
            The world coordinates units to use for the spatial
            starting position. The default value, None, selects
            the intrinsic world coordinates of the cube.

        """
        start = np.empty(3)
        start[0] = self.wave.get_start(unit_wave)
        start[1:] = self.wcs.get_start(unit_wcs)
        return start

    def get_end(self, unit_wave=None, unit_wcs=None):
        """Return [lbda,y,x] at the center of pixel (-1,-1,-1).

        Parameters
        ----------
        unit_wave : `astropy.units.Unit`
            The wavelenth units to use for the starting wavelength.
            The default value, None, selects the intrinsic wavelength
            units of the cube.
        unit_wcs : `astropy.units.Unit`
            The world coordinates units to use for the spatial
            starting position. The default value, None, selects
            the intrinsic world coordinates of the cube.

        """
        end = np.empty(3)
        end[0] = self.wave.get_end(unit_wave)
        end[1:] = self.wcs.get_end(unit_wcs)
        return end

    def get_rot(self, unit=u.deg):
        """Return the rotation angle of the images of the cube,
        defined such that a rotation angle of zero aligns north
        along the positive Y axis, and a positive rotation angle
        rotates north away from the Y axis, in the sense of a
        rotation from north to east.

        Note that the rotation angle is defined in a flat
        map-projection of the sky. It is what would be seen if
        the pixels of the image were drawn with their pixel
        widths scaled by the angular pixel increments returned
        by the get_axis_increments() method.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The unit to give the returned angle (degrees by default).

        Returns
        -------
        out : float
            The angle between celestial north and the Y axis of
            the image, in the sense of an eastward rotation of
            celestial north from the Y-axis.

        """
        return self.wcs.get_rot(unit)

    def sum(self, axis=None, weights=None):
        """Return a sum over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
            Axis or axes along which a sum is performed:

            - The default (axis = None) performs a sum over all the
              dimensions of the cube and returns a float.
            - axis = 0 performs a sum over the wavelength dimension and
              returns an image.
            - axis = (1,2) performs a sum over the (X,Y) axes and returns
              a spectrum.

        weights : ndarray, np.ma.array, float
            When an array of weights is provided via this argument, it
            used to perform a weighted sum. This involves obtaining a
            weighted mean using the Cube.mean() function, then scaling
            this by the number of points that were averaged along the
            specified axes. The number of points that is used to scale
            the mean to a sum, is the total number of points along the
            averaged axes, not the number of unmasked points that had
            finite weights. As a result, the sum behaves as though all
            pixels along the averaged axes had values equal to the
            mean, regardless of whether any of these were masked or
            had zero weight.

            The weights array can have the same shape as the cube, or
            they can be 1-D if axis=(1,2), or 2-D if axis=0. If
            weights=None, then all non-masked data points are given a
            weight equal to one. Finally, if a scalar float is given,
            then the data are all weighted equally. This can be used
            to get an unweighted sum that behaves as though masked
            pixels in the input cube had been filled with the mean
            along the averaged axes before the sum was performed.

            If the Cube provides an array of variances for each
            data-point, then a good choice for the array of weights is
            the reciprocal of this array, (ie. weights=1.0/cube.var).
            However note that not all data-sets provide variance
            information, so check that cube.var is not None before
            trying this.

            Any weight elements that are masked, infinite or nan, are
            replaced with zero. As a result, if the weights are
            specified as 1.0/cube.var, then any zero-valued variances
            will not produce infinite weights.

        """

        # Should a weighted sum be performed?
        doweight = weights is not None
        if doweight and is_number(weights):
            weights = None      # This requests unit weights.

        # Sum all pixels to yield a single value?
        if axis is None:
            if doweight:
                return self.mean(axis=axis, weights=weights) * np.prod(self.shape)
            else:
                return self.data.sum()

        # Sum along the spectral axis to yield an image?
        elif axis == 0:
            if doweight:
                return self.mean(axis=axis, weights=weights) * self.shape[0]
            else:
                data = ma.sum(self.data, axis=0)
                if self._var is not None:
                    var = ma.sum(self.var, axis=0)
                else:
                    var = None
                return Image.new_from_obj(self, data=data, var=var)

        # Sum along the image X and Y axes to yield a spectrum?
        elif axis == (1, 2):
            if doweight:
                return self.mean(axis=axis, weights=weights) * np.prod(self.shape[1:])
            else:
                data = ma.sum(ma.sum(self.data, axis=1), axis=1)
                if self._var is not None:
                    var = ma.sum(ma.sum(self.var, axis=1), axis=1).filled(np.inf)
                else:
                    var = None
                return Spectrum(wave=self.wave, unit=self.unit, data=data,
                                var=var, copy=False)
        else:
            raise ValueError('Invalid axis argument')

    def mean(self, axis=None, weights=None):
        """Return a weighted or unweighted mean over a given axis or axes.

        The mean is computed as follows. Note that weights of 1.0 are
        implicitly used for each data-point if the weights option is None.
        Given N data points of values, d[i], with weights, w[i], the weighted
        mean of d[0..N-1] is given by::

            mean = Sum(d[i] * w[i]) / Sum(w[i])  for i=0..N-1

        If data point d[i] has a variance of v[i], then the variance
        of the mean is given by::

            variance = Sum(v[i] * w[i]**2) / Sum(w[i])**2   for i=0..N-1

        Note that if one substitutes 1/v[i] for w[i] in this equation, the
        result is a variance of 1/Sum(1/v[i]). If all the variances, v[i],
        happen to have the same value, v, then this further simplifies to
        v / N, which is equivalent to a standard deviation of sigma/sqrt(N).
        This is the familiar result that the signal-to-noise ratio of a mean of
        N samples increases as sqrt(N).

        Parameters
        ----------
        axis : None or int or tuple of ints
            The axis or axes along which the mean is to be performed.

            The default (axis = None) performs a mean over all the
            dimensions of the cube and returns a float.

            axis = 0 performs a mean over the wavelength dimension and
            returns an image.

            axis = (1,2) performs a mean over the (X,Y) axes and
            returns a spectrum.

        weights : numpy.ndarray or numpy.ma.core.MaskedArray
            When an array of weights is provided via this argument, it
            used to perform a weighted mean, as described in the
            introductory documentation above.

            The weights array can have the same shape as the cube, or
            they can be 1-D if axis=(1,2), or 2-D if axis=0. If
            weights is None then all non-masked data points are given
            a weight equal to one.

            If the Cube provides an array of variances for each
            data-point, then a good choice for the array of weights is
            the reciprocal of this array, (ie. weights=1.0/cube.var).
            However beware that not all data-sets provide variance
            information.

        """

        if weights is not None:
            # Convert the weights array to a non-masked array with
            # masked, infinite and nan values replaced with zero weights.
            if isinstance(weights, ma.MaskedArray):
                weights = weights.filled(0.0)
            weights = np.where(np.isfinite(weights), weights, 0.0)

            # If the dimensions of the weights array does not match
            # the dimensions of the data, remedy this if possible using
            # the rules given in the description of the weights argument.
            if not np.array_equal(weights.shape, self.shape):
                msg = 'Wrong dimensions for the weights (%s) (should be (%s))'
                if weights.ndim == 3:
                    raise ValueError(msg % (weights.shape, self.shape))
                elif weights.ndim == 2:
                    if np.array_equal(weights.shape, self.shape[1:]):
                        weights = np.tile(weights, (self.shape[0], 1, 1))
                    else:
                        raise ValueError(msg % (weights.shape, self.shape[1:]))
                elif weights.ndim == 1:
                    if weights.shape[0] == self.shape[0]:
                        weights = (np.ones_like(self._data) *
                                   weights[:, np.newaxis, np.newaxis])
                    else:
                        raise ValueError(msg % (weights.shape[0],
                                                self.shape[0]))
                else:
                    raise ValueError(msg % (None, self.shape))

        if axis is None:
            return ma.average(self.data, weights=weights)

        data = self.data
        var = None if self._var is None else self.var

        # To average over the two dimensions of an image, it is
        # necessary to combine the two image dimensions into a single
        # dimension and average over that axis, because
        # np.ma.average() can only average along one axis.
        if axis == (1, 2) or axis == [1, 2]:
            shape = (self.shape[0], self.shape[1] * self.shape[2])
            data = data.reshape(shape)
            if weights is not None:
                weights = weights.reshape(shape)
            if var is not None:
                var = var.reshape(shape)
            axis = 1

        # Average the data over the specified axis. Note that the
        # wsum return value holds the sum of weights for each of the
        # returned data points. When weights=None, this is the
        # number of unmasked points that contributed to the average.
        data, wsum = ma.average(data, axis=axis, weights=weights,
                                returned=True)

        if var is not None:
            # Compute the variance of each averaged data-point,
            # using the equation given in the docstring of
            # this function. When weights=None, the effective
            # weights are all unity, so we don't need to multiply
            # the data by the square of the weights in that case.
            if weights is None:
                var = ma.sum(var, axis=axis) / wsum**2
            else:
                var = ma.sum(var * weights**2, axis=axis) / wsum**2

        if axis is None:
            return data
        elif axis == 0:
            return Image.new_from_obj(self, data=data, var=var)
        elif axis == 1:
            return Spectrum.new_from_obj(self, data=data, var=var)
        else:
            raise ValueError('Invalid axis argument')

    def median(self, axis=None):
        """Return the median over a given axis.

        Beware that if the pixels of the cube have associated variances, these
        are discarded by this function, because there is no way to estimate the
        effects of a median on the variance.

        Parameters
        ----------
        axis : None or int or tuple of ints
            The axis or axes along which a median is performed.

            The default (axis = None) performs a median over all the
            dimensions of the cube and returns a float.

            axis = 0 performs a median over the wavelength dimension and
            returns an image.

            axis = (1,2) performs a median over the (X,Y) axes and
            returns a spectrum.

        """
        if axis is None:
            return np.ma.median(self.data)
        elif axis == 0:
            # return an image
            data = np.ma.median(self.data, axis)
            return Image.new_from_obj(self, data=data, var=False, copy=False)
        elif axis == (1, 2):
            # return a spectrum
            data = np.ma.median(np.ma.median(self.data, axis=1), axis=1)
            return Spectrum.new_from_obj(self, data=data, var=False,
                                         copy=False)
        else:
            raise ValueError('Invalid axis argument')

    def max(self, axis=None):
        """Return the maximum over a given axis.

        Beware that if the pixels of the cube have associated variances, these
        are discarded by this function, because there is no way to estimate the
        effects of a maximum on the variance.

        Parameters
        ----------
        axis : None or int or tuple of ints
            The axis or axes along which the maximum is computed.

            The default (axis = None) computes the maximum over all the
            dimensions of the cube and returns a float.

            axis = 0 computes the maximum over the wavelength dimension and
            returns an image.

            axis = (1,2) computes the maximum over the (X,Y) axes and
            returns a spectrum.

        """
        if axis is None:
            return np.ma.amax(self.data)
        elif axis == 0:
            # return an image
            data = np.ma.amax(self.data, axis)
            return Image.new_from_obj(self, data=data, var=False, copy=False)
        elif axis == (1, 2):
            # return a spectrum
            data = np.ma.amax(np.ma.amax(self.data, axis=1), axis=1)
            return Spectrum.new_from_obj(self, data=data, var=False,
                                         copy=False)
        else:
            raise ValueError('Invalid axis argument')

    def min(self, axis=None):
        """Return the minimum over a given axis.

        Beware that if the pixels of the cube have associated variances, these
        are discarded by this function, because there is no way to estimate the
        effects of a minimum on the variance.

        Parameters
        ----------
        axis : None or int or tuple of ints
            The axis or axes along which the minimum is computed.

            The default (axis = None) computes the minimum over all the
            dimensions of the cube and returns a float.

            axis = 0 computes the minimum over the wavelength dimension and
            returns an image.

            axis = (1,2) computes the minimum over the (X,Y) axes and
            returns a spectrum.

        """
        if axis is None:
            return np.ma.amin(self.data)
        elif axis == 0:
            # return an image
            data = np.ma.amin(self.data, axis)
            return Image.new_from_obj(self, data=data, var=False, copy=False)
        elif axis == (1, 2):
            # return a spectrum
            data = np.ma.amin(np.ma.amin(self.data, axis=1), axis=1)
            return Spectrum.new_from_obj(self, data=data, var=False,
                                         copy=False)
        else:
            raise ValueError('Invalid axis argument')

    def truncate(self, coord, mask=True, unit_wave=u.angstrom, unit_wcs=u.deg):
        """Return a sub-cube bounded by specified wavelength and spatial
        world-coordinates.

        Note that unless unit_wcs is None, the y-axis and x-axis
        boundary coordinates are along sky axes such as declination
        and right-ascension, which may not be parallel to the image
        array-axes. When they are not parallel, the returned image
        area will contain some pixels that are outside the requested
        range. These are masked by default. To prevent them from being
        masked, pass False to the mask argument.

        Parameters
        ----------
        coord : array
           The coordinate boundaries, arranged into an array
           as follows:

             [lbda_min, y_min, x_min, lbda_max, y_max, x_max]

           Note that this is the order of the values returned by
           mpdaf.obj.cube.get_range(), so when these functions are
           used together, then can be used to extract a subcube whose
           bounds match those of an existing smaller cube.
        mask : bool
            If True, pixels outside [y_min,y_max] and [x_min,x_max]
            are masked. This can be useful when the world-coordinate
            X and Y axis are not parallel with the image array-axes.
        unit_wave : `astropy.units.Unit`
            The wavelength units of lbda_min and lbda_max elements
            of the coord array.  If None, lbda_min and lbda_max are
            interpretted as pixel indexes along the wavelength axis.
        unit_wcs : `astropy.units.Unit`
            The wavelength units of x_min,x_max,y_min and y_max
            elements of the coord array.  If None, these values are
            interpretted as pixel indexes along the image axes.

        Returns
        -------
        out : `mpdaf.obj.Cube`
            A Cube object that contains the requested sub-cube.
        """
        lmin, ymin, xmin, lmax, ymax, xmax = coord

        # Get the coordinates of the corners of the requested
        # sub-image, ordered as though sequentially drawing the lines
        # of a polygon.
        skycrd = [[ymin, xmin], [ymax, xmin],
                  [ymax, xmax], [ymin, xmax]]

        # Convert corner coordinates to pixel indexes.
        if unit_wcs is None:
            pixcrd = np.array(skycrd)
        else:
            pixcrd = self.wcs.sky2pix(skycrd, unit=unit_wcs)

        # Round the corners to the integer center of the containing pixel.
        pixcrd = np.floor(pixcrd + 0.5).astype(int)

        # Obtain the range of Y-axis pixel-indexes of the region.
        imin = max(np.min(pixcrd[:, 0]), 0)
        imax = min(np.max(pixcrd[:, 0]), self.shape[1] - 1)

        # Obtain the range of X-axis pixel-indexes of the region.
        jmin = max(np.min(pixcrd[:, 1]), 0)
        jmax = min(np.max(pixcrd[:, 1]), self.shape[2] - 1)

        # Convert the wavelength range to floating-point pixel indexes.
        if unit_wave is not None:
            lmin = self.wave.pixel(lmin, unit=unit_wave)
            lmax = self.wave.pixel(lmax, unit=unit_wave)

        # Round the wavelength limits to the integer center of the
        # containing wavelength pixel.
        kmin = max(np.floor(lmin + 0.5).astype(int), 0)
        kmax = min(np.floor(lmax + 0.5).astype(int), self.shape[0] - 1)

        # Complain if the truncation would leave nothing.
        if imin > imax or jmin > jmax or kmin > kmax:
            raise ValueError('The requested area is not within the cube.')

        # Extract the requested part of the cube.
        res = self[kmin:kmax + 1, imin:imax + 1, jmin:jmax + 1]

        # Mask pixels outside the specified ranges? This is only pertinent
        # to regions that are specified in world coordinates.
        if mask and unit_wcs is not None:
            res.mask_polygon(pixcrd - np.array([imin, jmin]), unit_poly=None,
                             inside=False)
        return res

    def rebin(self, factor, margin='center', inplace=False):
        """Combine neighboring pixels to reduce the size of a cube by integer
        factors along each axis.

        Each output pixel is the mean of n pixels, where n is the
        product of the reduction factors in the factor argument.

        Parameters
        ----------
        factor : int or (int,int,int)
            The integer reduction factors along the wavelength, z
            array axis, and the image y and x array axes,
            respectively. Python notation: (nz,ny,nx).
        margin : 'center'|'right'|'left'|'origin'
            When the dimensions of the input cube are not integer
            multiples of the reduction factor, the cube is truncated
            to remove just enough pixels that its dimensions are
            multiples of the reduction factor. This subcube is then
            rebinned in place of the original cube. The margin
            parameter determines which pixels of the input cube are
            truncated, and which remain.

            The options are:
              'origin' or 'center':
                 The starts of the axes of the output cube are
                 coincident with the starts of the axes of the input
                 cube.
              'center':
                 The center of the output cube is aligned with the
                 center of the input cube, within one pixel along
                 each axis.
              'right':
                 The ends of the axes of the output cube are
                 coincident with the ends of the axes of the input
                 cube.
        inplace : bool
            If False, return a rebinned copy of the cube (the default).
            If True, rebin the original cube in-place, and return that.

        """
        factor = np.asarray(factor)
        return self._rebin(factor, margin, inplace)

    def loop_spe_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """Use multiple processes to run a function on each spectrum of a cube.

        The provided function must accept a Spectrum object as its first
        argument, such as a method function of the Spectrum class. There are
        three options for the return value of the function.

        1. The return value of the provided function can be another
           Spectrum, in which case loop_spe_multiprocessing() returns
           a Cube of the processed spectra.

        2. Alternatively if each call to the function returns a scalar
           number, then these numbers are assembled into an Image that
           has the same world coordinates as the Cube. This Cube is
           returned by loop_spe_multiprocessing().

        3. Finally, if each call to the provided function returns a
           value that can be stored as an element of a numpy array,
           then a numpy array of these values is returned. This array
           has the shape of the images in the cube, such that pixel of
           [i,j] of the returned array contains the result of processing
           the spectrum of pixel [i,j] in the cube.

        Parameters
        ----------
        f : function or `~mpdaf.obj.Spectrum` method
            The function to be applied to each spectrum of the cube.
            This function must either be a method of the Spectrum
            class, or it must be a top-level function that accepts an
            Spectrum object as its first argument. It should return
            either an Spectrum, a number, or a value that can be
            recorded as an element of a numpy array. Note that the
            function must not be a lambda function, or a function
            that is defined within another function.
        cpu : int
            The desired number of CPUs to use, or None to select
            the number of available CPUs. By default, the available
            number of CPUs is equal to the number of CPU cores on the
            computer, minus 1 CPU for the main process. However
            the variable, `mpdaf.CPU` can be assigned a smaller number
            by the user to limit the number that are available to MPDAF.
        verbose : bool
            If True, a progress report is printed every 5 seconds.
        kargs : kargs
            An optional list of arguments to be passed to the function
            f(). The datatypes of all of the arguments in this list
            must support python pickling.

        Returns
        -------
        out : `~mpdaf.obj.Cube` if f returns `~mpdaf.obj.Spectrum`,
        out : `~mpdaf.obj.Image` if f returns a float or int,
        out : np.array(dtype=object) for all others cases.

        """
        return _loop_multiprocessing(self, f, 'spe', cpu=cpu,
                                     verbose=verbose, **kargs)

    def loop_ima_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """Use multiple processes to run a function on each image of a cube.

        The provided function must accept an Image object as its first
        argument, such as a method function of the Image class. There are
        three options for the return value of the function.

        1. The return value of the provided function can be another
           Image, in which case loop_ima_multiprocessing() returns a Cube
           of the processed images.

        2. Alternatively if each call to the function returns a scalar
           number, then these numbers are assembled into a Spectrum
           that has the same spectral coordinates as the Cube.
           This Cube is returned by loop_ima_multiprocessing().

        3. Finally, if each call to the provided function returns a
           value that can be stored as an element of a numpy array,
           then a numpy array of these values is returned, ordered
           such that element k of the array contains the return value
           for spectral channel k of the cube.

        Parameters
        ----------
        f : function or `~mpdaf.obj.Image` method
            The function to be applied to each image of the cube.
            This function must either be a method of the Image class,
            or it must be a top-level function that accepts an Image object
            as its first argument. It should return either an Image,
            a number, or a value that can be recorded as an element
            of a numpy array. Note that the function must not be a
            lambda function, or a function that is defined within
            another function.
        cpu : int
            The desired number of CPUs to use, or None to select
            the number of available CPUs. By default, the available
            number of CPUs is equal to the number of CPU cores on the
            computer, minus 1 CPU for the main process. However
            the variable, `mpdaf.CPU` can be assigned a smaller number
            by the user to limit the number that are available to MPDAF.
        verbose : bool
            If True, a progress report is printed every 5 seconds.
        kargs : kargs
            An optional list of arguments to be passed to the function
            f(). The datatypes of all of the arguments in this list
            must support python pickling.

        Returns
        -------
        out : `~mpdaf.obj.Cube` if f returns `~mpdaf.obj.Image`,
        out : `~mpdaf.obj.Spectrum` if f returns a float or int.
        out : np.array(dtype=object) for all others cases.

        """
        return _loop_multiprocessing(self, f, 'ima', cpu=cpu,
                                     verbose=verbose, **kargs)

    def get_image(self, wave, is_sum=False, subtract_off=False, margin=10.,
                  fband=3., unit_wave=u.angstrom, method="mean"):
        """Generate an image aggregating over a wavelength range.

        This method creates an image aggregating all the slices between
        a wavelength range.

        Parameters
        ----------
        wave : (float, float)
            The (lbda1,lbda2) interval of wavelength in angstrom.
        unit_wave : `astropy.units.Unit`
            The wavelength units of the lbda1, lbda2 and margin
            arguments (angstrom by default). If None, lbda1, lbda2,
            and margin should be in pixels.
        is_sum : bool
            If True, compute the sum of the images. Deprecated, use "sum"
            as aggregation method.
        subtract_off : bool
            If True, subtract off a background image that is estimated
            from combining some images from both below and above the
            chosen wavelength range. If the number of images between
            lbda1 and lbda2 is denoted, N, then the number of
            background images taken from below and above the wavelength
            range are::

                nbelow = nabove = (fband * N) / 2   [rounded up to an integer]

            where fband is a parameter of this function.

            The wavelength ranges of the two groups of background
            images below and above the chosen wavelength range are
            separated from lower and upper edges of the chosen
            wavelength range by a value, margin, which is an argument
            of this function.

            This scheme was developed by Jarle Brinchmann
            (jarle@strw.leidenuniv.nl)

            The background is removed from the wavelength region of interest
            before aggregating it.
        margin : float
            The wavelength or pixel offset of the centers of the
            ranges of background images below and above the chosen
            wavelength range. This has the units specified by the
            unit_wave argument. The zero points of the margin are one
            pixel below and above the chosen wavelength range.
            The default value is 10 angstroms.
        fband : float
            The ratio of the number of images used to form a
            background image and the number of images that are being
            combined.  The default value is 3.0.
        method: str
            Name of the Cube method used to aggregate the data. This method
            must accept the axis=0 parameter and return an image. Example:
            mean, sum, max.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        if is_sum:
            warnings.warn(
                "The 'is_sum' parameter is deprecated. Use method='sum' "
                "instead. Aggregation function set to sum.", MpdafWarning)
            method = "sum"

        # Convert the wavelength range to pixel indexes.
        if unit_wave is None:
            k1, k2 = wave
        else:
            k1, k2 = np.rint(self.wave.pixel(wave, unit=unit_wave)).astype(int)

        # Clip the wavelength range to the available range of pixels.
        k1 = max(k1, 0)
        k2 = min(k2, self.shape[0] - 1)

        # Obtain the effective wavelength range of the chosen
        # wavelength pixels.
        l1 = self.wave.coord(k1 - 0.5)
        l2 = self.wave.coord(k1 + 0.5)

        # Sub-cube on the wavelength range
        data_cube = self[k1:k2 + 1, :, :]

        # Subtract off a background image?
        if subtract_off:

            # Convert the margin to a number of pixels.
            if unit_wave is not None:
                margin = np.rint(
                    margin / self.wave.get_step(unit=unit_wave)).astype(int)

            # How many images were combined above?
            nim = k2 + 1 - k1

            # Calculate the indexes of the last pixel of the lower range
            # of background images and the first pixel of the upper range
            # of background images.
            lower_maxpix = max(k1 - 1 - margin, 0)
            upper_minpix = min(k2 + 1 + margin, self.shape[0])

            # Calculate the number of images to separately select from
            # below and above the chosen wavelength range.
            nhalf = np.ceil(nim * fband / 2.0).astype(int)

            # Start by assuming that we will be combining equal numbers
            # of images from below and above the chosen wavelength range.
            nabove = nhalf
            nbelow = nhalf

            # If the chosen wavelength range is too close to one edge of
            # the cube's wavelength range, reduce the number to fit.
            if lower_maxpix - nbelow < 0:
                nbelow = lower_maxpix
            elif upper_minpix + nabove > self.shape[0]:
                nabove = self.shape[0] - upper_minpix

            # If there was too little room both below and above the
            # chosen wavelength range to compute the background, give up.
            if(lower_maxpix - nbelow < 0 or
               upper_minpix + nabove > self.shape[0]):
                raise ValueError('Insufficient space outside the wavelength '
                                 'range to estimate a background')

            # Calculate slices that select the wavelength pixels below
            # and above the chosen wavelength range.
            below = slice(lower_maxpix - nbelow, lower_maxpix)
            above = slice(upper_minpix, upper_minpix + nabove)

            # The background is the mean of the background below and the
            # background above (may be different of the mean of above and
            # below pixels if the number of pixels is different above and
            # below).
            background = (self[below, :, :].mean(axis=0) +
                          self[above, :, :].mean(axis=0)) / 2

            # Adding and Image to a Cube takes care of variance propagation.
            data_cube = data_cube - background

        # Aggregating using the Cube method takes care of the variance
        # propagation.
        ima = getattr(data_cube, method)(axis=0)

        # add input in header
        unit = 'pix' if unit_wave is None else str(unit_wave)
        f = '' if self.filename is None else os.path.basename(self.filename)
        add_mpdaf_method_keywords(
            ima.primary_header, "cube.get_image",
            ['cube', 'lbda1', 'lbda2', 'method', 'subtract_off', 'margin',
             'fband'],
            [f, l1, l2, method, subtract_off, margin, fband],
            ['cube', 'min wavelength (%s)' % str(unit),
             'max wavelength (%s)' % str(unit), 'aggregation method',
             'subtracting off nearby data', 'off-band margin', 'off_band size']
        )

        return ima

    def get_band_image(self, name):
        """Generate an image using a known filter.

        Parameters
        ----------
        name : str
            Filter name. Must exist in the filter file taken from the MUSE DRS
            (`lib/mpdaf/obj/filters/filter_list.fits`).  Available filters:
            Johnson_B, Johnson_V, Cousins_R, Cousins_I, SDSS_u, SDSS_g, SDSS_r,
            SDSS_i, SDSS_z, ACS_F475W, ACS_F550M, ACS_F555W, ACS_F606W,
            ACS_F625W, ACS_F775W, ACS_F814W, WFPC2_F555W, WFPC2_F675W,
            WFPC2_F814W, WFC3_F502N, WFC3_F555W, WFC3_F606W, WFC3_F625W,
            WFC3_F656N, WFC3_F775W, WFC3_F814W, Kron_V

        """
        FILTERS = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'filters', 'filter_list.fits')

        with fits.open(FILTERS) as hdul:
            if name not in hdul:
                filter_names = ', '.join(hdu.name for hdu in hdul[1:])
                raise ValueError("requested filter '{}' not found. Available "
                                 "filters: {}".format(name, filter_names))
            wave = hdul[name].data['lambda']
            throughput = hdul[name].data['throughput']

        im = self.bandpass_image(wave, throughput, unit_wave=u.angstrom,
                                 interpolation="linear")
        # as we use the DRS filters, add the same keyword allowing to find
        # which filter was used.
        im.primary_header['ESO DRS MUSE FILTER NAME'] = (name,
                                                         'filter name used')
        add_mpdaf_method_keywords(im.primary_header, "cube.get_band_image",
                                  ['name'], [name], ['filter name used'])
        return im

    def bandpass_image(self, wavelengths, sensitivities, unit_wave=u.angstrom,
                       interpolation="linear"):
        """Given a cube of images versus wavelength and the bandpass
        filter-curve of a wide-band monochromatic instrument, extract
        an image from the cube that has the spectral response of the
        monochromatic instrument.

        For example, this can be used to create a MUSE image that has
        the same spectral characteristics as an HST image. The MUSE
        image can then be compared to the HST image without having to
        worry about any differences caused by different spectral
        sensitivities.

        For each channel n of the cube, the filter-curve is integrated
        over the width of that channel to obtain a weight, w[n]. The
        output image is then given by the following weighted mean::

            output_image = sum(w[n] * cube_image[n]) / sum(w[n])

        In practice, to accomodate masked pixels, the w[n] array is
        expanded into a cube w[n,y,x], and the weights of individual
        masked pixels in the cube are zeroed before the above equation
        is applied.

        If the wavelength axis of the cube only partly overlaps the
        bandpass of the filter-curve, the filter curve is truncated to
        fit within the bounds of the wavelength axis. A warning is
        printed to stderr if this occurs, because this results in an
        image that lacks flux from some of the wavelengths of the
        requested bandpass.

        Parameters
        ----------
        wavelengths : numpy.ndarray
            An array of the wavelengths of the filter curve,
            listed in ascending order of wavelength. Outside
            the listed wavelengths the filter-curve is assumed
            to be zero.
        sensitivities : numpy.ndarray
            The relative flux sensitivities at the wavelengths
            in the wavelengths array. These sensititivies will be
            normalized, so only their relative values are important.
        unit_wave : `astropy.units.Unit`
            The units used in the array of wavelengths. The default is
            angstroms. To specify pixel units, pass None.
        interpolation : str
            The form of interpolation to use to integrate over the
            filter curve. This should be one of::

              "linear"     : Linear interpolation
              "cubic"      : Cubic spline interpolation (very slow)

            The default is linear interpolation. If the filter curve
            is well sampled and its sampling interval is narrower than
            the wavelength pixels of the cube, then this should be
            sufficient. Alternatively, if the sampling interval is
            significantly wider than the wavelength pixels of the
            cube, then cubic interpolation should be used instead.
            Beware that cubic interpolation is much slower than linear
            interpolation.

        Returns
        -------
        out : `~mpdaf.obj.Image`
            An image formed from the filter-weighted mean
            of channels in the cube that overlap the bandpass
            of the filter curve.

        """

        wavelengths = np.asarray(wavelengths, dtype=float)
        sensitivities = np.asarray(sensitivities, dtype=float)

        if (wavelengths.ndim != 1 or sensitivities.ndim != 1 or
                len(wavelengths) != len(sensitivities)):
            raise ValueError('The wavelengths and sensititivies arguments'
                             ' should be 1D arrays of equal length')

        if unit_wave is None:
            pixels = wavelengths.copy()
        else:
            pixels = self.wave.pixel(wavelengths, unit=unit_wave)

        # Get the integer indexes of the pixels that contain the above
        # floating point pixel indexes.
        indexes = np.rint(pixels).astype(int)

        # If there is no overlap between the bandpass filter curve
        # and the wavelength coverage of the cube, complain.
        if indexes[0] >= self.shape[0] or indexes[-1] < 0:
            raise ValueError("The filter curve does not overlap the "
                             "wavelength coverage of the cube.")

        # To correctly reproduce an image taken through a specified
        # filter, the bandpass curve should be completely encompassed
        # by the wavelength axis of the cube. If the overlap is
        # incomplete, emit a warning, then truncate the bandpass curve
        # to the edge of the wavelength range of the cube.
        if indexes[0] < 0 or indexes[-1] >= self.shape[0]:

            # Work out the start and stop indexes of the slice needed
            # to truncate the arrays of the bandpass filter curve.
            if indexes[0] < 0:
                start = np.searchsorted(indexes, 0, 'left')
            else:
                start = 0
            if indexes[-1] >= self.shape[0]:
                stop = np.searchsorted(indexes, self.shape[0], 'left')
            else:
                stop = indexes.shape[0]

            # Integrate the overal bandpass filter curve.
            total = integrate.trapz(sensitivities, wavelengths)

            # Also integrate over just the truncated parts of the curve.
            lost = 0.0
            if start > 0:
                s = slice(0, start)
                lost += integrate.trapz(sensitivities[s], wavelengths[s])
            if stop < indexes.shape[0]:
                s = slice(stop, indexes.shape[0])
                lost += integrate.trapz(sensitivities[s], wavelengths[s])

            # Compute the fraction of the integrated bandpass response
            # that has been truncated.
            lossage = lost / total

            # Truncate the bandpass filter curve.
            indexes = indexes[start:stop]
            pixels = pixels[start:stop]
            sensitivities = sensitivities[start:stop]

            # Report the loss if it is over 0.5%.
            if lossage > 0.005:
                self._logger.warning(
                    "%.2g%% of the integrated " % (lossage * 100.0) +
                    "filter curve is beyond the edges of the cube.")

        # Get the range of indexes along the wavelength axis that
        # encompass the filter bandpass within the cube.
        kmin = indexes[0]
        kmax = indexes[-1]

        # Obtain an interpolator of the bandpass curve.
        spline = interpolate.interp1d(x=pixels, y=sensitivities,
                                      kind=interpolation)

        # Integrate the bandpass over the range of each spectral pixel
        # to determine the weights of each pixel. For the moment skip
        # the first and last pixels, which need special treatment.
        # Integer pixel indexes refer to the centers of pixels,
        # so for integer pixel index k, we need to integrate from
        # k-0.5 to k+0.5.
        w = np.empty((kmax + 1 - kmin))
        for k in range(kmin + 1, kmax):
            w[k - kmin], err = integrate.quad(spline, k - 0.5, k + 0.5)

        # Start the integration of the weight of the first channel
        # from the lower limit of the bandpass.
        w[0], err = integrate.quad(spline, pixels[0], kmin + 0.5)

        # End the integration of the weight of the final channel
        # at the upper limit of the bandpass.
        w[-1], err = integrate.quad(spline, kmax - 0.5, pixels[-1])

        # Normalize the weights.
        w /= w.sum()

        # Create a sub-cube of the selected channels.
        subcube = self[kmin:kmax + 1, :, :]

        # To accomodate masked pixels, create a cube of the above
        # weights, but with masked pixels given zero weight.
        if subcube._mask is ma.nomask:
            wcube = w[:, np.newaxis, np.newaxis] * np.ones(subcube.shape)
        else:
            wcube = w[:, np.newaxis, np.newaxis] * ~subcube._mask

        # Get an image which is the sum of the weights along the spectral axis.
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
            var = np.ma.sum(subcube.var * wcube**2, axis=0) / wsum**2
        else:
            var = False

        return Image.new_from_obj(subcube, data=data, var=var)

    def subcube(self, center, size, lbda=None, unit_center=u.deg,
                unit_size=u.arcsec, unit_wave=u.angstrom):
        """Return a subcube view around a position and for a wavelength range.

        Note: as this is a view on the original cube, both the cube and the
        sub-cube will be modified at the same time.  If you need to make
        changes only to the sub-cube, copy it before.

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
        # If only the width is given, give the height the same size.
        if np.isscalar(size):
            size = np.array([size, size])
        else:
            size = np.asarray(size)
        if size[0] <= 0.0 or size[1] <= 0.0:
            raise ValueError('Size must be positive')

        # Get the central position in pixels.
        center = np.asarray(center)
        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]

        # Get the image pixel steps in the units of the size argument.
        if unit_size is None:
            step = np.array([1.0, 1.0])     # Pixel counts
        else:
            step = self.wcs.get_step(unit=unit_size)

        # Select the whole wavelength range?
        if lbda is None:
            lmin = 0
            lmax = self.shape[0] - 1
        else:
            # Get the wavelength range.
            lmin, lmax = lbda

            # Convert the minimum wavelength to a wavelength pixel-index.
            if unit_wave is not None:
                lmin = self.wave.pixel(lmin, nearest=True, unit=unit_wave)

            # Convert the maximum wavelength to a wavelength pixel-index.
            if unit_wave is not None:
                lmax = self.wave.pixel(lmax, nearest=True, unit=unit_wave)

            # Check that the wavelength bounds are usable and in ascending
            # order.
            if lmin >= self.shape[0] or lmax <= 0:
                raise ValueError("Wavelength range not in cube")
            elif lmin > lmax:
                lmin, lmax = lmax, lmin

        # Create a slice that selects the above wavelength range,
        # clipped to the extent of the cube.
        sl = slice(lmin if lmin >= 0 else 0,
                   lmax + 1 if lmax < self.shape[0] else self.shape[0])

        # Get Y-axis and X-axis slice objects that bound the rectangular area.
        [sy, sx], [uy, ux], center = bounding_box(
            form="rectangle", center=center, radii=size / 2.0,
            shape=self.shape[1:], step=step)

        if (sx.start >= self.shape[2] or sx.stop < 0 or sx.start == sx.stop or
                sy.start >= self.shape[1] or sy.stop < 0 or
                sy.start == sy.stop):
            raise ValueError('Sub-cube boundaries are outside the cube')

        # Extract the requested part of the cube.
        res = self[sl, sy, sx]

        # If the image region was not clipped by the edges of the
        # parent cube, then return the subcube.
        if sy == uy and sx == ux:
            return res

        # Since the subcube is smaller than requested, due to clipping,
        # create new data and variance arrays of the required size.
        shape = (sl.stop - sl.start, uy.stop - uy.start, ux.stop - ux.start)
        data = np.zeros(shape, dtype=self.dtype)
        var = None if self._var is None else np.zeros(shape, dtype=self.dtype)

        # Create the mask (ignoring nomask) as we need it to mask the regions
        # outside of the subcube
        mask = np.ones(shape, dtype=bool)

        # Calculate the slices where the clipped subcube should go in
        # the new arrays.
        slices = (slice(0, shape[0]),
                  slice(sy.start - uy.start, sy.stop - uy.start),
                  slice(sx.start - ux.start, sx.stop - ux.start))

        # Copy the clipped subcube into unclipped arrays.
        data[slices] = res._data[:]
        if var is not None:
            var[slices] = res._var[:]
        if res._mask is not ma.nomask:
            mask[slices] = res._mask[:]
        else:
            mask[slices] = False

        # Create a new WCS object for the unclipped subcube.
        wcs = res.wcs
        wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + slices[2].start)
        wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + slices[1].start)
        wcs.naxis1 = shape[2]
        wcs.naxis2 = shape[1]

        # Create a new wavelength description object.
        wave = self.wave[sl]

        # Create the new unclipped sub-cube.
        return Cube(wcs=wcs, wave=wave, unit=self.unit, copy=False,
                    data=data, var=var, mask=mask,
                    data_header=fits.Header(self.data_header),
                    primary_header=fits.Header(self.primary_header),
                    filename=self.filename)

    def subcube_circle_aperture(self, center, radius, lbda=None,
                                unit_center=u.deg, unit_radius=u.arcsec,
                                unit_wave=u.angstrom):
        """Extract a sub-cube that encloses a circular aperture of
        a specified radius and for a given wavelength range.

        Pixels outside the circle are masked.

        Parameters
        ----------
        center : (float,float)
            The center of the aperture (y,x)
        radius : float
            The radius of the aperture.
        lbda : (float, float) or None
            If not None, tuple giving the wavelength range.
        unit_center : `astropy.units.Unit`
            The units of the center coordinates (degrees by default)
            The special value, None, indicates that the center is a
            2D array index.
        unit_radius : `astropy.units.Unit`
            The units of the radius argument (arcseconds by default)
            The special value, None, indicates that the radius is
            specified in pixels.
        unit_wave : `astropy.units.Unit`
            Wavelengths unit (angstrom by default)
            If None, inputs are in pixels

        Returns
        -------
        out : `~mpdaf.obj.Cube`

        """
        subcub = self.subcube(center, radius * 2, unit_center=unit_center,
                              lbda=lbda, unit_size=unit_radius,
                              unit_wave=unit_wave).copy()
        # Mask the region outside the circle. Work on a copy to avoid modifying
        # the original cube.
        center = np.array(subcub.shape[1:]) / 2.0
        subcub.mask_region(center, radius, inside=False,
                           unit_center=None, unit_radius=unit_radius)
        return subcub

    def aperture(self, center, radius, unit_center=u.deg,
                 unit_radius=u.arcsec, is_sum=True):
        """Extract the spectrum of a circular aperture of given radius.

        A spectrum is formed by summing/averaging the pixels within a specified
        circular region of each wavelength image. This yields a spectrum
        that has the same length as the wavelength axis of the cube.

        Parameters
        ----------
        center : (float,float)
            The center of the aperture (y,x).
        radius : float
            The radius of the aperture.
            If the radius is None, or <= 0, the spectrum of
            the nearest image pixel to the specified center is
            returned.
        unit_center : `astropy.units.Unit`
            The units of the center coordinates (degrees by default)
            The special value, None, indicates that center is a 2D
            pixel index.
        unit_radius : `astropy.units.Unit`
            The units of the radius argument (arcseconds by default)
            The special value, None, indicates that the radius is
            specified in pixels.
        is_sum : bool
            If True, compute the sum of the pixels, otherwise compute
            the arithmetic mean of the pixels.

        Returns
        -------
        out : `~mpdaf.obj.Spectrum`
        """

        # Sum over multiple image pixels?
        if radius is not None and radius > 0:
            cub = self.subcube_circle_aperture(center, radius,
                                               unit_center=unit_center,
                                               unit_radius=unit_radius)
            if is_sum:
                spec = cub.sum(axis=(1, 2))
            else:
                spec = cub.mean(axis=(1, 2))
            self._logger.info('%d spaxels used', cub.shape[1] * cub.shape[2])
        # Sum over a single image pixel?
        else:
            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
            spec = self[:, int(center[0] + 0.5), int(center[1] + 0.5)]
            self._logger.info('returning spectrum at nearest spaxel')
        return spec

    def convolve(self, other, inplace=False):
        """Convolve a Cube with a 3D array or another Cube, using the
        discrete convolution equation.

        This function, which uses the discrete convolution equation, is
        usually slower than Cube.fftconvolve(). However it can be faster when
        other.data.size is small, and it always uses much less memory, so it
        is sometimes the only practical choice.

        Masked values in self.data and self.var are replaced with zeros before
        the convolution is performed, but they are masked again after the
        convolution.

        If self.var exists, the variances are propagated using the equation::

            result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived by
        applying the usual rules of error-propagation to the discrete
        convolution equation.

        The speed of this function scales as O(Nd x No) where
        Nd=self.data.size and No=other.data.size.

        Uses `scipy.signal.convolve`.

        Parameters
        ----------
        other : Cube or np.ndarray
            The 3D array with which to convolve the cube in self.data.
            This can be an 3D array of the same size as self, or it
            can be a smaller array, such as a small 3D gaussian to use to
            smooth the larger cube.

            When ``other`` contains a symmetric filtering function, such
            as a 3-dimensional gaussian, the center of the function
            should be placed at the center of pixel:

             ``(other.shape - 1) // 2``

            If other is an MPDAF Cube object, note that only its data
            array is used. Masked values in this array are treated
            as zero, and any variances found in other.var are ignored.
        inplace : bool
            If False (the default), return the results in a new Cube.
            If True, record the result in self and return that.

        Returns
        -------
        out : `~mpdaf.obj.Cube`

        """
        return self._convolve(signal.convolve, other=other, inplace=inplace)

    def fftconvolve(self, other, inplace=False):
        """Convolve a Cube with a 3D array or another Cube, using the
        Fourier convolution theorem.

        This function, which performs the convolution by multiplying the
        Fourier transforms of the two arrays, is usually much faster than
        Cube.convolve(), except when other.data.size is small. However it uses
        much more memory, so Cube.convolve() is sometimes a better choice.

        Masked values in self.data and self.var are replaced with zeros before
        the convolution is performed, but they are masked again after the
        convolution.

        If self.var exists, the variances are propagated using the equation::

            result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived by
        applying the usual rules of error-propagation to the discrete
        convolution equation.

        The speed of this function scales as O(Nd x log(Nd)) where
        Nd=self.data.size.  It temporarily allocates a pair of arrays that
        have the sum of the shapes of self.shape and other.shape, rounded up
        to a power of two along each axis. This can involve a lot of memory
        being allocated. For this reason, when other.shape is small,
        Cube.convolve() may be more efficient than Cube.fftconvolve().

        Uses `scipy.signal.fftconvolve`.

        Parameters
        ----------
        other : Cube or np.ndarray
            The 3D array with which to convolve the cube in self.data.
            This array can be the same size as self, or it can be a
            smaller array, such as a small 3D gaussian to use to
            smooth the larger cube.

            When ``other`` contains a symmetric filtering function, such as a
            3-dimensional gaussian, the center of the function should be
            placed at the center of pixel:

             ``(other.shape - 1) // 2``

            If ``other`` is an MPDAF Cube object, note that only its data
            array is used. Masked values in this array are treated as
            zero, and any variances found in other.var are ignored.
        inplace : bool
            If False (the default), return the results in a new Cube.
            If True, record the result in self and return that.

        Returns
        -------
        out : `~mpdaf.obj.Cube`

        """
        return self._convolve(signal.fftconvolve, other=other, inplace=inplace)

    def spatial_erosion(self, npixels, inplace=False):
        """Remove n pixels around the masked white image.

        Parameters
        ----------
        npixels : integer
            Erosion width in pixels
        inplace : bool
            If False (the default), return the results in a new Cube.
            If True, record the result in self and return that.

        """
        # Change the input cube or change a copy of it?
        res = self if inplace else self.copy()
        white = self.sum(axis=0)
        mask = ~ndi.binary_erosion(white._data, mask=~white._mask,
                                   iterations=npixels)
        res._mask = (np.resize(mask, self.shape) | self._mask)
        return res


def _is_method(func, cls):
    """Check if func is a method of cls.

    The previous way to do this using isinstance(types.MethodType) is not
    compatible with Python 3 (which no more has unbound methods). So one way to
    do this is to check if func is an attribute of cls, and has a __name__
    attribute.

    """
    try:
        getattr(cls, func.__name__)
    except AttributeError:
        return False
    else:
        return True


def _multiproc_worker(arglist):
    """Worker process for loop_{spe/ima}_multiprocessing"""
    try:
        pos, f, obj, kwargs = arglist
        # If the function is an Spectrum method, attach it to the spectrum
        # object that we are processing and execute this.
        if isinstance(f, types.FunctionType):
            return pos, f(obj, **kwargs)
        else:
            return pos, getattr(obj, f)(**kwargs)
    except Exception as inst:
        raise inst.__class__(
            '{}\n The error occurred while processing {} {}'
            .format(str(inst), obj.__class__.__name__, pos))


def _loop_multiprocessing(self, f, loop_type, cpu=None, verbose=True, **kargs):
    # Determine the number of processes:
    # - default: all CPUs except one.
    # - mdaf.CPU
    # - cpu_count parameter
    from mpdaf import CPU
    cpu_count = multiprocessing.cpu_count() - 1
    if CPU > 0 and CPU < cpu_count:
        cpu_count = CPU
    if cpu is not None and cpu < cpu_count:
        cpu_count = cpu

    pool = multiprocessing.Pool(processes=cpu_count)

    # If the provided function is an Image or Spectru method, get its name
    if (loop_type == 'ima' and _is_method(f, Image)) or \
            (loop_type == 'spe' and _is_method(f, Spectrum)):
        f = f.__name__

    if loop_type == 'ima':
        # There will be one task per image
        processlist = [(k, f, self[k, :, :], kargs)
                       for k in range(self.shape[0])]
    elif loop_type == 'spe':
        # There will be one task per spectrum
        processlist = [((p, q), f, self[:, p, q], kargs)
                       for p, q in np.ndindex(self.shape[1:])]
    else:
        raise ValueError('unsupported way to slice the cube')

    # Start passing tasks to the worker processes. The return
    # value is an iterator that will hereafter return a new value
    # each time that a worker process finishes one task.
    results = pool.imap_unordered(_multiproc_worker, processlist)

    # Tell the worker pool that no more tasks will be passed to it.
    pool.close()

    # How many images are there to be processed as individual tasks?
    ntasks = len(processlist)

    # Report what is being done.
    if verbose:
        self._logger.info('loop_%s_multiprocessing (%s): %i tasks',
                          loop_type, f, ntasks)
        reporter = _MultiprocessReporter(ntask=ntasks)

    # Wait for the results from each task and collect them into the appropriate
    # object. If verbose, also emit a progress report every few seconds.
    init = True
    while True:
        try:
            # Wait for the next result. When verbose=True, interrupt
            # this wait every few seconds to allow a progress-report
            # to be written to the user's terminal.
            if verbose:
                k, out = results.next(timeout=reporter.countdown())
                reporter.note_completed_task()
            else:
                k, out = results.next()

            if isinstance(out, (Image, Spectrum)):
                # If the function returns images or spectra, make a cube
                if init:
                    if loop_type == 'ima':
                        cshape = (self.shape[0], out.shape[0], out.shape[1])
                        wcs = out.wcs
                        wave = self.wave
                    elif loop_type == 'spe':
                        cshape = (out.shape[0], self.shape[1], self.shape[2])
                        wcs = self.wcs
                        wave = out.wave

                    result = Cube(
                        wcs=wcs, wave=wave,
                        data=np.zeros(cshape),
                        unit=out.unit,
                        data_header=self.data_header.copy(),
                        primary_header=self.primary_header.copy()
                    )
                    if self.var is not None:
                        result._var = np.zeros(cshape)
                    init = False

                if loop_type == 'ima':
                    result[k, :, :] = out
                elif loop_type == 'spe':
                    p, q = k
                    result[:, p, q] = out

            elif is_number(out):
                # If it returns numbers, assemble these into a spectrum/image
                if init:
                    if loop_type == 'ima':
                        result = Spectrum(wave=self.wave, unit=self.unit,
                                          data=np.zeros(self.shape[0],
                                                        dtype=type(out)))
                    elif loop_type == 'spe':
                        result = Image(wcs=self.wcs, unit=self.unit,
                                       data=np.zeros(self.shape[1:],
                                                     dtype=type(out)))
                    init = False
                result[k] = out

            else:
                # If the function returns anything else, make a numpy array
                if init:
                    if loop_type == 'ima':
                        result = np.empty(self.shape[0], dtype=type(out))
                    elif loop_type == 'spe':
                        result = np.empty(self.shape[1:], dtype=type(out))
                    init = False
                result[k] = out

        except multiprocessing.TimeoutError:
            # Is it time for a new report to be made to the terminal?
            pass
        except StopIteration:
            break
        if verbose:
            # If the time for the next report has been reached, report
            # the progress through the tasks to the terminal.
            reporter.report_if_needed()

    return result
