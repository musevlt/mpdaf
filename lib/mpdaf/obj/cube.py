"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>
Copyright (c) 2014-2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c) 2012-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c)      2016 Roland Bacon <roland.bacon@univ-lyon1.fr>

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

from __future__ import absolute_import, division

import astropy.units as u
import multiprocessing
import numpy as np
import os.path
import sys
import time
import types

from astropy.io import fits
from matplotlib.path import Path
from numpy import ma
from six.moves import range, zip
from scipy import integrate, interpolate

from .arithmetic import ArithmeticMixin
from .coords import WCS, WaveCoord
from .data import DataArray
from .image import Image
from .objs import is_int, bounding_box, is_number
from .spectrum import Spectrum
from ..tools import deprecated, add_mpdaf_method_keywords

__all__ = ('iter_spe', 'iter_ima', 'Cube')


def iter_spe(cube, index=False):
    """An iterator over the spectra of successive image pixels in a Cube

    Each call to the iterator returns the spectrum of one pixel of the
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

# A class that is used by loop_ima_multiprocessing and loop_spe_multiprocessing
# to make periodic completion reports to the terminal while tasks are being
# performed by external processes.
class _MultiprocessReporter(object):
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
            output = "\r Completed %d of %d tasks (%d%%) in %.1fs" % (self.completed, self.ntask, self.completed * 100.0 / self.ntask, now - self.start_time)
            sys.stdout.write("\r\x1b[K" + output.__str__())
            sys.stdout.flush()

    def _update_report_time(self):
        """Calculate the time at which the next report should be made."""
        self.report_time = self.start_time + (self.reports+1) * self.interval

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
            return self.mask_ellipse(center=center, radius=radius, posangle=0.0,
                                     lmin=lmin, lmax=lmax, inside=inside,
                                     unit_center=unit_center,
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
            poly = np.array([[-hw*s-hh*c, -hw*c+hh*s],
                             [-hw*s+hh*c, -hw*c-hh*s],
                             [+hw*s+hh*c, +hw*c-hh*s],
                             [+hw*s-hh*c, +hw*c+hh*s]]) / step + center
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
        [sy, sx], unclipped, center = bounding_box(form="rectangle", center=center,
                                                   radii=radius, posangle=0.0,
                                                   shape=self.shape[1:], step=step)

        # Mask pixels inside the region.
        if inside:
            self.data[lmin:lmax, sy, sx] = ma.masked

        # Mask pixels outside the region.
        else:
            self.data[:lmin, :, :] = ma.masked
            self.data[lmax:, :, :] = ma.masked
            self.data[lmin:lmax, 0:sy.start, :] = np.ma.masked
            self.data[lmin:lmax, sy.stop:,   :] = np.ma.masked
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
        [sy, sx], unclipped, center = bounding_box(form="ellipse", center=center,
                                                   radii=radii, posangle=posangle,
                                                   shape=self.shape[1:], step=step)

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
        x, y = np.meshgrid((np.arange(sx.start, sx.stop) - center[1]) * step[1],
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
        self._mask[lmin:lmax,:,:] = np.logical_or(self._mask[lmin:lmax,:,:], c)

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

    def get_lambda(self, lbda_min, lbda_max=None, unit_wave=u.angstrom):
        """Return the sub-cube corresponding to a wavelength range.

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
            pix_max = min(self.shape[0], int(self.wave.pixel(lbda_max, unit=unit_wave)) + 1)

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

    def mean(self, axis=None, weights=None):
        """Return a weighted or unweighted mean over a given axis or axes.

        The mean is computed as follows. Note that weights of 1.0 are
        implicitly used for each data-point if the weights option is
        None.  Given N data points of values, d[i], with weights,
        w[i], the weighted mean of d[0..N-1] is given by:

        mean = Sum(d[i] * w[i]) / Sum(w[i])  for i=0..N-1

        If data point d[i] has a variance of v[i], then the variance
        of the mean is given by:

        variance = Sum(v[i] * w[i]**2) / Sum(w[i])**2   for i=0..N-1

        Note that if one substitutes 1/v[i] for w[i] in this equation,
        the result is a variance of 1/Sum(1/v[i]). If all the
        variances, v[i], happen to have the same value, v, then this
        further simplifies to v / N, which is equivalent to a standard
        deviation of sigma/sqrt(N).  This is the familiar result that
        the signal-to-noise ratio of a mean of N samples increases as
        sqrt(N).

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

            Other cases return None.
        weights : numpy.ndarray or numpy.ma.core.MaskedArray
            When an array of weights is provided via this argument, it
            used to perform a weighted mean, as described in the
            introductory documentation above. If the Cube provides an
            array of variances for each data-point, then a good choice
            for the array of weights is the reciprocal of this array,
            (ie. weights=1.0/cube.var). However beware that not all
            data-sets provide variance information. If a different
            weighting array is provided, note that it must have the
            same shape as the cube. The default value of this argument
            is None, which indicates that an unweighted mean should be
            performed.

        """

        # Check the shape of any weighting array that is provided.
        if(weights is not None and
           not np.array_equal(np.asarray(weights.shape),
                              np.asarray(self.shape))):
            raise ValueError('The weights array has the wrong shape')

        # Average the whole array to a single number?
        if axis is None:
            return ma.average(self.data, weights=weights)

        # Get the data and variances to be processed.
        data = self.data
        var = None if self._var is None else self.var

        # To average over the two dimensions of an image, it is
        # necessary to combine the two image dimensions into a single
        # dimension and average over that axis, because
        # np.ma.average() can only average along one axis.
        if axis == (1,2) or axis == [1,2]:
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
        # number of unmasked points that contributed to the
        # average.
        data, wsum = ma.average(data, axis=axis, weights=weights,
                                returned=True)

        # Does the input cube have variances to be updated?
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

        # Return the average in an appropriate object.
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

        Beware that if the pixels of the cube have associated
        variances, these are discarded by this function, because
        there is no way to estimate the effects of a median on
        the variance.

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

            Other cases return None.

        """
        if axis is None:
            return np.ma.median(self.data)
        elif axis == 0:
            # return an image
            data = np.ma.median(self.data, axis)
            return Image.new_from_obj(self, data=data, var=False)
        elif axis == (1, 2):
            # return a spectrum
            data = np.ma.median(np.ma.median(self.data, axis=1), axis=1)
            return Spectrum.new_from_obj(self, data=data, var=False)
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
        imax = min(np.max(pixcrd[:, 0]), self.shape[1] -1)

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
        res = self[kmin:kmax+1, imin:imax+1, jmin:jmax+1]

        # Mask pixels outside the specified ranges? This is only pertinent
        # to regions that are specified in world coordinates.
        if mask and unit_wcs is not None:
            res.mask_polygon(pixcrd - np.array([imin,jmin]), unit_poly=None,
                             inside=False)
        return res

    def _rebin_(self, factor):
        """Shrink the size of the cube by factor. New size must be an integer
        multiple of the original size.

        Parameters
        ----------
        factor : (int,int,int)
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

    def _rebin(self, factor, margin='center', flux=False):
        """Shrink the size of the cube by factor.

        Parameters
        ----------
        factor : int or (int,int,int)
            Factor in z, y and x. Python notation: (nz,ny,nx).
        flux : bool
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
            self._rebin_(factor)
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
            cub._rebin_(factor)

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

    def rebin(self, factor, margin='center', flux=False, inplace=False):
        """Shrink the size of the cube by factor.

        Parameters
        ----------
        factor : int or (int,int,int)
            Factor in z, y and x. Python notation: (nz,ny,nx).
        flux : bool
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
        factor = np.clip(factor, (1, 1, 1), self.shape)
        res = self if inplace else self.copy()
        res._rebin(factor, margin, flux)
        return res

    def loop_spe_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """Use multiple processes to run a function on each spectrum of a cube.

        The provided function must accept a Spectrum object as its
        first argument, such as a method function of the Spectrum
        class. There are three options for the return value of the
        function.

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
        from mpdaf import CPU

        # Start by assuming that child processes will be created for
        # all CPUs except one.
        cpu_count = multiprocessing.cpu_count() - 1

        # If mpdaf.CPU has been given a smaller value, limit the available
        # CPUs to this number.
        if CPU > 0 and CPU < cpu_count:
            cpu_count = CPU

        # If a smaller number of CPUs has been specified, use this number.
        if cpu is not None and cpu < cpu_count:
            cpu_count = cpu

        # Create a pool of cpu_count worker processes.
        pool = multiprocessing.Pool(processes=cpu_count)

        # If the provided function is an Spectrum method, get its name
        # without the object that it is attached to, so that it can be
        # attached to new Spectrum objects in _process_spe().
        if _is_method(f, Spectrum):
            f = f.__name__

        # Get the attributes that will be passed to the _process_ima()
        # in the worker processes.
        data = self._data
        var = self._var
        mask = self._mask
        header = self.wave.to_header()
        pv, qv = np.meshgrid(list(range(self.shape[1])),
                             list(range(self.shape[2])),
                             sparse=False, indexing='ij')
        pv = pv.ravel()
        qv = qv.ravel()

        # There will be one task per image of the cube. Assemble a
        # list of the argument-lists that will be passed to each of
        # these tasks.
        if var is None:
            processlist = [((p, q), f, header, data[:, p, q], mask[:, p, q],
                            None, self.unit, kargs)
                           for p, q in zip(pv, qv)]
        else:
            processlist = [((p, q), f, header, data[:, p, q], mask[:, p, q],
                            var[:, p, q], self.unit, kargs)
                           for p, q in zip(pv, qv)]

        # Start passing tasks to the worker processes. The return
        # value is an iterator that will hereafter return a new value
        # each time that a worker process finishes one task.
        results = pool.imap_unordered(_process_spe, processlist)

        # Tell the worker pool that no more tasks will be passed to it.
        pool.close()

        # How many spectra are there to be processed as individual tasks?
        ntasks = len(processlist)

        # Report what is being done.
        if verbose:
            self._logger.info('loop_spe_multiprocessing (%s): %i tasks',
                              f, ntasks)
            reporter = _MultiprocessReporter(ntask=ntasks)

        # Wait for the results from each task and collect them into the
        # appropriate object. If verbose is True, also emit a progress
        # report every few seconds.
        init = True
        while True:
            try:
                # Wait for the next result. When verbose=True, interrupt
                # this wait every few seconds to allow a progress-report
                # to be written to the user's terminal.
                if verbose:
                    (p, q), dtype, out = results.next(timeout=reporter.countdown())
                    reporter.note_completed_task()
                else:
                    (p, q), dtype, out = results.next()

                # If the function returns spectra, then accumulate a cube
                # of these spectra.
                if dtype == 'spectrum':
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

                    result.data_header = self.data_header.copy()
                    result.primary_header = self.primary_header.copy()
                    result[:, p, q] = spe

                # If the function returns numbers, assemble these into
                # an image.
                elif is_number(out[0]):
                    if init:
                        result = Image(wcs=self.wcs.copy(),
                                       data=np.zeros((self.shape[1],
                                                      self.shape[2])),
                                       unit=self.unit)
                        init = False
                    result[p, q] = out[0]

                # If the function returns anything else, make a numpy
                # array of them, giving this array the shape of the
                # images in the cube.
                else:
                    if init:
                        result = np.empty((self.shape[1], self.shape[2]),
                                          dtype=type(out[0]))
                        init = False
                    result[p, q] = out[0]

            # Is it time for a new report to be made to the terminal?
            except multiprocessing.TimeoutError:
                pass

            # Have we now processed the last of the images?
            except StopIteration:
                break

            # If the time for the next report has been reached, report
            # the progress through the tasks to the terminal.
            if verbose:
                reporter.report_if_needed()

        return result

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
        from mpdaf import CPU

        # Start by assuming that child processes will be created for
        # all CPUs except one.
        cpu_count = multiprocessing.cpu_count() - 1

        # If mpdaf.CPU has been given a smaller value, limit the available
        # CPUs to this number.
        if CPU > 0 and CPU < cpu_count:
            cpu_count = CPU

        # If a smaller number of CPUs has been specified, use this number.
        if cpu is not None and cpu < cpu_count:
            cpu_count = cpu

        # Create a pool of cpu_count worker processes.
        pool = multiprocessing.Pool(processes=cpu_count)

        # If the provided function is an Image method, get its name
        # without the object that it is attached to, so that it can be
        # attached to new Image objects in _process_ima().
        if _is_method(f, Image):
            f = f.__name__

        # Get the attributes that will be passed to the _process_ima()
        # in the worker processes.
        header = self.wcs.to_header()
        data = self._data
        mask = self._mask
        var = self._var

        # There will be one task per image of the cube. Assemble a
        # list of the argument-lists that will be passed to each of
        # these tasks.
        if var is None:
            processlist = [(k, f, header, data[k, :, :], mask[k, :, :],
                            None, self.unit, kargs)
                           for k in range(self.shape[0])]
        else:
            processlist = [(k, f, header, data[k, :, :], mask[k, :, :],
                            var[k, :, :], self.unit, kargs)
                           for k in range(self.shape[0])]

        # Start passing tasks to the worker processes. The return
        # value is an iterator that will hereafter return a new value
        # each time that a worker process finishes one task.
        results = pool.imap_unordered(_process_ima, processlist)

        # Tell the worker pool that no more tasks will be passed to it.
        pool.close()

        # How many images are there to be processed as individual tasks?
        ntasks = len(processlist)

        # Report what is being done.
        if verbose:
            self._logger.info('loop_ima_multiprocessing (%s): %i tasks',
                              f, ntasks)
            reporter = _MultiprocessReporter(ntask=ntasks)


        # Wait for the results from each task and collect them into the
        # appropriate object. If verbose is True, also emit a progress
        # report every few seconds.
        init = True
        while True:
            try:
                # Wait for the next result. When verbose=True, interrupt
                # this wait every few seconds to allow a progress-report
                # to be written to the user's terminal.
                if verbose:
                    k, dtype, out = results.next(timeout=reporter.countdown())
                    reporter.note_completed_task()
                else:
                    k, dtype, out = results.next()

                # If the function returns images, then accumulate a cube
                # of these images.
                if dtype == 'image':
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
                    result.data_header = self.data_header.copy()
                    result.primary_header = self.primary_header.copy()

                # If the function returns numbers, assemble these into
                # a spectrum.
                elif is_number(out[0]):
                    if init:
                        result = Spectrum(wave=self.wave.copy(),
                                          data=np.zeros(self.shape[0]),
                                          unit=self.unit)
                        init = False
                    result[k] = float(out[0])

                # If the function returns anything else, make a numpy
                # array of them.
                else:
                    if init:
                        result = np.empty(self.shape[0], dtype=type(out[0]))
                        init = False
                    result[k] = out[0]

            # Is it time for a new report to be made to the terminal?
            except multiprocessing.TimeoutError:
                pass

            # Have we now processed the last of the images?
            except StopIteration:
                break

            # If the time for the next report has been reached, report
            # the progress through the tasks to the terminal.
            if verbose:
                reporter.report_if_needed()

        return result

    def get_image(self, wave, is_sum=False, subtract_off=False, margin=10.,
                  fband=3., unit_wave=u.angstrom):
        """Form the average or sum of images over given wavelength range.

        Parameters
        ----------
        wave : (float, float)
            The (lbda1,lbda2) interval of wavelength in angstrom.
        unit_wave : `astropy.units.Unit`
            The wavelength units of the lbda1, lbda2 and margin
            arguments (angstrom by default). If None, lbda1, lbda2,
            and margin should be in pixels.
        is_sum : bool
            If True, compute the sum of the images, otherwise compute
            the arithmetic mean of the images.
        subtract_off : bool
            If True, subtract off a background image that is estimated
            from combining some images from both below and above the
            chosen wavelength range. If the number of images between
            lbda1 and lbda2 is denoted, N, then the number of
            background images taken from below and above the wavelength
            range are:

              nbelow = nabove = (fband * N) / 2   [rounded up to an integer]

            where fband is an optional argument of this function.

            The wavelength ranges of the two groups of background
            images below and above the chosen wavelength range are
            separated from lower and upper edges of the chosen
            wavelength range by a value, margin, which is an argument
            of this function.

            When is_sum is True, the sum of the background images is
            multiplied by N/nbg to produce a background image that has
            the same flux scale as the N images being combined.

            This scheme was developed by Jarle Brinchmann
            (jarle@strw.leidenuniv.nl)
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

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """

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

        # Obtain the sum of the images within the specified range
        # of wavelength pixels.
        if is_sum:
            ima = self[k1:k2 + 1, :, :].sum(axis=0)
        else:
            ima = self[k1:k2 + 1, :, :].mean(axis=0)

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
                raise ValueError("Insufficient space outside the wavelength range to estimate a background")

            # Calculate slices that select the wavelength pixels below
            # and above the chosen wavelength range.
            below = slice(lower_maxpix - nbelow, lower_maxpix)
            above = slice(upper_minpix, upper_minpix + nabove)

            # Combine the background images, rescaling when summing, to
            # obtain the same unit scaling as the combination of the 'nim'
            # foreground images.
            if is_sum:
                off_im = (self[below, :, :].sum(axis=0) +
                          self[above, :, :].sum(axis=0)) * float(nim)/float(nbelow + nabove)
            else:
                off_im = (self[below, :, :].mean(axis=0) +
                          self[above, :, :].mean(axis=0)) / 2.0

            # Subtract the background image from the combined images.
            ima.data -= off_im.data
            if ima._var is not None:
                ima._var += off_im._var

        # add input in header
        unit = 'pix' if unit_wave is None else str(unit_wave)
        f = '' if self.filename is None else os.path.basename(self.filename)
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
        output image is then given by the following weighted mean::

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

        # The sensitivities and wavelengths arrays must be one
        # dimensional and have the same length.

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

        k = np.arange(kmin, kmax + 1, dtype=int)
        w = np.empty((kmax + 1 - kmin))

        # Integrate the bandpass over the range of each spectral pixel
        # to determine the weights of each pixel. For the moment skip
        # the first and last pixels, which need special treatment.
        # Integer pixel indexes refer to the centers of pixels,
        # so for integer pixel index k, we need to integrate from
        # k-0.5 to k+0.5.

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
            var = np.ma.sum(subcube.var * wcube**2, axis=0) / wsum**2
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
        [sy, sx], [uy, ux], center = bounding_box(form = "rectangle",
                                                  center = center,
                                                  radii = size / 2.0,
                                                  posangle = 0.0,
                                                  shape = self.shape[1:],
                                                  step = step)
        if (sx.start >= self.shape[2] or sx.stop < 0 or sx.start==sx.stop or
            sy.start >= self.shape[1] or sy.stop < 0 or sy.start==sy.stop):
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
        data = np.zeros(shape)
        if self._var is None:
            var = None
        else:
            var = np.zeros(shape)

        # If no mask is currently in use, start with every pixel of
        # the new array filled with nans. Otherwise create a mask that
        # initially flags all pixels.
        if self._mask is ma.nomask:
            mask = ma.nomask
            data[:] = np.nan
            if var is not None:
                var[:] = np.nan
        else:
            mask = np.ones(shape, dtype=bool)

        # Calculate the slices where the clipped subcube should go in
        # the new arrays.
        slices = [slice(0, shape[0]),
                  slice(sy.start - uy.start, sy.stop - uy.start),
                  slice(sx.start - ux.start, sx.stop - ux.start)]

        # Copy the clipped subcube into unclipped arrays.
        data[slices] = res._data[:]
        if var is not None:
            var[slices] = res._var[:]
        if mask is not None:
            mask[slices] = res._mask[:]

        # Create a new WCS object for the unclipped subcube.
        wcs = res.wcs
        wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + slices[2].start)
        wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + slices[1].start)
        wcs.set_naxis1(shape[2])
        wcs.set_naxis2(shape[1])

        # Create a new wavelength description object.
        wave = self.wave[sl]

        # Create the new unclipped sub-cube.
        return Cube(wcs=wcs, wave=wave, unit=self.unit, copy=False,
                    data=data, var=var, mask=mask,
                    data_header=fits.Header(self.data_header),
                    primary_header=fits.Header(self.primary_header),
                    filename=self.filename)

    def subcube_circle_aperture(self, center, radius, unit_center=u.deg,
                                unit_radius=u.arcsec):
        """Extract a sub-cube that encloses a circular aperture of
        a specified radius.

        Pixels outside the circle are masked.

        Parameters
        ----------
        center : (float,float)
            The center of the aperture (y,x)
        radius : float
            The radius of the aperture.
        unit_center : `astropy.units.Unit`
            The units of the center coordinates (degrees by default)
            The special value, None, indicates that the center is a
            2D array index.
        unit_radius : `astropy.units.Unit`
            The units of the radius argument (arcseconds by default)
            The special value, None, indicates that the radius is
            specified in pixels.

        Returns
        -------
        out : `~mpdaf.obj.Cube`

        """

        # Extract a subcube of a square image area of 2*radius x 2*radius.
        subcub = self.subcube(center, radius * 2, unit_center=unit_center,
                              unit_size=unit_radius)

        # Mask the region outside the circle.
        center = np.array(subcub.shape[1:]) / 2.0
        subcub.mask_region(center, radius, inside=False,
                           unit_center=None, unit_radius=unit_radius)
        return subcub

    def aperture(self, center, radius, unit_center=u.deg,
                 unit_radius=u.arcsec):
        """Extract the spectrum of a circular aperture of given radius.

        A spectrum is formed by summing the pixels within a specified
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

        Returns
        -------
        out : `~mpdaf.obj.Spectrum`
        """

        # Sum over multiple image pixels?
        if radius is not None and radius > 0:
            cub = self.subcube_circle_aperture(center, radius,
                                               unit_center=unit_center,
                                               unit_radius=unit_radius)
            spec = cub.sum(axis=(1, 2))
            self._logger.info('%d spaxels summed', cub.shape[1] * cub.shape[2])

        # Sum over a single image pixel?
        else:
            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            else:
                center = np.array(center)
            spec = self[:, int(center[0] + 0.5), int(center[1] + 0.5)]
            self._logger.info('returning spectrum at nearest spaxel')
        return spec

    @deprecated('rebin_mean method is deprecated, use rebin instead')
    def rebin_mean(self, factor, margin='center'):
        return self.rebin(factor, margin)

    @deprecated('rebin_median method is deprecated, use rebin instead')
    def rebin_median(self, factor, margin='center'):
        return self.rebin(factor, margin)

    @deprecated('rebin_factor method is deprecated, use rebin instead')
    def rebin_factor(self, factor, margin='center'):
        return self.rebin(factor, margin)

    @deprecated('subcube_aperture method is deprecated: use '
                'subcube_circle_aperture instead')
    def subcube_aperture(self, center, radius, unit_center=u.deg,
                         unit_radius=u.angstrom):
        return self.subcube_circle_aperture(center, radius,
                                            unit_center, unit_radius)


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


def _process_spe(arglist):
    """This function is the function that is executed in worker processes
    by pool.imap_unordered() to do the work of loop_spe_multiprocessing()."""

    try:
        # Expand the argument-list that was passed to pool.imap_unordered().
        pos, f, header, data, mask, var, unit, kargs = arglist

        # Reconstruct the world coordinate information and the Spectrum object.
        wave = WaveCoord(header, shape=data.shape[0])
        spe = Spectrum(wave=wave, unit=unit, data=data, var=var, mask=mask)

        # If the function is an Spectrum method, attach it to the spectrum
        # object that we are processing and execute this.
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
                         '\n The error occurred while processing spectrum '
                         '[:,%i,%i]' % (pos[0], pos[1]))


def _process_ima(arglist):
    """This function is the function that is executed in worker processes
    by pool.imap_unordered() to do the work of loop_ima_multiprocessing()."""

    try:

        # Expand the argument-list that was passed to pool.imap_unordered().
        k, f, header, data, mask, var, unit, kargs = arglist

        # Reconstruct the world coordinate information and the Image object.
        wcs = WCS(header, shape=data.shape)
        obj = Image(wcs=wcs, unit=unit, data=data, var=var, mask=mask)

        # If the function is an Image method, attach it to the image
        # object that we are processing and execute this.
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
                         'while processing image [%i,:,:]' % k)
