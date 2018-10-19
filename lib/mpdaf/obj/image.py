"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2012-2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2012-2017 Roland Bacon <roland.bacon@univ-lyon1.fr>
Copyright (c) 2014-2016 Johan Richard <jrichard@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2015 Floriane Leclercq <floriane.leclercq@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>
Copyright (c)      2017 Ole Streicher <ole@aip.de>
Copyright (c)      2018 David Carton <cartondj@gmail.com>
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

import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

import astropy.units as u
from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
from astropy.visualization import ZScaleInterval
from scipy import interpolate, signal
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import affine_transform
from scipy.optimize import leastsq

from .arithmetic import ArithmeticMixin
from .coords import WCS
from .data import DataArray
from .fitting import Gauss2D, Moffat2D
from .objs import is_int, is_number, bounding_box, UnitMaskedArray, UnitArray

__all__ = ('Image', 'plot_rgb', 'gauss_image', 'moffat_image',
           'SpatialFrequencyLimits')


class Image(ArithmeticMixin, DataArray):

    """Manage image, optionally including a variance and a bad pixel mask.

    Parameters
    ----------
    filename : str
        Possible filename (.fits, .png or .bmp).
    ext : int or (int,int) or string or (string,string)
        Number/name of the data extension or numbers/names
        of the data and variance extensions.
    wcs : `mpdaf.obj.WCS`
        World coordinates.
    unit : str or `astropy.units.Unit`
        The physical units of the data values. Defaults to
        `astropy.units.dimensionless_unscaled`.
    data : float array
        Array containing the pixel values of the image.  None by default.
    var : float array
        Array containing the variance. None by default.
    copy : bool
        If true (default), then the data and variance arrays are copied.
    dtype : numpy.dtype
        Type of the data (int, float)

    Attributes
    ----------
    filename : str
        Possible FITS filename.
    primary_header : `astropy.io.fits.Header`
        FITS primary header instance.
    data_header : `astropy.io.fits.Header`
        FITS data header instance.
    wcs : `mpdaf.obj.WCS`
        World coordinates.
    unit : `astropy.units.Unit`
        Physical units of the data values.
    dtype : numpy.dtype
        Type of the data (int, float)

    """

    _ndim_required = 2
    _has_wcs = True

    def __init__(self, filename=None, ext=None, wcs=None, data=None, var=None,
                 unit=u.dimensionless_unscaled, copy=True, dtype=None,
                 **kwargs):
        self._spflims = None

        super(Image, self).__init__(
            filename=filename, ext=ext, wcs=wcs, unit=unit, data=data, var=var,
            copy=copy, dtype=dtype, **kwargs)

    def copy(self):
        """Return a new copy of an Image object."""
        obj = super(Image, self).copy()

        # Make a deep copy of the spatial-frequency limits.
        if self._spflims is not None:
            obj._spflims = self._spflims.deepcopy()
        return obj

    def get_step(self, unit=None):
        """Return the angular height and width of a pixel along the
        Y and X axes of the image array.

        In MPDAF, images are sampled on a regular grid of square
        pixels that represent a flat projection of the celestial
        sphere. The get_step() method returns the angular width and
        height of these pixels on the sky.

        See also get_axis_increments().

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The angular units of the returned values.

        Returns
        -------
        out : numpy.ndarray
           (dy,dx). These are the angular height and width of pixels
           along the Y and X axes of the image. The returned values are
           either in the unit specified by the 'unit' input parameter,
           or in the unit specified by the self.unit property.
        """

        if self.wcs is not None:
            return self.wcs.get_step(unit)

    def get_axis_increments(self, unit=None):
        """Return the displacements on the sky that result from
        incrementing the array indexes of the image by one along the Y
        and X axes, respectively.

        In MPDAF, images are sampled on a regular grid of square
        pixels that represent a flat projection of the celestial
        sphere. The get_axis_increments() method returns the angular
        width and height of these pixels on the sky, with signs that
        indicate whether the angle increases or decreases as one
        increments the array indexes. To keep plots consistent,
        regardless of the rotation angle of the image on the sky, the
        returned height is always positive, but the returned width is
        negative if a plot of the image with pixel 0,0 at the bottom
        left would place east anticlockwise of north, and positive
        otherwise.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The angular units of the returned values.

        Returns
        -------
        out : numpy.ndarray
           (dy,dx). These are the angular increments of pixels along
           the Y and X axes of the image. The returned values are
           either in the unit specified by the 'unit' input parameter,
           or in the unit specified by the self.unit property.

        """

        if self.wcs is not None:
            return self.wcs.get_axis_increments(unit)

    def get_range(self, unit=None):
        """Return the minimum and maximum right-ascensions and declinations
        in the image array.

        Specifically a list is returned with the following contents:

         [dec_min, ra_min, dec_max, ra_max]

        Note that if the Y axis of the image is not parallel to the
        declination axis, then the 4 returned values will all come
        from different corners of the image. In particular, note that
        this means that the coordinates [dec_min,ra_min] and
        [dec_max,ra_max] will only coincide with pixels in the image
        if the Y axis is aligned with the declination axis. Otherwise
        they will be outside the bounds of the image.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The units of the returned angles.

        Returns
        -------
        out : numpy.ndarray
           The range of right ascensions and declinations, arranged as
           [dec_min, ra_min, dec_max, ra_max]. The returned values are
           either in the units specified in the 'unit' input parameter,
           or in the units stored in the self.unit property.


        """
        if self.wcs is not None:
            return self.wcs.get_range(unit)

    def get_start(self, unit=None):
        """Return [y,x] corresponding to pixel (0,0).

        Parameters
        ----------
        unit : `astropy.units.Unit`
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
        unit : `astropy.units.Unit`
            type of the world coordinates

        Returns
        -------
        out : float array
        """
        if self.wcs is not None:
            return self.wcs.get_end(unit)

    def get_rot(self, unit=u.deg):
        """Return the rotation angle of the image, defined such that a
        rotation angle of zero aligns north along the positive Y axis,
        and a positive rotation angle rotates north away from the Y
        axis, in the sense of a rotation from north to east.

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

        if self.wcs is not None:
            return self.wcs.get_rot(unit)

    def mask_region(self, center, radius, unit_center=u.deg,
                    unit_radius=u.arcsec, inside=True, posangle=0.0):
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
        unit_center : `astropy.units.Unit`
            The units of the coordinates of the center argument
            (degrees by default).  If None, the units of the center
            argument are assumed to be pixels.
        unit_radius : `astropy.units.Unit`
            The units of the radius argument (arcseconds by default).
            If None, the units are assumed to be pixels.
        inside : bool
            If inside is True, pixels inside the region are masked.
            If inside is False, pixels outside the region are masked.
        posangle : float
            When the region is rectangular, this is the counter-clockwise
            rotation angle of the rectangle in degrees. When posangle is
            0.0 (the default), the X and Y axes of the ellipse are along
            the X and Y axes of the image.

        """
        center = np.array(center)

        # If the radius argument is a scalar value, this requests
        # that a circular region be masked. Delegate this to mask_ellipse().
        if np.isscalar(radius):
            return self.mask_ellipse(center=center, radius=radius,
                                     posangle=0.0, unit_center=unit_center,
                                     unit_radius=unit_radius, inside=inside)

        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]

        # Get the pixel sizes in the units of the radius argument.
        if unit_radius is None:
            step = np.array([1.0, 1.0])     # Pixel counts
        else:
            step = self.wcs.get_step(unit=unit_radius)

        # Treat rotated rectangles as polygons.
        if not np.isclose(posangle, 0.0):
            cos = np.cos(np.radians(posangle))
            sin = np.sin(np.radians(posangle))
            hw, hh = radius
            poly = np.array([[-hw * sin - hh * cos, -hw * cos + hh * sin],
                             [-hw * sin + hh * cos, -hw * cos - hh * sin],
                             [+hw * sin + hh * cos, +hw * cos - hh * sin],
                             [+hw * sin - hh * cos, +hw * cos + hh * sin]])
            return self.mask_polygon(poly / step + center, unit=None,
                                     inside=inside)

        # Get Y-axis and X-axis slice objects that bound the rectangular area.
        sy, sx = bounding_box(form="rectangle", center=center,
                              radii=radius, shape=self.shape, step=step)[0]

        if inside:
            self.data[sy, sx] = np.ma.masked
        else:
            self.data[0:sy.start, :] = np.ma.masked
            self.data[sy.stop:, :] = np.ma.masked
            self.data[sy, 0:sx.start] = np.ma.masked
            self.data[sy, sx.stop:] = np.ma.masked

    def mask_ellipse(self, center, radius, posangle, unit_center=u.deg,
                     unit_radius=u.arcsec, inside=True):
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
            The counter-clockwise rotation angle of the ellipse in
            degrees. When posangle is zero, the X and Y axes of the
            ellipse are along the X and Y axes of the image.
        unit_center : `astropy.units.Unit`
            The units of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : `astropy.units.Unit`
            The units of the radius argument. Arcseconds by default.
            (use None for radius in pixels)
        inside : bool
            If inside is True, pixels inside the described region are masked.
            If inside is False, pixels outside the described region are masked.

        """
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

        # Obtain Y and X axis slice objects that select the rectangular
        # region that just encloses the rotated ellipse.
        [sy, sx], _, center = bounding_box(
            form="ellipse", center=center, radii=radii,
            shape=self.shape, posangle=posangle, step=step)

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
            self.data[sy, sx][ksel < 1] = np.ma.masked
        else:
            self.data[0:sy.start, :] = np.ma.masked
            self.data[sy.stop:, :] = np.ma.masked
            self.data[sy, 0:sx.start] = np.ma.masked
            self.data[sy, sx.stop:] = np.ma.masked
            self.data[sy, sx][ksel > 1] = np.ma.masked

    def mask_polygon(self, poly, unit=u.deg, inside=True):
        """Mask values inside or outside a polygonal region.

        Parameters
        ----------
        poly : (float, float)
            An array of (float,float) containing a set of (p,q) or (dec,ra)
            values for the polygon vertices.
        unit : `astropy.units.Unit`
            The units of the polygon coordinates (by default in degrees).
            Use unit=None to have polygon coordinates in pixels.
        inside : bool
            If inside is True, pixels inside the polygonal region are masked.
            If inside is False, pixels outside the polygonal region are masked.

        """

        # Convert DEC,RA (deg) values coming from poly into Y,X value (pixels)
        if unit is not None:
            poly = np.array([
                [self.wcs.sky2pix((val[0], val[1]), unit=unit)[0][0],
                 self.wcs.sky2pix((val[0], val[1]), unit=unit)[0][1]]
                for val in poly])

        b = np.mgrid[:self.shape[0], :self.shape[1]].reshape(2, -1).T

        # Use a matplotlib method to create a polygon path and check if points
        # are within the polygon. The ouput is a boolean table.
        from matplotlib.path import Path
        polymask = Path(poly)
        c = polymask.contains_points(b)
        if not inside:
            c = ~c

        self._mask |= c.reshape(self.shape)
        return poly

    def truncate(self, y_min, y_max, x_min, x_max, mask=True, unit=u.deg,
                 inplace=False):
        """Return a sub-image that contains a specified area of the sky.

        The ranges x_min to x_max and y_min to y_max, specify a rectangular
        region of the sky in world coordinates. The truncate function returns
        the sub-image that just encloses this region. Note that if the world
        coordinate axes are not parallel to the array axes, the region will
        appear to be a rotated rectangle within the sub-image. In such cases,
        the corners of the sub-image will contain pixels that are outside the
        region. By default these pixels are masked. However this can be
        disabled by changing the optional mask argument to False.

        Parameters
        ----------
        y_min : float
            The minimum Y-axis world-coordinate of the selected
            region. The Y-axis is usually Declination, which may not
            be parallel to the Y-axis of the image array.
        y_max : float
            The maximum Y-axis world coordinate of the selected region.
        x_min : float
            The minimum X-axis world-coordinate of the selected
            region. The X-axis is usually Right Ascension, which may
            not be parallel to the X-axis of the image array.
        x_max : float
            The maximum X-axis world coordinate of the selected region.
        mask : bool
            If True, any pixels in the sub-image that remain outside the
            range x_min to x_max and y_min to y_max, will be masked.
        unit : `astropy.units.Unit`
            The units of the X and Y world-coordinates (degrees by default).
        inplace : bool
            If False, return a truncated copy of the image (the default).
            If True, truncate the original image in-place, and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """

        # Get the sky and pixel coordinates of the corners of the rectangular
        # region that is bounded by x_min..x_max and y_min..y_max.
        skycrd = np.array([[y_min, x_min],
                           [y_min, x_max],
                           [y_max, x_min],
                           [y_max, x_max]])

        if unit is not None:
            pixcrd = self.wcs.sky2pix(skycrd, unit=unit)
        else:
            pixcrd = skycrd

        # The sides of the selected region may not be parallel with the
        # array axes. Determine the pixel bounds of a rectangular
        # region of the array that contains the requested region.
        imin = max(0, int(np.min(pixcrd[:, 0]) + 0.5))
        imax = min(self.shape[0], int(np.max(pixcrd[:, 0]) + 0.5) + 1)
        jmin = max(0, int(np.min(pixcrd[:, 1]) + 0.5))
        jmax = min(self.shape[1], int(np.max(pixcrd[:, 1]) + 0.5) + 1)

        # Extract the rectangular area that contains the requested region.
        subima = self[imin:imax, jmin:jmax]
        if inplace:
            self._data = subima._data
            if self._var is not None:
                self._var = subima._var
            self._mask = subima._mask
            self.wcs = subima.wcs
            out = self
        else:
            out = subima.copy()

        # If the region is rotated relative to the image array axes
        # then the rectangular sub-image that contains this will has
        # some pixels outside this region. Should these be masked?
        if mask:
            # Get the indexes of all of the pixels in the "out" array,
            # ordered like: [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]...]
            pixcrd = np.mgrid[:out.shape[0], :out.shape[1]].reshape(2, -1).T

            if unit is None:
                skycrd = pixcrd
            else:
                skycrd = np.array(out.wcs.pix2sky(pixcrd, unit=unit))

            # Reshape the array of coordinates to have the shape of
            # the output array.
            x = skycrd[:, 1].reshape(out.shape)
            y = skycrd[:, 0].reshape(out.shape)

            # Test the X and Y coordinates of each pixel against the
            # requested range of X and Y coordinates, and mask pixels
            # that are outside this range.
            test_x = np.logical_or(x < x_min, x > x_max)
            test_y = np.logical_or(y < y_min, y > y_max)
            test = np.logical_or(test_x, test_y)
            out._mask = np.logical_or(out._mask, test)

            # Remove any array margins that are now completely masked.
            out.crop()

        return out

    def subimage(self, center, size, unit_center=u.deg, unit_size=u.arcsec,
                 minsize=2.0):
        """Return a view on a square or rectangular part.

        This method returns a square or rectangular sub-image whose center and
        size are specified in world coordinates.  Note that this is a view on
        the original map and that both will be modified at the same time. If
        you need to modify only the sub-image, copy() the result of the
        method.

        Parameters
        ----------
        center : (float,float)
            The center (dec, ra) of the square region. If this position
            is not within the parent image, None is returned.
        size : float or (float,float)
            The width of a square region, or the width and height of
            a rectangular region.
        unit_center : `astropy.units.Unit`
            The units of the center coordinates.
            Degrees are assumed by default. To specify the center
            in pixels, assign None to unit_center.
        unit_size : `astropy.units.Unit`
            The units of the size and minsize arguments.
            Arcseconds are assumed by default (use None to specify
            sizes in pixels).
        minsize : float
            The minimum width of the output image along both the Y and
            X axes. This function returns None if size is smaller than
            minsize, or if the part of the square that lies within the
            parent image is smaller than minsize along either axis.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """

        # If just one size is given, use it for both axes.
        if np.isscalar(size):
            size = np.array([size, size])
        else:
            size = np.asarray(size)
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError('Size must be positive')

#         # Require the center to be within the parent image.
#         if not self.inside(center, unit_center):
#             raise ValueError('The center must be within the image')

        # Convert the center position from world-coordinates to pixel indexes.
        center = np.asarray(center)
        if unit_center is not None:
            center = self.wcs.sky2pix(center, unit=unit_center)[0]

        # Get the pixel sizes in the units of the size argument.
        if unit_size is None:
            step = np.array([1.0, 1.0])     # Pixel counts
        else:
            step = self.wcs.get_step(unit=unit_size)

        # Convert the minimum size from world coordinates to pixel counts,
        # taking account of the possibility that pixels can be rectangular.
        minsize /= step

        # Convert the width and height of the region to radii, and
        # get Y-axis and X-axis slice objects that select this region.
        radius = size / 2.
        [sy, sx], [uy, ux], center = bounding_box(
            form="rectangle", center=center, radii=radius,
            shape=self.shape, step=step)

        if (sx.start >= self.shape[1] or
                sx.stop < 0 or
                sx.start == sx.stop or
                sy.start >= self.shape[0] or
                sy.stop < 0 or
                sy.start == sy.stop):
            raise ValueError('Sub-image boundaries are outside the cube: '
                             'center: {}, shape: {}, size: {}'
                             .format(center, self.shape, size))

        # Require that the image be at least minsize x minsize pixels.
        if (sy.stop - sy.start + 1) < minsize[0] or \
           (sx.stop - sx.start + 1) < minsize[1]:
            return None

        # Extract the requested part of the image.
        res = self[sy, sx]

        # If the image region was not clipped at the edges of the parent cube,
        # then return the subcube.
        if sy == uy and sx == ux:
            return res

        # Since the subimage is smaller than requested, due to clipping,
        # create new data and variance arrays of the required size.
        shape = (uy.stop - uy.start, ux.stop - ux.start)
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
        slices = (slice(sy.start - uy.start, sy.stop - uy.start),
                  slice(sx.start - ux.start, sx.stop - ux.start))

        # Copy the clipped subcube into unclipped arrays.
        data[slices] = res._data[:]
        if var is not None:
            var[slices] = res._var[:]
        if mask is not None:
            mask[slices] = res._mask[:]

        # Create a new WCS object for the unclipped subcube.
        wcs = res.wcs
        wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + slices[1].start)
        wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + slices[0].start)
        wcs.naxis1 = shape[1]
        wcs.naxis2 = shape[0]

        # Create the new unclipped sub-cube.
        return Image(wcs=wcs, unit=self.unit, copy=False,
                     data=data, var=var, mask=mask,
                     data_header=fits.Header(self.data_header),
                     primary_header=fits.Header(self.primary_header),
                     filename=self.filename)

    def _rotate(self, theta=0.0, interp='no', reshape=False, order=1,
                pivot=None, unit=u.deg, regrid=None, flux=False, cutoff=0.25):

        # In general it isn't possible to both anchor a point in the
        # image while reshaping the image so that it fits.
        if reshape and pivot is not None:
            raise ValueError("The pivot and reshape options can't be combined")

        # Turn off the sampling filter when orders of less than 2 are selected.
        prefilter = order > 1

        # Convert the rotation angle to radians.
        angle = (theta * unit).to(u.rad).value

        # Get the current rotation angle of the image in radians.
        oldrot = self.wcs.get_rot(unit=u.rad)

        # Where should north end up after we have rotated the image?
        newrot = oldrot + angle

        # Get the current pixel size.
        oldinc = self.wcs.get_axis_increments()

        # If no value has been specified for the regrid option, regrid
        # unless asked not to reshape the array.
        if regrid is None:
            regrid = reshape

        # Have we been asked to adjust pixel dimensions to avoid undersampling
        # and oversampling?
        if regrid:

            # Determine the spatial frequencies that need to be sampled
            # along the rotated Y and X axes.
            newfmax = self.get_spatial_fmax(np.rad2deg(newrot))

            # Calculate the pixel increments along the X and Y axes
            # that will be needed to adequately sample these
            # frequencies.
            newinc = 0.5 / newfmax * np.sign(oldinc)

        else:
            # Keep pixel sizes fixed?
            newinc = oldinc

        # Get the coordinate reference pixel of the input image,
        # arranged as a column vector in python (Y,X) order. Note that
        # crpix contains FITS pixel indexes which are 1 greater than
        # the corresponding python pixel indexes.
        oldcrpix = np.array([[self.wcs.get_crpix2() - 1],
                             [self.wcs.get_crpix1() - 1]])

        # Create a similar matrix that would scale a column vector in
        # (X,Y) axis order by the rotated X-axis and Y-axis pixel
        # increments.
        newscale = np.array([[newinc[1], 0.0],
                             [0.0, newinc[0]]])

        # Get the current WCS coordinate transformation matrix (which
        # transforms pixel coordinates to intermediate sky
        # coordinates).
        oldcd = self.wcs.get_cd()

        # Create a rotation matrix that multiplies the sky by the
        # above angle.
        sinq = np.sin(newrot)
        cosq = np.cos(newrot)
        sky_mrot = np.array([[cosq, -sinq],
                             [sinq, cosq]])

        # Compute the coordinate transformation matrix that will
        # pertain to the output image. We can interpolate to any grid,
        # so take the opportunity to zero any shear terms that were in
        # the original CD matrix, and just create a CD matrix that
        # rotates and scales the sky.
        newcd = np.dot(sky_mrot, newscale)

        # To fill the pixels of the output image we need a coordinate
        # transformation matrix to transform pixel indexes of the
        # rotated image back to pixel indexes of the input image. To
        # do this, we apply the new CD matrix to convert the rotated
        # indexes to intermediate sky coordinates, then apply the
        # inverse of the old CD matrix, to convert these back to
        # indexes of the original image.
        wcs_remap = np.dot(np.linalg.inv(oldcd), newcd)

        # The above matrix was computed from the WCS CD matrix, which
        # is designed to multiply a column vector in FORTRAN (X,Y)
        # axis order. Rearrange it to the equivalent matrix for
        # multiplying a column vector in python (Y,X) axis order.
        new2old = np.array([[wcs_remap[1, 1], wcs_remap[1, 0]],
                            [wcs_remap[0, 1], wcs_remap[0, 0]]])

        # Also compute the inverse of this, so that we can convert
        # from input image indexes to rotated image indexes.
        old2new = np.linalg.inv(new2old)

        # Have we been asked to reshape the image array to just encompass
        # the rotated image?
        if reshape:

            # Determine where the corners of the input image end up in the
            # output image with CRPIX set to [0,0].
            corners = np.array(
                [[0, 0, self.shape[0] - 1, self.shape[0] - 1],  # Y indexes
                 [0, self.shape[1] - 1, 0, self.shape[1] - 1]],  # X indexes
                dtype=float)
            pix = np.dot(old2new, (corners - oldcrpix))

            # Get the ranges of indexes occupied by the input image in the
            # rotated image.
            ymin = min(pix[0, :])
            ymax = max(pix[0, :])
            xmin = min(pix[1, :])
            xmax = max(pix[1, :])

            # Calculate the indexes of the coordinate reference pixel of
            # the rotated image, such that pixel [xmin,ymin] is moved to
            # array index [0,0]. Use (Y,X) axis ordering.
            newcrpix = np.array([[-ymin], [-xmin]])

            # Calculate the dimensions of the output image in (Y,X) order.
            # The dimensions are ymax-ymin+1 rounded up, and xmax-xmin+1
            # rounded up.
            newdims = np.array([int(ymax - ymin + 1.5),
                                int(xmax - xmin + 1.5)])

        # If not asked to reshape the image array, keep the image
        # dimensions the same, and choose the reference pixel such
        # that the rotation appears to occur around a specified pixel,
        # or the central pixel of the image.
        else:
            newdims = np.asarray(self.shape)

            # If no pivot pixel has been specified, substitute the
            # central pixel of the input image.
            if pivot is None:
                pivot = np.asarray(self.shape, dtype=float) / 2.0
            else:
                pivot = np.asarray(pivot, dtype=float)

            # Convert the pivot indexes to a column vector.
            pivot = pivot[np.newaxis, :].T

            # If the new coordinate reference pixel were zero, where
            # would the pivot pixel end up if we rotated the image
            # around oldcrpix?
            pix = np.dot(old2new, (pivot - oldcrpix))

            # Calculate the indexes of the coordinate reference pixel of
            # the rotated image, such that pixel pix is moved to
            # pivot. Use (Y,X) axis ordering.
            newcrpix = pivot - pix

        # The affine_transform() function calculates the pixel index
        # of the input image that corresponds to a given pixel index
        # of the rotated image, as follows:
        #
        #  oldpixel = new2old * newpixel + offset
        #
        # The coordinate reference pixels of the rotated and input
        # images must refer to the same position on the sky, so:
        #
        #  oldcrpix = new2old * newcrpix + offset
        #
        # Thus the value of 'offset' has to be:
        #
        #  offset = oldcrpix - new2old * newcrpix

        offset = oldcrpix - np.dot(new2old, newcrpix)

        # Get a copy of the current image array with masked values filled.
        newdata = self._prepare_data(interp)

        # For each pixel of the rotated image, use the new2old affine
        # transformation matrix to determine where that pixel
        # originates in the input image, then interpolate a value from
        # the pixels of the input image surrounding that point.
        newdata = affine_transform(newdata, matrix=new2old,
                                   offset=offset.flatten(), cval=0.0,
                                   output_shape=newdims, output=float,
                                   order=order, prefilter=prefilter)

        # Zero the current data array and then fill its masked pixels
        # with floating point 1.0s, so that we can rotate this in the
        # the same way as the data to see where the masked areas end up.
        self._data[:, :] = 0.0
        newmask = np.ma.filled(self.data, 1.0)

        # Rotate the array of 1s that represent masked pixels, and fill
        # corners that weren't mapped from the input array with 1s, so
        # that we end up flagging them too.
        newmask = affine_transform(newmask, matrix=new2old,
                                   offset=offset.flatten(), cval=1.0,
                                   output_shape=newdims, output=float,
                                   order=order, prefilter=prefilter)

        # Create a new boolean mask in which all pixels that had an
        # integrated contribution of more than 'cutoff' originally
        # masked pixels are masked. Note that setting the cutoff to
        # the "obvious" value of zero results in lots of pixels being
        # masked that are far away from any masked pixels, due to
        # precision errors in the affine_transform() function.
        # Limit the minimum value of the cutoff to avoid this.
        newmask = np.greater(newmask, max(cutoff, 1e-6))

        # If the image has an associated array of variances, rotate it too.
        if self._var is not None:
            newvar = affine_transform(self._var, matrix=new2old,
                                      offset=offset.flatten(), cval=0.0,
                                      output_shape=newdims, output=float,
                                      order=order, prefilter=prefilter)
        else:
            newvar = None

        # Compute the number of old pixel areas per new pixel, if the
        # pixel dimensions have been changed.
        if regrid:
            n = newinc.prod() / oldinc.prod()

            # Scale the flux per pixel by the multiplicative increase in the
            # area of a pixel?
            if flux:

                # Scale the pixel fluxes by the increase in the area.
                newdata *= n

                # Each output pixel is an interpolation between the
                # nearest neighboring pixels, so the variance is unchanged
                # by resampling. Scaling the pixel values by n, however,
                # increases the variances by n**2.
                if newvar is not None:
                    newvar *= n**2

        # Install the rotated data array, mask and variances.
        self._data = newdata
        self._mask = newmask
        self._var = newvar

        # Install the new world-coordinate transformation matrix, along
        # with the new reference pixel.
        self.wcs.set_cd(newcd)
        self.wcs.naxis1 = newdims[1]
        self.wcs.naxis2 = newdims[0]

        # Record the new value of the coordinate reference pixel,
        # being careful to convert from python 0-relative pixel
        # indexes to FITS 1-relative pixel indexes.
        self.wcs.set_crpix1(newcrpix[1] + 1)
        self.wcs.set_crpix2(newcrpix[0] + 1)

        # If allowed to reshape the array, crop away any entirely
        # masked margins.
        if reshape:
            self.crop()

    def rotate(self, theta=0.0, interp='no', reshape=False, order=1,
               pivot=None, unit=u.deg, regrid=None, flux=False, cutoff=0.25,
               inplace=False):
        """Rotate the sky within an image in the sense of a rotation from
        north to east.

        For example if the image rotation angle that is currently
        returned by image.get_rot() is zero, image.rotate(10.0) will
        rotate the northward direction of the image 10 degrees
        eastward of where it was, and self.get_rot() will thereafter
        return 10.0.

        Uses `scipy.ndimage.affine_transform`.

        Parameters
        ----------
        theta : float
            The angle to rotate the image (degrees). Positive
            angles rotate features in the image in the sense of a
            rotation from north to east.
        interp : 'no' | 'linear' | 'spline'
            If 'no', replace masked data with the median value of the
            image. This is the default.
            If 'linear', replace masked values using a linear
            interpolation between neighboring values.
            if 'spline', replace masked values using a spline
            interpolation between neighboring values.
        reshape : bool
            If True, the size of the output image array is adjusted
            so that the input image is contained completely in the
            output. The default is False.
        order : int
            The order of the prefilter that is applied by the affine
            transform function. Prefiltering is not really needed for
            band-limited images, but this option is retained for
            backwards compatibility with an older version of the
            image.rotate method. In general orders > 1 tend to
            generate ringing at sharp edges, such as those of CCD
            saturation spikes, so this argument is best left with
            its default value of 1.
        pivot : float,float or None
            When the reshape option is True, or the pivot argument is
            None, the image is rotated around its center.
            Alternatively, when the reshape option is False, the pivot
            argument can be used to indicate which pixel index [y,x]
            the image will be rotated around. Integer pixel indexes
            specify the centers of pixels. Non-integer values can be
            used to indicate positions between pixel centers.

            On the sky, the rotation always occurs around the
            coordinate reference position of the observation. However
            the rotated sky is then mapped onto the pixel array of the
            image in such a way as to keep the sky position of the
            pivot pixel at the same place. This makes the image appear
            to rotate around that pixel.
        unit : `astropy.units.Unit`
            The angular units of the rotation angle, theta.
        regrid : bool
            When this option is True, the pixel sizes along each axis
            are adjusted to avoid undersampling or oversampling any
            direction in the original image that would otherwise be
            rotated onto a lower or higher resolution axis. This is
            particularly important for images whose pixels have
            different angular dimensions along the X and Y axes, but
            it can also be important for images with square pixels,
            because the diagonal of an image with square pixels has
            higher resolution than the axes of that image.

            If this option is left with its default value of None,
            then it is given the value of the reshape option.
        flux : bool
            This tells the function whether the pixel units of the
            image are flux densities (flux=True), such as
            erg/s/cm2/Hz, or whether they are per-steradian brightness
            units (flux=False), such as erg/s/cm2/Hz/steradian. It
            needs to know this when it changes the pixel size, because
            when pixel sizes change, resampled flux densities need to
            be corrected for the change in the area per pixel, where
            resampled brightnesses don't.
        cutoff : float
            Mask each output pixel where at least this fraction of the
            pixel was interpolated from dummy values given to masked
            input pixels.
        inplace : bool
            If False, return a rotated copy of the image (the default).
            If True, rotate the original image in-place, and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """

        res = self if inplace else self.copy()
        res._rotate(theta=theta, interp=interp, reshape=reshape, order=order,
                    pivot=pivot, unit=unit, regrid=regrid, flux=flux,
                    cutoff=cutoff)
        return res

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
        self._data *= norm
        if self._var is not None:
            self._var *= (norm * norm)

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
        return tab.mean(), tab.std()

    def peak_detection(self, nstruct, niter, threshold=None):
        """Return a list of peak locations.

        Parameters
        ----------
        nstruct : int
            Size of the structuring element used for the erosion.
        niter : int
            Number of iterations used for the erosion and the dilatation.
        threshold : float
            Threshold value. If None, it is initialized with background value.

        Returns
        -------
        out : np.array

        """
        if threshold is None:
            background, std = self.background()
            threshold = background + 10 * std

        def _struct(n):
            struct = np.zeros([n, n])
            for i in range(0, n):
                dist = abs(i - (n // 2))
                struct[i][dist: abs(n - dist)] = 1
            return struct

        selec = self.data > threshold
        selec.fill_value = False
        struct = _struct(nstruct)
        selec = ndi.binary_erosion(selec, structure=struct, iterations=niter)
        selec = ndi.binary_dilation(selec, structure=struct, iterations=niter)
        selec = ndi.binary_fill_holes(selec)
        structure = ndi.generate_binary_structure(2, 2)
        label = ndi.measurements.label(selec, structure)
        pos = ndi.measurements.center_of_mass(self.data, label[0],
                                              np.arange(label[1]) + 1)
        return np.array(pos)

    def peak(self, center=None, radius=0, unit_center=u.deg,
             unit_radius=u.arcsec, dpix=2, background=None, plot=False):
        """Find image peak location.

        Used `scipy.ndimage.measurements.maximum_position` and
        `scipy.ndimage.measurements.center_of_mass`.

        Parameters
        ----------
        center : (float,float)
            Center (y,x) of the explored region.
            If center is None, the full image is explored.
        radius : float or (float,float)
            Radius defined the explored region.
        unit_center : `astropy.units.Unit`
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : `astropy.units.Unit`
            Radius unit.
            Arcseconds by default (use None for radius in pixels)
        dpix : int
            Half size of the window (in pixels) to compute the center of
            gravity.
        background : float
            Background value. If None, it is computed.
        plot : bool
            If True, the peak center is overplotted on the image.

        Returns
        -------
        out : dict {'y', 'x', 'p', 'q', 'data'}
            Containing the peak position and the peak intensity.

        """
        if center is None or radius == 0:
            d = self.data
            imin = 0
            jmin = 0
        else:
            if is_number(radius):
                radius = (radius, radius)

            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            if unit_radius is not None:
                radius = radius / self.wcs.get_step(unit=unit_radius)

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
             unit_radius=u.arcsec):
        """Compute the fwhm.

        Parameters
        ----------
        center : (float,float)
            Center of the explored region.
            If center is None, the full image is explored.
        radius : float or (float,float)
            Radius defined the explored region.
        unit_center : `astropy.units.Unit`
            type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : `astropy.units.Unit`
            Radius unit.  Arcseconds by default (use None for radius in pixels)

        Returns
        -------
        out : array of float
            [fwhm_y,fwhm_x], returned in unit_radius (arcseconds by default).

        """
        if center is None or radius == 0:
            img = self
        else:
            size = ((radius * 2, radius * 2) if is_number(radius)
                    else (radius[0] * 2, radius[1] * 2))
            img = self.subimage(center, size, unit_center=unit_center,
                                unit_size=unit_radius)

        width = img.moments(unit=unit_radius)
        return width / 2 * gaussian_sigma_to_fwhm

    def ee(self, center=None, radius=0, unit_center=u.deg,
           unit_radius=u.arcsec, frac=False, cont=0):
        """Compute ensquared/encircled energy.

        Parameters
        ----------
        center : (float,float)
            Center of the explored region.
            If center is None, the full image is explored.
        radius : float or (float,float)
            Radius defined the explored region.
            If float, it defined a circular region (encircled energy).
            If (float,float), it defined a rectangular region (ensquared
            energy).
        unit_center : `astropy.units.Unit`
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : `astropy.units.Unit`
            Radius unit. Arcseconds by default (use None for radius in pixels)
        frac : bool
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
            if is_number(radius):
                circular = True
                radius2 = radius * radius
                radius = (radius, radius)
            else:
                circular = False

            if unit_center is not None:
                center = self.wcs.sky2pix(center, unit=unit_center)[0]
            if unit_radius is not None:
                radius = radius / self.wcs.get_step(unit=unit_radius)
                radius2 = radius[0] * radius[1]

            imin = max(0, center[0] - radius[0])
            imax = min(center[0] + radius[0] + 1, self.shape[0])
            jmin = max(0, center[1] - radius[1])
            jmax = min(center[1] + radius[1] + 1, self.shape[1])
            ima = self[imin:imax, jmin:jmax]

            if circular:
                xaxis = np.arange(ima.shape[0], dtype=float) \
                    - ima.shape[0] / 2.
                yaxis = np.arange(ima.shape[1], dtype=float) \
                    - ima.shape[1] / 2.
                gridx = np.empty(ima.shape, dtype=float)
                gridy = np.empty(ima.shape, dtype=float)
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
        unit_center : `astropy.units.Unit`
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_radius : `astropy.units.Unit`
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
            i = self.shape[0] // 2
            j = self.shape[1] // 2
        elif unit_center is None:
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
            step = np.mean(self.get_step(unit=unit_radius))
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
        unit : `astropy.units.Unit`
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        etot : float
            Total energy used to comute the ratio.
                      If etot is not set, it is computed from the full image.
        frac : float in ]0,1]
            Fraction of energy.
        cont : float
            continuum value
        unit_center : `astropy.units.Unit`
            Type of the center coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_size : `astropy.units.Unit`
            Size unit.  Arcseconds by default (use None for sier in pixels).

        Returns
        -------
        out : float array
        """
        if center is None:
            i = self.shape[0] // 2
            j = self.shape[1] // 2
        elif unit_center is None:
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
            `scipy.interpolate.griddata`), or if True: spline
            interpolation (uses `scipy.interpolate.bisplrep` and
            `scipy.interpolate.bisplev`).

        """
        if self.mask is np.ma.nomask:
            x, y = np.mgrid[:self.shape[0], :self.shape[1]].reshape(2, -1)
            data = self._data
        else:
            x, y = np.where(~self._mask)
            data = self._data[x, y]

        grid = np.array(grid)

        if spline:
            if self.var is not None:
                var = self.var.filled(np.inf)
                weight = 1 / np.sqrt(np.abs(var[x, y]))
            else:
                weight = None

            tck = interpolate.bisplrep(x, y, data, w=weight)
            res = interpolate.bisplev(grid[0], grid[1], tck)
            return res
        else:
            # FIXME - check if this is still needed :
            # scipy 0.9 griddata - interpolate.interp2d segfaults when there
            # are too many data points
            # f = interpolate.interp2d(x, y, data)
            res = interpolate.griddata((x, y), data, grid.T, method='linear')
            return res

    def _interp_data(self, spline=False):
        """Return data array with interpolated values for masked pixels.

        Parameters
        ----------
        spline : bool
            False: bilinear interpolation (it uses
            `scipy.interpolate.griddata`), True: spline interpolation (it
            uses `scipy.interpolate.bisplrep` and
            `scipy.interpolate.bisplev`).

        """
        if not self._mask.any():
            return self._data
        else:
            ksel = np.where(self._mask)
            data = self._data.__copy__()
            data[ksel] = self._interp(ksel, spline)
            return data

    def _prepare_data(self, interp='no'):
        """Return a copy of the data array in which masked values
        have been filled, either with the median value of the image,
        or by interpolating neighboring pixels.

        Parameters
        ----------
        interp : 'no' | 'linear' | 'spline'
            If 'no', replace masked data with the median image value.
            If 'linear', replace masked values using a linear
            interpolation between neighboring values.
            if 'spline', replace masked values using a spline
            interpolation between neighboring values.

        Returns
        -------
        out : numpy.ndarray
           A patched copy of the data array.

        """

        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))
        return data

    def moments(self, unit=u.arcsec):
        """Return [width_y, width_x] first moments of the 2D gaussian.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            Unit of the returned moments (arcseconds by default).
            If None, moments will be in pixels.

        Returns
        -------
        out : float array

        """
        total = np.abs(self.data).sum()
        P, Q = np.indices(self.data.shape)
        # python convention: reverse x,y numpy.indices
        p = np.argmax((Q * np.abs(self.data)).sum(axis=1) / total)
        q = np.argmax((P * np.abs(self.data)).sum(axis=0) / total)

        # FIXME: check from where does this formula comes. Should be equivalent
        # to scipy.stats.moment(..., moment=2) ??
        col = self.data[int(p), :]
        width_q = np.sqrt(np.abs((np.arange(col.size) - p) * col).sum() /
                          np.abs(col).sum())
        row = self.data[:, int(q)]
        width_p = np.sqrt(np.abs((np.arange(row.size) - q) * row).sum() /
                          np.abs(row).sum())
        mom = np.array([width_p, width_q])
        if unit is not None:
            mom *= self.wcs.get_step(unit=unit)
        return mom

    def _prepare_fit_parameters(self, pos_min, pos_max, weight=True,
                                center=None, unit_center=u.deg,
                                fwhm=None, unit_fwhm=u.arcsec):
        pmin, qmin = 0, 0
        pmax, qmax = self.shape

        if unit_center is None:
            if pos_min is not None:
                pmin, qmin = pos_min
            if pos_max is not None:
                pmax, qmax = pos_max
        else:
            if pos_min is not None:
                pmin, qmin = self.wcs.sky2pix(pos_min, unit=unit_center,
                                              nearest=True)[0]
            if pos_max is not None:
                pmax, qmax = self.wcs.sky2pix(pos_max, unit=unit_center,
                                              nearest=True)[0]
            if pmin > pmax:
                pmin, pmax = pmax, pmin
            if qmin > qmax:
                qmin, qmax = qmax, qmin

        pmin = int(max(0, pmin))
        qmin = int(max(0, qmin))
        pmax = int(pmax)
        qmax = int(qmax)
        ima = self[pmin:pmax, qmin:qmax]

        N = ima.data.count()
        if N == 0:
            raise ValueError('empty sub-image')
        data = ima.data.compressed()
        p, q = np.where(~ima._mask)

        # weight
        if ima.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(ima.var[p, q].filled(np.inf)))
        else:
            wght = np.ones(N)

        # initial gaussian peak position
        if center is None:
            imax = data.argmax()
            center = np.array([p[imax], q[imax]])
        elif unit_center is not None:
            center = ima.wcs.sky2pix(center, unit=unit_center)[0]
        else:
            center = np.array(center)
            center[0] -= pmin
            center[1] -= qmin

        # initial moment value
        if fwhm is None:
            width = ima.moments(unit=None)
            fwhm = width * gaussian_sigma_to_fwhm
        else:
            fwhm = np.asarray(fwhm) / self.wcs.get_step(unit=unit_fwhm)

        return ima, pmin, pmax, qmin, qmax, data, wght, p, q, center, fwhm

    def gauss_fit(self, pos_min=None, pos_max=None, center=None, flux=None,
                  fwhm=None, circular=False, cont=0, fit_back=True, rot=0,
                  peak=False, factor=1, weight=True, plot=False,
                  unit_center=u.deg, unit_fwhm=u.arcsec, maxiter=0,
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
            The unit is given by ``unit_fwhm`` (arcseconds by default).
        circular : bool
            True: circular gaussian, False: elliptical gaussian
        cont : float
            continuum value, 0 by default.
        fit_back : bool
            False: continuum value is fixed,
            True: continuum value is a fit parameter.
        rot : float
            Initial rotation in degree.
            If None, rotation is fixed to 0.
        peak : bool
            If true, flux contains a gaussian peak value.
        factor : int
            If factor<=1, gaussian value is computed in the center of each
            pixel. If factor>1, for each pixel, gaussian value is the sum of
            the gaussian values on the factor*factor pixels divided by the
            pixel area.
        weight : bool
            If weight is True, the weight is computed as the inverse of
            variance.
        unit_center : `astropy.units.Unit`
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_fwhm : `astropy.units.Unit`
            FWHM unit. Arcseconds by default (use None for radius in pixels)
        maxiter : int
            The maximum number of iterations during the sum of square
            minimization.
        plot : bool
            If True, the gaussian is plotted.
        verbose : bool
            If True, the Gaussian parameters are printed at the end of the
            method.
        full_output : bool
            True-zero to return a `mpdaf.obj.Gauss2D` object containing
            the gauss image.

        Returns
        -------
        out : `mpdaf.obj.Gauss2D`

        """
        ima, pmin, pmax, qmin, qmax, data, wght, p, q, center, fwhm = \
            self._prepare_fit_parameters(
                pos_min, pos_max, weight=weight,
                center=center, unit_center=unit_center,
                fwhm=fwhm, unit_fwhm=unit_fwhm)

        # initial gaussian integrated flux
        if flux is None:
            peak = ima._data[int(center[0]), int(center[1])] - cont
        elif peak is True:
            peak = flux - cont

        N = len(p)
        width = fwhm * gaussian_fwhm_to_sigma
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
            pixcrd = np.array(list(zip(fp, fq)))

            e_gauss_fit = lambda v, p, q, data, w: \
                w * (((gaussfit(v, p, q)).reshape(N, factor * factor).sum(1)
                      / factor / factor).T.ravel() - data)
            v, covar, info, mesg, success = \
                leastsq(e_gauss_fit, v0[:],
                        args=(pixcrd[:, 0], pixcrd[:, 1], data, wght),
                        maxfev=maxiter, full_output=1)
        else:
            e_gauss_fit = lambda v, p, q, data, w: \
                w * (gaussfit(v, p, q) - data)
            v, covar, info, mesg, success = \
                leastsq(e_gauss_fit, v0[:], args=(p, q, data, wght),
                        maxfev=maxiter, full_output=1)

        if success not in [1, 2, 3, 4]:
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
        p_fwhm = p_width * gaussian_sigma_to_fwhm
        q_fwhm = q_width * gaussian_sigma_to_fwhm
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
            err_p_fwhm = err_p_width * gaussian_sigma_to_fwhm
            err_q_fwhm = err_q_width * gaussian_sigma_to_fwhm
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

            err_center = np.array([err_p_peak, err_q_peak]) * \
                self.wcs.get_step(unit=unit_center)
        else:
            center = (p_peak, q_peak)
            err_center = (err_p_peak, err_q_peak)

        step = self.wcs.get_step(unit=unit_fwhm)
        fwhm = np.array([p_fwhm, q_fwhm]) * step
        err_fwhm = np.array([err_p_fwhm, err_q_fwhm]) * step

        gauss = Gauss2D(center, flux, fwhm, cont, rot, peak, err_center,
                        err_flux, err_fwhm, err_cont, err_rot, err_peak)

        if verbose:
            gauss.print_param()
        if full_output:
            ima = gauss_image(shape=self.shape, wcs=self.wcs, gauss=gauss,
                              unit_center=unit_center, unit_fwhm=unit_fwhm)
            gauss.ima = ima
        return gauss

    def moffat_fit(self, pos_min=None, pos_max=None, center=None, fwhm=None,
                   flux=None, n=2.0, circular=False, cont=0, fit_back=True,
                   rot=0, peak=False, factor=1, weight=True, plot=False,
                   unit_center=u.deg, unit_fwhm=u.arcsec,
                   verbose=True, full_output=0, fit_n=True, maxiter=0):
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
        circular : bool
            True: circular moffat, False: elliptical moffat
        cont : float
            continuum value, 0 by default.
        fit_back : bool
            False: continuum value is fixed,
            True: continuum value is a fit parameter.
        rot : float
            Initial angle position in degree.
        peak : bool
            If true, flux contains a gaussian peak value.
        factor : int
            If factor<=1, gaussian is computed in the center of each pixel.
            If factor>1, for each pixel, gaussian value is the sum of the
            gaussian values on the factor*factor pixels divided by the pixel
            area.
        weight : bool
            If weight is True, the weight is computed as the inverse of
            variance.
        plot : bool
            If True, the gaussian is plotted.
        unit_center : `astropy.units.Unit`
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_fwhm : `astropy.units.Unit`
            FWHM unit. Arcseconds by default (use None for radius in pixels)
        full_output : bool
            True to return a `mpdaf.obj.Moffat2D` object containing the
            moffat image.
        fit_n : bool
            False: n value is fixed,
            True: n value is a fit parameter.
        maxiter : int
            The maximum number of iterations during the sum of square
            minimization.

        Returns
        -------
        out : `mpdaf.obj.Moffat2D`

        """
        ima, pmin, pmax, qmin, qmax, data, wght, p, q, center, fwhm = \
            self._prepare_fit_parameters(
                pos_min, pos_max, weight=weight,
                center=center, unit_center=unit_center,
                fwhm=fwhm, unit_fwhm=unit_fwhm)

        N = len(p)
        a = fwhm[0] / (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        e = fwhm[0] / fwhm[1]

        # initial gaussian integrated flux
        if flux is None:
            I = ima.data.data[int(center[0]), int(center[1])] - cont
        elif peak is True:
            I = flux - cont
        else:
            I = flux * (n - 1) / (np.pi * a * a * e)

        def moffat(c, x, y, amplitude, x_0, y_0, alpha, beta, e):
            """Two dimensional Moffat model function"""
            rr_gg = (((x - x_0) / alpha) ** 2 + ((y - y_0) / alpha / e) ** 2)
            return c + amplitude * (1 + rr_gg) ** (-beta)

        # def ellpt_moffat(c, x, y, amplitude, x_0, y_0, alpha, beta, e, theta):
        #     """Two dimensional elliptical Moffat model function"""
        #     cost = np.cos(theta)
        #     sint = np.sin(theta)
        #     xdiff = x - x_0
        #     ydiff = y - y_0
        #     rr_gg = (((xdiff * cost - ydiff * sint) / alpha) ** 2 +
        #              ((xdiff * sint + ydiff * cost) / alpha / e) ** 2)
        #     return c + amplitude * (1 + rr_gg) ** (-beta)

        if circular:
            rot = None
            if not fit_back:
                if fit_n:
                    moffatfit = lambda v, p, q: moffat(
                        cont, p, q, v[0], v[1], v[2], v[3], v[4], 1)
                    v0 = [I, center[0], center[1], a, n]
                else:
                    moffatfit = lambda v, p, q: moffat(
                        cont, p, q, v[0], v[1], v[2], v[3], n, 1)
                    v0 = [I, center[0], center[1], a]
            else:
                # 2d moffat function
                if fit_n:
                    moffatfit = lambda v, p, q: moffat(
                        v[5], p, q, v[0], v[1], v[2], v[3], v[4], 1)
                    v0 = [I, center[0], center[1], a, n, cont]
                else:
                    moffatfit = lambda v, p, q: moffat(
                        v[4], p, q, v[0], v[1], v[2], v[3], n, 1)
                    v0 = [I, center[0], center[1], a, cont]
        else:
            if not fit_back:
                if rot is None:
                    if fit_n:
                        moffatfit = lambda v, p, q: moffat(
                            cont, p, q, v[0], v[1], v[2], v[3], v[4], v[5])
                        v0 = [I, center[0], center[1], a, n, e]
                    else:
                        moffatfit = lambda v, p, q: moffat(
                            cont, p, q, v[0], v[1], v[2], v[3], n, v[5])
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
                        moffatfit = lambda v, p, q: moffat(
                            v[6], p, q, v[0], v[1], v[2], v[3], v[4], v[5])
                        v0 = [I, center[0], center[1], a, n, e, cont]
                    else:
                        moffatfit = lambda v, p, q: moffat(
                            v[5], p, q, v[0], v[1], v[2], v[3], n, v[4])
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
            pixcrd = np.array(list(zip(fp, fq)))

            e_moffat_fit = lambda v, p, q, data, w: \
                w * (((moffatfit(v, p, q)).reshape(N, factor * factor).sum(1)
                      / factor / factor).T.ravel() - data)
            v, covar, info, mesg, success = \
                leastsq(e_moffat_fit, v0[:], args=(pixcrd[:, 0], pixcrd[:, 1],
                                                   data, wght),
                        maxfev=maxiter, full_output=1)
        else:
            e_moffat_fit = lambda v, p, q, data, w: \
                w * (moffatfit(v, p, q) - data)
            v, covar, info, mesg, success = \
                leastsq(e_moffat_fit, v0[:],
                        args=(p, q, data, wght),
                        maxfev=maxiter, full_output=1)

        if success not in [1, 2, 3, 4]:
            self._logger.warning(mesg)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i])) *
                            np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = np.zeros_like(v)
            err[:] = np.abs(v[:] - v0[:])

        # center in pixel in the input image
        v[1] += int(pmin)
        v[2] += int(qmin)

        if plot:
            pp = np.arange(pmin, pmax, float(pmax - pmin) / 100)
            qq = np.arange(qmin, qmax, float(qmax - qmin) / 100)
            ff = np.empty((np.shape(pp)[0], np.shape(qq)[0]))
            for i in range(np.shape(pp)[0]):
                ff[i, :] = moffatfit(v, pp[i], qq[:])
            self._ax.contour(qq, pp, ff, 5)

        # Moffat2D object in pixels
        I, p_peak, q_peak = v[:3]
        a = np.abs(v[3])
        v = list(v[4:])

        # v0 = [I, center[0], center[1], a, n, e, rot, cont]
        if fit_back:
            # If present, cont is always the last parameter
            cont = v.pop()

        if fit_n:
            n = v.pop(0)

        _fwhm = a * (2 * np.sqrt(2 ** (1.0 / n) - 1.0))

        if circular:
            rot = 0
            fwhm = (_fwhm, _fwhm)
        else:
            e = v.pop(0)
            if e < 1:
                fwhm = (_fwhm, _fwhm * e)
            else:
                fwhm = (_fwhm * e, _fwhm)
            if rot is None:
                rot = 0
            else:
                if e < 1:
                    rot = (v[0] * 180.0 / np.pi) % 180
                else:
                    rot = (v[0] * 180.0 / np.pi + 90) % 180

        flux = I / (n - 1) * (np.pi * a * a * e)

        if err is not None:
            err_I, err_p_peak, err_q_peak = err[:3]
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
                    else:
                        err_cont = 0
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
                    else:
                        err_cont = 0
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
            err_center = np.array([err_p_peak, err_q_peak]) * \
                self.wcs.get_step(unit=unit_center)

        fwhm = np.array(fwhm)

        if unit_fwhm is not None:
            step0 = self.wcs.get_step(unit=unit_fwhm)[0]
            a = a * step0
            err_a = err_a * step0
            fwhm = fwhm * step0
            err_fwhm = err_fwhm * step0

        result = Moffat2D(center, flux, fwhm, cont, n,
                          rot, I, err_center, err_flux, err_fwhm,
                          err_cont, err_n, err_rot, err_I)

        if verbose:
            result.print_param()
        if full_output:
            ima = moffat_image(shape=self.shape, wcs=self.wcs, moffat=result,
                               unit_center=unit_center, unit_fwhm=unit_fwhm)
            result.ima = ima
        return result

    def rebin(self, factor, margin='center', inplace=False):
        """Combine neighboring pixels to reduce the size of an image by
        integer factors along each axis.

        Each output pixel is the mean of n pixels, where n is the
        product of the reduction factors in the factor argument.

        Parameters
        ----------
        factor : int or (int,int)
            The integer reduction factor along the y and x array axes.
            Note the conventional python ordering of the axes.
        margin : 'center'|'right'|'left'|'origin'
            When the dimensions of the input image are not integer
            multiples of the reduction factor, the image is truncated
            to remove just enough pixels that its dimensions are
            multiples of the reduction factor. This subimage is then
            rebinned in place of the original image. The margin
            parameter determines which pixels of the input image are
            truncated, and which remain.

            The options are:
              'origin' or 'center':
                 The starts of the axes of the output image are
                 coincident with the starts of the axes of the input
                 image.
              'center':
                 The center of the output image is aligned with the
                 center of the input image, within one pixel along
                 each axis.
              'right':
                 The ends of the axes of the output image are
                 coincident with the ends of the axes of the input
                 image.
        inplace : bool
            If False, return a rebinned copy of the image (the default).
            If True, rebin the original image in-place, and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """

        # Delegate the rebinning to the generic DataArray function.
        res = self._rebin(factor, margin, inplace)

        # If the spatial frequency band-limits of the image have been
        # reduced by the changes in the Y and X sampling intervals,
        # record this.
        res.update_spatial_fmax(0.5 / res.wcs.get_step())

        return res

    def resample(self, newdim, newstart, newstep, flux=False,
                 order=1, interp='no', unit_start=u.deg, unit_step=u.arcsec,
                 antialias=True, inplace=False, window="blackman"):
        """Resample an image of the sky to select its angular resolution and
        to specify which sky position appears at the center of pixel [0,0].

        This function is a simplified interface to the `mpdaf.obj.Image.regrid`
        function, which it calls with the following arguments::

            regrid(newdim, newstart, [0.0, 0.0],
                   [abs(newstep[0]),-abs(newstep[1])]
                   flux=flux, order=order, interp=interp, unit_pos=unit_start,
                   unit_inc=unit_step, inplace=inplace)

        When this function is used to resample an image to a lower
        resolution, a low-pass anti-aliasing filter is applied to the
        image before it is resampled, to remove all spatial frequencies
        below half the new sampling rate. This is required to satisfy
        the Nyquist sampling constraint. It prevents high
        spatial-frequency noise and edges from being folded into lower
        frequency artefacts in the resampled image. The removal of
        this noise improves the signal to noise ratio of the resampled
        image.

        Parameters
        ----------
        newdim : int or (int,int)
            The desired new dimensions. Python notation: (ny,nx)
        newstart : float or (float, float)
            The sky position (dec,ra) that should appear at the center
            of pixel [0,0].

            If None, the value of self.get_start() is substituted,
            so that the sky position that appears at the center of pixel
            [0,0] is unchanged by the resampling operation.
        newstep : float or (float, float)
            The desired angular size of the image pixels on the sky.
            The size is expressed as either one number to request
            square pixels on the sky with that width and height, or
            two numbers that specify the height and width of
            rectangular pixels on the sky. In the latter case, the two
            numbers are the size along the Y axis of the image array
            followed by the size along the X axis.
        flux : bool
            This tells the function whether the pixel units of the
            image are flux densities (flux=True), such as
            erg/s/cm2/Hz, or whether they are per-steradian brightness
            units (flux=False), such as erg/s/cm2/Hz/steradian. It
            needs to know this when it changes the pixel size, because
            when pixel sizes change, resampled flux densities need to
            be corrected for the change in the area per pixel, where
            resampled brightnesses don't.
        order : int
            The order of the spline interpolation. This can take any
            value from 0-5. The default is 1 (linear interpolation).
            When this function is used to lower the resolution of
            an image, the low-pass anti-aliasing filter that is applied,
            makes linear interpolation sufficient.
            Conversely, when this function is used to increase the
            image resolution, order=3 might be useful. Higher
            orders than this will tend to introduce ringing artefacts.
        interp : 'no' | 'linear' | 'spline'
            If 'no', replace masked data with the median image value.
            If 'linear', replace masked values using a linear
            interpolation between neighboring values.
            if 'spline', replace masked values using a spline
            interpolation between neighboring values.
        unit_start : `astropy.units.Unit`
            The angular units of the newstart coordinates. Degrees by default.
        unit_step : `astropy.units.Unit`
            The angular units of the step argument. Arcseconds by default.
        antialias : bool
            By default, when the resolution of an image axis is about
            to be reduced, a low pass filter is first applied to suppress
            high spatial frequencies that can not be represented by the
            reduced sampling interval. If this is not done, high-frequency
            noise and sharp edges get folded back to lower frequencies,
            where they increase the noise level of the image and introduce
            ringing artefacts next to sharp edges, such as CCD saturation
            spikes. This filtering can be disabled by passing False to
            the antialias argument.
        inplace : bool
            If False, return a rotated copy of the image (the default).
            If True, rotate the original image in-place, and return that.
        window : str
            The type of window function to use for antialiasing
            in the Fourier plane. The following windows are supported:

            blackman
               This window suppresses ringing better than any other
               window, at the expense of lowered image resolution. In
               the image plane, the PSF of this window is
               approximately gaussian, with a standard deviation of
               around 0.96*newstep, and a FWHM of about 2.3*newstep.

            gaussian
               A truncated gaussian window. This has a smaller PSF
               than the blackman window, however gaussians never fall
               to zero, so either significant ringing will be seen due
               to truncation of the gaussian, or low-level aliasing
               will occur, depending on the spatial frequency coverage
               of the image beyond the folding frequency. It can be a
               good choice for images that only contain smoothly
               varying features. It is equivalent to a convolution of
               the image with both an airy profile and a gaussian of
               standard deviation 0.724*newstep (FWHM 1.704*newstep).

            rectangle
               This window simply zeros all spatial frequencies above
               the highest that can be correctly sampled by the new
               pixel size.  This gives the best resolution of any of
               the windows, but this is marred by the strong sidelobes
               of the resulting airy-profile, especially near bright
               point sources and CCD saturation lines.

        Returns
        -------
        out : `~mpdaf.obj.Image`
           The resampled image.

        """
        # Convert newstep to the newinc argument used by regrid(), being
        # careful to preserve the signs of the existing coordinate increments.
        step_signs = np.sign(self.get_axis_increments())
        if is_number(newstep):
            newinc = step_signs * abs(newstep)
        else:
            newinc = step_signs * abs(np.asarray(newstep))

        # Convert newstart to the refpos,refpix arguments expected by regrid().
        refpix = None if newstart is None else [0.0, 0.0]

        return self.regrid(newdim, newstart, refpix, newinc, flux=flux,
                           order=order, interp=interp, unit_pos=unit_start,
                           unit_inc=unit_step, antialias=antialias,
                           inplace=inplace, window=window)

    def regrid(self, newdim, refpos, refpix, newinc, flux=False, order=1,
               interp='no', unit_pos=u.deg, unit_inc=u.arcsec, antialias=True,
               inplace=False, cutoff=0.25, window="blackman"):
        """Resample an image of the sky to select its angular resolution,
        to specify the position of the sky in the image array, and
        optionally to reflect one or more of its axes.

        This function can be used to decrease or increase the
        resolution of an image. It can also shift the contents of an
        image to place a specific (dec,ra) position at a specific
        fractional pixel position.  Finally, it can be used to invert
        the direction of one or both of the array axes on the sky.

        When this function is used to resample an image to a lower
        resolution, a low-pass anti-aliasing filter is applied to the
        image before it is resampled, to remove all spatial
        frequencies below half the new sampling rate. This is required
        to satisfy the Nyquist sampling constraint. It prevents high
        spatial-frequency noise and edges from being aliased to lower
        frequency artefacts in the resampled image. The removal of
        this noise improves the signal to noise ratio of the resampled
        image.

        Parameters
        ----------
        newdim : int or (int,int)
            The desired new dimensions. Python notation: (ny,nx)
        refpos : (float, float)
            The sky position (dec,ra) to place at the pixel specified
            by the refpix argument.

            If refpix and refpos are both None, the sky position at
            the bottom corner of the input image is placed at the
            bottom left corner of the output image. Note that refpix
            and refpos must either both be given values, or both
            be None.
        refpix : (float, float)
            The [Y, X] indexes of the output pixel where the sky
            position, refpos, should be placed. Y and X are
            interpreted as floating point indexes, where integer
            values indicate pixel centers and integer values +/- 0.5
            indicate the edges of pixels.

            If refpix and refpos are both None, the sky position at
            the bottom corner of the input image is placed at the
            bottom left corner of the output image. Note that refpix
            and refpos must either both be given values, or both
            be None.
        newinc : float or (float, float)
            The signed increments of the angle on the sky from one
            pixel to the next, given as either a single increment for
            both image axes, or two numbers (dy,dx) for the Y and X
            axes respectively.

            The signs of these increments are interpreted as described
            in the documentation of the Image.get_axis_increments()
            function. In particular, note that dy is typically
            positive and dx is usually negative, such that when the
            image is plotted, east appears anticlockwise of north, and
            east is towards the left of the plot when the image
            rotation angle is zero.

            If either of the signs of the two newinc numbers is
            different from the sign of the increments of the original
            image (queryable with image.get_axis_increments()), then
            the image will be reflected about that axis. In this case
            the value of the refpix argument should be chosen with
            care, because otherwise the sampled part of the image may
            end up being reflected outside the limits of the image
            array, and the result will be a blank image.

            If only one number is given for newinc then both axes
            are given the same resolution, but the signs of the
            increments are kept the same as the pixel increments
            of the original image.
        flux : bool
            This tells the function whether the pixel units of the
            image are flux densities (flux=True), such as
            erg/s/cm2/Hz, or whether they are per-steradian brightness
            units (flux=False), such as erg/s/cm2/Hz/steradian. It
            needs to know this when it changes the pixel size, because
            when pixel sizes change, resampled flux densities need to
            be corrected for the change in the area per pixel, where
            resampled brightnesses don't.
        order : int
            The order of the spline interpolation. This can take any
            value from 0-5. The default is 1 (linear interpolation).
            When this function is used to lower the resolution of
            an image, the low-pass anti-aliasing filter that is applied,
            makes linear interpolation sufficient.
            Conversely, when this function is used to increase the
            image resolution, order=3 might be useful. Higher
            orders than this will tend to introduce ringing artefacts.
        interp : 'no' | 'linear' | 'spline'
            If 'no', replace masked data with the median image value.
            If 'linear', replace masked values using a linear
            interpolation between neighboring values.
            if 'spline', replace masked values using a spline
            interpolation between neighboring values.
        unit_pos : `astropy.units.Unit`
            The units of the refpos coordinates.  Degrees by default.
        unit_inc : `astropy.units.Unit`
            The units of newinc.  Arcseconds by default.
        antialias : bool
            By default, when the resolution of an image axis is about
            to be reduced, a low pass filter is first applied to suppress
            high spatial frequencies that can not be represented by the
            reduced sampling interval. If this is not done, high-frequency
            noise and sharp edges get folded back to lower frequencies,
            where they increase the noise level of the image and introduce
            ringing artefacts next to sharp edges, such as CCD saturation
            spikes. This filtering can be disabled by passing False to
            the antialias argument.
        inplace : bool
            If False, return a resampled copy of the image (the default).
            If True, resample the original image in-place, and return that.
        cutoff : float
            Mask each output pixel where at least this fraction of the
            pixel was interpolated from dummy values given to masked
            input pixels.
        window : str
            The type of window function to use for antialiasing
            in the Fourier plane. The following windows are supported:

            blackman
               This window suppresses ringing better than any other
               window, at the expense of lowered image resolution. In
               the image plane, the PSF of this window is
               approximately gaussian, with a standard deviation of
               around 0.96*newstep, and a FWHM of about 2.3*newstep.

            gaussian
               A truncated gaussian window. This has a smaller PSF
               than the blackman window, however gaussians never fall
               to zero, so either significant ringing will be seen due
               to truncation of the gaussian, or low-level aliasing
               will occur, depending on the spatial frequency coverage
               of the image beyond the folding frequency. It can be a
               good choice for images that only contain smoothly
               varying features. It is equivalent to a convolution of
               the image with both an airy profile and a gaussian of
               standard deviation 0.724*newstep (FWHM 1.704*newstep).

            rectangle
               This window simply zeros all spatial frequencies above
               the highest that can be correctly sampled by the new
               pixel size.  This gives the best resolution of any of
               the windows, but this is marred by the strong sidelobes
               of the resulting airy-profile, especially near bright
               point sources and CCD saturation lines.

        Returns
        -------
        out : `~mpdaf.obj.Image`
            The resampled image is returned.

        """
        if is_int(newdim):
            newdim = (newdim, newdim)
        newdim = np.asarray(newdim, dtype=int)

        if refpos is None and refpix is None:
            # If neither refpos nor refpix have values, substitute values
            # that will place the current sky position of the bottom left
            # corner of the image at the bottom left corner of the output
            # image.

            # Use the pixel index of the bottom left corner of the image.
            refpix = np.array([-0.5, -0.5])
            refpos = self.wcs.pix2sky(refpix)
        elif refpos is not None and refpix is not None:
            # Were refpos and refpix both given values?

            # If necessary convert refpos to a numpy array and convert
            # it's units to the current WCS units.
            refpos = np.asarray(refpos, dtype=float)
            if unit_pos is not None:
                refpos = UnitArray(refpos, unit_pos, self.wcs.unit)

            # If necessary convert refpix to a floating point numpy array.
            refpix = np.asarray(refpix, dtype=float)
        else:
            # Complain if just one of refpos and refpix is None.
            raise ValueError('The refpos and refpix arguments should both be '
                             'None or both have values.')

        # Get the current index increments of the 2 axes.
        oldinc = self.wcs.get_axis_increments()

        # Use a common increment for both axes? If so, give them
        # the same size, but with signs matching the current
        # pixel increments.
        if is_number(newinc):
            size = abs(newinc)
            newinc = (size * np.sign(oldinc[0]), size * np.sign(oldinc[1]))

        # Ensure that newinc is an array of values that have the
        # same units as the WCS object.
        newinc = np.asarray(newinc, dtype=float)
        if unit_inc is not None:
            newinc = UnitArray(newinc, unit_inc, self.wcs.unit)

        # Get a copy of the data array with masked values filled.
        data = self._prepare_data(interp)

        # If the angular pixel increments along either axis are being
        # increased, then low-pass filter the data along that axis to
        # prevent aliasing in the resampled data.
        if antialias:
            data, newfmax = _antialias_filter_image(
                data, abs(oldinc), abs(newinc), self.get_spatial_fmax(),
                window)
        else:
            newfmax = 0.5 / abs(newinc)

        # For each pixel in the output image, the affine_transform
        # function calculates the index of the equivalent pixel in the
        # input image, and interpolates a value of the output pixel
        # from the surrounding pixels of the input image. It calculates
        # the input index from the output index as follows:
        #
        #   oldpixel = new2old * newpixel + offset
        #
        # where new2old is a 2x2 affine transform matrix designed to
        # multiply a column vector in axis order (Y,X). In our case
        # the matrix is:
        #
        #  new2old = |newinc[0]/oldinc[0],          0           |
        #            |          0          , newinc[1]/oldinc[0]|
        #
        # This scales an output index by newinc to calculate the
        # corresponding angular offset of that pixel from the origin
        # of the output array, then divides this by oldinc to compute
        # the equivalent index offset in the input array.
        new2old = np.array([[newinc[0] / oldinc[0], 0],
                            [0, newinc[1] / oldinc[1]]])

        # Also work out the inverse, so that we can convert from
        # pixels in the current image to the equivalent pixel of the
        # resampled image.
        old2new = np.linalg.inv(new2old)

        # We have been asked to locate sky position 'refpos' at pixel
        # 'refpix' of the resampled array. Reproducing the equation
        # from above:
        #
        #  oldpixel = new2old * newpixel + offset
        #
        # In this case oldpixel is the pixel index of the input array
        # where the sky position 'refpos' is currently located, and
        # newpixel is refpix.
        #
        #   sky2pix(refpos) = new2old * refpix + offset
        #
        # Thus the appropriate value for the offset parameter of
        # affine_transform() is:
        #
        #   offset = sky2pix(refpos) - new2old * refpix
        offset = (self.wcs.sky2pix(refpos).T[:, :1] -
                  np.dot(new2old, refpix[np.newaxis, :].T))

        # For each pixel of the output image, map its index to the
        # equivalent index of the input image and interpolate a value
        # for the new pixel from there.
        data = affine_transform(data, new2old, offset.flatten(),
                                output_shape=newdim, order=order,
                                prefilter=order >= 3)

        # Create a floating point version of the mask in which masked
        # elements are 1.0 and unmasked elements are 0.0.
        mask = self._mask.astype(float)

        # Resample the floating point version of the mask array.
        mask = affine_transform(mask, new2old, offset.flatten(), cval=1.0,
                                output_shape=newdim, output=float)

        # Create new boolean mask in which all pixels that had an
        # integrated contribution of more than 'cutoff' originally
        # masked pixels are masked. Note that setting the cutoff to
        # the "obvious" value of zero results in lots of pixels being
        # masked that are far away from any masked pixels, due to
        # precision errors in the affine_transform() function.
        # Limit the minimum value of the cutoff to avoid this.
        mask = np.greater(mask, max(cutoff, 1.0e-6))

        # Also repeat the procedure for the array of variances, if any.
        if self._var is not None:
            var = affine_transform(self._var, new2old, offset.flatten(),
                                   output_shape=newdim, order=order,
                                   prefilter=order >= 3)

        else:
            var = None

        # Compute the absolute changes in the size of the pixels
        # along the X and Y axes.
        xs = abs(newinc[1] / oldinc[1])
        ys = abs(newinc[0] / oldinc[0])

        # Compute the number of input pixels per output pixel.
        n = xs * ys

        # Scale the flux per pixel by the multiplicative increase in the
        # area of a pixel?

        if flux:
            # Scale the pixel fluxes by the increase in the area.
            data *= n

            # The variances of the output pixels depend on whether an
            # anti-aliasing filter was applied, as follows.
            #
            # 1. An anti-aliasing filter is applied before resampling
            #    when increasing the pixel size. This filter
            #    effectively averages together n neighboring
            #    pixels. The affine_transform() samples these averages
            #    when it interpolates the output pixel values, so the
            #    output pixels are effectively the average of n
            #    independent pixels of the input image. Multiplying
            #    these pixel values by n, then turns each output pixel
            #    value into the sum of n pixels.  The variance of a
            #    sum of n samples of variance v, is n*v.
            # 2. No anti-aliasing filter is applied when decreasing
            #    the pixel size, so in this case affine_transform()
            #    samples raw pixel values.  The variances of these
            #    output pixels are thus identical to those of the
            #    input pixels. If we then multiply this by n, then the
            #    variance of each output pixel is n**2 times the
            #    variance of the input pixels.
            # 3. If the pixel sizes along one axis are increased,
            #    while those of the other axis are decreased, then we
            #    have a mix of the above two cases.

            if var is not None:
                # Scale the variance according to the prescription described
                # above.
                var *= ((xs if xs > 1.0 and antialias else xs**2) *
                        (ys if ys > 1.0 and antialias else ys**2))

        # If we haven't been asked to scale the fluxes by the increase
        # in the area of a pixel, the effect on the variances are as
        # explained above, but without the flux scaling. If
        # anti-aliasing was applied to both axes, then each output
        # pixel is effectively the average of n input pixels, and the
        # variance of a mean of n samples of variance v is v/n. If no
        # anti-aliasing was applied, then there is no change to the
        # variance.

        else:
            if var is not None and (xs > 1.0 or ys > 1.0):
                var *= ((1 / xs if xs > 1.0 and antialias else 1.0) *
                        (1 / ys if ys > 1.0 and antialias else 1.0))

        # Get the coordinate reference pixel of the input image,
        # arranged as a column vector in python (Y,X) order. Note that
        # crpix contains FITS pixel indexes which are 1 greater than
        # the corresponding python pixel indexes.
        oldcrpix = np.array([[self.wcs.get_crpix2() - 1],
                             [self.wcs.get_crpix1() - 1]])

        # Compute the updated value of the coordinate reference pixel
        # in (Y,X) axis order.
        newcrpix = np.dot(old2new, (oldcrpix - offset))

        # Make a copy of the WCS object of the image to use as a template
        # for the WCS object of the resampled image.
        wcs = self.wcs.copy()

        # Install the new increments and image dimensions.
        wcs.set_axis_increments(newinc)
        wcs.naxis1 = newdim[1]
        wcs.naxis2 = newdim[0]

        # Record the new value of the coordinate reference pixel,
        # being careful to convert from python 0-relative pixel
        # indexes to FITS 1-relative pixel indexes.
        wcs.set_crpix1(newcrpix[1] + 1)
        wcs.set_crpix2(newcrpix[0] + 1)

        # Install the resampled data, mask and variance arrays, either
        # within self, or in a new Image object.
        out = self if inplace else self.clone()
        out._data = data
        out._mask = mask
        out._var = var
        out.wcs = wcs

        # If the spatial frequency band-limits of the image have been
        # reduced by the changes in the Y and X sampling intervals,
        # record this.
        out.update_spatial_fmax(newfmax)

        return out

    def align_with_image(self, other, flux=False, inplace=False, cutoff=0.25,
                         antialias=True, window="blackman"):
        """Resample the image to give it the same orientation, position,
        resolution and size as a given image.

        The image is first rotated to give it the same orientation on
        the sky as the other image. The resampling process also
        eliminates any shear terms from the original image, so that
        its pixels can be correctly drawn on a rectangular grid.

        Secondly the image is resampled. This changes its resolution,
        shifts the image such that the same points on the sky appear
        in the same pixels as in the other image, and changes the
        dimensions of the image array to match that of the other
        image.

        The rotation and resampling processes are performed as
        separate steps because the anti-aliasing filter that needs to
        be applied in the resampling step reduces the resolution, is
        difficult to implement before the axes have been rotated to
        the final orientation.

        Parameters
        ----------
        other : `~mpdaf.obj.Image`
            The image to be aligned with.
        flux : bool
            This tells the function whether the pixel units of the
            image are flux densities (flux=True), such as
            erg/s/cm2/Hz, or whether they are per-steradian brightness
            units (flux=False), such as erg/s/cm2/Hz/steradian. It
            needs to know this when it changes the pixel size, because
            when pixel sizes change, resampled flux densities need to
            be corrected for the change in the area per pixel, where
            resampled brightnesses don't.
        inplace : bool
            If False, return an aligned copy of the image (the default).
            If True, align the original image in-place, and return that.
        cutoff : float
            Mask each output pixel where at least this fraction of the
            pixel was interpolated from dummy values given to masked
            input pixels.
        antialias : bool
            By default, when the resolution of an image axis is about
            to be reduced, a low pass filter is first applied to suppress
            high spatial frequencies that can not be represented by the
            reduced sampling interval. If this is not done, high-frequency
            noise and sharp edges get folded back to lower frequencies,
            where they increase the noise level of the image and introduce
            ringing artefacts next to sharp edges, such as CCD saturation
            spikes and bright unresolved stars. This filtering can be
            disabled by passing False to the antialias argument.
        window : str
            The type of window function to use for antialiasing
            in the Fourier plane. The following windows are supported:

            blackman
               This window suppresses ringing better than any other
               window, at the expense of lowered image resolution. In
               the image plane, the PSF of this window is
               approximately gaussian, with a standard deviation of
               around 0.96*newstep, and a FWHM of about 2.3*newstep.

            gaussian
               A truncated gaussian window. This has a smaller PSF
               than the blackman window, however gaussians never fall
               to zero, so either significant ringing will be seen due
               to truncation of the gaussian, or low-level aliasing
               will occur, depending on the spatial frequency coverage
               of the image beyond the folding frequency. It can be a
               good choice for images that only contain smoothly
               varying features. It is equivalent to a convolution of
               the image with both an airy profile and a gaussian of
               standard deviation 0.724*newstep (FWHM 1.704*newstep).

            rectangle
               This window simply zeros all spatial frequencies above
               the highest that can be correctly sampled by the new
               pixel size.  This gives the best resolution of any of
               the windows, but this is marred by the strong sidelobes
               of the resulting airy-profile, especially near bright
               point sources and CCD saturation lines.

        """

        # Do nothing if the images are already aligned.
        if self.wcs.isEqual(other.wcs):
            return self if inplace else self.copy()

        # Determine the ranges of right-ascension and declination
        # covered by the target image grid plus an extra pixel at
        # each edge.
        pixsky = other.wcs.pix2sky([[-1, -1],
                                    [other.shape[0], -1],
                                    [-1, other.shape[1]],
                                    [other.shape[0], other.shape[1]]],
                                   unit=u.deg)
        dec_min, ra_min = pixsky.min(axis=0)
        dec_max, ra_max = pixsky.max(axis=0)

        # Truncate the input image to just enclose the above ranges of
        # right-ascension and declination.
        out = self.truncate(dec_min, dec_max, ra_min, ra_max, mask=False,
                            unit=u.deg, inplace=inplace)

        # Rotate the image to have the same orientation as the other
        # image. Note that the rotate function has a side effect of
        # correcting the image for shear terms in the CD matrix, so we
        # perform this step even if no rotation is otherwise needed.
        out._rotate(other.wcs.get_rot() - out.wcs.get_rot(), reshape=True,
                    regrid=True, flux=flux, cutoff=cutoff)

        # Get the pixel index and Dec,Ra coordinate at the center of
        # the image that we are aligning with.
        centerpix = np.asarray(other.shape) / 2.0
        centersky = other.wcs.pix2sky(centerpix)[0]

        # Re-sample the rotated image to have the same axis
        # increments, offset and number of pixels as the image that we
        # are aligning it with.
        out.regrid(other.shape, centersky, centerpix,
                   other.wcs.get_axis_increments(unit=u.deg),
                   flux, unit_inc=u.deg, inplace=True, cutoff=cutoff,
                   antialias=antialias, window=window)
        return out

    def estimate_coordinate_offset(self, ref, nsigma=1.0):
        """Given a reference image of the sky that is expected to
        overlap with the current image, attempt to fit for any offset
        between the sky coordinate system of the current image and
        that of the reference image. The returned value is designed to
        be added to the coordinate reference pixel values of self.wcs.

        This function performs the following steps:

        1. The align_with_image() method is called to resample the
           reference image onto the same coordinate grid as the
           current image.

        2. The two images are then cross-correlated, after zeroing all
           background values in the images below nsigma standard
           deviations above the mean.

        3. The peak in the auto-correlation image is found and its
           sub-pixel position is estimated by a simple quadratic
           interpolation. This position, relative to the center of the
           auto-correlation image, gives the average position offset
           between similar features in the two images.

        Parameters
        ----------
        ref : `~mpdaf.obj.Image`
            The image of the sky that is to be used as the coordinate
            reference. The sky coverage of this image should overlap
            with that of self. Ideally the resolution of this image
            should be at least as good as the resolution of self.
        nsigma : float
            Only values that exceed this many standard deviations
            above the mean of each image will be used.

        Returns
        -------
        out : float,float
            The pixel offsets that would need to be added to the
            coordinate reference pixel values, crpix2 and crpix1, of
            self.wcs to make the features in self line up with those
            in the reference image.

        """

        # Resample the reference sky image onto the same coordinate
        # grid as our image.
        ref = ref.align_with_image(self)

        # Before cross-correlating the images we need to make sure
        # that any areas that are masked in one image are also masked
        # in the other. Otherwise if one image has a very bright
        # source in an area that is masked in the other, then this
        # will produce false correlations.
        #
        # First get the union of the masked areas of the two images.
        mask = np.ma.mask_or(self._mask, ref._mask)

        # Place both image arrays into masked array containers that
        # share the above mask.
        sdata = np.ma.array(data=self._data, mask=mask)
        rdata = np.ma.array(data=ref._data, mask=mask)

        # Get copies of the above arrays with masked pixels filled
        # with the median values of the images.
        sdata = np.ma.filled(sdata, np.ma.median(sdata))
        rdata = np.ma.filled(rdata, np.ma.median(rdata))

        # When we cross-correlate the images, any constant or noisy
        # background will bias the result towards the origin of the
        # correlation, so remove most of the noisy background by
        # zeroing all values that are less than nsigma standard
        # deviations above the mean.
        mask = sdata < sdata.mean() + nsigma * sdata.std()
        sdata[mask] = 0
        mask = rdata < rdata.mean() + nsigma * rdata.std()
        rdata[mask] = 0

        # Sometimes a bright artefact or a bright star with
        # appreciable proper motion biases the correlation. To avoid
        # this take the log of the thresholded data to prevent very
        # bright features from dominating the correlation.
        sdata = np.log(1.0 + sdata)
        rdata = np.log(1.0 + rdata)

        # Cross correlate our image with the reference image, by
        # convolving our image with an axis-reversed version of the
        # reference image. Use mode="same" to only keep the inner half
        # of the array. We don't expect the peak to be outside this
        # area, and this avoids edge effects where there is incomplete
        # data.
        cc = signal.fftconvolve(sdata, rdata[::-1, ::-1], mode="same")

        # Find the position of the maximum value in the correlation image.
        py, px = np.unravel_index(np.argmax(cc), cc.shape)

        # Quadratically interpolate a more precise peak position from three
        # points along the X and Y axes, centered on the position found above.
        py2 = py - 1 + _find_quadratic_peak(cc[py - 1: py + 2, px])
        px2 = px - 1 + _find_quadratic_peak(cc[py, px - 1: px + 2])

        # Compute the offset of the peak relative to the central pixel
        # of the correlation image. This yields the offset between the
        # two images.
        dy = py2 - float(cc.shape[0] // 2)
        dx = px2 - float(cc.shape[1] // 2)

        return dy, dx

    def adjust_coordinates(self, ref, nsigma=1.0, inplace=False):
        """Given a reference image of the sky that is expected to
        overlap with the current image, attempt to fit for any offset
        between the sky coordinate system of the current image and
        that of the reference image. Apply this offset to the
        coordinates of the current image, to bring it into line with
        the reference image.

        This function calls self.estimate_coordinate_offset() to
        fit for the offset between the coordinate systems of the
        two images, then adjusts the coordinate reference pixel of
        the current image to bring its coordinates into line with
        those of the reference image.

        Parameters
        ----------
        ref : `~mpdaf.obj.Image`
            The image of the sky that is to be used as the coordinate
            reference. The sky coverage of this image should overlap
            with that of self. Ideally the resolution of this image
            should be at least as good as the resolution of self.
        nsigma : float
            Only values that exceed this many standard deviations
            above the mean of each image will be used.
        inplace : bool
            If False, return a shifted copy of the image (the default).
            If True, shift the original image in-place, and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`
            A version of self in which the sky coordinates have been
            shifted to match those of the reference image.

        """

        out = self if inplace else self.copy()

        # Determine the pixel offset of features in the current
        # image relative to features in the reference image.
        dy, dx = out.estimate_coordinate_offset(ref, nsigma)

        # Offset the WCS of the current image by the pixel shift found above.
        out.wcs.set_crpix1(out.wcs.get_crpix1() + dx)
        out.wcs.set_crpix2(out.wcs.get_crpix2() + dy)

        # Calculate the resulting shift in pixel coordinates, for display
        # to the user.
        units = u.arcsec if self.wcs.unit is u.deg else self.wcs.unit
        offset = np.array([-dy, -dx]) * self.wcs.get_axis_increments(units)
        self._logger.info("Shifted the coordinates by dy=%.3g dx=%.3g %s" %
                          (offset[0], offset[1], units))
        return out

    def gaussian_filter(self, sigma=3, interp='no', inplace=False):
        """Return an image containing Gaussian filter applied to the current
        image.

        Uses `scipy.ndimage.gaussian_filter`.

        Parameters
        ----------
        sigma : float
            Standard deviation for Gaussian kernel
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.
        inplace : bool
            If False, return a filtered copy of the image (the default).
            If True, filter the original image in-place, and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """

        out = self if inplace else self.copy()

        # Get a copy of the data array with masked values filled.
        data = out._prepare_data(interp)
        out._data = ndi.gaussian_filter(data, sigma)
        if out._var is not None:
            out._var = ndi.gaussian_filter(out._var, sigma)
        return out

    def segment(self, shape=(2, 2), minsize=20, minpts=None,
                background=20, interp='no', median=None):
        """Segment the image in a number of smaller images.

        Returns a list of images. Uses
        `scipy.ndimage.generate_binary_structure`,
        `scipy.ndimage.grey_dilation`, `scipy.ndimage.measurements.label`, and
        `scipy.ndimage.measurements.find_objects`.

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
            If not None (default), size of the window to apply a median filter
            on the image.

        Returns
        -------
        out : list of `Image`

        """
        data = self._prepare_data(interp)
        if median is not None:
            data = np.ma.array(ndi.median_filter(data, median),
                               mask=self._mask)
        expanded = ndi.grey_dilation(data, (minsize, minsize))
        expanded[expanded < background] = 0

        structure = ndi.generate_binary_structure(shape[0], shape[1])
        labels, nlabels = ndi.measurements.label(expanded, structure)
        slices = ndi.measurements.find_objects(labels)

        return [self[slices[i]] for i in range(nlabels)
                if minpts is None or len(data[labels == i + 1]) >= minpts]

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
        # Get a copy of the data array with masked values filled.
        data = self._prepare_data(interp)
        self._data = np.random.normal(data, sigma)

        if self._var is None:
            self._var = np.ones((self.shape)) * sigma * sigma
        else:
            self._var *= (sigma * sigma)

    def inside(self, coord, unit=u.deg):
        """Return True if coord is inside image.

        Parameters
        ----------
        coord : (float,float)
                coordinates (y,x).
        unit : `astropy.units.Unit`
                Type of the coordinates (degrees by default)

        Returns
        -------
        out : bool
        """
        if unit is not None:
            pixcrd = self.wcs.sky2pix([coord[0], coord[1]], unit=unit)[0]
        else:
            pixcrd = coord
        if (pixcrd >= -self.wcs.get_step(unit=unit) / 100).all() and \
                (pixcrd < self.shape + self.wcs.get_step(unit=unit) / 100).all():
            return True
        else:
            return False

    def convolve(self, other, inplace=False):
        """Convolve an Image with a 2D array or another Image, using the
        discrete convolution equation.

        This function, which uses the discrete convolution equation, is
        usually slower than Image.fftconvolve(). However it can be faster when
        other.data.size is small, and it always uses much less memory, so it
        is sometimes the only practical choice.

        Masked values in self.data and self.var are replaced with zeros before
        the convolution is performed, but they are masked again after the
        convolution.

        If self.var exists, the variances are propagated using the equation:

          result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived by
        applying the usual rules of error-propagation to the discrete
        convolution equation.

        The speed of this function scales as O(Nd x No) where
        Nd=self.data.size and No=other.data.size.

        Uses `scipy.signal.convolve`.

        Parameters
        ----------
        other : Image or np.ndarray
            The 2D array with which to convolve the image in self.data.
            This array can be an image of the same size as self, or it
            can be a smaller image, such as a small gaussian to use to
            smooth the larger image.

            When ``other`` contains a symmetric filtering function, such
            as a two-dimensional gaussian, the center of the function
            should be placed at the center of pixel:

             ``(other.shape - 1) // 2``

            If other is an MPDAF Image object, note that only its data
            array is used. Masked values in this array are treated
            as zero. Any variances found in other.var are ignored.
        inplace : bool
            If False (the default), return the results in a new Image.
            If True, record the result in self and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        # Delegate the task to DataArray._convolve()
        return self._convolve(signal.convolve, other=other, inplace=inplace)

    def fftconvolve(self, other, inplace=False):
        """Convolve an Image with a 2D array or another Image, using the
        Fourier convolution theorem.

        This function, which performs the convolution by multiplying the
        Fourier transforms of the two images, is usually much faster than
        Image.convolve(), except when other.data.size is small. However it
        uses much more memory, so Image.convolve() is sometimes a better
        choice.

        Masked values in self.data and self.var are replaced with zeros before
        the convolution is performed, but they are masked again after the
        convolution.

        If self.var exists, the variances are propagated using the equation:

          result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived by
        applying the usual rules of error-propagation to the discrete
        convolution equation.

        The speed of this function scales as O(Nd x log(Nd)) where
        Nd=self.data.size.  It temporarily allocates a pair of arrays that
        have the sum of the shapes of self.shape and other.shape, rounded up
        to a power of two along each axis. This can involve a lot of memory
        being allocated. For this reason, when other.shape is small,
        Image.convolve() may be more efficient than Image.fftconvolve().

        Uses `scipy.signal.fftconvolve`.

        Parameters
        ----------
        other : Image or np.ndarray
            The 2D array with which to convolve the image in self.data.  This
            array can be an image of the same size as self, or it can be a
            smaller image, such as a small 2D gaussian to use to smooth the
            larger image.

            When ``other`` contains a symmetric filtering function, such as a
            two-dimensional gaussian, the center of the function should be
            placed at the center of pixel:

             ``(other.shape - 1) // 2``

            If other is an MPDAF Image object, note that only its data array
            is used. Masked values in this array are treated as zero. Any
            variances found in other.var are ignored.
        inplace : bool
            If False (the default), return the results in a new Image.
            If True, record the result in self and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        # Delegate the task to DataArray._convolve()
        return self._convolve(signal.fftconvolve, other=other, inplace=inplace)

    def fftconvolve_gauss(self, center=None, flux=1., fwhm=(1., 1.),
                          peak=False, rot=0., factor=1, unit_center=u.deg,
                          unit_fwhm=u.arcsec, inplace=False):
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
        peak : bool
            If true, flux contains a gaussian peak value.
        rot : float
            Angle position in degree.
        factor : int
            If factor<=1, gaussian value is computed in the center of each
            pixel.  If factor>1, for each pixel, gaussian value is the sum of
            the gaussian values on the factor*factor pixels divided by the
            pixel area.
        unit_center : `astropy.units.Unit`
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_fwhm : `astropy.units.Unit`
            FWHM unit. Arcseconds by default (use None for radius in pixels)
        inplace : bool
            If False, return a convolved copy of the image (default value).
            If True, convolve the original image in-place, and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        ima = gauss_image(self.shape, wcs=self.wcs, center=center,
                          flux=flux, fwhm=fwhm, peak=peak, rot=rot,
                          factor=factor, gauss=None, unit_center=unit_center,
                          unit_fwhm=unit_fwhm, cont=0, unit=self.unit)

        # Normalize the total flux of the Gaussian.
        ima.norm(typ='sum')
        return self.fftconvolve(ima, inplace=inplace)

    def fftconvolve_moffat(self, center=None, flux=1., a=1.0, q=1.0,
                           n=2, peak=False, rot=0., factor=1,
                           unit_center=u.deg, unit_a=u.arcsec, inplace=False):
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
        peak : bool
            If true, flux contains a gaussian peak value.
        unit_center : `astropy.units.Unit`
            type of the center and position coordinates.
            Degrees by default (use None for coordinates in pixels).
        unit_a : `astropy.units.Unit`
            a unit. Arcseconds by default (use None for radius in pixels)
        inplace : bool
            If False, return a convolved copy of the image (default value).
            If True, convolve the original image in-place, and return that.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        fwhmy = a * (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        fwhmx = fwhmy / q
        ima = moffat_image(self.shape, wcs=self.wcs, factor=factor,
                           center=center, flux=flux, fwhm=(fwhmy, fwhmx), n=n,
                           rot=rot, peak=peak, unit_center=unit_center,
                           unit_fwhm=unit_a, unit=self.unit)
        ima.norm(typ='sum')
        return self.fftconvolve(ima, inplace=inplace)

    def correlate2d(self, other, interp='no'):
        """Return the cross-correlation of the image with an array/image

        Uses `scipy.signal.correlate2d`.

        Parameters
        ----------
        other : 2d-array or Image
            Second Image or 2d-array.
        interp : 'no' | 'linear' | 'spline'
            if 'no', data median value replaced masked values.
            if 'linear', linear interpolation of the masked values.
            if 'spline', spline interpolation of the masked values.

        """
        if not isinstance(other, DataArray):
            # Get a copy of the data array with masked values filled.
            data = self._prepare_data(interp)

            res = self.copy()
            res._data = signal.correlate2d(data, other, mode='same',
                                           boundary='symm')
            if res._var is not None:
                res._var = signal.correlate2d(res._var, other, mode='same',
                                              boundary='symm')
            return res
        elif other.ndim == 2:
            # Get copies of the data arrays with masked values filled.
            data = self._prepare_data(interp)
            other_data = other._prepare_data(interp)
            other_data = UnitMaskedArray(other_data, other.unit, self.unit)
            res = self.copy()
            res._data = signal.correlate2d(data, other_data, mode='same')

            if res._var is not None:
                res._var = signal.correlate2d(res._var, other_data,
                                              mode='same')
            return res
        else:
            raise IOError('Operation forbidden')

    def plot(self, title=None, scale='linear', vmin=None, vmax=None,
             zscale=False, colorbar=None, var=False, show_xlabel=False,
             show_ylabel=False, ax=None, unit=u.deg, use_wcs=False, **kwargs):
        """Plot the image with axes labeled in pixels.

        If either axis has just one pixel, plot a line instead of an image.

        Colors are assigned to each pixel value as follows. First each
        pixel value, ``pv``, is normalized over the range ``vmin`` to ``vmax``,
        to have a value ``nv``, that goes from 0 to 1, as follows::

            nv = (pv - vmin) / (vmax - vmin)

        This value is then mapped to another number between 0 and 1 which
        determines a position along the colorbar, and thus the color to give
        the displayed pixel. The mapping from normalized values to colorbar
        position, color, can be chosen using the scale argument, from the
        following options:

        - 'linear': ``color = nv``
        - 'log': ``color = log(1000 * nv + 1) / log(1000 + 1)``
        - 'sqrt': ``color = sqrt(nv)``
        - 'arcsinh': ``color = arcsinh(10*nv) / arcsinh(10.0)``

        A colorbar can optionally be drawn. If the colorbar argument is given
        the value 'h', then a colorbar is drawn horizontally, above the plot.
        If it is 'v', the colorbar is drawn vertically, to the right of the
        plot.

        By default the image is displayed in its own plot. Alternatively
        to make it a subplot of a larger figure, a suitable
        ``matplotlib.axes.Axes`` object can be passed via the ``ax`` argument.
        Note that unless matplotlib interative mode has previously been enabled
        by calling ``matplotlib.pyplot.ion()``, the plot window will not appear
        until the next time that ``matplotlib.pyplot.show()`` is called. So to
        arrange that a new window appears as soon as ``Image.plot()`` is
        called, do the following before the first call to ``Image.plot()``::

            import matplotlib.pyplot as plt
            plt.ion()

        Parameters
        ----------
        title : str
            An optional title for the figure (None by default).
        scale : 'linear' | 'log' | 'sqrt' | 'arcsinh'
            The stretch function to use mapping pixel values to
            colors (The default is 'linear'). The pixel values are
            first normalized to range from 0 for values <= vmin,
            to 1 for values >= vmax, then the stretch algorithm maps
            these normalized values, nv, to a position p from 0 to 1
            along the colorbar, as follows:
            linear:  p = nv
            log:     p = log(1000 * nv + 1) / log(1000 + 1)
            sqrt:    p = sqrt(nv)
            arcsinh: p = arcsinh(10*nv) / arcsinh(10.0)
        vmin : float
            Pixels that have values <= vmin are given the color
            at the dark end of the color bar. Pixel values between
            vmin and vmax are given colors along the colorbar according
            to the mapping algorithm specified by the scale argument.
        vmax : float
            Pixels that have values >= vmax are given the color
            at the bright end of the color bar. If None, vmax is
            set to the maximum pixel value in the image.
        zscale : bool
            If True, vmin and vmax are automatically computed
            using the IRAF zscale algorithm.
        colorbar : str
            If 'h', a horizontal colorbar is drawn above the image.
            If 'v', a vertical colorbar is drawn to the right of the image.
            If None (the default), no colorbar is drawn.
        var : bool
              If true variance array is shown in place of data array
        ax : matplotlib.axes.Axes
            An optional Axes instance in which to draw the image,
            or None to have one created using ``matplotlib.pyplot.gca()``.
        unit : `astropy.units.Unit`
            The units to use for displaying world coordinates
            (degrees by default). In the interactive plot, when
            the mouse pointer is over a pixel in the image the
            coordinates of the pixel are shown using these units,
            along with the pixel value.
        use_wcs : bool
            If True, use `astropy.visualization.wcsaxes` to get axes
            with world coordinates.
        kwargs : matplotlib.artist.Artist
            Optional extra keyword/value arguments to be passed to
            the ``ax.imshow()`` function.

        Returns
        -------
        out : matplotlib AxesImage

        """
        cax = None
        # Default X and Y axes are labeled in pixels.
        xlabel = 'q (pixel)'
        ylabel = 'p (pixel)'

        if ax is None:
            if use_wcs:
                ax = plt.subplot(projection=self.wcs.wcs)
                xlabel = 'ra'
                ylabel = 'dec'
            else:
                ax = plt.gca()
        elif use_wcs:
            self._logger.warning(
                'use_wcs does not work when giving also an axis (ax)')

        if var:
            data_plot = self.var
        else:
            data_plot = self.data

        # If either axis has just one pixel, plot it as a line-graph.
        if self.shape[1] == 1:
            # Plot a column as a line-graph
            yaxis = np.arange(self.shape[0], dtype=float)
            ax.plot(yaxis, data_plot)
            xlabel = 'p (pixel)'
            ylabel = self.unit
        elif self.shape[0] == 1:
            # Plot a row as a line-graph
            xaxis = np.arange(self.shape[1], dtype=float)
            ax.plot(xaxis, data_plot.T)
            xlabel = 'q (pixel)'
            ylabel = self.unit
        else:
            # Plot a 2D image.

            # get image normalization
            norm = get_plot_norm(data_plot, vmin=vmin, vmax=vmax,
                                 zscale=zscale, scale=scale)

            # Display the image.
            cax = ax.imshow(data_plot, interpolation='nearest',
                            origin='lower', norm=norm, **kwargs)

            # Create a colorbar
            import matplotlib.axes as maxes
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            if colorbar == "h":
                cax2 = divider.append_axes("top", size="5%", pad=0.2,
                                           axes_class=maxes.Axes)
                cbar = plt.colorbar(cax, cax=cax2, orientation='horizontal')
                for t in cbar.ax.xaxis.get_major_ticks():
                    t.tick1On = True
                    t.tick2On = True
                    t.label1On = False
                    t.label2On = True
            elif colorbar == "v":
                cax2 = divider.append_axes("right", size="5%", pad=0.05,
                                           axes_class=maxes.Axes)
                plt.colorbar(cax, cax=cax2)

            # Keep the axis to allow other functions to overplot
            # the image with contours etc.
            self._ax = ax

        # Label the axes if requested.
        if show_xlabel:
            ax.set_xlabel(xlabel)
        if show_ylabel:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        # Change the way that plt.show() displays coordinates when the pointer
        # is over the image, such that world coordinates are displayed with the
        # specified unit, and pixel values are displayed with their native
        # units.
        ax.format_coord = FormatCoord(self, data_plot)
        self._unit = unit
        return cax

    def get_spatial_fmax(self, rot=None):
        """Return the spatial-frequency band-limits of the image along
        the Y and X axes.

        See the documentation of set_spatial_fmax() for an explanation
        of what the band-limits are used for.

        If no band limits have been specified yet, this function has the
        side-effect of setting them to the band-limits dictated by the
        sampling interval of the image array. Specifically, an X axis
        with a sampling interval of dx can sample spatial frequencies of
        up to 0.5/dx cycles per unit of dx without aliasing.

        Parameters
        ----------
        rot : float or None
            Either None, to request band-limits that pertain to the
            Y and X axes of the current image without any rotation,
            or, if the band-limits pertain to a rotated version of
            the image, the rotation angle of its Y axis westward of north
            (degrees). This is defined such that if image.wcs.get_rot()
            is passed to this function, the band limits for the Y and
            X axes of the current image axes will be returned.

        Returns
        -------
        out : np.ndarray
            The spatial-frequency band-limits of the image along
            the Y and X axes of the image in cycles per self.wcs.unit.

        """

        # If no image angle was provided, get the current rotation angle.

        if rot is None:
            rot = self.wcs.get_rot()

        # If no band-limits have been specified, initialize them to the
        # limits currently dictated by the sampling intervals of the image.

        if self._spflims is None:
            self.set_spatial_fmax(0.5 / self.get_step(), self.wcs.get_rot())

        # Return the frequency limits that pertain to the specified
        # rotation angle.

        return self._spflims.get_fmax(rot)

    def update_spatial_fmax(self, newfmax, rot=None):
        """Update the spatial-frequency band-limits recorded for the
        current image.

        See the documentation of set_spatial_fmax() for an explanation
        of what the band-limits are used for.

        If either of the new limits is less than an existing
        band-limit, and the rotation angle of the new limits is
        the same as the angle of the recorded limits, then the smaller
        limits replace the originals.

        If either of the new limits is smaller than the existing
        limits, but the rotation angle for the new limits differs from
        the recorded limits, then both of the original limits are
        discarded and replaced by the new ones at the specified angle.

        Parameters
        ----------
        newfmax : numpy.ndarray
            The frequency limits along the Y and X axes, respectively,
            specified in cycles per the angular unit in self.wcs.unit.
        rot : float or None
            Either None, to specify band-limits that pertain to the Y
            and X axes of the current image without any rotation, or,
            if the band-limits pertain to a rotated version of the
            image, the rotation angle of its Y axis westward of north
            (degrees). This is defined such that if
            image.wcs.get_rot() is passed to this function, the
            band-limit newfmax[0] will be along the Y axis of the
            image and newfmax[1] will be along its X axis.

        """

        # If no image rotation angle was specified, assume the
        # current angle.

        if rot is None:
            rot = self.wcs.get_rot()

        # If no band-limits have been set yet, record the new limits.

        if self._spflims is None:
            self.set_spatial_fmax(newfmax, rot)
        else:

            # Get the existing spatial-frequency band limits at the
            # specified angle.

            oldfmax = self._spflims.get_fmax(rot)

            # Are either of the new limits smaller than the old ones?

            if np.any(newfmax < oldfmax):

                # If the rotation angle of the recorded limits is the
                # same as the rotation angle of the new limits, keep
                # existing axis limits that are smaller than the new
                # limits.

                if np.isclose(rot, self._spflims.rot):
                    newfmax = np.minimum(newfmax, oldfmax)

                # Record the new limits.

                self.set_spatial_fmax(newfmax, rot)

    def set_spatial_fmax(self, newfmax=None, rot=None):
        """Specify the spatial-frequency band-limits of the image along
        the Y and X axis. This function completely replaces any existing
        band-limits. See also update_spatial_fmax().

        The recorded limits are used to avoid redundantly performing
        anti-aliasing measures such as low-pass filtering an image
        before resampling to a lower resolution, or decreasing pixel
        sizes before rotating high resolution axes onto low resolution
        axes.

        Parameters
        ----------
        newfmax : numpy.ndarray
            The new frequency limits along the Y and X axes or a
            band-limiting ellipse, specified in cycles per the angular
            unit in self.wcs.unit.
        rot : float or None
            Either None, to specify band-limits that pertain to the Y
            and X axes of the current image without any rotation, or,
            if the band-limits pertain to a rotated version of the
            image, the rotation angle of its Y axis westward of north
            (degrees). This is defined such that if
            image.wcs.get_rot() is passed to this function, the
            band-limit newfmax[0] will be along the Y axis of the
            image and newfmax[1] will be along its X axis.

        """

        if rot is None:
            rot = self.wcs.get_rot()
        self._spflims = SpatialFrequencyLimits(newfmax, rot)


class FormatCoord(object):
    """Alter mouse-over coordinates displayed by plt.show()"""

    def __init__(self, image, data):
        self.image = image
        self.data = data

    def __call__(self, x, y):  # pragma: no cover
        """Tell the interactive plotting window how to display the sky
        coordinates and pixel values of an image.

        Parameters
        ----------
        x : float
            The X-axis pixel index of the mouse pointer.
        y : float
            The Y-axis pixel index of the mouse pointer.

        Returns
        -------
        out : str
            The string to be displayed when the mouse pointer is
            over pixel x,y.

        """
        # Find the pixel indexes closest to the specified position.
        col = int(x + 0.5)
        row = int(y + 0.5)

        # Is the mouse pointer within the image?
        im = self.image
        if (im.wcs is not None and row >= 0 and row < im.shape[0]
                and col >= 0 and col < im.shape[1]):
            yc, xc = im.wcs.pix2sky([row, col], unit=im._unit)[0]
            val = self.data[row, col]
            if np.isscalar(val):
                return 'y= %g x=%g p=%i q=%i data=%g' % (yc, xc, row, col, val)
            else:
                return 'y= %g x=%g p=%i q=%i data=%s' % (yc, xc, row, col, val)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)


def get_plot_norm(data, vmin=None, vmax=None, zscale=False, scale='linear'):
    from astropy import visualization as viz
    from astropy.visualization.mpl_normalize import ImageNormalize

    # Choose vmin and vmax automatically?
    if zscale:
        interval = ZScaleInterval()
        if data.dtype == np.float64:
            try:
                vmin, vmax = interval.get_limits(data.filled(np.nan))
            except:
                # catch failure on all NaN
                if np.all(np.isnan(data.filled(np.nan))):
                    vmin, vmax = (np.nan, np.nan)
                else:
                    raise
        else:
            vmin, vmax = interval.get_limits(data.filled(0))

    # How are values between vmin and vmax mapped to corresponding
    # positions along the colorbar?
    if scale == 'linear':
        stretch = viz.LinearStretch
    elif scale == 'log':
        stretch = viz.LogStretch
    elif scale in ('asinh', 'arcsinh'):
        stretch = viz.AsinhStretch
    elif scale == 'sqrt':
        stretch = viz.SqrtStretch
    else:
        raise ValueError('Unknown scale: {}'.format(scale))

    # Create an object that will be used to map pixel values
    # in the range vmin..vmax to normalized colormap indexes.
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch())

    return norm


def plot_rgb(images, title=None, scale='linear', vmin=None, vmax=None,
             zscale=False, show_xlabel=False, show_ylabel=False, ax=None,
             unit=u.deg, use_wcs=False, **kwargs):
    """Plot the RGB composite image with axes labeled in pixels.

    For each color, final intensity values are assigned to each pixel as
    follows. First each pixel value, ``pv``, is normalized over the range
    ``vmin`` to ``vmax``, to have a value ``nv``, that goes from 0 to 1, as
    follows::

        nv = (pv - vmin) / (vmax - vmin)

    This value is then mapped to another number between 0 and 1 which
    determines the final value to give the displayed pixel. The mapping from
    normalized values to final value can be chosen using the scale argument,
    from the following options:

    - 'linear': ``color = nv``
    - 'log': ``color = log(1000 * nv + 1) / log(1000 + 1)``
    - 'sqrt': ``color = sqrt(nv)``
    - 'arcsinh': ``color = arcsinh(10*nv) / arcsinh(10.0)``

    By default the image is displayed in its own plot. Alternatively
    to make it a subplot of a larger figure, a suitable
    ``matplotlib.axes.Axes`` object can be passed via the ``ax`` argument.
    Note that unless matplotlib interative mode has previously been enabled
    by calling ``matplotlib.pyplot.ion()``, the plot window will not appear
    until the next time that ``matplotlib.pyplot.show()`` is called. So to
    arrange that a new window appears as soon as ``plot_rgb`` is
    called, do the following before the first call to ``plot_rgb``::

        import matplotlib.pyplot as plt
        plt.ion()

    Parameters
    ----------
    images : [`~mpdaf.obj.Image`, `~mpdaf.obj.Image`, `~mpdaf.obj.Image`]
        The three [blue, green, red] images to be used. i.e. ordered by
        increasing wavelength.
    title : str
        An optional title for the figure (None by default).
    scale : 'linear' | 'log' | 'sqrt' | 'arcsinh'
        The stretch function to use mapping pixel values to
        final values (The default is 'linear'). The same scaling is applied to
        all three imasges. The pixel values are
        first normalized to range from 0 for values <= vmin,
        to 1 for values >= vmax, then the stretch algorithm maps
        these normalized values, nv, to a position p from 0 to 1
        along the colorbar, as follows:
        linear:  p = nv
        log:     p = log(1000 * nv + 1) / log(1000 + 1)
        sqrt:    p = sqrt(nv)
        arcsinh: p = arcsinh(10*nv) / arcsinh(10.0)
    vmin : [float, float, float]
        Lower limits corresponing to the [blue, green, red] images.
        Pixels that have values <= vmin are assigned a value of 0.
        Pixel values between vmin and vmax are scaled according
        to the mapping algorithm specified by the scale argument.
    vmax : [float, float, float]
        Upper limits corresponing to the [blue, green, red] images.
        Pixels that have values >= vmax are assigned a value of 1.
        Pixel values between vmin and vmax are scaled according
        to the mapping algorithm specified by the scale argument.
    zscale : bool
        If True, vmin and vmax are automatically computed
        using the IRAF zscale algorithm.
    ax : matplotlib.axes.Axes
        An optional Axes instance in which to draw the image,
        or None to have one created using ``matplotlib.pyplot.gca()``.
    unit : `astropy.units.Unit`
        The units to use for displaying world coordinates
        (degrees by default). In the interactive plot, when
        the mouse pointer is over a pixel in the image the
        coordinates of the pixel are shown using these units,
        along with the pixel value.
    use_wcs : bool
        If True, use `astropy.visualization.wcsaxes` to get axes
        with world coordinates.
    kwargs : matplotlib.artist.Artist
        Optional extra keyword/value arguments to be passed to
        the ``ax.imshow()`` function.

    Returns
    -------
    ax : matplotlib AxesImage
    images_aligned : `~mpdaf.obj.Image`, `~mpdaf.obj.Image`, `~mpdaf.obj.Image`
        The input images, but all aligned to that with the highest resolution.

    """
    if vmin is None:
        vmin = [None, None, None]

    if vmax is None:
        vmax = [None, None, None]

    # Default X and Y axes are labeled in pixels.
    xlabel = 'q (pixel)'
    ylabel = 'p (pixel)'

    if ax is None:
        if use_wcs:
            ax = plt.subplot(projection=images[0].wcs.wcs)
            xlabel = 'ra'
            ylabel = 'dec'
        else:
            ax = plt.gca()
    elif use_wcs:
        logging.getLogger(__name__).warning(
            'use_wcs does not work when giving also an axis (ax)')

    # find which image has the highest pixel resolution
    # also find bbox that encloses all 3 images
    steps = np.full([3, 2], np.nan, dtype=float)
    corners = np.full([3, 4, 2], np.nan, dtype=float)
    for i_im, im in enumerate(images):
        wcs = im.wcs
        step = wcs.get_axis_increments(unit=u.deg)
        corn = wcs.wcs.calc_footprint(axes=[wcs.naxis1, wcs.naxis2])

        steps[i_im] = step
        corners[i_im] = corn

    idx_best_res = np.argmin(np.mean(np.abs(steps), 1))
    im_best_res = images[idx_best_res]  # image with highest res

    # get bounding pixel coords in best image
    corners = np.vstack(corners)
    corners = im_best_res.wcs.wcs.all_world2pix(corners, 0)
    new_shape = np.array([[np.min(corners[:, 0]), np.max(corners[:, 0])],
                          [np.min(corners[:, 1]), np.max(corners[:, 1])]])
    new_shape = np.around(new_shape).astype(int)

    new_dim = new_shape[:, 1] - new_shape[:, 0] + 1
    new_start = new_shape[:, 0].reshape(1, 2)
    new_start = im_best_res.wcs.wcs.all_pix2world(new_start, 0)[0]

    new_dim = new_dim[::-1]  # naxis2, naxis1
    new_start = new_start[::-1]  # dec, ra
    old_inc = im_best_res.get_axis_increments(unit=u.deg)

    # expand the reference image so that it now covers the footprints of the
    # other 2 images
    im_best_res = im_best_res.resample(new_dim, new_start, old_inc,
                                       unit_step=u.deg)

    # create BGR stack
    data_stack = np.full(im_best_res.shape + (3,), np.nan, dtype=float)
    data_stack = np.ma.array(data_stack)

    images_aligned = []
    for i, im in enumerate(images):
        # align all images to image with best res
        im = im.align_with_image(im_best_res)
        images_aligned.append(im)
        data = im.data

        norm = get_plot_norm(data, vmin=vmin[i], vmax=vmax[i], zscale=zscale,
                             scale=scale)

        data = norm(data)

        data_stack[:, :, i] = data

    data_stack = np.ma.clip(data_stack, 0, 1)
    data_stack = data_stack.filled(np.nan)

    # reverse BGR to RGB order
    data_stack = data_stack[:, :, ::-1]

    # mask all NaNs and plot transparent
    mask = np.all(np.isnan(data_stack), axis=2)
    alpha = ~mask * 1.  # no transparency where data is good
    data_stack = np.concatenate([data_stack, alpha[..., np.newaxis]], axis=2)

    # Display the RGBA image.
    ax.imshow(data_stack, interpolation='nearest', origin='lower', **kwargs)

    # Keep the axis to allow other functions to overplot
    # the image with contours etc.
    for im in images_aligned:
        im._ax = ax

    # Label the axes if requested.
    if show_xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # Change the way that plt.show() displays coordinates when the pointer
    # is over the image, such that world coordinates are displayed with the
    # specified unit, and pixel values are displayed with their native
    # units.
    ax.format_coord = FormatCoord(images_aligned[0], data_stack)
    for im in images_aligned:
        im._unit = unit
    return ax, images_aligned


def gauss_image(shape=(101, 101), wcs=None, factor=1, gauss=None,
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
    wcs : `mpdaf.obj.WCS`
        World coordinates.
    factor : int
        If factor<=1, gaussian value is computed in the center of each pixel.
        If factor>1, for each pixel, gaussian value is the sum of the gaussian
        values on the factor*factor pixels divided by the pixel area.
    gauss : `mpdaf.obj.Gauss2D`
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
    peak : bool
        If true, flux contains a gaussian peak value.
    rot : float
        Angle position in degree.
    cont : float
        Continuum value. 0 by default.
    unit_center : `astropy.units.Unit`
        type of the center and position coordinates.
        Degrees by default (use None for coordinates in pixels).
    unit_fwhm : `astropy.units.Unit`
        FWHM unit.  Arcseconds by default (use None for radius in pixels)

    Returns
    -------
    out : `~mpdaf.obj.Image`

    """
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)

    wcs = wcs or WCS()
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
        center = (np.array(shape) - 1) / 2.0
    else:
        if unit_center is not None:
            center = wcs.sky2pix(center, unit=unit_center)[0]

    if unit_fwhm is not None:
        fwhm = np.array(fwhm) / wcs.get_step(unit=unit_fwhm)

    # data = np.empty(shape=shape, dtype=float)

    if fwhm[1] == 0 or fwhm[0] == 0:
        raise ValueError('fwhm equal to 0')
    p_width = fwhm[0] * gaussian_fwhm_to_sigma
    q_width = fwhm[1] * gaussian_fwhm_to_sigma

    # rotation angle in rad
    theta = np.pi * rot / 180.0

    if peak is True:
        norm = flux * 2 * np.pi * p_width * q_width
    else:
        norm = flux

    def gauss(p, q):
        cost = np.cos(theta)
        sint = np.sin(theta)
        xdiff = p - center[0]
        ydiff = q - center[1]
        return (
            norm / (2 * np.pi * p_width * q_width) *
            np.exp(-(xdiff * cost - ydiff * sint) ** 2 / (2 * p_width ** 2)) *
            np.exp(-(xdiff * sint + ydiff * cost) ** 2 / (2 * q_width ** 2))
        )

    if factor > 1:
        if rot == 0:
            from scipy import special

            X, Y = np.meshgrid(range(shape[0]), range(shape[1]))
            pixcrd_min = np.array(list(zip(X.ravel(), Y.ravel()))) - 0.5
            # pixsky_min = wcs.pix2sky(pixcrd)
            xmin = (pixcrd_min[:, 1] - center[1]) / np.sqrt(2.0) / q_width
            ymin = (pixcrd_min[:, 0] - center[0]) / np.sqrt(2.0) / p_width

            pixcrd_max = np.array(list(zip(X.ravel(), Y.ravel()))) + 0.5
            # pixsky_max = wcs.pix2sky(pixcrd)
            xmax = (pixcrd_max[:, 1] - center[1]) / np.sqrt(2.0) / q_width
            ymax = (pixcrd_max[:, 0] - center[0]) / np.sqrt(2.0) / p_width

            dx = pixcrd_max[:, 1] - pixcrd_min[:, 1]
            dy = pixcrd_max[:, 0] - pixcrd_min[:, 0]
            data = norm * 0.25 / dx / dy \
                * (special.erf(xmax) - special.erf(xmin)) \
                * (special.erf(ymax) - special.erf(ymin))
            data = np.reshape(data, (shape[1], shape[0])).T
        else:
            yy, xx = np.mgrid[:shape[0] * factor, :shape[1] * factor] / factor
            data = gauss(yy, xx)
            data = data.reshape(shape[0], 2, shape[1], 2).sum(axis=(1, 3))
            data /= factor ** 2
    else:
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        data = gauss(yy, xx)

    return Image(data=data + cont, wcs=wcs, unit=unit, copy=False, dtype=None)


def moffat_image(shape=(101, 101), wcs=None, factor=1, moffat=None,
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
    wcs : `mpdaf.obj.WCS`
        World coordinates.
    factor : int
        If factor<=1, moffat value is computed in the center of each pixel.
        If factor>1, for each pixel, moffat value is the sum
        of the moffat values on the factor*factor pixels divided
        by the pixel area.
    moffat : `mpdaf.obj.Moffat2D`
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
    peak : bool
        If true, flux contains a gaussian peak value.
    n : int
        Atmospheric scattering coefficient. 2 by default.
    rot : float
        Angle position in degree.
    cont : float
        Continuum value. 0 by default.
    unit_center : `astropy.units.Unit`
        type of the center and position coordinates.
        Degrees by default (use None for coordinates in pixels).
    unit_fwhm : `astropy.units.Unit`
        FWHM unit. Arcseconds by default (use None for radius in pixels)

    Returns
    -------
    out : `~mpdaf.obj.Image`

    """
    n = float(n)
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)

    wcs = wcs or WCS()
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
    e = fwhm[1] / fwhm[0]

    if unit_fwhm is not None:
        a = a / wcs.get_step(unit=unit_fwhm)[0]

    if peak:
        norm = flux
    else:
        norm = flux * (n - 1) / (np.pi * a * a * e)

    if center is None:
        center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])
    else:
        if unit_center is not None:
            center = wcs.sky2pix(center, unit=unit_center)[0]

    # rotation angle in rad
    theta = np.pi * rot / 180.0

    def moffat(p, q):
        cost = np.cos(theta)
        sint = np.sin(theta)
        xdiff = p - center[0]
        ydiff = q - center[1]
        return (
            norm * (1 +
                    ((xdiff * cost - ydiff * sint) / a) ** 2 +
                    ((xdiff * sint + ydiff * cost) / a / e) ** 2) ** (-n)
        )

    if factor > 1:
        X, Y = np.meshgrid(range(shape[0] * factor),
                           range(shape[1] * factor))
        factor = float(factor)
        pixcrd = np.array(list(zip(X.ravel() / factor, Y.ravel() / factor)))
        data = moffat(pixcrd[:, 0], pixcrd[:, 1])
        data = (data.reshape(shape[1], factor, shape[0], factor)
                .sum(1).sum(2) / factor / factor).T
    else:
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        data = moffat(yy, xx)

    return Image(data=data + cont, wcs=wcs, unit=unit, copy=False, dtype=None)


def _antialias_filter_image(data, oldstep, newstep, oldfmax=None,
                            window="blackman"):
    """Apply an anti-aliasing prefilter to an image to prepare
    it for subsampling.

    Parameters
    ----------
    data : np.ndimage
        The 2D image to be filtered.
    oldstep: float or (float, float)
        The cell size of the input image. This can be a single
        number for both the X and Y axes, or it can be two
        numbers in an iterable, ordered like (ystep,xstep)
    newstep: float or (float, float)
        The cell size of the output image. This can be a single
        number for both the X and Y axes, or it can be two
        numbers in an iterable, ordered like (ystep,xstep)
    oldfmax : float,float or None
        When an image has previously been filtered, this
        argument can be used to indicate the frequency cutoffs
        that were applied at that time along the Y and X axes,
        respectively, in units of cycles per the unit of oldstep
        and newstep. Image axes that have already been sufficiently
        filtered will then not be refiltered redundantly. If no
        band-limits have previously been established, pass this
        argument as None.
    window : str
        The type of window function to use to filter the
        FFT, chosen from:

        blackman
           This window suppresses ringing better than any other
           window, at the expense of lowered image resolution. In
           the image plane, the PSF of this window is
           approximately gaussian, with a standard deviation of
           around 0.96*newstep, and a FWHM of about 2.3*newstep.

        gaussian
           A truncated gaussian window. This has a smaller PSF
           than the blackman window, however gaussians never fall
           to zero, so either significant ringing will be seen due
           to truncation of the gaussian, or low-level aliasing
           will occur, depending on the spatial frequency coverage
           of the image beyond the folding frequency. It can be a
           good choice for images that only contain smoothly
           varying features. It is equivalent to a convolution of
           the image with both an airy profile and a gaussian of
           standard deviation 0.724*newstep (FWHM 1.704*newstep).

        rectangle
           This window simply zeros all spatial frequencies above
           the highest that can be correctly sampled by the new
           pixel size.  This gives the best resolution of any of
           the windows, but this is marred by the strong sidelobes
           of the resulting airy-profile, especially near bright
           point sources and CCD saturation lines.

    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        The filtered version of the 2D input image, followed by
        a 2-element array that contains the new band-limits
        along the Y and X axes, respectively.

    """

    # Convert oldstep into a numpy array of two float elements.
    if is_number(oldstep):
        oldstep = (oldstep, oldstep)
    oldstep = abs(np.asarray(oldstep, dtype=float))

    # Convert newstep into a numpy array of two float elements.
    if is_number(newstep):
        newstep = (newstep, newstep)
    newstep = abs(np.asarray(newstep, dtype=float))

    # If no band-limits have been specified, substitute the
    # band-limits dictated by the current sampling interval.
    if oldfmax is None:
        oldfmax = 0.5 / oldstep
    else:
        oldfmax = np.minimum(oldfmax, 0.5 / oldstep)

    # Calculate the maximum frequencies that will be sampled by
    # the new pixel sizes along the Y and X axes.
    newfmax = 0.5 / newstep

    # Which axes need to be filtered?
    filter_axes = newfmax < oldfmax

    # Return the original image if neither axis needs filtering.
    if np.all(np.logical_not(filter_axes)):
        return data, oldfmax

    # Get the extent of the input image as a pair of slices.
    image_slice = (slice(0, data.shape[0]), slice(0, data.shape[1]))

    # FFT algorithms can be extremely slow for arrays whose
    # dimensions are not powers of 2. The conventional way to avoid this
    # is to copy the image into a new array whose dimensions
    # are powers of 2, and fill the extra pixels with zeros.

    shape = 2**(np.ceil(np.log(np.asarray(data.shape)) /
                        np.log(2.0))).astype(int)
    if data.shape[0] != shape[0] or data.shape[1] != shape[1]:
        tmp = np.zeros(shape)
        tmp[image_slice] = data
        data = tmp

    # Get the new dimensions of the zero-padded image.
    ny, nx = shape

    # Obtain the FFT of the image.
    fft = np.fft.rfft2(data)
    del data

    # The new pixel sizes along the X and Y axes can only correctly
    # sample spatial frequencies up to the values in newfmax. Set the
    # cutoff frequencies for the window functions along the x and y
    # axes to those frequencies.
    fycut, fxcut = newfmax

    # Create an array which, for each pixel in the FFT image, holds
    # the radial spatial-frequency of the pixel center, divided by
    # the cutoff frequency. These values will later be used to index
    # the 1D window-function.

    wr = np.sqrt((np.fft.rfftfreq(nx, oldstep[1]) / fxcut)**2 +
                 (np.fft.fftfreq(ny, oldstep[0]) / fycut)[np.newaxis, :].T**2)

    # Get the requested window function as a function of frequency
    # divided by its cutoff frequency.

    if window is None or window == "blackman":
        winfn = lambda r: np.where(r <= 1.0,
                                   0.42 + 0.5 * np.cos(np.pi * r) +
                                   0.08 * np.cos(2 * np.pi * r),
                                   0.0)

    # For the gaussian window the standard deviation, sigma, is
    # as a fraction of the normalized cutoff frequency. Note that
    # in the image plane the corresponding gaussian standard
    # deviation should be newstep/(pi*sigma).

    elif window == "gaussian":
        sigma = 0.44
        winfn = lambda r: np.exp(-0.5 * (r / sigma)**2)

    # For the rectangular window, just multiply all pixels below the
    # cutoff frequency by one, and the rest by zero.

    elif window == "rectangle":
        winfn = lambda r: np.where(r <= 1.0, 1.0, 0.0)

    # Apply the window function to the FFT to remove frequencies above the
    # cutoff frequencies.

    fft *= winfn(wr)
    del wr

    # Perform an inverse Fourier transform to get the filtered image
    data = np.fft.irfft2(fft)
    del fft

    # Crop the antialiased image to remove the zero-padded pixels, and
    # return this along with the new spatial-frequency limits.
    return data[image_slice], np.where(filter_axes, newfmax, oldfmax)


def _find_quadratic_peak(y):
    """Given an array of 3 numbers in which the first and last numbers are
    less than the central number, determine the array index at which a
    quadratic curve through the 3 points reaches its peak value.

    Parameters
    ----------
    y  : float,float,float
      The values of the curve at x=0,1,2 respectively. Note that y[1]
      must be greater than both y[0] and y[2]. Otherwise +/- infinity
      will be returned.

    Returns
    -------
    xpeak : float
      The floating point array index of the peak of the quadratic. This
      will always be in the range 0.0 to 2.0, provided that y[0]<y[1] and
      y[2]<y[1].

    """

    # Given the three equations:
    #
    #  a * x0**2 + b * x0 + c = y0
    #  a * x1**2 + b * x1 + c = y1
    #  a * x2**2 + b * x2 + c = y2
    #
    # a, b, and c are given by:
    #
    #  a =  0.5 * y0 - y1 + 0.5 * y2
    #  b = -1.5 * y0 + 2.0 * y1 - 0.5 * y2
    #  c = y0

    a = 0.5 * y[0] - y[1] + 0.5 * y[2]
    b = -1.5 * y[0] + 2.0 * y[1] - 0.5 * y[2]

    # Quadratic curves peak at:  x = -b / (2*a)
    return -b / (2 * a)


class SpatialFrequencyLimits(object):

    """Allow to keep track of the spatial frequency limits of an image.

    Such that before resampling an image it can see if anything needs to be
    done to avoid undersampling and generating aliasing artefacts in the output
    image.

    The band-limits are recorded as an ellipse. Most telescopes have circularly
    symmetric PSFs and thus circularly symmetric spatial-frequency band limits,
    but this spatial-frequency profile may become elliptical if an image is
    resampled to have a lower resolution along one axis.

    The ellipse is defined in its own X,Y coordinate system as follows::

        xe(t)=xs*cos(t)
        ye(t)=ys*sin(t)

    The ye axis of the ellipse is at self.rot degrees west of north in the
    image. For the Y axis of a coordinate system where Y is rot degrees west of
    north, the ellipse thus has to be rotated by ``psi = (rot - self.rot)``
    degrees anticlockwise to calculate the X and Y values of the ellipse in
    that coordinate system::

        |x(t)| = |cos(psi), -sin(psi)| |xe(t)|
        |y(t)|   |sin(psi),  cos(psi)| |ye(t)|

    Parameters
    ----------
    fmax : float, float
        The frequency limits along the Y-axis and X-axis of an
        elliptically shaped band-limit (eg. cycles per degree).
    rot : float
        The rotation angle of the Y axis of the ellipse westward of
        north (degrees). This is defined such that if
        image.wcs.get_rot() is passed to this function, the Y axis of
        the ellipse will be aligned with the Y axis of the image.

    """

    def __init__(self, fmax, rot):
        # Store the Y and X axes of the band-limiting ellipse.
        self.fmax = np.array(fmax, dtype=float, copy=True)

        # Record the rotation angle in degrees of the ellipse, after
        # wrapping the angle into the range -180 to 180, to make it
        # easy to compare with angles returned by wcs.get_rot().
        self.rot = float(rot - 360.0 * np.floor(rot / 360.0 + 0.5))

    def deepcopy(self):
        return SpatialFrequencyLimits(self.fmax, self.rot)

    def get_fmax(self, rot):
        """Return the spatial-frequency band-limits along a Y axis that is
        'rot' degrees west of north, and an X axis that is 90 degrees
        away from this Y axis in the sense of a rotation from north to east.

        Parameters
        ----------
        rot : float
           The angle of the target Y axis west of north (degrees).

        Returns
        -------
        out : np.ndarray()
           The maximum spatial frequencies along the Y and X axes at
           rotation angle rot, in the same units as were used to
           initialize the object.

        """

        # Extract the Y and X axis radii of the ellipse.
        ys, xs = self.fmax

        # Compute the rotation angle of the ellipse in radians.
        psi = np.deg2rad(rot - self.rot)

        # Precalculate sin and cos of the ellipse rotation angle.
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Calculate the ellipse phases where the X and Y coordinates
        # of the ellipse locus are maximized. These equations come from
        # calculating d[x(t)]/dt=0 and d[y(t)]/dt=0 using the definitions
        # of x(t) and y(t) that are shown in the class documentation.
        t_xmax = np.arctan2(-ys * sin_psi, xs * cos_psi)
        t_ymax = np.arctan2(ys * cos_psi, xs * sin_psi)

        # Get the maximum X and Y coordinates of the rotated ellipse.
        xmax = xs * np.cos(t_xmax) * cos_psi - ys * np.sin(t_xmax) * sin_psi
        ymax = xs * np.cos(t_ymax) * sin_psi + ys * np.sin(t_ymax) * cos_psi

        return np.array([ymax, xmax], dtype=float)

    def ellipse_locus(self, t, rot):
        """Return the Y,X coordinates of the band-limiting ellipse
        at ellipse phase t.

        Parameters
        ----------
        t : float
            The elliptical phase at which the calculate the
            coordinates (radians).
        rot : float
            The rotation angle of the Y axis of the ellipse west
            of north (degrees).

        Returns
        -------
        out : np.ndarray
            The Y and X coordinates of the band-limiting ellipse.
        """

        # Extract the Y and X axis radii of the ellipse.
        ys, xs = self.fmax

        # Compute the rotation angle of the ellipse in radians.
        psi = np.deg2rad(rot - self.rot)

        # Precalculate sin and cos of the ellipse rotation angle.
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Precalculate sin and cos of the phase of the ellipse.
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        # Calculate the locus of the ellipse at phase t, using
        # the equations shown in the class documentation.
        x = xs * cos_t * cos_psi - ys * sin_t * sin_psi
        y = xs * cos_t * sin_psi + ys * sin_t * cos_psi

        return np.array([y, x], dtype=float)
