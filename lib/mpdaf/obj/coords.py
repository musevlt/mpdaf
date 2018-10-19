"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2012-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>

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
import astropy.wcs as pywcs
import logging
import numbers
import numpy as np

from astropy.coordinates import Angle
from astropy.io import fits

from .objs import UnitArray
from ..tools import fix_unit_read

__all__ = ('deg2sexa', 'sexa2deg', 'deg2hms', 'hms2deg', 'deg2dms', 'dms2deg',
           'image_angle_from_cd', 'axis_increments_from_cd',
           'WCS', 'WaveCoord', 'determine_refframe')


def deg2sexa(x):
    """Transform equatorial coordinates from degrees to sexagesimal strings.

    Parameters
    ----------
    x : float array
        Either a single coordinate in a 1D array like ``[dec, ra]``,
        or a 2D array of multiple (dec,ra) coordinates, ordered like
        ``[[dec1,ra1], [dec2,ra2], ...]``. All coordinates must be in degrees.

    Returns
    -------
    out : array of strings
        The array of dec,ra coordinates as sexagesimal strings, stored in an
        array of the same dimensions as the input array. Declination values
        are written like degrees:minutes:seconds. Right-ascension values are
        written like hours:minutes:seconds.

    """
    x = np.asarray(x)
    ndim = x.ndim
    x = np.atleast_2d(x)

    # FIXME: can be replaced with SkyCoord.to_string('hmsdms', sep=':')
    result = []
    for i in range(np.shape(x)[0]):
        ra = deg2hms(x[i][1])
        dec = deg2dms(x[i][0])
        result.append(np.array([dec, ra]))
    return np.array(result) if ndim > 1 else result[0]


def sexa2deg(x):
    """Transform equatorial coordinates from sexagesimal strings to degrees.

    Parameters
    ----------
    x : string array
        Either a single pair of coordinate strings in a 1D array like ``[dec,
        ra]``, or a 2D array of multiple (dec,ra) coordinate strings, ordered
        like ``[[dec1,ra1], [dec2,ra2], ...]``. In each coordinate pair, the
        declination string should be written like degrees:minutes:seconds, and
        the right-ascension string should be written like
        hours:minutes:seconds.

    Returns
    -------
    out : array of numbers
        The array of ra,dec coordinates in degrees, returned in an
        array of the same dimensions as the input array.

    """
    x = np.asarray(x)
    ndim = x.ndim
    x = np.atleast_2d(x)

    result = []
    for i in range(np.shape(x)[0]):
        ra = hms2deg(x[i][1])
        dec = dms2deg(x[i][0])
        result.append(np.array([dec, ra]))
    return np.array(result) if ndim > 1 else result[0]


def deg2hms(x):
    """Transform degrees to *hours:minutes:seconds* strings.

    Parameters
    ----------
    x : float
        The degree value to be written as a sexagesimal string.

    Returns
    -------
    out : str
        The input angle written as a sexagesimal string, in the
        form, hours:minutes:seconds.

    """
    ac = Angle(x, unit='degree')
    hms = ac.to_string(unit='hour', sep=':', pad=True)
    return str(hms)


def hms2deg(x):
    """Transform *hours:minutes:seconds* strings to degrees.

    Parameters
    ----------
    x : str
        The input angle, written in the form, hours:minutes:seconds

    Returns
    -------
    out : float
        The angle as a number of degrees.

    """
    ac = Angle(x, unit='hour')
    deg = float(ac.to_string(unit='degree', decimal=True))
    return deg


def deg2dms(x):
    """Transform degrees to *degrees:arcminutes:arcseconds* strings.

    Parameters
    ----------
    x : float
        The degree value to be converted.

    Returns
    -------
    out : str
        The input angle as a string, written as degrees:minutes:seconds.

    """
    ac = Angle(x, unit='degree')
    dms = ac.to_string(unit='degree', sep=':', pad=True)
    return str(dms)


def dms2deg(x):
    """Transform *degrees:arcminutes:arcseconds* strings to degrees.

    Parameters
    ----------
    x : str
        The input angle written in the form, degrees:arcminutes:arcseconds

    Returns
    -------
    out : float
        The input angle as a number of degrees.

    """
    ac = Angle(x, unit='degree')
    deg = float(ac.to_string(unit='degree', decimal=True))
    return deg


def _wcs_from_header(hdr, naxis=None):
    if 'CD1_1' in hdr and 'CDELT3' in hdr and 'CD3_3' not in hdr:
        hdr['CD3_3'] = hdr['CDELT3']
    if 'PC1_1' in hdr and 'CDELT3' in hdr and 'PC3_3' not in hdr:
        hdr['PC3_3'] = 1
    # WCS object from data header
    return pywcs.WCS(hdr, naxis=naxis)


def image_angle_from_cd(cd, unit=u.deg):
    """Return the rotation angle of the image.

    Defined such that a rotation angle of zero aligns north along the positive
    Y axis, and a positive rotation angle rotates north away from the Y axis,
    in the sense of a rotation from north to east.

    Note that the rotation angle is defined in a flat map-projection of the
    sky. It is what would be seen if the pixels of the image were drawn with
    their pixel widths scaled by the angular pixel increments returned by the
    axis_increments_from_cd() method.

    If the CD matrix was derived from the archaic CROTA and CDELT FITS
    keywords, then the angle returned by this function is equal to CROTA.

    Parameters
    ----------
    cd : numpy.ndarray
        The 2x2 coordinate conversion matrix, with its elements
        ordered for multiplying a column vector in FITS (x,y) axis order.
    unit : `astropy.units.Unit`
        The unit to give the returned angle (degrees by default).

    Returns
    -------
    out : float
        The angle between celestial north and the Y axis of the image,
        in the sense of an eastward rotation of celestial north from
        the Y-axis. The angle is returned in the range -180 to 180
        degrees (or the equivalent for the specified unit).

    """

    # Get the angular increments of pixels along the Y and X axes
    step = axis_increments_from_cd(cd)

    # Get the determinant of the coordinate transformation matrix.
    cddet = np.linalg.det(cd)

    # The angle of a northward vector from the origin can be calculated by
    # first using the inverse of the CD matrix to calculate the equivalent
    # vector in pixel indexes, then calculating the angle of this vector to the
    # Y axis of the image array.
    north = np.arctan2(-cd[0, 1] * step[1] / cddet,
                       cd[0, 0] * step[0] / cddet)

    # Return the angle with the specified units.
    return (north * u.rad).to(unit).value


def axis_increments_from_cd(cd):
    """Return the angular increments of pixels along the Y and X axes
    of an image array whose coordinates are described by a specified
    FITS CD matrix.

    In MPDAF, images are a regular grid of square pixels on a flat
    projection of the celestial sphere. This function returns the
    angular width and height of these pixels on the sky, with signs
    that indicate whether the angle increases or decreases as one
    steps along the corresponding array axis. To keep plots
    consistent, regardless of the rotation angle of the image on the
    sky, the returned height is always positive, but the returned
    width is negative if a plot of the image with pixel 0,0 at the
    bottom left would place east anticlockwise of north, and positive
    otherwise.

    Parameters
    ----------
    cd : numpy.ndarray
        The 2x2 coordinate conversion matrix, with its elements
        ordered for multiplying a column vector in FITS (x,y) axis
        order.
    unit : `astropy.units.Unit`
        The angular units of the returned values.

    Returns
    -------
    out : numpy.ndarray
        (dy,dx). These are the angular increments of pixels along the
        Y and X axes of the image, returned with the same units as
        the contents of the CD matrix.

    """

    # The pixel dimensions are determined as follows. First note
    # that the coordinate transformation matrix looks as follows:
    #
    #    |r| = |M[0,0], M[0,1]| |col - get_crpix1()|
    #    |d|   |M[1,0], M[1,1]| |row - get_crpix2()|
    #
    # In this equation [col,row] are the indexes of a pixel in the
    # image array and [r,d] are the coordinates of this pixel on a
    # flat map-projection of the sky. If the celestial coordinates
    # of the observation are right ascension and declination, then d
    # is parallel to declination, and r is perpendicular to this,
    # pointing east. When the column index is incremented by 1, the
    # above equation indicates that r and d change by:
    #
    #    col_dr = M[0,0]   col_dd = M[1,0]
    #
    # The length of the vector from (0,0) to (col_dr,col_dd) is
    # the angular width of pixels along the X axis.
    #
    #    dx = sqrt(M[0,0]^2 + M[1,1]^2)
    #
    # Similarly, when the row index is incremented by 1, r and d
    # change by:
    #
    #    row_dr = M[0,1]   row_dd = M[1,1]
    #
    # The length of the vector from (0,0) to (row_dr,row_dd) is
    # the angular width of pixels along the Y axis.
    #
    #    dy = sqrt(M[0,1]^2 + M[1,1]^2)
    #
    # Calculate the width and height of the pixels as described above.

    dx = np.sqrt(cd[0, 0]**2 + cd[1, 0]**2)
    dy = np.sqrt(cd[0, 1]**2 + cd[1, 1]**2)

    # To decide what sign to give the step in X, we need to know
    # whether east is clockwise or anticlockwise of north when the
    # image is plotted with pixel 0,0 at the bottom left of the
    # plot. The angle of a northward vector from the origin can be
    # calculated by first using the inverse of the CD matrix to
    # calculate the equivalent vector in pixel indexes, then
    # calculating the angle of this vector to the Y axis of the
    # image array. Start by calculating the determinant of the CD
    # matrix.

    cddet = np.linalg.det(cd)

    # Calculate the rotation angle of a unit northward vector
    # clockwise of the Y axis.

    north = np.arctan2(-cd[0, 1] / cddet, cd[0, 0] / cddet)

    # Calculate the rotation angle of a unit eastward vector
    # clockwise of the Y axis.

    east = np.arctan2(cd[1, 1] / cddet, -cd[1, 0] / cddet)

    # Wrap the difference east-north into the range -pi to pi radians.

    delta = Angle((east - north) * u.rad).wrap_at(np.pi * u.rad).value

    # If east is anticlockwise of north make the X-axis pixel increment
    # negative.

    if delta < 0.0:
        dx *= -1.0

    # Return the axis increments in python array-indexing order.

    return np.array([dy, dx])


def determine_refframe(phdr):
    """Determine the reference frame and equinox in standard FITS WCS terms.

    Parameters
    ----------
    phdr : `astropy.io.fits.Header`
        Primary Header of an observation

    Returns
    -------
    out : str, float
        Reference frame ('ICRS', 'FK5', 'FK4') and equinox

    """
    # MUSE files should have RADECSYS='FK5' and EQUINOX=2000.0
    equinox = phdr.get('EQUINOX')
    radesys = phdr.get('RADESYS') or phdr.get('RADECSYS')

    if radesys == 'FK5' and equinox == 2000.0:
        return 'FK5', equinox
    elif radesys:
        return radesys, None
    elif equinox is not None:
        return 'FK4' if equinox < 1984. else 'FK5', equinox
    else:
        return None, None


class WCS(object):

    """The WCS class manages the world coordinates of the spatial axes of
    MPDAF images, using the pywcs package.

    Note that MPDAF images are stored in python arrays that are
    indexed in [y,x] axis order. In general the axes of these arrays
    are not along celestial axes such as right-ascension and
    declination. They are cartesian axes of a flat map projection of
    the sky around the observation center, and they may be rotated
    away from the celestial axes. When their rotation angle is zero,
    the Y axis is parallel to the declination axis. However the X axis
    is only along the right ascension axis for observations at zero
    declination.

    Pixels in MPDAF images are not generally square on the sky. To
    scale index offsets in the image to angular distances in the map
    projection, the Y-axis and X-axis index offsets must be scaled by
    different numbers. These numbers can be obtained by calling the
    get_axis_increments() method, which returns the angular increment
    per pixel increment along the Y and X axes of the array. The
    Y-axis increment is always positive, but the X-axis increment is
    negative if east is anti-clockwise of north when the X-axis pixels
    are plotted from left to right,

    The rotation angle of the map projection, relative to the sky, can
    be obtained by calling the get_rot() method. This returns the
    angle between celestial north and the Y axis of the image, in the
    sense of an eastward rotation of celestial north from the Y-axis.

    When the linearized coordinates of the map projection are
    insufficient, the celestial coordinates of one or more pixels can
    be queried by calling the pix2sky() method, which returns
    coordinates in the [dec,ra] axis order. In the other direction,
    the [y,x] indexes of the pixel of a given celestial coordinate can
    be obtained by calling the sky2pix() method.

    Parameters
    ----------
    hdr : astropy.fits.CardList
        A FITS header. If the hdr parameter is not None, the WCS
        object is created from the data header, and the remaining
        parameters are ignored.
    crpix : float or (float,float)
        The FITS array indexes of the reference pixel of the image,
        given in the order (y,x). Note that the first pixel of the
        FITS image is [1,1], whereas in the python image array it is
        [0,0]. Thus to place the reference pixel at [ry,rx] in the
        python image array would require crpix=(ry+1,rx+1).

        If both crpix and shape are None, then crpix is given the
        value (1.0,1.0) and the reference position is at index [0,0]
        in the python array of the image.

        If crpix is None and shape is not None, then crpix is set to
        (shape + 1.0)/2.0, which places the reference point at the
        center of the image.
    crval : float or (float,float)
        The celestial coordinates of the reference pixel
        (ref_dec,ref_ra). If this paramater is not provided, then
        (0.0,0.0) is substituted.
    cdelt : float or (float,float)
        If the hdr and cd parameters are both None, then this argument
        can be used to specify the pixel increments along the Y and X
        axes of the image, respectively.  If this parameter is not
        provided, (1.0,1.0) is substituted. Note that it is
        conventional for cdelt[1] to be negative, such that east is
        plotted towards the left when the image rotation angle is
        zero.
    deg : bool
        If True, then cdelt and crval are in decimal degrees
        (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg').
        If False (the default), the celestial coordinates are linear
        (CTYPE1=CTYPE2='LINEAR').
    rot : float
        If the hdr and cd paramters are both None, then this argument
        can be used to specify a value for the rotation angle of the
        image. This is the angle between celestial north and the Y
        axis of the image, in the sense of an eastward rotation of
        celestial north from the Y-axis.

        Along with the cdelt parameter, the rot parameter is used to
        construct a FITS CD rotation matrix. This is done as described
        in equation 189 of Calabretta, M. R., and Greisen, E. W,
        Astronomy & Astrophysics, 395, 1077-1122, 2002, where it
        serves as the value of the CROTA term.
    shape : integer or (integer,integer)
        The dimensions of the image axes (optional). The dimensions
        are given in python order (ny,nx).
    cd : numpy.ndarray
         This parameter can optionally be used to specify the FITS CD
         rotation matrix. By default this parameter is None. However if
         a matrix is provided and hdr is None, then it is used
         instead of cdelt and rot, which are then ignored. The matrix
         should be ordered like

           cd = numpy.array([[CD1_1, CD1_2],
                             [CD2_1, CD2_2]]),

         where CDj_i are the names of the corresponding FITS keywords.

    Attributes
    ----------
    wcs : astropy.wcs.WCS
        The underlying object that performs most of the world coordinate
        conversions.

    """

    def __init__(self, hdr=None, crpix=None, crval=(1.0, 1.0),
                 cdelt=(1.0, 1.0), deg=False, rot=0, shape=None, cd=None,
                 frame=None, equinox=None):
        self._logger = logging.getLogger(__name__)

        if hdr is not None:
            # Initialize the WCS object from a FITS header?
            self.wcs = _wcs_from_header(hdr, naxis=2)

            if frame is not None:
                self.wcs.wcs.radesys = frame
            if equinox is not None:
                self.wcs.wcs.equinox = equinox
        else:
            # Initialize the WCS object

            # Define a function that checks that 2D attributes are
            # either a 2-element tuple of float or int, or a float or
            # int scalar which is converted to a 2-element tuple.
            def check_attrs(val, types=numbers.Number):
                if isinstance(val, types):
                    return (val, val)
                elif len(val) > 2:
                    raise ValueError('dimension > 2')
                else:
                    return val

            crval = check_attrs(crval)
            cdelt = check_attrs(cdelt)

            if crpix is not None:
                crpix = check_attrs(crpix)

            if shape is not None:
                shape = check_attrs(shape, types=int)

            self.wcs = pywcs.WCS(naxis=2)

            # Get the FITS array indexes of the reference pixel.  Beware that
            # FITS array indexes are offset by 1 from python array indexes.
            if crpix is not None:
                self.wcs.wcs.crpix = np.array([crpix[1], crpix[0]])
            elif shape is None:
                self.wcs.wcs.crpix = np.array([1.0, 1.0])
            else:
                self.wcs.wcs.crpix = (np.array([shape[1], shape[0]]) + 1) / 2.

            # Get the world coordinate value of reference pixel.
            self.wcs.wcs.crval = np.array([crval[1], crval[0]])
            if deg:  # in decimal degree
                self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
                self.wcs.wcs.cunit = ['deg', 'deg']
                if frame is not None:
                    self.wcs.wcs.radesys = frame
                if equinox is not None:
                    self.wcs.wcs.equinox = equinox
            else:   # in pixel or arcsec
                self.wcs.wcs.ctype = ['LINEAR', 'LINEAR']
                self.wcs.wcs.cunit = ['pixel', 'pixel']

            if cd is not None and cd.shape[0] == 2 and cd.shape[1] == 2:
                # If a CD rotation matrix has been provided by the caller,
                # install it.
                self.set_cd(cd)
            else:
                # If no CD matrix was provided, construct one from the cdelt
                # and rot parameters, following the official prescription given
                # by equation 189 of Calabretta, M. R., and Greisen, E. W,
                # Astronomy & Astrophysics, 395, 1077-1122, 2002.
                rho = np.deg2rad(rot)
                sin_rho = np.sin(rho)
                cos_rho = np.cos(rho)
                self.set_cd(np.array([
                    [cdelt[1] * cos_rho, -cdelt[0] * sin_rho],
                    [cdelt[1] * sin_rho, cdelt[0] * cos_rho]]))

            # Update the wcs object to accomodate the new value of
            # the CD matrix.
            self.wcs.wcs.set()

            # Set the dimensions of the image array.
            if shape is not None:
                self.naxis1 = shape[1]
                self.naxis2 = shape[0]

    @property
    def naxis1(self):
        return self.wcs._naxis1

    @naxis1.setter
    def naxis1(self, value):
        self.wcs._naxis1 = value

    @property
    def naxis2(self):
        return self.wcs._naxis2

    @naxis2.setter
    def naxis2(self, value):
        self.wcs._naxis2 = value

    def copy(self):
        """Return a copy of a WCS object."""
        out = WCS()
        out.wcs = self.wcs.deepcopy()
        return out

    def __repr__(self):
        return repr(self.wcs)

    def info(self):
        """Print information about a WCS object."""
        try:
            dy, dx = self.get_step(unit=u.arcsec)
            sizex = dx * self.naxis1  # ra
            sizey = dy * self.naxis2  # dec
            # center in sexadecimal
            xc = (self.naxis1 - 1) / 2.
            yc = (self.naxis2 - 1) / 2.
            pixsky = self.pix2sky([yc, xc], unit=u.deg)[0]
            dec, ra = deg2sexa(pixsky)
            self._logger.info(
                'center:(%s,%s) size:(%0.3f",%0.3f") '
                'step:(%0.3f",%0.3f") rot:%0.1f deg frame:%s',
                dec, ra, sizey, sizex, dy, dx, self.get_rot(),
                self.wcs.wcs.radesys)
        except Exception:
            # FIXME: when is that useful ?
            pixcrd = [[0, 0], [self.naxis2 - 1, self.naxis1 - 1]]
            pixsky = self.pix2sky(pixcrd)
            dy, dx = self.get_step()
            self._logger.info(
                'spatial coord (%s): min:(%0.1f,%0.1f) max:(%0.1f,%0.1f) '
                'step:(%0.1f,%0.1f) rot:%0.1f deg', self.unit,
                pixsky[0, 0], pixsky[0, 1], pixsky[1, 0], pixsky[1, 1],
                dy, dx, self.get_rot())

    def to_header(self):
        """Generate an astropy.fits header object containing the WCS
        information.
        """
        has_cd = self.wcs.wcs.has_cd()
        hdr = self.wcs.to_header()
        if has_cd:
            for ci in range(1, 3):
                cdelt = hdr.pop('CDELT%i' % ci, 1)
                for cj in range(1, 3):
                    try:
                        val = cdelt * hdr.pop('PC%i_%i' % (ci, cj))
                    except KeyError:
                        if ci == cj:
                            val = cdelt
                        else:
                            val = 0.
                    hdr['CD%i_%i' % (ci, cj)] = val
        return hdr

    def sky2pix(self, x, nearest=False, unit=None):
        """Convert world coordinates (dec,ra) to image pixel indexes (y,x).

        If nearest=True; returns the nearest integer pixel.

        Parameters
        ----------
        x : array
            An (n,2) array of dec- and ra- world coordinates.
        nearest : bool
            If nearest is True returns the nearest integer pixel
            in place of the decimal pixel.
        unit : `astropy.units.Unit`
            The units of the world coordinates

        Returns
        -------
        out : (n,2) array of image pixel indexes. These are
              python array indexes, ordered like (y,x) and with
              0,0 denoting the lower left pixel of the image.

        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape == (2,):
            x = x.reshape(1, 2)
        elif len(x.shape) != 2 or x.shape[1] != 2:
            raise IOError('invalid input coordinates for sky2pix')

        if unit is not None:
            x[:, 1] = UnitArray(x[:, 1], unit, self.unit)
            x[:, 0] = UnitArray(x[:, 0], unit, self.unit)

        # Tell world2pix to convert the world coordinates to
        # zero-relative array indexes.
        ax, ay = self.wcs.wcs_world2pix(x[:, 1], x[:, 0], 0)
        res = np.array([ay, ax]).T

        if nearest:
            res += 0.5
            res = res.astype(int)
            if self.naxis1 != 0 and self.naxis2 != 0:
                np.clip(res, (0, 0), (self.naxis2 - 1, self.naxis1 - 1),
                        out=res)
        return res

    def pix2sky(self, x, unit=None):
        """Convert image pixel indexes (y,x) to world coordinates (dec,ra).

        Parameters
        ----------
        x : array
            An (n,2) array of image pixel indexes. These should be
            python array indexes, ordered like (y,x) and with
            0,0 denoting the lower left pixel of the image.
        unit : `astropy.units.Unit`
            The units of the world coordinates.

        Returns
        -------
        out : (n,2) array of dec- and ra- world coordinates.

        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape == (2,):
            x = x.reshape(1, 2)
        elif len(x.shape) != 2 or x.shape[1] != 2:
            raise IOError('invalid input coordinates for pix2sky')

        # Tell world2pix to treat the pixel indexes as zero relative
        # array indexes.
        ra, dec = self.wcs.wcs_pix2world(x[:, 1], x[:, 0], 0)
        if unit is not None:
            ra = UnitArray(ra, self.unit, unit)
            dec = UnitArray(dec, self.unit, unit)

        return np.array([dec, ra]).T

    def isEqual(self, other):
        """Return True if other and self have the same attributes.

        Beware that if the two wcs objects have the same world coordinate
        characteristics, but come from images of different dimensions, the
        objects will be considered different.

        Parameters
        ----------
        other : WCS
            The wcs object to be compared to self.

        Returns
        -------
        out : bool
            True if the two WCS objects have the same attributes.

        """
        if not isinstance(other, WCS):
            return False

        return (self.sameStep(other) and
                self.naxis1 == other.naxis1 and
                self.naxis2 == other.naxis2 and
                np.allclose(self.get_start(), other.get_start(),
                            atol=1E-3, rtol=0) and
                np.allclose(self.get_rot(), other.get_rot(),
                            atol=1E-3, rtol=0))

    def sameStep(self, other):
        """Return True if other and self have the same pixel sizes.

        Parameters
        ----------
        other : WCS
            The wcs object to compare to self.

        Returns
        -------
        out : bool
            True if the two arrays of axis step increments are equal.
        """

        if not isinstance(other, WCS):
            return False

        steps1 = self.get_step()
        steps2 = other.get_step(unit=self.unit)
        return np.allclose(steps1, steps2, atol=1E-7, rtol=0)

    def __getitem__(self, item):
        """Return a WCS object of a 2D slice"""
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('Invalid index, a 2D slice is expected')

        if not isinstance(item[1], slice) and not isinstance(item[0], slice):
            return None

        # See if a slice object was sent for the X axis.
        if isinstance(item[1], slice):
            # If a start index was provided, limit it to the extent of
            # the x-axis. If no start index was provided, default to
            # zero.
            if item[1].start is None:
                imin = 0
            else:
                imin = int(item[1].start)
                if imin < 0:
                    imin = self.naxis1 + imin
                if imin > self.naxis1:
                    imin = self.naxis1

            # If a stop index was provided, limit it to the extent of the
            # X axis. Otherwise substitute the size of the X-axis.
            if item[1].stop is None:
                imax = self.naxis1
            else:
                imax = int(item[1].stop)
                if imax < 0:
                    imax = self.naxis1 + imax
                if imax > self.naxis1:
                    imax = self.naxis1

            # If a step was provided and it isn't 1, complain
            # because we can't accomodate gaps between pixels.
            if item[1].step is not None and item[1].step != 1:
                raise ValueError('Index steps are not supported')

        else:
            # If a slice object wasn't sent, then maybe a single index
            # was passed for the X axis. If so, select the specified
            # single pixel.
            imin = int(item[1])
            imax = int(item[1] + 1)

        # See if a slice object was sent for the Y axis.
        if isinstance(item[0], slice):
            # If a start index was provided, limit it to the extent of
            # the y-axis. If no start index was provided, default to
            # zero.
            if item[0].start is None:
                jmin = 0
            else:
                jmin = int(item[0].start)
                if jmin < 0:
                    jmin = self.naxis2 + jmin
                if jmin > self.naxis2:
                    jmin = self.naxis2

            # If a stop index was provided, limit it to the extent of the
            # Y axis. Otherwise substitute the size of the Y-axis.
            if item[0].stop is None:
                jmax = self.naxis2
            else:
                jmax = int(item[0].stop)
                if jmax < 0:
                    jmax = self.naxis2 + jmax
                    if jmax > self.naxis2:
                        jmax = self.naxis2

            # If an index step was provided and it isn't 1, reject
            # the call, because we can't accomodate gaps between selected
            # pixels.
            if item[0].step is not None and item[0].step != 1:
                raise ValueError('Index steps are not supported')

        else:
            # If a slice object wasn't sent, then maybe a single index
            # was passed for the Y axis. If so, select the specified
            # single pixel.
            jmin = int(item[0])
            jmax = int(item[0] + 1)

        # Compute the array indexes of the coordinate reference
        # pixel in the sliced array. Note that this can indicate a
        # pixel outside the slice.
        crpix = (self.wcs.wcs.crpix[0] - imin, self.wcs.wcs.crpix[1] - jmin)

        # Get a copy of the original WCS object.
        res = self.copy()

        # Record the new coordinate reference pixel index and the
        # reduced dimensions of the selected sub-image.
        res.wcs.wcs.crpix = np.array(crpix)
        res.naxis1 = int(imax - imin)
        res.naxis2 = int(jmax - jmin)
        res.wcs.wcs.set()

        return res

    def get_step(self, unit=None):
        """Return the angular height and width of a pixel along the
        Y and X axes of the image array.

        In MPDAF, images are sampled on a regular grid of square pixels that
        represent a flat projection of the celestial sphere. The get_step()
        method returns the angular width and height of these pixels on the sky.

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
        # The pixel dimensions are determined as follows. First note
        # that the coordinate transformation matrix looks as follows:
        #
        #    |r| = |M[0,0], M[0,1]| |col - get_crpix1()|
        #    |d|   |M[1,0], M[1,1]| |row - get_crpix2()|
        #
        # In this equation [col,row] are the indexes of a pixel in the
        # image array and [r,d] are the coordinates of this pixel on a
        # flat map-projection of the sky. If the celestial coordinates
        # of the observation are right ascension and declination, then d
        # is parallel to declination, and r is perpendicular to this,
        # pointing east. When the column index is incremented by 1, the
        # above equation indicates that r and d change by:
        #
        #    col_dr = M[0,0]   col_dd = M[1,0]
        #
        # The length of the vector from (0,0) to (col_dr,col_dd) is
        # the angular width of pixels along the X axis.
        #
        #    dx = sqrt(M[0,0]^2 + M[1,1]^2)
        #
        # Similarly, when the row index is incremented by 1, r and d
        # change by:
        #
        #    row_dr = M[0,1]   row_dd = M[1,1]
        #
        # The length of the vector from (0,0) to (row_dr,row_dd) is
        # the angular width of pixels along the Y axis.
        #
        #    dy = sqrt(M[0,1]^2 + M[1,1]^2)
        #
        # Calculate the width and height of the pixels as described above.

        cd = self.get_cd()
        dx = np.sqrt(cd[0, 0]**2 + cd[1, 0]**2)
        dy = np.sqrt(cd[0, 1]**2 + cd[1, 1]**2)

        steps = np.array([dy, dx])

        if unit is not None:
            steps = (steps * self.unit).to(unit).value

        return steps

    def get_axis_increments(self, unit=None):
        """Return the displacements on the sky that result from
        incrementing the array indexes of the image by one along the Y
        and X axes, respectively.

        In MPDAF, images are sampled on a regular grid of square pixels that
        represent a flat projection of the celestial sphere. The
        get_axis_increments() method returns the angular width and height of
        these pixels on the sky, with signs that indicate whether the angle
        increases or decreases as one increments the array indexes. To keep
        plots consistent, regardless of the rotation angle of the image on the
        sky, the returned height is always positive, but the returned width is
        negative if a plot of the image with pixel 0,0 at the bottom left would
        place east anticlockwise of north, and positive otherwise.

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
        # Get the axis increments that are configured by the CD matrix.
        cd = self.get_cd()
        increments = axis_increments_from_cd(cd)

        if unit is not None:
            increments = (increments * self.unit).to(unit).value

        return increments

    def get_range(self, unit=None):
        """Return the minimum and maximum right-ascensions and declinations
        in the image array.

        Specifically a list is returned with the following contents::

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
        pixcrd = [[0, 0], [self.naxis2 - 1, 0], [0, self.naxis1 - 1],
                  [self.naxis2 - 1, self.naxis1 - 1]]
        pixsky = self.pix2sky(pixcrd, unit=unit)
        return np.hstack([pixsky.min(axis=0), pixsky.max(axis=0)])

    def get_start(self, unit=None):
        """Return the [dec,ra] coordinates of pixel (0,0).

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The angular units of the returned coordinates.

        Returns
        -------
        out : numpy.ndarray
           The equatorial coordinate of pixel [0,0], ordered as:
           [dec,ra]. If a value was given to the optional 'unit'
           argument, the angular unit specified there will be used for
           the return value. Otherwise the unit stored in the
           self.unit property will be used.

        """
        return self.pix2sky([0, 0], unit=unit)[0]

    def get_end(self, unit=None):
        """Return the [dec,ra] coordinates of pixel (-1,-1).

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The angular units of the returned coordinates.

        Returns
        -------
        out : numpy.ndarray
           The equatorial coordinate of pixel [-1,-1], ordered as,
           [dec,ra]. If a value was given to the optional 'unit'
           argument, the angular unit specified there will be used for
           the return value. Otherwise the unit stored in the
           self.unit property will be used.

        """
        return self.pix2sky([self.naxis2 - 1, self.naxis1 - 1], unit=unit)[0]

    def get_rot(self, unit=u.deg):
        """Return the rotation angle of the image.

        The angle is defined such that a rotation angle of zero aligns north
        along the positive Y axis, and a positive rotation angle rotates north
        away from the Y axis, in the sense of a rotation from north to east.

        Note that the rotation angle is defined in a flat map-projection of the
        sky. It is what would be seen if the pixels of the image were drawn
        with their pixel widths scaled by the angular pixel increments returned
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
        cd = self.get_cd()
        return image_angle_from_cd(cd, unit)

    def get_cd(self):
        """Return the coordinate conversion matrix (CD).

        This is a 2x2 matrix that can be used to convert from the column and
        row indexes of a pixel in the image array to a coordinate within a flat
        map-projection of the celestial sphere. For example, if the celestial
        coordinates of the observation are right-ascension and declination, and
        r and d denote their gnonomic TAN projection onto a flat plane, then
        a pixel at row and column [col,row] has [r,d] coordinates given by::

            (r,d) = np.dot(get_cd(), (col - get_crpix1(), row - get_crpix2())

        Returns
        -------
        out : nump.ndarray
           A 2D array containing the coordinate transformation matrix,
           arranged such that the elements described in the FITS
           standard are arranged as follows::

               [[CD_11, CD_12]
                [CD_21, CD_22]]

        """

        # The documentation for astropy.wcs.Wcsprm indicates that
        # get_cdelt() and get_pc() work:
        #
        # "even when the header specifies the linear transformation
        #  matrix in one of the alternative CDi_ja or CROTAia
        #  forms. This is useful when you want access to the linear
        #  transformation matrix, but don't care how it was specified
        #  in the header."
        #
        # So to ensure that get_cd() always returns the current CD
        # matrix, get CDELT and PC and convert them to the equivalent
        # CD matrix. Note that self.wcs.wcs.piximg_matrix is not used
        # because sometimes pywcs doesn't create it when a WCS object
        # is not initialized from a FITS header.

        return np.dot(np.diag(self.wcs.wcs.get_cdelt()), self.wcs.wcs.get_pc())

    def get_crpix1(self):
        """Return the value of the FITS CRPIX1 parameter.

        CRPIX1 contains the index of the reference position of the image along
        the X-axis of the image. Beware that this is a FITS array index, which
        is 1 greater than the corresponding python array index. For example,
        a crpix value of 1 denotes a python array index of 0. The reference
        pixel index is a floating point value that can indicate a position
        between two pixels. It can also indicate an index that is outside the
        bounds of the array.

        Returns
        -------
        out : float
           The value of the FITS CRPIX1 parameter.

        """
        return self.wcs.wcs.crpix[0]

    def get_crpix2(self):
        """Return the value of the FITS CRPIX2 parameter.

        CRPIX2 contains the index of the reference position of the image along
        the Y-axis of the image. Beware that this is a FITS array index, which
        is 1 greater than the corresponding python array index. For example,
        a crpix value of 1 denotes a python array index of 0. The reference
        pixel index is a floating point value that can indicate a position
        between two pixels. It can also indicate an index that is outside the
        bounds of the array.

        Returns
        -------
        out : float
           The value of the FITS CRPIX2 parameter.

        """
        return self.wcs.wcs.crpix[1]

    def get_crval1(self, unit=None):
        """Return the value of the FITS CRVAL1 parameter.

        CRVAL1 contains the coordinate reference value of the first image axis
        (eg. right-ascension).

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The angular units to give the return value.

        Returns
        -------
        out : float
           The value of CRVAL1 in the specified angular units. If the
           units are not given, then the unit in the self.unit
           property is used.

        """
        if unit is None:
            return self.wcs.wcs.crval[0]
        else:
            return (self.wcs.wcs.crval[0] * self.unit).to(unit).value

    def get_crval2(self, unit=None):
        """Return the value of the FITS CRVAL2 parameter.

        CRVAL2 contains the coordinate reference value of the second image axis
        (eg. declination).

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The angular units to give the return value.

        Returns
        -------
        out : float
           The value of CRVAL2 in the specified angular units. If the
           units are not given, then the unit in the self.unit
           property is used.

        """
        if unit is None:
            return self.wcs.wcs.crval[1]
        else:
            return (self.wcs.wcs.crval[1] * self.unit).to(unit).value

    @property
    def unit(self):
        """Return the default angular unit used for sky coordinates.

        Returns
        -------
        out : `astropy.units.Unit`
           The unit to use for coordinate angles.

        """
        if self.wcs.wcs.cunit[0] != self.wcs.wcs.cunit[1]:
            self._logger.warning('different units on x- and y-axes')
        return self.wcs.wcs.cunit[0]

    def set_cd(self, cd):
        """Install a new coordinate transform matrix.

        This is a 2x2 matrix that is used to convert from the row and column
        indexes of a pixel in the image array to a coordinate within a flat
        map-projection of the celestial sphere. It is formerly described in the
        FITS standard. The matrix should be ordered like::

            cd = numpy.array([[CD1_1, CD1_2],
                              [CD2_1, CD2_2]]),

        where CDj_i are the names of the corresponding FITS keywords.

        Parameters
        ----------
        cd : numpy.ndarray
            The 2x2 coordinate conversion matrix, with its elements
            ordered for multiplying a column vector in FITS (x,y) axis order.

        """

        # Wcslib supports three different ways to configure the linear
        # coordinate transformation. Once one of these has been
        # established by reading in a FITS header, it can't reliably
        # be changed by configuring one of the other methods, so
        # record the specified CD matrix in the currently established format.

        if self.wcs.wcs.has_pc():      # PC matrix + CDELT
            self.wcs.wcs.pc = cd
            self.wcs.wcs.cdelt = np.array([1.0, 1.0])
        elif self.wcs.wcs.has_cd():    # CD matrix
            self.wcs.wcs.cd = cd
        elif self.wcs.wcs.has_crota():  # CROTA + CDELT
            self.wcs.wcs.crota = [0., image_angle_from_cd(cd, u.deg)]
            self.wcs.wcs.cdelt = axis_increments_from_cd(cd)[::-1]

        self.wcs.wcs.set()

    def set_crpix1(self, x):
        """Set the value of the FITS CRPIX1 parameter.

        This sets the reference pixel index along the X-axis of the image.

        This is a floating point value which can denote a position between
        pixels. It is specified with the FITS indexing convention, where FITS
        pixel 1 is equivalent to pixel 0 in python arrays. In general subtract
        1 from x to get the corresponding floating-point pixel index along axis
        1 of the image array.  In cases where x is an integer, the
        corresponding row in the python data array that contains the image is
        ``data[:, x-1]``.

        Parameters
        ----------
        x : float
            The index of the reference pixel along the X axis.

        """
        self.wcs.wcs.crpix[0] = x
        self.wcs.wcs.set()

    def set_crpix2(self, y):
        """Set the value of the FITS CRPIX2 parameter.

        This sets the reference pixel index along the Y-axis of the image.

        This is a floating point value which can denote a position between
        pixels. It is specified with the FITS indexing convention, where FITS
        pixel 1 is equivalent to pixel 0 in python arrays. In general subtract
        1 from y to get the corresponding floating-point pixel index along axis
        0 of the image array.  In cases where y is an integer, the
        corresponding column in the python data array that contains the image
        is ``data[y-1, :]``.

        Parameters
        ----------
        y : float
            The index of the reference pixel along the Y axis.

        """
        self.wcs.wcs.crpix[1] = y
        self.wcs.wcs.set()

    def set_crval1(self, x, unit=None):
        """Set the value of the CRVAL1 keyword.

        It indicates the coordinate reference value along the first image axis
        (eg. right ascension).

        Parameters
        ----------
        x : float
            The value of the reference pixel on the first axis.
        unit : `astropy.units.Unit`
            The angular units of the world coordinates.

        """
        if unit is None:
            self.wcs.wcs.crval[0] = x
        else:
            self.wcs.wcs.crval[0] = (x * unit).to(self.unit).value
        self.wcs.wcs.set()

    def set_crval2(self, x, unit=None):
        """Set the value of the CRVAL2 keyword.

        It indicates the coordinate reference value along the second image axis
        (eg. declination).

        Parameters
        ----------
        x : float
            The value of the reference pixel on the second axis.
        unit : `astropy.units.Unit`
            The angular units of the world coordinates.

        """
        if unit is None:
            self.wcs.wcs.crval[1] = x
        else:
            self.wcs.wcs.crval[1] = (x * unit).to(self.unit).value
        self.wcs.wcs.set()

    def set_step(self, step, unit=None):
        """Set the height and width of pixels on the sky.

        In MPDAF, images are sampled on a regular grid of square pixels that
        represent a flat projection of the celestial sphere. The set_step()
        method changes the angular width and height of these pixels on the sky.

        Parameters
        ----------
        step : array-like
           (h,w). These are the desired angular height and width of pixels
           along the Y and X axes of the image. These should either be in the
           unit specified by the 'unit' input parameter, or, if unit=None, in
           the unit specified by the self.unit property.
        unit : `astropy.units.Unit`
            The angular units of the specified increments.

        """
        step = abs(np.asarray(step))
        if unit is not None:
            step = (step * unit).to(self.unit).value

        old_step = self.get_step()
        ratio = step / old_step
        cd = self.get_cd()

        # Scaling the 1st column of the CD matrix, scales the X-axis pixel
        # sizes. Scaling the 2nd column scales the Y-axis pixel sizes.
        self.set_cd(np.dot(cd, np.array([[ratio[1], 0.0],
                                         [0.0, ratio[0]]])))

    def set_axis_increments(self, increments, unit=None):
        """Set the displacements on the sky that result from
        incrementing the array indexes of the image by one along the Y
        and X axes, respectively.

        In MPDAF, images are sampled on a regular grid of square pixels that
        represent a flat projection of the celestial sphere. The
        set_axis_increments() method changes the angular width and height of
        these pixels on the sky, with signs that indicate whether the angle
        increases or decreases as one increments the array indexes. To keep
        plots consistent, regardless of the rotation angle of the image on the
        sky, the height should always be positive, and the width should be
        negative if a plot of the image with pixel 0,0 at the bottom left would
        place east anticlockwise of north, and positive otherwise.

        Parameters
        ----------
        increments : numpy.ndarray
           (dy,dx). These are the desired angular increments of pixels
           along the Y and X axes of the image. These should
           either be in the unit specified by the 'unit' input parameter,
           or, if unit=None, in the unit specified by the self.unit property.
        unit : `astropy.units.Unit`
            The angular units of the specified increments.

        """
        increments = np.asarray(increments)
        if unit is not None:
            increments = (increments * unit).to(self.unit).value

        cd = self.get_cd()
        old_increments = axis_increments_from_cd(cd)
        ratio = increments / old_increments

        # Scaling the 1st column of the CD matrix, scales the X-axis pixel
        # sizes. Scaling the 2nd column scales the Y-axis pixel sizes.
        self.set_cd(np.dot(cd, np.array([[ratio[1], 0.0],
                                         [0.0, ratio[0]]])))

    def rotate(self, theta):
        """Rotate WCS coordinates to new orientation given by theta.

        Analog to ``astropy.wcs.WCS.rotateCD``, which is deprecated since
        version 1.3 (see https://github.com/astropy/astropy/issues/5175).

        Parameters
        ----------
        theta : float
            Rotation in degree.

        """
        theta = np.deg2rad(theta)
        sinq = np.sin(theta)
        cosq = np.cos(theta)
        mrot = np.array([[cosq, -sinq],
                         [sinq, cosq]])

        if self.wcs.wcs.has_cd():    # CD matrix
            newcd = np.dot(mrot, self.wcs.wcs.cd)
            self.wcs.wcs.cd = newcd
            self.wcs.wcs.set()
        elif self.wcs.wcs.has_pc():      # PC matrix + CDELT
            newpc = np.dot(mrot, self.wcs.wcs.get_pc())
            self.wcs.wcs.pc = newpc
            self.wcs.wcs.set()
        else:
            raise TypeError("Unsupported wcs type (need CD or PC matrix)")

    def rebin(self, factor):
        """Rebin to a new coordinate system.

        This is a helper function for the Image and Cube rebin() methods.

        Parameters
        ----------
        factor : (integer,integer)
            Factor in y and x.

        Returns
        -------
        out : WCS

        """
        res = self.copy()

        # Record the increased pixel sizes.
        res.set_step(self.get_step() * np.asarray(factor))

        # Compute the new coordinate reference pixel, noting that for
        # the FITS crpix value, the center of the first pixel is
        # defined to be 1, not 0. The original crpix index denotes a
        # pixel that is (crpix-0.5) of the original pixel widths from
        # the start of the first pixel of the original array. This
        # corresponds to (crpix-0.5)/factor new pixel widths from the
        # start of the first pixel, so this has a pixel index of
        # (oldcrpix-0.5)/factor+0.5 in the rebinned array.
        res.wcs.wcs.crpix[0] = (res.wcs.wcs.crpix[0] - 0.5) / factor[1] + 0.5
        res.wcs.wcs.crpix[1] = (res.wcs.wcs.crpix[1] - 0.5) / factor[0] + 0.5

        # Record the new dimensions of the image.
        res.naxis1 = res.naxis1 // factor[1]
        res.naxis2 = res.naxis2 // factor[0]
        res.wcs.wcs.set()

        return res

    def is_deg(self):
        """Return True if world coordinates are in decimal degrees.

        (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg)

        """
        try:
            return self.wcs.wcs.ctype[0] not in ('LINEAR', 'PIXEL')
        except Exception:
            return True

    def to_cube_header(self, wave):
        """Generate an astropy.fits header object with WCS information and
        wavelength information."""
        hdr = self.to_header()
        if wave is not None:
            hdr.update(wave.to_header(naxis=3, use_cd='CD1_1' in hdr))
        return hdr


class WaveCoord(object):

    """WaveCoord class manages world coordinates in spectral direction.

    Parameters
    ----------
    hdr : astropy.fits.CardList
        A FITS header. If hdr is not None, WaveCoord object is created from
        this header and other parameters are not used.
    crpix : float
        Reference pixel coordinates. 1.0 by default. Note that for crpix
        definition, the first pixel in the spectrum has pixel coordinates.
    cdelt : float
        Step in wavelength (1.0 by default).
    crval : float
        Coordinates of the reference pixel (0.0 by default).
    cunit : u.unit
        Wavelength unit (Angstrom by default).
    ctype : string
        Type of the coordinates.
    shape : integer or None
        Size of spectrum (no mandatory).

    Attributes
    ----------
    shape : integer
        Size of spectrum.
    wcs : astropy.wcs.WCS
        Wavelength coordinates.

    """

    def __init__(self, hdr=None, crpix=1.0, cdelt=1.0, crval=1.0,
                 cunit=u.angstrom, ctype='LINEAR', shape=None):
        self._logger = logging.getLogger(__name__)
        self.shape = shape
        self.unit = cunit

        if hdr is not None:
            hdr = hdr.copy()
            try:
                n = hdr['NAXIS']
                self.shape = hdr['NAXIS%d' % n]
            except KeyError:
                n = hdr['WCSAXES']

            axis = 1 if n == 1 else 3
            # Get the unit and remove it from the header so that wcslib does
            # not convert the values.
            self.unit = u.Unit(fix_unit_read(hdr.pop('CUNIT%d' % axis)))
            self.wcs = _wcs_from_header(hdr).sub([axis])
        else:
            self.unit = u.Unit(cunit)
            self.wcs = pywcs.WCS(naxis=1)
            self.wcs.wcs.crpix[0] = crpix
            self.wcs.wcs.cdelt[0] = cdelt
            self.wcs.wcs.ctype[0] = ctype
            self.wcs.wcs.crval[0] = crval
            self.wcs.wcs.set()

    def copy(self):
        """Copie WaveCoord object in a new one and returns it."""
        out = WaveCoord(shape=self.shape, cunit=self.unit)
        out.wcs = self.wcs.deepcopy()
        return out

    def __repr__(self):
        return repr(self.wcs)

    def info(self, unit=None):
        """Print information."""
        unit = unit or self.unit
        start = self.get_start(unit=unit)
        step = self.get_step(unit=unit)

        if self.shape is None:
            self._logger.info('wavelength: min:%0.2f step:%0.2f %s',
                              start, step, unit)
        else:
            end = self.get_end(unit=unit)
            self._logger.info('wavelength: min:%0.2f max:%0.2f step:%0.2f %s',
                              start, end, step, unit)

    def isEqual(self, other):
        """Return True if other and self have the same attributes."""
        if not isinstance(other, WaveCoord):
            return False

        l1 = self.coord(0, unit=self.unit)
        l2 = other.coord(0, unit=self.unit)
        return (self.shape == other.shape and
                np.allclose(l1, l2, atol=1E-2, rtol=0) and
                np.allclose(self.get_step(), other.get_step(unit=self.unit),
                            atol=1E-2, rtol=0) and
                self.wcs.wcs.ctype[0] == other.wcs.wcs.ctype[0])

    def coord(self, pixel=None, unit=None):
        """Return the coordinate corresponding to pixel.

        If pixel is None (default value), the full coordinate array is
        returned.

        Parameters
        ----------
        pixel : int, array or None.
            pixel value.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        Returns
        -------
        out : float or array of float

        """
        if pixel is None and self.shape is None:
            raise IOError("wavelength coordinates without dimension")

        if pixel is None:
            pixelarr = np.arange(self.shape, dtype=float)
        else:
            pixelarr = np.atleast_1d(pixel)

        res = self.wcs.wcs_pix2world(pixelarr, 0)[0]
        if unit is not None:
            res = (res * self.unit).to(unit).value
        return res[0] if np.isscalar(pixel) else res

    def pixel(self, lbda, nearest=False, unit=None):
        """Return the decimal pixel corresponding to the wavelength lbda.

        If nearest=True; returns the nearest integer pixel.

        Parameters
        ----------
        lbda : float or array
            wavelength value.
        nearest : bool
            If nearest is True returns the nearest integer pixel
            in place of the decimal pixel.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        Returns
        -------
        out : float or integer

        """
        lbdarr = np.atleast_1d(lbda)
        if unit is not None:
            lbdarr = UnitArray(lbdarr, unit, self.unit)
        pix = self.wcs.wcs_world2pix(lbdarr, 0)[0]
        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out=pix)
            if self.shape is not None:
                np.minimum(pix, self.shape - 1, out=pix)
        return pix[0] if np.isscalar(lbda) else pix

    def __getitem__(self, item):
        """Return the coordinate corresponding to pixel if item is an integer
        Return the corresponding WaveCoord object if item is a slice."""

        noshape_msg = 'negative index cannot be used without a shape'

        if item is None:
            return self
        elif isinstance(item, int):
            if item >= 0:
                lbda = self.coord(pixel=item)
            else:
                if self.shape is None:
                    raise ValueError(noshape_msg)
                else:
                    lbda = self.coord(pixel=self.shape + item)
            return WaveCoord(crpix=1.0, cdelt=0, crval=lbda,
                             cunit=self.unit, shape=1,
                             ctype=self.wcs.wcs.ctype[0])
        elif isinstance(item, slice):
            if item.start is None:
                start = 0
            elif item.start >= 0:
                start = item.start
            else:
                if self.shape is None:
                    raise ValueError(noshape_msg)
                else:
                    start = self.shape + item.start
            if item.stop is None:
                if self.shape is None:
                    raise ValueError(noshape_msg)
                else:
                    stop = self.shape
            elif item.stop >= 0:
                stop = item.stop
            else:
                if self.shape is None:
                    raise ValueError(noshape_msg)
                else:
                    stop = self.shape + item.stop
            newlbda = self.coord(pixel=np.arange(start, stop, item.step))
            dim = newlbda.shape[0]
            if dim < 2:
                raise ValueError('Spectrum with dim < 2')
            cdelt = newlbda[1] - newlbda[0]
            return WaveCoord(crpix=1.0, cdelt=cdelt, crval=newlbda[0],
                             cunit=self.unit, shape=dim,
                             ctype=self.wcs.wcs.ctype[0])
        else:
            raise ValueError('Operation forbidden')

    def resample(self, step, start, unit=None):
        """Resample to a new coordinate system.

        Parameters
        ----------
        start : float
            New wavelength for the pixel 0.
        step : float
            New step.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        Returns
        -------
        out : WaveCoord

        """
        if unit is not None:
            step = (step * unit).to(self.unit).value
            if start is not None:
                start = (start * unit).to(self.unit).value

        cdelt = self.get_step()
        if start is None:
            pix0 = self.coord(0)
            start = pix0 - 0.5 * cdelt + 0.5 * step

        old_start = self.get_start()
        res = self.copy()
        res.wcs.wcs.crpix[0] = 1.0
        res.wcs.wcs.crval[0] = start
        if self.wcs.wcs.has_cd():
            res.wcs.wcs.cd[0][0] = step
        else:
            res.wcs.wcs.cdelt[0] = 1.0
            res.wcs.wcs.pc[0][0] = step
        res.wcs.wcs.set()
        res.shape = int(np.ceil((self.shape * cdelt - start + old_start) /
                                step))
        return res

    def rebin(self, factor):
        """Rebin to a new coordinate system (in place).

        Parameters
        ----------
        factor : integer
            Factor.

        Returns
        -------
        out : WaveCoord

        """
        old_cdelt = self.get_step()

        if self.wcs.wcs.has_cd():
            self.wcs.wcs.cd = self.wcs.wcs.cd * factor
        else:
            self.wcs.wcs.cdelt = self.wcs.wcs.cdelt * factor
        self.wcs.wcs.set()
        cdelt = self.get_step()

        crpix = self.wcs.wcs.crpix[0]
        crpix = (crpix * old_cdelt - old_cdelt / 2.0 + cdelt / 2.0) / cdelt
        self.wcs.wcs.crpix[0] = crpix
        self.shape = self.shape // factor
        self.wcs.wcs.set()

    def get_step(self, unit=None):
        """Return the step in wavelength.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        """
        if self.wcs.wcs.has_cd():
            step = self.wcs.wcs.cd[0][0]
        else:
            cdelt = self.wcs.wcs.get_cdelt()[0]
            pc = self.wcs.wcs.get_pc()[0, 0]
            step = (cdelt * pc)

        if unit is not None:
            step = (step * self.unit).to(unit).value
        return step

    def set_step(self, x, unit=None):
        """Return the step in wavelength.

        Parameters
        ----------
        x : float
            Step value
        unit : `astropy.units.Unit`
            type of the wavelength coordinates
        """
        if unit is not None:
            step = (x * unit).to(self.unit).value
        else:
            step = x

        if self.wcs.wcs.has_cd():
            self.wcs.wcs.cd[0][0] = step
        else:
            pc = self.wcs.wcs.get_pc()[0, 0]
            self.wcs.wcs.cdelt[0] = step / pc
        self.wcs.wcs.set()

    def get_start(self, unit=None):
        """Return the value of the first pixel.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        """
        return self.coord(0, unit)

    def get_end(self, unit=None):
        """Return the value of the last pixel.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        """
        if self.shape is None:
            raise IOError("wavelength coordinates without dimension")
        else:
            return self.coord(self.shape - 1, unit)

    def get_range(self, unit=None):
        """Return the wavelength range [Lambda_min,Lambda_max].

        Parameters
        ----------
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        """
        if self.shape is None:
            raise IOError("wavelength coordinates without dimension")
        else:
            return self.coord([0, self.shape - 1], unit)

    def get_crpix(self):
        """CRPIX getter (reference pixel on the wavelength axis)."""
        return self.wcs.wcs.crpix[0]

    def set_crpix(self, x):
        """CRPIX setter (reference pixel on the wavelength axis)."""
        self.wcs.wcs.crpix[0] = x
        self.wcs.wcs.set()

    def get_crval(self, unit=None):
        """CRVAL getter (value of the reference pixel on the wavelength axis).

        Parameters
        ----------
        unit : `astropy.units.Unit`
            type of the wavelength coordinates

        """
        if unit is None:
            return self.wcs.wcs.crval[0]
        else:
            return (self.wcs.wcs.crval[0] * self.unit).to(unit).value

    def set_crval(self, x, unit=None):
        """CRVAL getter (value of the reference pixel on the wavelength axis).

        Parameters
        ----------
        x : float
            value of the reference pixel on the wavelength axis
        unit : `astropy.units.Unit`
            type of the wavelength coordinates
        """
        if unit is None:
            self.wcs.wcs.crval[0] = x
        else:
            self.wcs.wcs.crval[0] = (x * unit).to(self.unit).value
        self.wcs.wcs.set()

    def get_ctype(self):
        """Return the type of wavelength coordinates."""
        return self.wcs.wcs.ctype[0]

    def to_header(self, naxis=1, use_cd=False):
        """Generate a astropy.fits header object with the WCS information."""
        hdr = fits.Header()
        hdr['WCSAXES'] = (naxis, 'Number of coordinate axes')
        hdr['CRVAL%d' % naxis] = (self.get_crval(),
                                  'Coordinate value at reference point')
        hdr['CRPIX%d' % naxis] = (self.get_crpix(),
                                  'Pixel coordinate of reference point')
        hdr['CUNIT%d' % naxis] = (self.unit.to_string('fits'),
                                  'Units of coordinate increment and value')
        hdr['CTYPE%d' % naxis] = (self.get_ctype(),
                                  'Coordinate type code')

        if use_cd and naxis == 3:
            hdr['CD3_3'] = self.get_step()
            hdr['CD1_3'] = 0.
            hdr['CD2_3'] = 0.
            hdr['CD3_1'] = 0.
            hdr['CD3_2'] = 0.
        else:
            hdr['CDELT%d' % naxis] = (self.get_step(),
                                      'Coordinate increment at reference '
                                      'point')

        return hdr
