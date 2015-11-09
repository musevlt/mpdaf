"""coords.py Manages coordinates."""

from astropy.coordinates import Angle
from astropy.io import fits
import astropy.units as u
import astropy.wcs as pywcs
import logging
import numpy as np

from .objs import is_float, is_int


def deg2sexa(x):
    """Transform the values of n coordinates from degrees to sexagesimal.

    Parameters
    ----------
    x : float array
        An (n,2) array of dec- and ra- coordinates in degrees.

    Returns
    -------
    out : (n,2) array of dec- and ra- coordinates in sexagesimal (string)

    """
    x = np.array(x)
    if len(np.shape(x)) == 1 and np.shape(x)[0] == 2:
        ra = deg2hms(x[1])
        dec = deg2dms(x[0])
        return np.array([dec, ra])
    elif len(np.shape(x)) == 2 and np.shape(x)[1] == 2:
        result = []
        for i in range(np.shape(x)[0]):
            ra = deg2hms(x[i][1])
            dec = deg2dms(x[i][0])
            result.append(np.array([dec, ra]))
        return np.array(result)
    else:
        raise ValueError('Operation forbidden')


def sexa2deg(x):
    """Transform the values of n coordinates from sexagesimal to degrees.

    Parameters
    ----------
    x : string array
        An (n,2) array of dec- and ra- coordinates in sexagesimal.

    Returns
    -------
    out : (n,2) array of dec- and ra- coordinates in degrees.

    """
    x = np.array(x)
    if len(np.shape(x)) == 1 and np.shape(x)[0] == 2:
        ra = hms2deg(x[1])
        dec = dms2deg(x[0])
        return np.array([dec, ra])
    elif len(np.shape(x)) == 2 and np.shape(x)[1] == 2:
        result = []
        for i in range(np.shape(x)[0]):
            ra = hms2deg(x[i][1])
            dec = dms2deg(x[i][0])
            result.append(np.array([dec, ra]))
        return np.array(result)
    else:
        raise ValueError('Operation forbidden')


def deg2hms(x):
    """Transform a degree value to a string representation
    of the coordinate as hours:minutes:seconds.

    Parameters
    ----------
    x : float
        degree value.

    Returns
    -------
    out : string

    """
    ac = Angle(x, unit='degree')
    hms = ac.to_string(unit='hour', sep=':', pad=True)
    return str(hms)


def hms2deg(x):
    """Transform a string representation of the coordinate
    as hours:minutes:seconds to a float degree value.

    Parameters
    ----------
    x : string
        hours:minutes:seconds

    Returns
    -------
    out : float

    """
    ac = Angle(x, unit='hour')
    deg = float(ac.to_string(unit='degree', decimal=True))
    return deg


def deg2dms(x):
    """Transform a degree value to a string representation
    of the coordinate as degrees:arcminutes:arcseconds.

    Parameters
    ----------
    x : float
        degree value.

    Returns
    -------
    out : string

    """
    ac = Angle(x, unit='degree')
    dms = ac.to_string(unit='degree', sep=':', pad=True)
    return str(dms)


def dms2deg(x):
    """Transform a string representation of the coordinate
    as degrees:arcminutes:arcseconds to a float degree value.

    Parameters
    ----------
    x : string
        degrees:arcminutes:arcseconds

    Returns
    -------
    out : float

    """
    ac = Angle(x, unit='degree')
    deg = float(ac.to_string(unit='degree', decimal=True))
    return deg


def wcs_from_header(hdr, naxis=None):
    if 'CD1_1' in hdr and 'CDELT3' in hdr and 'CD3_3' not in hdr:
        hdr['CD3_3'] = hdr['CDELT3']
    if 'PC1_1' in hdr and 'CDELT3' in hdr and 'PC3_3' not in hdr:
        hdr['PC3_3'] = 1
    try:
        # WCS object from data header
        return pywcs.WCS(hdr, naxis=naxis)
    except ValueError as e:
        # Workaround for https://github.com/astropy/astropy/issues/4089
        logger = logging.getLogger(__name__)
        logger.warning('Failed to create WCS object: "%s". Trying to fix the '
                       'header', e)
        for key in ('WAT0_001', 'WAT1_001', 'WAT2_001'):
            if key in hdr:
                del hdr[key]
        return pywcs.WCS(hdr, naxis=naxis)


class WCS(object):

    """WCS class manages world coordinates in spatial direction (pywcs package
    is used). Python notation is used (dec,ra).

    Parameters
    ----------
    hdr : astropy.fits.CardList
        A FITS header. If hdr is not equal to None, WCS object is created from
        data header and other parameters are not used.
    crpix : float or (float,float)
        Reference pixel coordinates.
        If crpix is None and shape is None crpix = 1.0 and
        the reference point is the first pixel of the image.
        If crpix is None and shape is not None crpix = (shape + 1.0)/2.0
        and the reference point is the center of the image.
    crval : float or (float,float)
        Coordinates of the reference pixel (ref_dec,ref_ra).
        (0.0,0.0) by default.
    cdelt : float or (float,float)
        Sizes of one pixel (dDec,dRa). (1.0,1.0) by default.
    deg : bool
        If True, world coordinates are in decimal degrees
        (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg').
        If False (by default), world coordinates are linear
        (CTYPE1=CTYPE2='LINEAR').
    rot : float
        Rotation angle in degree.
    shape : integer or (integer,integer)
        Dimensions. No mandatory.

    Attributes
    ----------
    wcs : pywcs.WCS
        World coordinates.

    """

    def __init__(self, hdr=None, crpix=None, crval=(1.0, 1.0),
                 cdelt=(1.0, 1.0), deg=False, rot=0, shape=None):
        self._logger = logging.getLogger(__name__)
        if hdr is not None:
            self.wcs = wcs_from_header(hdr, naxis=2)
            try:
                self.naxis1 = hdr['NAXIS1']
                self.naxis2 = hdr['NAXIS2']
            except:
                if shape is not None:
                    self.naxis1 = shape[1]
                    self.naxis2 = shape[0]
                else:
                    self.naxis1 = 0
                    self.naxis2 = 0
            # bug if naxis=3
            # http://mail.scipy.org/pipermail/astropy/2011-April/001242.html
        else:
            def check_attrs(val, types=(int, float)):
                """check attribute dimensions."""
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

            # create pywcs object
            self.wcs = pywcs.WCS(naxis=2)

            # reference pixel
            if crpix is not None:
                self.wcs.wcs.crpix = np.array([crpix[1], crpix[0]])
            elif shape is None:
                self.wcs.wcs.crpix = np.array([1.0, 1.0])
            else:
                self.wcs.wcs.crpix = (np.array([shape[1], shape[0]]) + 1) / 2.

            # value of reference pixel
            self.wcs.wcs.crval = np.array([crval[1], crval[0]])
            if deg:  # in decimal degree
                self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
                self.wcs.wcs.cunit = ['deg', 'deg']
                self.wcs.wcs.cd = np.array([[-cdelt[1], 0], [0, cdelt[0]]])
            else:   # in pixel or arcsec
                self.wcs.wcs.ctype = ['LINEAR', 'LINEAR']
                self.wcs.wcs.cunit = ['pixel', 'pixel']
                self.wcs.wcs.cd = np.array([[cdelt[1], 0], [0, cdelt[0]]])

            # rotation
            self.wcs.rotateCD(-rot)
            self.wcs.wcs.set()

            # dimensions
            if shape is not None:
                self.naxis1 = shape[1]
                self.naxis2 = shape[0]
            else:
                self.naxis1 = 0
                self.naxis2 = 0

    def copy(self):
        """Copy WCS object in a new one and returns it."""
        out = WCS()
        out.wcs = self.wcs.deepcopy()
        out.naxis1 = self.naxis1
        out.naxis2 = self.naxis2
        return out

    def info(self):
        """Print information."""
        try:
            dy, dx = self.get_step(unit=u.arcsec)
            sizex = dx * self.naxis1  # ra
            sizey = dy * self.naxis2  # dec
            # center in sexadecimal
            xc = (self.naxis1 - 1) / 2.
            yc = (self.naxis2 - 1) / 2.
            pixsky = self.pix2sky([yc, xc], unit=u.deg)
            sexa = deg2sexa(pixsky)
            ra = sexa[0][1]
            dec = sexa[0][0]
            self._logger.info('center:(%s,%s) size in arcsec:(%0.3f,%0.3f) '
                              'step in arcsec:(%0.3f,%0.3f) rot:%0.1f deg',
                              dec, ra, sizey, sizex, dy, dx, self.get_rot())
        except:
            pixcrd = [[0, 0], [self.naxis2 - 1, self.naxis1 - 1]]
            pixsky = self.pix2sky(pixcrd)
            dy, dx = self.get_step()
            self._logger.info(
                'spatial coord (%s): min:(%0.1f,%0.1f) max:(%0.1f,%0.1f) '
                'step:(%0.1f,%0.1f) rot:%0.1f deg', self.unit,
                pixsky[0, 0], pixsky[0, 1], pixsky[1, 0], pixsky[1, 1],
                dy, dx, self.get_rot())

    def to_header(self):
        """Generate a astropy.fits header object with the WCS information."""
        has_cd = self.wcs.wcs.has_cd()
        hdr = self.wcs.to_header()
        if has_cd:
            for c in ['1_1', '1_2', '2_1', '2_2']:
                try:
                    val = hdr['PC' + c]
                    del hdr['PC' + c]
                except KeyError:
                    if c == '1_1' or c == '2_2':
                        val = 1.
                    else:
                        val = 0.
                hdr['CD' + c] = val
        return hdr

    def sky2pix(self, x, nearest=False, unit=None):
        """Convert world coordinates (dec,ra) to pixel coordinates.

        If nearest=True; returns the nearest integer pixel.

        Parameters
        ----------
        x : array
            An (n,2) array of dec- and ra- world coordinates.
        nearest : bool
            If nearest is True returns the nearest integer pixel
            in place of the decimal pixel.
        unit : astropy.units
            type of the world coordinates

        Returns
        -------
        out : (n,2) array of pixel coordinates.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape == (2,):
            x = x.reshape(1, 2)
        elif len(x.shape) != 2 or x.shape[1] != 2:
            raise IOError('invalid input coordinates for sky2pix')

        if unit is not None:
            x[:, 1] = (x[:, 1] * unit).to(self.unit).value
            x[:, 0] = (x[:, 0] * unit).to(self.unit).value

        ax, ay = self.wcs.wcs_world2pix(x[:, 1], x[:, 0], 0)
        res = np.array([ay, ax]).T

        if nearest:
            res = (res + 0.5).astype(int)
            if self.naxis1 != 0 and self.naxis2 != 0:
                np.minimum(res, [self.naxis2 - 1, self.naxis1 - 1], out=res)
                np.maximum(res, [0, 0], out=res)
        return res

    def pix2sky(self, x, unit=None):
        """Convert pixel coordinates to world coordinates.

        Parameters
        ----------
        x : array
            An (n,2) array of pixel coordinates (python notation).
        unit : astropy.units
            type of the world coordinates

        Returns
        -------
        out : (n,2) array of dec- and ra- world coordinates.

        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape == (2,):
            x = x.reshape(1, 2)
        elif len(x.shape) != 2 or x.shape[1] != 2:
            raise IOError('invalid input coordinates for pix2sky')

        ra, dec = self.wcs.wcs_pix2world(x[:, 1], x[:, 0], 0)
        if unit is not None:
            ra = (ra * self.unit).to(unit).value
            dec = (dec * self.unit).to(unit).value

        return np.array([dec, ra]).T

    def isEqual(self, other):
        """Return True if other and self have the same attributes."""
        if not isinstance(other, WCS):
            return False

        cdelt1 = self.get_step()
        cdelt2 = other.get_step(unit=self.unit)
        x1 = self.pix2sky([0, 0])[0]
        x2 = other.pix2sky([0, 0], unit=self.unit)[0]
        return (self.naxis1 == other.naxis1 and
                self.naxis2 == other.naxis2 and
                np.allclose(x1, x2, atol=1E-3, rtol=0) and
                np.allclose(cdelt1, cdelt2, atol=1E-3, rtol=0) and
                np.allclose(self.get_rot(), other.get_rot(), atol=1E-3,
                            rtol=0))

    def sameStep(self, other):
        """Return True if other and self have the same steps."""
        if not isinstance(other, WCS):
            return False

        cdelt1 = self.get_step()
        cdelt2 = other.get_step(unit=self.unit)
        return np.allclose(cdelt1, cdelt2, atol=1E-7, rtol=0)

    def __getitem__(self, item):
        """Return the corresponding WCS."""
        if isinstance(item, tuple) and len(item) == 2:
            try:
                if item[1].start is None:
                    imin = 0
                else:
                    imin = int(item[1].start)
                    if imin < 0:
                        imin = self.naxis1 + imin
                    if imin > self.naxis1:
                        imin = self.naxis1
                if item[1].stop is None:
                    imax = self.naxis1
                else:
                    imax = int(item[1].stop)
                    if imax < 0:
                        imax = self.naxis1 + imax
                    if imax > self.naxis1:
                        imax = self.naxis1
            except:
                imin = int(item[1])
                imax = int(item[1] + 1)
            try:
                if item[0].start is None:
                    jmin = 0
                else:
                    jmin = int(item[0].start)
                    if jmin < 0:
                        jmin = self.naxis2 + jmin
                    if jmin > self.naxis2:
                        jmin = self.naxis2
                if item[0].stop is None:
                    jmax = self.naxis2
                else:
                    jmax = int(item[0].stop)
                    if jmax < 0:
                        jmax = self.naxis2 + jmax
                        if jmax > self.naxis2:
                            jmax = self.naxis2
            except:
                jmin = int(item[0])
                jmax = int(item[0] + 1)

            crpix = (self.wcs.wcs.crpix[0] - imin,
                     self.wcs.wcs.crpix[1] - jmin)

            res = self.copy()
            res.wcs.wcs.crpix = np.array(crpix)
            res.naxis1 = int(imax - imin)
            res.naxis2 = int(jmax - jmin)

            res.wcs.wcs.set()

            return res
        else:
            raise ValueError('Operation forbidden')

    def get_step(self, unit=None):
        """Return [dDec,dRa].

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        """
        try:
            dy, dx = np.sqrt(np.sum(self.wcs.wcs.cd ** 2, axis=1))[::-1]
        except:
            try:
                cdelt = self.wcs.wcs.get_cdelt()
                pc = self.wcs.wcs.get_pc()
                dx = cdelt[0] * np.sqrt(pc[0, 0] ** 2 + pc[0, 1] ** 2)
                dy = cdelt[1] * np.sqrt(pc[1, 0] ** 2 + pc[1, 1] ** 2)
            except:
                raise IOError('No standard WCS')

        if unit:
            dx = (dx * self.unit).to(unit).value
            dy = (dy * self.unit).to(unit).value
        return np.array([dy, dx])

    def get_range(self, unit=None):
        """Return [ [dec_min,ra_min], [dec_max,ra_max] ]

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        """
        pixcrd = [[0, 0], [self.naxis2 - 1, 0], [0, self.naxis1 - 1],
                  [self.naxis2 - 1, self.naxis1 - 1]]
        pixsky = self.pix2sky(pixcrd, unit=unit)
        return np.vstack([pixsky.min(axis=0), pixsky.max(axis=0)])

    def get_start(self, unit=None):
        """Return [dec,ra] corresponding to pixel (0,0).

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        """
        pixcrd = [[0, 0]]
        pixsky = self.pix2sky(pixcrd, unit=unit)
        return np.array([pixsky[0, 0], pixsky[0, 1]])

    def get_end(self, unit=None):
        """Return [dec,ra] corresponding to pixel (-1,-1).

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        """
        pixcrd = [[self.naxis2 - 1, self.naxis1 - 1]]
        pixsky = self.pix2sky(pixcrd, unit=unit)
        return np.array([pixsky[0, 0], pixsky[0, 1]])

    def get_rot(self, unit=u.deg):
        """Return the rotation angle.

        Parameters
        ----------
        unit : astropy.units
            type of the angle coordinate, degree by default

        """
        try:
            theta = np.arctan2(self.wcs.wcs.cd[1, 0], self.wcs.wcs.cd[1, 1])
        except:
            try:
                pc = self.wcs.wcs.get_pc()
                theta = np.arctan2(pc[1, 0], pc[1, 1])
                # return np.rad2deg(np.arctan2(self.wcs.wcs.pc[1, 0], \
                #                          self.wcs.wcs.pc[1, 1]))
            except:
                raise IOError('No standard WCS')
        return (theta * u.rad).to(unit).value

    def get_cd(self):
        """Return the CD matrix."""
        try:
            return self.wcs.wcs.cd
        except:
            try:
                # cd = self.wcs.wcs.pc
                # cd[0,:] *= self.wcs.wcs.cdelt[0]
                # cd[1,:] *= self.wcs.wcs.cdelt[1]
                cdelt = self.wcs.wcs.get_cdelt()
                cd = self.wcs.wcs.get_pc().__copy__()
                cd[0, :] *= cdelt[0]
                cd[1, :] *= cdelt[1]
                return cd
            except:
                raise IOError('No standard WCS')

    def get_naxis1(self):
        """NAXIS1 getter (first dimension of an image)."""
        return self.naxis1

    def get_naxis2(self):
        """NAXIS2 getter (second dimension of an image)."""
        return self.naxis2

    def get_crpix1(self):
        """CRPIX1 getter (reference pixel on the first axis)."""
        return self.wcs.wcs.crpix[0]

    def get_crpix2(self):
        """CRPIX2 getter (reference pixel on the second axis)."""
        return self.wcs.wcs.crpix[1]

    def get_crval1(self, unit=None):
        """CRVAL1 getter (value of the reference pixel on the first axis).

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        """
        if unit is None:
            return self.wcs.wcs.crval[0]
        else:
            return (self.wcs.wcs.crval[0] * self.unit).to(unit).value

    def get_crval2(self, unit=None):
        """CRVAL2 getter (value of the reference pixel on the second axis).

        Parameters
        ----------
        unit : astropy.units
            type of the world coordinates

        """
        if unit is None:
            return self.wcs.wcs.crval[1]
        else:
            return (self.wcs.wcs.crval[1] * self.unit).to(unit).value

    @property
    def unit(self):
        if self.wcs.wcs.cunit[0] != self.wcs.wcs.cunit[1]:
            self._logger.warning('different units on x- and y-axes')
        return self.wcs.wcs.cunit[0]

    def set_naxis1(self, n):
        """NAXIS1 setter (first dimension of an image)."""
        self.naxis1 = n

    def set_naxis2(self, n):
        """NAXIS2 setter (second dimension of an image)."""
        self.naxis2 = n

    def set_crpix1(self, x):
        """CRPIX1 setter (reference pixel on the first axis)."""
        self.wcs.wcs.crpix[0] = x
        self.wcs.wcs.set()

    def set_crpix2(self, x):
        """CRPIX2 setter (reference pixel on the second axis)."""
        self.wcs.wcs.crpix[1] = x
        self.wcs.wcs.set()

    def set_crval1(self, x, unit=None):
        """CRVAL1 setter (value of the reference pixel on the first axis).

        Parameters
        ----------
        x : float
            Value of the reference pixel on the first axis
        unit : astropy.units
            type of the world coordinates

        """
        if unit is None:
            self.wcs.wcs.crval[0] = x
        else:
            self.wcs.wcs.crval[0] = (x * unit).to(self.unit).value
        self.wcs.wcs.set()

    def set_crval2(self, x, unit=None):
        """CRVAL2 setter (value of the reference pixel on the second axis).

        Parameters
        ----------
        x : float
            Value of the reference pixel on the second axis
        unit : astropy.units
            type of the world coordinates
        """
        if unit is None:
            self.wcs.wcs.crval[1] = x
        else:
            self.wcs.wcs.crval[1] = (x * unit).to(self.unit).value
        self.wcs.wcs.set()

    def set_step(self, step, unit=None):
        """Update the step in the CD matrix or in the PC matrix."""
        if unit is not None:
            step[0] = (step[0] * unit).to(self.unit).value
            step[1] = (step[1] * unit).to(self.unit).value

        theta = self.get_rot()
        if np.abs(theta) > 1E-3:
            self.rotate(-theta)
        if self.is_deg():  # in decimal degree
            self.wcs.wcs.cd = np.array([[-step[1], 0], [0, step[0]]])
        else:   # in pixel or arcsec
            self.wcs.wcs.cd = np.array([[step[1], 0], [0, step[0]]])
        self.wcs.wcs.set()
        if np.abs(theta) > 1E-3:
            self.rotate(theta)

    def rotate(self, theta):
        """Rotate WCS coordinates to new orientation given by theta.

        Parameters
        ----------
        theta : float
            Rotation in degree.
        """
        # rotation matrix of -theta
        _theta = np.deg2rad(theta)
        _mrot = np.zeros(shape=(2, 2), dtype=np.double)
        _mrot[0] = (np.cos(_theta), -np.sin(_theta))
        _mrot[1] = (np.sin(_theta), np.cos(_theta))
        try:
            new_cd = np.dot(self.wcs.wcs.cd, _mrot)
            self.wcs.wcs.cd = new_cd
            self.wcs.wcs.set()
        except:
            try:
                # new_pc = np.dot(self.wcs.wcs.pc, _mrot)
                new_pc = np.dot(self.wcs.wcs.get_pc(), _mrot)
                self.wcs.wcs.pc = new_pc
                self.wcs.wcs.set()
            except:
                raise StandardError("problem with wcs rotation")

    def resample(self, step, start, unit=None):
        """Resample to a new coordinate system.

        Parameters
        ----------
        start : float or (float, float)
            New positions (dec,ra) for the pixel (0,0).
            If None, old position is used.
        step : float or (float, float)
            New step (ddec,dra).
        unit : astropy.units
            type of the world coordinates for the start and step parameters.

        Returns
        -------
        out : WCS

        """
        if unit is not None:
            step[0] = (step[0] * unit).to(self.unit).value
            step[1] = (step[1] * unit).to(self.unit).value
            if start is not None:
                start[0] = (start[0] * unit).to(self.unit).value
                start[1] = (start[1] * unit).to(self.unit).value

        cdelt = self.get_step()
        if start is None:
            xc = 0
            yc = 0
            pixsky = self.pix2sky([xc, yc])
            start = (pixsky[0][0] - 0.5 * cdelt[0] + 0.5 * step[0],
                     pixsky[0][1] - 0.5 * cdelt[1] + 0.5 * step[1])

        old_start = self.get_start()
        res = self.copy()
        res.set_crpix1(1.0)
        res.set_crpix2(1.0)
        res.set_crval1(start[1], unit=None)
        res.set_crval2(start[0], unit=None)
        res.set_step(step, unit=None)
        res.naxis1 = int(np.ceil((self.naxis1 * cdelt[1] - start[1] +
                                  old_start[1]) / step[1]))
        res.naxis2 = int(np.ceil((self.naxis2 * cdelt[0] - start[0] +
                                  old_start[0]) / step[0]))
        return res

    def rebin(self, factor):
        """Rebin to a new coordinate system.

        Parameters
        ----------
        factor : (integer,integer)
            Factor in y and x.

        Returns
        -------
        out : WCS
        """
        res = self.copy()
        factor = np.array(factor)

        try:
            cd = res.wcs.wcs.cd
            cd[0, :] *= factor[1]
            cd[1, :] *= factor[0]
            res.wcs.wcs.cd = cd
        except:
            try:
                cdelt = res.wcs.wcs.cdelt
                cdelt[0] *= factor[1]
                cdelt[1] *= factor[0]
                res.wcs.wcs.cdelt = cdelt
            except:
                raise StandardError("problem in wcs rebinning")
        res.wcs.wcs.set()
        old_cdelt = self.get_step()
        cdelt = res.get_step()

        crpix = res.wcs.wcs.crpix
        crpix[0] = (crpix[0] * old_cdelt[1] - old_cdelt[1] / 2.0 +
                    cdelt[1] / 2.0) / cdelt[1]
        crpix[1] = (crpix[1] * old_cdelt[0] - old_cdelt[0] / 2.0 +
                    cdelt[0] / 2.0) / cdelt[0]
        res.wcs.wcs.crpix = crpix
        res.naxis1 = res.naxis1 / factor[1]
        res.naxis2 = res.naxis2 / factor[0]
        res.wcs.wcs.set()

        return res

    def is_deg(self):
        """Return True if world coordinates are in decimal degrees
        (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg).
        """
        try:
            return self.wcs.wcs.ctype[0] not in ('LINEAR', 'PIXEL')
        except:
            return True

    def to_cube_header(self, wave):
        """Generate a astropy.fits header object with the WCS information and
        the wavelength information."""
        hdr = self.to_header()
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
            except:
                n = hdr['WCSAXES']

            axis = 1 if n == 1 else 3
            # Get the unit and remove it from the header so that wcslib does
            # not convert the values.
            self.unit = u.Unit(hdr.pop('CUNIT%d' % axis))
            self.wcs = wcs_from_header(hdr).sub([axis])
            if shape is not None:
                self.shape = shape
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
        # remove the  UnitsWarning: The unit 'Angstrom' has been deprecated in
        # the FITS standard.
        out = WaveCoord(shape=self.shape, cunit=self.unit)
        out.wcs = self.wcs.deepcopy()
        return out

    def info(self, unit=None):
        """Print information."""
        unit = unit or self.unit
        start = self.get_start(unit=unit)
        step = self.get_step(unit=unit)

        if self.shape is None:
            msg = 'wavelength: min:%0.2f step:%0.2f %s' % (start, step, unit)
        else:
            end = self.get_end(unit=unit)
            msg = 'wavelength: min:%0.2f max:%0.2f step:%0.2f %s' % (start, end, step, unit)
        self._logger.info(msg)

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
        """Return the coordinate corresponding to pixel. If pixel is None
        (default value), the full coordinate array is returned.

        Parameters
        ----------
        pixel : integer, array or None.
            pixel value.
        unit : astropy.units
            type of the wavelength coordinates

        Returns
        -------
        out : float or array of float

        """
        if pixel is None and self.shape is None:
            raise IOError("wavelength coordinates without dimension")

        if pixel is None:
            pixelarr = np.arange(self.shape, dtype=float)
        elif is_float(pixel) or is_int(pixel):
            pixelarr = np.ones(1) * pixel
        else:
            pixelarr = np.asarray(pixel)
        res = self.wcs.wcs_pix2world(pixelarr, 0)[0]
        if unit is not None:
            res = (res * self.unit).to(unit).value
        return res[0] if isinstance(pixel, (int, float)) else res

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
        unit : astropy.units
            type of the wavelength coordinates

        Returns
        -------
        out : float or integer

        """

        lbdarr = np.asarray([lbda] if isinstance(lbda, (int, float)) else lbda)
        if unit is not None:
            lbdarr = (lbdarr * unit).to(self.unit).value
        pix = self.wcs.wcs_world2pix(lbdarr, 0)[0]
        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out=pix)
            if self.shape is not None:
                np.minimum(pix, self.shape - 1, out=pix)
        return pix[0] if isinstance(lbda, (int, float)) else pix

    def __getitem__(self, item):
        """Return the coordinate corresponding to pixel if item is an integer
        Return the corresponding WaveCoord object if item is a slice."""

        if item is None:
            return self
        elif isinstance(item, int):
            if item >= 0:
                lbda = self.coord(pixel=item)
            else:
                if self.shape is None:
                    raise ValueError('wavelength coordinates without dimension')
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
                    raise ValueError('wavelength coordinates without dimension')
                else:
                    start = self.shape + item.start
            if item.stop is None:
                if self.shape is None:
                    raise ValueError('wavelength coordinates without dimension')
                else:
                    stop = self.shape
            elif item.stop >= 0:
                stop = item.stop
            else:
                if self.shape is None:
                    raise ValueError('wavelength coordinates without dimension')
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
        unit : astropy.units
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
        try:
            res.wcs.wcs.cd[0][0] = step
        except:
            try:
                res.wcs.wcs.cdelt[0] = 1.0
                res.wcs.wcs.pc[0][0] = step
            except:
                raise IOError('No standard WCS')
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

        try:
            self.wcs.wcs.cd = self.wcs.wcs.cd * factor
        except:
            try:
                self.wcs.wcs.cdelt = self.wcs.wcs.cdelt * factor
            except:
                raise StandardError("problem in wcs rebinning")
        self.wcs.wcs.set()
        cdelt = self.get_step()

        crpix = self.wcs.wcs.crpix[0]
        crpix = (crpix * old_cdelt - old_cdelt / 2.0 + cdelt / 2.0) / cdelt
        self.wcs.wcs.crpix[0] = crpix
        self.shape = self.shape / factor
        self.wcs.wcs.set()

    def get_step(self, unit=None):
        """Return the step in wavelength.

        Parameters
        ----------
        unit : astropy.units
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

    def get_start(self, unit=None):
        """Return the value of the first pixel.

        Parameters
        ----------
        unit : astropy.units
            type of the wavelength coordinates

        """
        return self.coord(0, unit)

    def get_end(self, unit=None):
        """Return the value of the last pixel.

        Parameters
        ----------
        unit : astropy.units
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
        unit : astropy.units
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
        """CRPIX1 setter (reference pixel on the first axis)."""
        self.wcs.wcs.crpix[0] = x
        self.wcs.wcs.set()

    def get_crval(self, unit=None):
        """CRVAL getter (value of the reference pixel on the wavelength axis).

        Parameters
        ----------
        x : float
            value of the reference pixel on the wavelength axis
        unit : astropy.units
            type of the wavelength coordinates

        """
        if unit is None:
            return self.wcs.wcs.crval[0]
        else:
            return (self.wcs.wcs.crval[0] * self.unit).to(unit).value

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
