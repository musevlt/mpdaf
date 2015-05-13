"""coords.py Manages coordinates."""

import astropy.wcs as pywcs
import logging
import numpy as np


def deg2sexa(x):
    """Transforms the values of n coordinates from degrees to sexagesimal.

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
    """Transforms the values of n coordinates from sexagesimal to degrees.

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
    """Transforms a degree value to a string representation
    of the coordinate as hours:minutes:seconds.

    Parameters
    ----------
    x : float
        degree value.

    Returns
    -------
    out : string
    """
    from astropy.coordinates import Angle
    ac = Angle(x, unit='degree')
    hms = ac.to_string(unit='hour', sep=':', pad=True)
    return str(hms)


def hms2deg(x):
    """Transforms a string representation of the coordinate
    as hours:minutes:seconds to a float degree value.

    Parameters
    ----------
    x : string
        hours:minutes:seconds

    Returns
    -------
    out : float
    """
    from astropy.coordinates import Angle
    ac = Angle(x, unit='hour')
    deg = float(ac.to_string(unit='degree', decimal=True))
    return deg


def deg2dms(x):
    """Transforms a degree value to a string representation
    of the coordinate as degrees:arcminutes:arcseconds.

    Parameters
    ----------
    x : float
        degree value.

    Returns
    -------
    out : string
    """
    from astropy.coordinates import Angle
    ac = Angle(x, unit='degree')
    dms = ac.to_string(unit='degree', sep=':', pad=True)
    return str(dms)


def dms2deg(x):
    """Transforms a string representation of the coordinate
    as degrees:arcminutes:arcseconds to a float degree value.

    Parameters
    ----------
    x : string
        degrees:arcminutes:arcseconds

    Returns
    -------
    out : float
    """
    from astropy.coordinates import Angle
    ac = Angle(x, unit='degree')
    deg = float(ac.to_string(unit='degree', decimal=True))
    return deg


class WCS(object):

    """WCS class manages world coordinates in spatial direction (pywcs package
    is used). Python notation is used (dec,ra).

    Parameters
    ----------
    hdr   : pyfits.CardList
            A FITS header.
            If hdr is not equal to None, WCS object is created from data header
            and other parameters are not used.
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
    deg   : bool
            If True, world coordinates are in decimal degrees
            (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg).
            If False (by default), world coordinates are linear
            (CTYPE1=CTYPE2='LINEAR').
    rot   : float
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
        """Creates a WCS object.

        Parameters
        ----------
        hdr   : pyfits.CardList
                A FITS header.
                If hdr is not equal to None, WCS object is created from data
                header and other parameters are not used.
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
        deg   : bool
                If True, world coordinates are in decimal degrees
                (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg).
                If False (by default), world coordinates are linear
                (CTYPE1=CTYPE2='LINEAR').
        rot   : float
                Rotation angle in degree.
        shape : integer or (integer,integer)
                Dimensions. No mandatory.
        """
        self.logger = logging.getLogger('mpdaf corelib')
        if hdr is not None:
            self.wcs = pywcs.WCS(hdr, naxis=2)  # WCS object from data header
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
                """check attribute dimensions"""
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
            else:
                if shape is None:
                    self.wcs.wcs.crpix = np.array([1.0, 1.0])
                else:
                    self.wcs.wcs.crpix = \
                        (np.array([shape[1], shape[0]]) + 1) / 2.

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
        """Copies WCS object in a new one and returns it."""
        out = WCS()
        out.wcs = self.wcs.deepcopy()
        out.naxis1 = self.naxis1
        out.naxis2 = self.naxis2
        return out

    def info(self):
        """Prints information."""
        # self.wcs.printwcs()
        d = {'class': 'WCS', 'method': 'info'}
        if not self.is_deg():
            pixcrd = [[0, 0], [self.naxis2 - 1, self.naxis1 - 1]]
            pixsky = self.pix2sky(pixcrd)
            cdelt = self.get_step()
            msg = 'spatial coord: min:(%0.1f,%0.1f) max:(%0.1f,%0.1f) '\
                'step:(%0.1f,%0.1f) rot:%0.1f' % (pixsky[0, 0], pixsky[0, 1],
                                                  pixsky[1, 0], pixsky[1, 1],
                                                  cdelt[0], cdelt[1],
                                                  self.get_rot())
            self.logger.info(msg, extra=d)
        else:
            # center in sexadecimal
            xc = (self.naxis1 - 1) / 2.
            yc = (self.naxis2 - 1) / 2.
            pixsky = self.pix2sky([yc, xc])
            sexa = deg2sexa(pixsky)
            ra = sexa[0][1]
            dec = sexa[0][0]
            # step in arcsec
            cdelt = self.get_step()
            dy = np.abs(cdelt[0] * 3600)
            dx = np.abs(cdelt[1] * 3600)
            sizex = self.naxis1 * dx
            sizey = self.naxis2 * dy
            msg = 'center:(%s,%s) size in arcsec:(%0.3f,%0.3f) '\
                'step in arcsec:(%0.3f,%0.3f) rot:%0.1f' % (dec, ra,
                                                            sizey, sizex,
                                                            dy, dx,
                                                            self.get_rot())
            self.logger.info(msg, extra=d)

    def to_header(self):
        """Generates a pyfits header object with the WCS information."""
        return self.wcs.to_header()

    def sky2pix(self, x, nearest=False):
        """Converts world coordinates (dec,ra) to pixel coordinates.

        If nearest=True; returns the nearest integer pixel.

        Parameters
        ----------
        x       : array
                An (n,2) array of dec- and ra- world coordinates.
        nearest : bool
                If nearest is True returns the nearest integer pixel
                in place of the decimal pixel.

        Returns
        -------
        out : (n,2) array of pixel coordinates.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape == (2,):
            x = x.reshape(1, 2)
        elif len(x.shape) != 2 or x.shape[1] != 2:
            raise IOError('invalid input coordinates for sky2pix')

        ax, ay = self.wcs.wcs_world2pix(x[:, 1], x[:, 0], 0)
        res = np.array([ay, ax]).T

        if nearest:
            res = (res + 0.5).astype(int)
            if self.naxis1 != 0 and self.naxis2 != 0:
                np.minimum(res, [self.naxis2 - 1, self.naxis1 - 1], out=res)
                np.maximum(res, [0, 0], out=res)
        return res

    def pix2sky(self, x):
        """Converts pixel coordinates to world coordinates.

        Parameters
        ----------
        x : array
            An (n,2) array of pixel coordinates (python notation).

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
        return np.array([dec, ra]).T

    def isEqual(self, other):
        """Returns True if other and self have the same attributes."""
        if not isinstance(other, WCS):
            return False

        cdelt1 = self.get_step()
        cdelt2 = other.get_step()
        x1 = self.pix2sky([0, 0])[0]
        x2 = other.pix2sky([0, 0])[0]
        return (self.naxis1 == other.naxis1 and
                self.naxis2 == other.naxis2 and
                np.allclose(x1, x2, atol=1E-3, rtol=0) and
                np.allclose(cdelt1, cdelt2, atol=1E-3, rtol=0) and
                self.get_rot() == other.get_rot())

    def __getitem__(self, item):
        """Returns the corresponding WCS."""
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

    def get_step(self):
        """Returns [dDec,dRa]."""
        try:
            return np.sqrt(np.sum(self.wcs.wcs.cd ** 2, axis=1))[::-1]
        except:
            try:
                cdelt = self.wcs.wcs.get_cdelt()
                pc = self.wcs.wcs.get_pc()
                dx = cdelt[0] * np.sqrt(pc[0, 0] ** 2 + pc[0, 1] ** 2)
                dy = cdelt[1] * np.sqrt(pc[1, 0] ** 2 + pc[1, 1] ** 2)
                return np.array([dy, dx])
            except:
                raise IOError('No standard WCS')

    def get_range(self):
        """Returns [ [dec_min,ra_min], [dec_max,ra_max] ]."""
        pixcrd = [[0, 0], [self.naxis2 - 1, 0], [0, self.naxis1 - 1],
                  [self.naxis2 - 1, self.naxis1 - 1]]
        pixsky = self.pix2sky(pixcrd)
        return np.vstack([pixsky.min(axis=0), pixsky.max(axis=0)])

    def get_start(self):
        """Returns [dec,ra] corresponding to pixel (0,0)."""
        pixcrd = [[0, 0]]
        pixsky = self.pix2sky(pixcrd)
        return np.array([pixsky[0, 0], pixsky[0, 1]])

    def get_end(self):
        """Returns [dec,ra] corresponding to pixel (-1,-1)."""
        pixcrd = [[self.naxis2 - 1, self.naxis1 - 1]]
        pixsky = self.pix2sky(pixcrd)
        return np.array([pixsky[0, 0], pixsky[0, 1]])

    def get_rot(self):
        """Returns the rotation angle."""
        try:
            return np.rad2deg(np.arctan2(self.wcs.wcs.cd[1, 0],
                                         self.wcs.wcs.cd[1, 1]))
        except:
            try:
                pc = self.wcs.wcs.get_pc()
                return np.rad2deg(np.arctan2(pc[1, 0], pc[1, 1]))
                # return np.rad2deg(np.arctan2(self.wcs.wcs.pc[1, 0], \
                #                          self.wcs.wcs.pc[1, 1]))
            except:
                raise IOError('No standard WCS')

    def get_cd(self):
        """Returns the CD matrix."""
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

    def set_crval1(self, x):
        """CRVAL1 setter (value of the reference pixel on the first axis)."""
        self.wcs.wcs.crval[0] = x
        self.wcs.wcs.set()

    def set_crval2(self, x):
        """CRVAL2 setter (value of the reference pixel on the second axis)."""
        self.wcs.wcs.crval[1] = x
        self.wcs.wcs.set()

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

    def get_crval1(self):
        """CRVAL1 getter (value of the reference pixel on the first axis)."""
        return self.wcs.wcs.crval[0]

    def get_crval2(self):
        """CRVAL2 getter (value of the reference pixel on the second axis)."""
        return self.wcs.wcs.crval[1]

    def rotate(self, theta):
        """Rotates WCS coordinates to new orientation given by theta.

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

    def rebin(self, step, start):
        """Rebins to a new coordinate system.

        Parameters
        ----------
        start : float or (float, float)
                New positions (dec,ra) for the pixel (0,0).
                If None, old position is used.
        step  : float or (float, float)
                New step (ddec,dra).

        Returns
        -------
        out : WCS
        """
        if start is None:
            xc = 0
            yc = 0
            pixsky = self.pix2sky([xc, yc])
            cdelt = self.get_step()
            start = (pixsky[0][0] - 0.5 * cdelt[0] + 0.5 * step[0],
                     pixsky[0][1] - 0.5 * cdelt[1] + 0.5 * step[1])

        res = WCS(crpix=1.0, crval=start, cdelt=step, deg=self.is_deg(),
                  rot=self.get_rot())
        return res

    def new_step(self, factor):
        try:
            self.wcs.wcs.cd[0, :] *= factor[1]
            self.wcs.wcs.cd[1, :] *= factor[0]
            self.wcs.wcs.set()
        except:
            try:
                self.wcs.wcs.cdelt[0] *= factor[1]
                self.wcs.wcs.cdelt[1] *= factor[0]
                self.wcs.wcs.set()
            except:
                raise StandardError("problem in wcs rebin")

    def rebin_factor(self, factor):
        """Rebins to a new coordinate system.

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
                raise StandardError("problem in wcs rebin")
        res.wcs.wcs.set()
        old_cdelt = self.get_step()
        cdelt = res.get_step()

        crpix = res.wcs.wcs.crpix
        crpix[0] = (crpix[0] * old_cdelt[1] - old_cdelt[1] / 2.0
                    + cdelt[1] / 2.0) / cdelt[1]
        crpix[1] = (crpix[1] * old_cdelt[0] - old_cdelt[0] / 2.0
                    + cdelt[0] / 2.0) / cdelt[0]
        res.wcs.wcs.crpix = crpix
        res.naxis1 = res.naxis1 / factor[1]
        res.naxis2 = res.naxis2 / factor[0]
        res.wcs.wcs.set()

        return res

    def is_deg(self):
        """Returns True if world coordinates are in decimal degrees
        (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg).
        """
        try:
            return self.wcs.wcs.ctype[0] not in ('LINEAR', 'PIXEL')
        except:
            return True


class WaveCoord(object):

    """WaveCoord class manages world coordinates in spectral direction.

    Parameters
    ----------
    crpix : float
            Reference pixel coordinates. 1.0 by default.

            Note that for crpix definition, the first pixel in the spectrum
            has pixel coordinates.
    cdelt : float
            Step in wavelength (1.0 by default).
    crval : float
            Coordinates of the reference pixel (0.0 by default).
    cunit : string
            Wavelength unit (Angstrom by default).
    shape : integer or None
            Size of spectrum (no mandatory).

    Attributes
    ----------
    dim   : integer
            Size of spectrum.
    crpix : float
            Reference pixel coordinates. Note that for crpix definition, the
            first pixel in the spectrum has pixel coordinates 1.0.
    crval : float
            Coordinates of the reference pixel.
    cdelt : float
            Step in wavelength.
    cunit : string
            Wavelength unit.
    """

    def __init__(self, crpix=1.0, cdelt=1.0, crval=1.0,
                 cunit='Angstrom', shape=None):
        """Creates a WaveCoord object.

        Parameters
        ----------
        crpix : float
                Reference pixel coordinates. 1.0 by default.

                Note that for crpix definition, the first pixel in the spectrum
                has pixel coordinates.
        cdelt : float
                Step in wavelength (1.0 by default).
        crval : float
                Coordinates of the reference pixel (0.0 by default).
        cunit : string
                Wavelength unit (Angstrom by default).
        shape : integer or None
                Size of spectrum (no mandatory).
        """
        self.logger = logging.getLogger('mpdaf corelib')
        self.shape = shape
        self.crpix = crpix
        self.cdelt = cdelt
        self.crval = crval
        self.cunit = cunit

    def copy(self):
        """Copies WaveCoord object in a new one and returns it."""
        return WaveCoord(shape=self.shape, crpix=self.crpix, cdelt=self.cdelt,
                         crval=self.crval, cunit=self.cunit)

    def info(self):
        """Prints information."""
        d = {'class': 'WaveCoord', 'method': 'info'}
        if self.shape is None:
            m = (1 - self.crpix) * self.cdelt + self.crval
            msg = 'wavelength: min:%0.2f step:%0.2f %s' % (m, self.cdelt,
                                                           self.cunit)
        else:
            msg = 'wavelength: min:%0.2f max:%0.2f step:%0.2f %s' % \
                (self.__getitem__(0), self.__getitem__(self.shape - 1),
                 self.cdelt, self.cunit)
        self.logger.info(msg, extra=d)

    def isEqual(self, other):
        """Returns True if other and self have the same attributes."""
        if not isinstance(other, WaveCoord):
            return False
        
        l1 = self.coord(0)
        l2 = other.coord(0)
        return (self.shape == other.shape and
                np.allclose(l1, l2, atol=1E-2, rtol=0) and
                np.allclose(self.cdelt, other.cdelt, atol=1E-2, rtol=0) and
                self.cunit == other.cunit)

    def coord(self, pixel=None):
        """Returns the coordinate corresponding to pixel. If pixel is None
        (default value), the full coordinate array is returned.

        Parameters
        ----------
        pixel : integer, array or None.
                pixel value.

        Returns
        -------
        out : float or array of float
        """
        if pixel is None and self.shape is None:
            raise IOError("wavelength coordinates without dimension")

        pixel = (np.asarray(pixel) if pixel is not None
                 else np.arange(self.shape, dtype=float))
        return (pixel - self.crpix + 1) * self.cdelt + self.crval

    def pixel(self, lbda, nearest=False):
        """ Returns the decimal pixel corresponding to the wavelength lbda.

        If nearest=True; returns the nearest integer pixel.

        Parameters
        ----------
        lbda    : float or array
                wavelength value.
        nearest : bool
                If nearest is True returns the nearest integer pixel
                in place of the decimal pixel.

        Returns
        -------
        out : float or integer
        """

        lbdarr = np.asarray([lbda] if isinstance(lbda, (int, float)) else lbda)
        pix = (lbdarr - self.crval) / self.cdelt + self.crpix - 1
        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out=pix)
            if self.shape is None:
                np.minimum(pix, self.shape - 1, out=pix)

        return pix[0] if isinstance(lbda, (int, float)) else pix

    def __getitem__(self, item):
        """Returns the coordinate corresponding to pixel if item is an integer
        Returns the corresponding WaveCoord object if item is a slice."""
        if self.shape is None:
            raise ValueError('wavelength coordinates without dimension')
        else:
            lbda = self.coord()

        if isinstance(item, int):
            return lbda[item]
        elif isinstance(item, slice):
            newlbda = lbda[item]
            dim = newlbda.shape[0]
            if dim < 2:
                raise ValueError('Spectrum with dim < 2')
            cdelt = newlbda[1] - newlbda[0]
            return WaveCoord(crpix=1.0, cdelt=cdelt, crval=newlbda[0],
                             cunit=self.cunit, shape=dim)
        else:
            raise ValueError('Operation forbidden')

    def rebin(self, step, start):
        """Rebins to a new coordinate system.

        Parameters
        ----------
        start : float
                New wavelength for the pixel 0.
        step  : float
                New step.

        Returns
        -------
        out : WaveCoord
        """
        # vector of pixel edges
        pix = np.arange(self.shape, dtype=np.float)
        lbda = (pix - self.crpix + 1) * self.cdelt + self.crval \
            - 0.5 * self.cdelt
        # vector of new pixel positions
        if start is None:
            start = lbda[0] + step * 0.5
        # pixel number necessary to cover old range
        dim = np.ceil((lbda[-1] + self.cdelt - (start - step * 0.5)) / step)
        res = WaveCoord(crpix=1.0, cdelt=step, crval=start,
                        cunit=self.cunit, shape=int(dim))
        return res

    def get_step(self):
        """Returns the step in wavelength."""
        return self.cdelt

    def get_start(self):
        """Returns the value of the first pixel."""
        return self.coord(0)

    def get_end(self):
        """Returns the value of the last pixel."""
        if self.shape is None:
            raise IOError("wavelength coordinates without dimension")
        else:
            return self.coord(self.shape - 1)

    def get_range(self):
        """Returns the wavelength range [Lambda_min,Lambda_max]."""
        if self.shape is None:
            raise IOError("wavelength coordinates without dimension")
        else:
            return self.coord([0, self.shape - 1])
