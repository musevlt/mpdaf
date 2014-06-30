""" coords.py Manages coordinates"""
import numpy as np
try:
    from astropy.io import fits as pyfits
except:
    import pyfits
try:
    import astropy.wcs as pywcs
except:
    import pywcs

from astropysics_coords import AstropysicsAngularCoordinate


def deg2sexa(x):
    """Transforms the values of n coordinates from degrees to sexagesimal.

    :param x: An (n,2) array of dec- and ra- coordinates in degrees.
    :type x: float array
    :rtype: (n,2) array of dec- and ra- coordinates in sexagesimal (string)
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

    :param x: An (n,2) array of dec- and ra- coordinates in sexagesimal.
    :type x: string array
    :rtype: (n,2) array of dec- and ra- coordinates in degrees.
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

    :param x: degree value.
    :type x: float
    :rtype: string
    """
    ac = AstropysicsAngularCoordinate(x)
    hms = ac.getHmsStr(canonical=True)
    return hms


def hms2deg(x):
    """Transforms a string representation of the coordinate
    as hours:minutes:seconds to a float degree value.

    :param x: hours:minutes:seconds
    :type x: string
    :rtype: float
    """
    ac = AstropysicsAngularCoordinate(x, sghms=True)
    deg = ac.d
    return deg


def deg2dms(x):
    """Transforms a degree value to a string representation
    of the coordinate as degrees:arcminutes:arcseconds.

    :param x: degree value.
    :type x: float
    :rtype: string
    """
    ac = AstropysicsAngularCoordinate(x)
    dms = ac.getDmsStr(canonical=True)
    return dms


def dms2deg(x):
    """Transforms a string representation of the coordinate
    as degrees:arcminutes:arcseconds to a float degree value.

    :param x: degrees:arcminutes:arcseconds
    :type x: string
    :rtype: float
    """
    ac = AstropysicsAngularCoordinate(x)
    deg = ac.d
    return deg


def deg2rad(deg):
    """Transforms a degree value to a radian value.

    :param deg: degree value.
    :type deg: float
    :rtype: float
    """
    return (deg * np.pi / 180.)


def rad2deg(rad):
    """Transforms a radian value to a degree value.

    :param rad: radian value.
    :type rad: float
    :rtype: float
    """
    return (rad * 180. / np.pi) % 180


class WCS(object):
    """WCS class manages world coordinates in spatial direction
    (pywcs package is used).
    Python notation is used (dec,ra).

    :param hdr: A FITS header.
    If hdr is not equal to None, WCS object is created from data header
    and other parameters are not used.
    :type hdr: pyfits.CardList
    :param crpix: Reference pixel coordinates.

            If crpix is None and shape is None crpix = 1.0 and
            the reference point is the first pixel of the image.

            If crpix is None and shape is not None crpix = (shape + 1.0)/2.0
            and the reference point is the center of the image.
    :type crpix: float or (float,float)
    :param crval: Coordinates of the reference pixel (ref_dec,ref_ra).
    (0.0,0.0) by default.
    :type crval: float or (float,float)
    :param cdelt: Sizes of one pixel (dDec,dRa). (1.0,1.0) by default.
    :type cdelt: float or (float,float)
    :param deg: If True, world coordinates are in decimal degrees
    (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg).
    If False (by default), world coordinates are linear
    (CTYPE1=CTYPE2='LINEAR').
    :type deg: bool
    :param rot: Rotation angle in degree.
    :type rot: float
    :param shape: Dimensions. No mandatory.
    :type shape: integer or (integer,integer)

    Attributes
    ----------

    wcs (pywcs.WCS) : World coordinates.

    """
    def __init__(self, hdr=None, crpix=None, crval=(1.0, 1.0), \
                 cdelt=(1.0, 1.0), deg=False, rot=0, shape=None):
        """Creates a WCS object.

        :param hdr: A FITS header.
        If hdr is not equal to None, WCS object is created from data header
        and other parameters are not used.
        :type hdr: pyfits.CardList
        :param crpix: Reference pixel coordinates.

            If crpix is None and shape is None crpix = 1.0
            and the reference point is the first pixel of the image.

            If crpix is None and shape is not None crpix = (shape + 1.0)/2.0
            and the reference point is the center of the image.
        :type crpix: float or (float,float)
        :param crval: Coordinates of the reference pixel (ref_dec,ref_ra).
        (1.0,1.0) by default.
        :type crval: float or (float,float)
        :param cdelt: Sizes of one pixel (dDec,dRa). (1.0,1.0) by default.
        :type cdelt: float or (float,float)
        :param deg: If True, world coordinates are in decimal degrees
        (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg).

                    If False (by default), world coordinates are linear
                    (CTYPE1=CTYPE2='LINEAR').
        :type deg: bool
        :param rot: Rotation angle in degree.
        :type rot: float
        :param shape: Dimensions. No mandatory.
        :type shape: integer or (integer,integer)

        """
        if hdr != None:
            self.wcs = pywcs.WCS(hdr, naxis=2)  # WCS object from data header
            self.naxis1 = hdr['NAXIS1']
            self.naxis2 = hdr['NAXIS2']
            # bug if naxis=3
            # http://mail.scipy.org/pipermail/astropy/2011-April/001242.html
        else:
            # check attribute dimensions
            if isinstance(crval, int) or isinstance(crval, float):
                crval = (crval, crval)
            elif len(crval) == 2:
                pass
            else:
                raise ValueError('crval with dimension > 2')
            if isinstance(cdelt, int) or isinstance(cdelt, float):
                cdelt = (cdelt, cdelt)
            elif len(cdelt) == 2:
                pass
            else:
                raise ValueError('cdelt with dimension > 2')
            if crpix is not None:
                if isinstance(crpix, int) or isinstance(crpix, float):
                    crpix = (crpix, crpix)
                elif len(crpix) == 2:
                    pass
                else:
                    raise ValueError('crpix with dimension > 2')
            if shape is not None:
                if isinstance(shape, int):
                    shape = (shape, shape)
                elif len(shape) == 2:
                    pass
                else:
                    raise ValueError('shape with dimension > 2')
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
            if shape != None:
                self.naxis1 = shape[1]
                self.naxis2 = shape[0]
            else:
                self.naxis1 = 1
                self.naxis2 = 1

    def copy(self):
        """Copies WCS object in a new one and returns it.
        """
        out = WCS()
        out.wcs = self.wcs.deepcopy()
        out.naxis1 = self.naxis1
        out.naxis2 = self.naxis2
        return out

    def info(self):
        """Prints information.
        """
        # self.wcs.printwcs()
        if not self.is_deg():
            pixcrd = [[0, 0], [self.naxis2 - 1, self.naxis1 - 1]]
            pixsky = self.pix2sky(pixcrd)
            cdelt = self.get_step()
            print 'spatial coord: min:(%0.1f,%0.1f) max:(%0.1f,%0.1f) '\
            'step:(%0.1f,%0.1f) rot:%0.1f' % (pixsky[0, 0], pixsky[0, 1], \
                                             pixsky[1, 0], pixsky[1, 1], \
                                             cdelt[0], cdelt[1], \
                                             self.get_rot())
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
            print 'center:(%s,%s) size in arcsec:(%0.3f,%0.3f) '\
            'step in arcsec:(%0.3f,%0.3f) rot:%0.1f' % (dec, ra, \
                                                       sizey, sizex, \
                                                       dy, dx, \
                                                       self.get_rot())

    def to_header(self):
        """Generates a pyfits header object with the WCS information.
        """
        return self.wcs.to_header()

    def sky2pix(self, x):
        """Converts world coordinates (dec,ra) to pixel coordinates.

        :param x: An (n,2) array of dec- and ra- world coordinates.
        :type x: array
        :rtype: (n,2) array of pixel coordinates.
        """
        x = np.array(x, np.float_)
        if len(np.shape(x)) == 1 and np.shape(x)[0] == 2:
            pixsky = np.array([[x[1], x[0]]])
        elif len(np.shape(x)) == 2 and np.shape(x)[1] == 2:
            pixsky = np.zeros(np.shape(x))
            pixsky[:, 0] = x[:, 1]
            pixsky[:, 1] = x[:, 0]
        else:
            raise InputError('invalid input coordinates for sky2pix')
        try:
            pixcrd = self.wcs.wcs_world2pix(pixsky, 0)
        except:
            pixcrd = self.wcs.wcs_sky2pix(pixsky, 0)
        res = np.array(pixcrd)
        res[:, 0] = pixcrd[:, 1]
        res[:, 1] = pixcrd[:, 0]
        return res

    def pix2sky(self, x):
        """Converts pixel coordinates to world coordinates.

        :param x: An (n,2) array of pixel coordinates (python notation).
        :type x: array
        :rtype: (n,2) array of dec- and ra- world coordinates.
        """
        x = np.array(x, np.float_)
        if len(np.shape(x)) == 1 and np.shape(x)[0] == 2:
            pixcrd = np.array([[x[1], x[0]]])
        elif len(np.shape(x)) == 2 and np.shape(x)[1] == 2:
            pixcrd = np.zeros(np.shape(x))
            pixcrd[:, 0] = x[:, 1]
            pixcrd[:, 1] = x[:, 0]
        else:
            raise InputError('invalid input coordinates for pix2sky')
        try:
            pixsky = self.wcs.wcs_pix2world(pixcrd, 0)
        except:
            pixsky = self.wcs.wcs_pix2sky(pixcrd, 0)
        res = np.array(pixsky)
        res[:, 0] = pixsky[:, 1]
        res[:, 1] = pixsky[:, 0]
        return res

    def isEqual(self, other):
        """Returns True if other and self have the same attributes.
        """
        if isinstance(other, WCS):
            cdelt1 = self.get_step()
            cdelt2 = other.get_step()
            x1 = self.pix2sky([0, 0])[0]
            x2 = other.pix2sky([0, 0])[0]
            if self.naxis1 == other.naxis1 and \
            self.naxis2 == other.naxis2 and \
               np.abs(x1[0] - x2[0]) < 1E-16 and \
               np.abs(x1[1] - x2[1]) < 1E-16 and\
               (cdelt1 == cdelt2).all() and self.get_rot() == other.get_rot():
                return True
            else:
                return False
        else:
            return False

    def __getitem__(self, item):
        """Returns the corresponding WCS.
        """
        if isinstance(item, tuple) and len(item) == 2:
            try:
                if item[1].start is None:
                    imin = 0
                else:
                    imin = item[1].start
                    if imin < 0:
                        imin = self.naxis1 + imin
                    if imin > self.naxis1:
                        imin = self.naxis1
                if item[1].stop is None:
                    imax = self.naxis1
                else:
                    imax = item[1].stop
                    if imax < 0:
                        imax = self.naxis1 + imax
                    if imax > self.naxis1:
                        imax = self.naxis1
            except:
                imin = item[1]
                imax = item[1] + 1
            try:
                if item[0].start is None:
                    jmin = 0
                else:
                    jmin = item[0].start
                    if jmin < 0:
                        jmin = self.naxis2 + jmin
                    if jmin > self.naxis2:
                        jmin = self.naxis2
                if item[0].stop is None:
                    jmax = self.naxis2
                else:
                    jmax = item[0].stop
                    if jmax < 0:
                        jmax = self.naxis2 + jmax
                        if jmax > self.naxis2:
                            jmax = self.naxis2
            except:
                jmin = item[0]
                jmax = item[0] + 1

            crpix = (self.wcs.wcs.crpix[0] - imin, \
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
        """Returns [dDec,dRa].
        """
        try:
            dx = np.sqrt(self.wcs.wcs.cd[0, 0] * self.wcs.wcs.cd[0, 0] \
                         + self.wcs.wcs.cd[0, 1] * self.wcs.wcs.cd[0][1])
            dy = np.sqrt(self.wcs.wcs.cd[1, 0] * self.wcs.wcs.cd[1, 0] \
                         + self.wcs.wcs.cd[1, 1] * self.wcs.wcs.cd[1][1])
            return np.array([dy, dx])
        except:
            try:
                dx = self.wcs.wcs.cdelt[0] \
                * np.sqrt(self.wcs.wcs.pc[0, 0] * self.wcs.wcs.pc[0, 0] \
                          + self.wcs.wcs.pc[0, 1] * self.wcs.wcs.pc[0][1])
                dy = self.wcs.wcs.cdelt[1] \
                * np.sqrt(self.wcs.wcs.pc[1, 0] * self.wcs.wcs.pc[1, 0] \
                          + self.wcs.wcs.pc[1, 1] * self.wcs.wcs.pc[1][1])
                return np.array([dy, dx])
            except:
                raise IOError('No standard WCS')

    def get_range(self):
        """Returns [ [dec_min,ra_min], [dec_max,ra_max] ].
        """
        pixcrd = [[0, 0], [self.naxis2 - 1, 0], [0, self.naxis1 - 1], \
                  [self.naxis2 - 1, self.naxis1 - 1]]
        pixsky = self.pix2sky(pixcrd)
        dec_min = np.min(pixsky[:, 0])
        ra_min = np.min(pixsky[:, 1])
        dec_max = np.max(pixsky[:, 0])
        ra_max = np.max(pixsky[:, 1])
        return np.array([[dec_min, ra_min], [dec_max, ra_max]])

    def get_start(self):
        """Returns [dec,ra] corresponding to pixel (0,0).
        """
        pixcrd = [[0, 0]]
        pixsky = self.pix2sky(pixcrd)
        return np.array([pixsky[0, 0], pixsky[0, 1]])

    def get_end(self):
        """Returns [dec,ra] corresponding to pixel (-1,-1).
        """
        pixcrd = [[self.naxis2 - 1, self.naxis1 - 1]]
        pixsky = self.pix2sky(pixcrd)
        return np.array([pixsky[0, 0], pixsky[0, 1]])

    def get_rot(self):
        """Returns the rotation angle.
        """
        try:
            return rad2deg(np.arctan2(self.wcs.wcs.cd[1, 0], \
                                      self.wcs.wcs.cd[1, 1]))
        except:
            try:
                return rad2deg(np.arctan2(self.wcs.wcs.pc[1, 0], \
                                          self.wcs.wcs.pc[1, 1]))
            except:
                raise IOError('No standard WCS')

    def get_cd(self):
        """Returns the CD matrix.
        """
        try:
            return self.wcs.wcs.cd
        except:
            try:
                cd = self.wcs.wcs.pc
                cd[0,:] *= self.wcs.wcs.cdelt[0]
                cd[1,:] *= self.wcs.wcs.cdelt[1]
                return cd
            except:
                raise IOError('No standard WCS')

    def set_naxis1(self, n):
        """NAXIS1 setter (first dimention of an image).
        """
        self.naxis1 = n

    def set_naxis2(self, n):
        """NAXIS2 setter (second dimention of an image).
        """
        self.naxis2 = n

    def set_crpix1(self, x):
        """CRPIX1 setter (reference pixel on the first axis).
        """
        self.wcs.wcs.crpix[0] = x
        self.wcs.wcs.set()

    def set_crpix2(self, x):
        """CRPIX2 setter (reference pixel on the second axis).
        """
        self.wcs.wcs.crpix[1] = x
        self.wcs.wcs.set()

    def set_crval1(self, x):
        """CRVAL1 setter (value of the reference pixel on the first axis).
        """
        self.wcs.wcs.crval[0] = x
        self.wcs.wcs.set()

    def set_crval2(self, x):
        """CRVAL2 setter (value of the reference pixel on the second axis).
        """
        self.wcs.wcs.crval[1] = x
        self.wcs.wcs.set()
        
    def get_naxis1(self):
        """NAXIS1 getter (first dimention of an image).
        """
        return self.naxis1

    def get_naxis2(self):
        """NAXIS2 getter (second dimention of an image).
        """
        return self.naxis2

    def get_crpix1(self):
        """CRPIX1 getter (reference pixel on the first axis).
        """
        return self.wcs.wcs.crpix[0]

    def get_crpix2(self):
        """CRPIX2 getter (reference pixel on the second axis).
        """
        return self.wcs.wcs.crpix[1]

    def get_crval1(self):
        """CRVAL1 getter (value of the reference pixel on the first axis).
        """
        return self.wcs.wcs.crval[0]

    def get_crval2(self):
        """CRVAL2 getter (value of the reference pixel on the second axis).
        """
        return self.wcs.wcs.crval[1]

    def rotate(self, theta):
        """Rotates WCS coordinates to new orientation given by theta.

        :param theta: Rotation in degree.
        :type theta: float
        """
        #rotation matrix of -theta
        _theta = deg2rad(theta)
        _mrot = np.zeros(shape=(2,2), dtype=np.double)
        _mrot[0] = (np.cos(_theta), -np.sin(_theta))
        _mrot[1] = (np.sin(_theta), np.cos(_theta))
        try:
            new_cd = np.dot(self.wcs.wcs.cd, _mrot)
            self.wcs.wcs.cd = new_cd
            self.wcs.wcs.set()
        except:
            try:
                new_pc = np.dot(self.wcs.wcs.pc, _mrot)
                self.wcs.wcs.pc = new_pc
                self.wcs.wcs.set()
            except:
                raise StandardError("problem with wcs rotation")

    def rebin(self, step, start):
        """Rebins to a new coordinate system.

        :param start: New positions (dec,ra) for the pixel (0,0).
        If None, old position is used.
        :type start: float or (float, float)
        :param step: New step (ddec,dra).
        :type step: float or (float, float)
        :rtype: WCS
        """
        if start == None:
            xc = 0
            yc = 0
            pixsky = self.pix2sky([xc, yc])
            cdelt = self.get_step()
            start = (pixsky[0][0] - 0.5 * cdelt[0] + 0.5 * step[0], \
                     pixsky[0][1] - 0.5 * cdelt[1] + 0.5 * step[1])

        res = WCS(crpix=1.0, crval=start, cdelt=step, deg=self.is_deg(), \
                  rot=self.get_rot())
        return res

    def new_step(self, factor):
        try:
            self.wcs.wcs.cd[0,:] *= factor[1]
            self.wcs.wcs.cd[1,:] *= factor[0]
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

        :param factor: Factor in y and x.
        :type factor: (integer,integer)
        :rtype: WCS
        """
        res = self.copy()
        factor = np.array(factor)

        try:
            cd = res.wcs.wcs.cd
            cd[0,:] *= factor[1]
            cd[1,:] *= factor[0]
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
        crpix[0] = (crpix[0] * old_cdelt[1] - old_cdelt[1] / 2.0 \
                    + cdelt[1] / 2.0) / cdelt[1]
        crpix[1] = (crpix[1] * old_cdelt[0] - old_cdelt[0] / 2.0 \
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
            if self.wcs.wcs.ctype[0] == 'LINEAR' \
            or self.wcs.wcs.ctype[0] == 'PIXEL':
                return False
            else:
                return True
        except:
            return True


class WaveCoord(object):
    """WaveCoord class manages world coordinates in spectral direction.

    :param crpix: Reference pixel coordinates. 1.0 by default.

    Note that for crpix definition, the first pixel in the spectrum
    has pixel coordinates.
    :type crpix: float
    :param cdelt: Step in wavelength (1.0 by default).
    :type cdelt: float
    :param crval: Coordinates of the reference pixel (0.0 by default).
    :type crval: float
    :param cunit: Wavelength unit (Angstrom by default).
    :type cunit: string
    :param shape: Size of spectrum (no mandatory).
    :type shape: integer or None

    Attributes
    ----------

    dim (integer) : Size of spectrum.

    crpix (float) : Reference pixel coordinates. Note that for crpix
    definition, the first pixel in the spectrum has pixel coordinates 1.0.

    crval (float) : Coordinates of the reference pixel.

    cdelt (float) : Step in wavelength.

    cunit (string) : Wavelength unit.
    """

    def __init__(self, crpix=1.0, cdelt=1.0, crval=1.0, \
                 cunit='Angstrom', shape=None):
        """Creates a WaveCoord object

        :param crpix: Reference pixel coordinates.
        1.0 by default. Note that for crpix definition,
        the first pixel in the spectrum has pixel coordinates.
        :type crpix: float
        :param cdelt: Step in wavelength (1.0 by default).
        :type cdelt: float
        :param crval: Coordinates of the reference pixel (1.0 by default).
        :type crval: float
        :param cunit: Wavelength unit (Angstrom by default).
        :type cunit: string
        :param shape: Size of spectrum (no mandatory).
        :type shape: integer or None
        """
        self.shape = shape
        self.crpix = crpix
        self.cdelt = cdelt
        self.crval = crval
        self.cunit = cunit

    def copy(self):
        """Copies WaveCoord object in a new one and returns it.
        """
        out = WaveCoord()
        out.shape = self.shape
        out.crpix = self.crpix
        out.cdelt = self.cdelt
        out.crval = self.crval
        out.cunit = self.cunit
        return out

    def info(self):
        """Prints information.
        """
        if self.shape is None:
            min = (1 - self.crpix) * self.cdelt + self.crval
            print 'wavelength: min:%0.2f step:%0.2f %s' % (min, self.cdelt, \
                                                           self.cunit)
        else:
            print 'wavelength: min:%0.2f max:%0.2f step:%0.2f %s' % \
            (self.__getitem__(0), self.__getitem__(self.shape - 1), \
             self.cdelt, self.cunit)

    def isEqual(self, other):
        """Returns True if other and self have the same attributes.
        """
        if isinstance(other,WaveCoord):
            if self.crpix == other.crpix and self.cdelt == other.cdelt and \
               self.crval == other.crval and self.cunit == other.cunit and \
               self.shape == other.shape:
                return True
            else:
                return False
        else:
            return False

    def coord(self, pixel=None):
        """Returns the coordinate corresponding to pixel. If pixel is None
        (default value), the full coordinate array is returned.

        :param pixel: pixel value.
        :type pixel: integer or None.
        :rtype: float or array of float
        """
        if pixel is None:
            if self.shape is None:
                raise IOError("wavelength coordinates without dimension")
            else:
                pix = np.arange(self.shape, dtype=np.float)
                lbda = (pix - self.crpix + 1) * self.cdelt + self.crval
                return lbda
        else:
            pixel = np.array(pixel)
            return (pixel - self.crpix + 1) * self.cdelt + self.crval

    def pixel(self, lbda, nearest=False):
        """ Returns the decimal pixel corresponding to the wavelength lbda.
        If nearest=True; returns the nearest integer pixel.

        :param lbda: wavelength value.
        :type lbda: float
        :param nearest: If nearest is True returns the nearest integer pixel
        in place of the decimal pixel.
        :type nearest: bool
        :rtype: float or integer
        """
        lbda = np.array(lbda)
        pix = (lbda - self.crval) / self.cdelt + self.crpix - 1
        if nearest:
            if self.shape is None:
                pix = max(int(pix + 0.5), 0)
            else:
                try:
                    pix = min(max(int(pix + 0.5), 0), self.shape - 1)
                except:
                    for i in range(len(pix)):
                        pix[i] = min(max(int(pix[i] + 0.5), 0), self.shape - 1)
        return pix

    def __getitem__(self, item):
        """ Returns the coordinate corresponding to pixel if item
        is an integer
            Returns the corresponding WaveCoord object if item is a slice
        """
        if self.shape is None:
            raise ValueError('wavelength coordinates without dimension')
        else:
            lbda = (np.arange(self.shape, dtype=np.float) - self.crpix + 1)\
             * self.cdelt + self.crval
        if isinstance(item, int):
            return lbda[item]
        elif isinstance(item, slice):
            newlbda = lbda[item]
            dim = newlbda.shape[0]
            if dim < 2:
                raise ValueError('Spectrum with dim < 2')
            cdelt = newlbda[1] - newlbda[0]
            res = WaveCoord(crpix=1.0, cdelt=cdelt, crval=newlbda[0], \
                            cunit=self.cunit, shape=dim)
            return res
        else:
            raise ValueError('Operation forbidden')

    def rebin(self, step, start):
        """Rebins to a new coordinate system.

        :param start: New wavelength for the pixel 0.
        :type start: float
        :param step: New step.
        :type step: float
        :rtype: WaveCoord
        """
        # vector of pixel edges
        pix = np.arange(self.shape, dtype=np.float)
        lbda = (pix - self.crpix + 1) * self.cdelt + self.crval \
        - 0.5 * self.cdelt
        # vector of new pixel positions
        if start == None:
            start = lbda[0] + step * 0.5
        # pixel number necessary to cover old range
        dim = np.ceil((lbda[-1] + self.cdelt - (start - step * 0.5)) / step)
        res = WaveCoord(crpix=1.0, cdelt=step, crval=start, \
                        cunit=self.cunit, shape=int(dim))
        return res

    def get_step(self):
        """Returns the step in wavelength.
        """
        return self.cdelt

    def get_start(self):
        """Returns the value of the first pixel.
        """
        return self.coord(0)

    def get_end(self):
        """Returns the value of the last pixel.
        """
        if self.shape is None:
            raise IOError("wavelength coordinates without dimension")
        else:
            return self.coord(self.shape - 1)

    def get_range(self):
        """Returns the wavelength range [Lambda_min,Lambda_max].
        """
        if self.shape is None:
            raise IOError("wavelength coordinates without dimension")
        else:
            return self.coord([0, self.shape - 1])
