"""pixtable.py Manages MUSE pixel table files."""


import ctypes
import datetime
import itertools
import logging
import os.path
import numpy as np
import warnings
from scipy import interpolate, ndimage

import astropy.units as u
from astropy.io import fits as pyfits
from astropy.io.fits import Column, ImageHDU

from ..obj import Image, Spectrum, WaveCoord, WCS
from ..obj.objs import is_float, is_int
from ..tools.fits import add_mpdaf_method_keywords

try:
    import numexpr
except:
    numexpr = False


class PixTableMask(object):

    """PixTableMask class.

    This class manages input/output for MUSE pixel mask files

    Parameters
    ----------
    filename : string or None
               Name of the FITS table containing the masked column.
               If PixTableMask object is loaded from a FITS file,
               the others parameters are not read but loaded from
               the FITS file.
    maskfile : string or None
               Name of the FITS image masking some objects.
    maskcol  : array of boolean or None
               pixtable's column corresponding to the mask
    pixtable : string or None
               Name of the corresponding pixel table.

    Attributes
    ----------
    filename : string
               Name of the FITS table containing the masked column.
    maskfile : string
               Name of the FITS image masking some objects.
    maskcol  : array of boolean
               pixtable's column corresponding to the mask
    pixtable : string
               Name of the corresponding pixel table.
    """

    def __init__(self, filename=None, maskfile=None, maskcol=None,
                 pixtable=None):
        """Create a PixTableMask object.

        Parameters
        ----------
        filename : string or None
                   Name of the FITS table containing the masked column.
                   If PixTableMask object is loaded from a FITS file,
                   the others parameters are not read but loaded from
                   the FITS file.
        maskfile : string or None
                   Name of the FITS image masking some objects.
        maskcol  : array of boolean or None
                   pixtable's column corresponding to the mask
        pixtable : string or None
                   Name of the corresponding pixel table.
        """
        if filename is None:
            self.maskfile = maskfile
            self.maskcol = maskcol
            self.pixtable = pixtable
        else:
            hdulist = pyfits.open(filename)
            self.maskfile = hdulist[0].header['mask']
            self.pixtable = hdulist[0].header['pixtable']
            self.maskcol = np.bool_(hdulist['maskcol'].data[:, 0])

    def write(self, filename):
        """Save the object in a FITS file.

        Parameters
        ----------
        filename    : string
                      The FITS filename.
        """
        prihdu = pyfits.PrimaryHDU()
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        add_mpdaf_method_keywords(prihdu.header,
                                  'mpdaf.drs.pixtable.mask_column',
                                  [], [], [])
        prihdu.header['pixtable'] = (os.path.basename(self.pixtable),
                                     'pixtable')
        prihdu.header['mask'] = (os.path.basename(self.maskfile),
                                 'file to mask out all bright obj')
        hdulist = [prihdu]
        nrows = self.maskcol.shape[0]
        hdulist.append(ImageHDU(
            name='maskcol', data=np.int32(self.maskcol.reshape((nrows, 1)))))
        hdu = pyfits.HDUList(hdulist)
        hdu[1].header['BUNIT'] = 'boolean'
        hdu.writeto(filename, clobber=True, output_verify='fix')


class PixTableAutoCalib(object):

    """PixTableAutoCalib class.

    This class manages input/output for file
    containing auto calibration results
    of MUSE pixel table files

    Parameters
    ----------
    filename : string
               The FITS file name.
               If PixTableAutoCalib object is loaded from a FITS file,
               the others parameters are not read but loaded from
               the FITS file.
    method   : string or None
               Name of the auto calibration method.
    maskfile : string or None
               Name of the FITS image masking some objects.
    skyref   : string or None
               sky reference spectrum.
    pixtable : string or None
               Name of the corresponding pixel table.
    ifu      : array of integer or None
               channel numbers.
    sli      : array of integer or None
               slice numbers.
    quad      : array of integer or None
               Detector quadrant numbers.
    npts     : array of integer or None
               number of remaining pixels.
    corr     : array of float or None
               correction value.

    Attributes
    ----------
    filename : string
               The FITS file name.
    method   : string
               Name of the auto calibration method.
    maskfile : string
               Name of the FITS image masking some objects.
    skyref   : string
               sky reference spectrum.
    pixtable : string
               Name of the corresponding pixel table.
    ifu      : array of integer
               channel numbers.
    sli      : array of integer
               slice numbers.
    quad      : array of integer or None
               Detector quadrant numbers.
    npts     : array of integer
               number of remaining pixels.
    corr     : array of float
               correction value.

    """

    def __init__(self, filename=None, method=None, maskfile=None, skyref=None,
                 pixtable=None, ifu=None, sli=None, quad=None, npts=None, corr=None):
        """Create a PixTableAutoCalib object.

        Parameters
        ----------
        filename : string
                   The FITS file name.
                   If PixTableAutoCalib object is loaded from a FITS file,
                   the others parameters are not read but loaded from
                   the FITS file.
        method   : string or None
                   Name of the auto calibration method.
        maskfile : string or None
                   Name of the FITS image masking some objects.
        skyref   : string or None
                   sky reference spectrum.
        pixtable : string or None
                   Name of the corresponding pixel table.
        ifu      : array of integer or None
                   channel numbers.
        sli      : array of integer or None
                   slice numbers.
        quad     : array of integer or None
                   Detector quadrant numbers.
        npts     : array of integer or None
                   number of remaining pixels.
        corr     : array of float or None
                   correction value.
        """
        if filename is None:
            self.method = method
            self.maskfile = maskfile
            self.skyref = skyref
            self.pixtable = pixtable
            self.ifu = ifu
            self.sli = sli
            self.quad = quad
            self.npts = npts
            self.corr = corr
        else:
            hdulist = pyfits.open(filename)
            self.method = hdulist[0].header['method']
            self.maskfile = hdulist[0].header['mask']
            self.skyref = hdulist[0].header['skyref']
            self.pixtable = hdulist[0].header['pixtable']
            self.ifu = hdulist['ifu'].data[:, 0]
            self.sli = hdulist['sli'].data[:, 0]
            try:
                self.quad = hdulist['quad'].data[:, 0]
            except:
                self.quad = None
            self.npts = hdulist['npts'].data[:, 0]
            self.corr = hdulist['corr'].data[:, 0]

    def write(self, filename):
        """Save the object in a FITS file.
        """
        prihdu = pyfits.PrimaryHDU()
        warnings.simplefilter("ignore")
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        add_mpdaf_method_keywords(prihdu.header,
                                  'mpdaf.drs.PixTableAutoCalib.write',
                                  [], [], [])
        prihdu.header['method'] = (self.method, 'auto calib method')
        prihdu.header['pixtable'] = (os.path.basename(self.pixtable),
                                     'pixtable')
        prihdu.header['mask'] = (os.path.basename(self.maskfile),
                                 'file to mask out all bright obj')
        prihdu.header['skyref'] = (os.path.basename(self.skyref),
                                   'reference sky spectrum')

        shape = (self.corr.shape[0], 1)
        hdulist = [
            prihdu,
            ImageHDU(name='ifu', data=np.int32(self.ifu.reshape(shape))),
            ImageHDU(name='sli', data=np.int32(self.sli.reshape(shape))),
            ImageHDU(name='quad', data=np.int32(self.quad.reshape(shape))),
            ImageHDU(name='npts', data=np.int32(self.npts.reshape(shape))),
            ImageHDU(name='corr', data=np.float64(self.corr.reshape(shape)))]
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True, output_verify='fix')
        warnings.simplefilter("default")


def write(filename, xpos, ypos, lbda, data, dq, stat, origin, weight=None,
          primary_header=None, save_as_ima=True, wcs=u.pix, wave=u.angstrom,
          unit_data=u.count):
    """Save the object in a FITS file.

    Parameters
    ----------
    filename    : string
                  The FITS filename.
    save_as_ima : bool
                  If True, pixtable is saved as multi-extension FITS
    """
    logger = logging.getLogger(__name__)
    pyfits.conf.extension_name_case_sensitive = True

    prihdu = pyfits.PrimaryHDU()
    warnings.simplefilter("ignore")
    if primary_header is not None:
        for card in primary_header.cards:
            try:
                card.verify('fix')
                prihdu.header[card.keyword] = (card.value, card.comment)
            except ValueError:
                if isinstance(card.value, str):
                    n = 80 - len(card.keyword) - 14
                    s = card.value[0:n]
                    prihdu.header['hierarch %s' % card.keyword] = \
                        (s, card.comment)
                else:
                    prihdu.header['hierarch %s' % card.keyword] = \
                        (card.value, card.comment)
            except:
                logger.warning("%s keyword not written", card.keyword)
                pass
    prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
    prihdu.header['author'] = ('MPDAF', 'origin of the file')

    if save_as_ima:
        nrows = xpos.shape[0]
        hdulist = [
            prihdu,
            ImageHDU(name='xpos', data=np.float32(xpos.reshape((nrows, 1)))),
            ImageHDU(name='ypos', data=np.float32(ypos.reshape((nrows, 1)))),
            ImageHDU(name='lambda', data=np.float32(lbda.reshape((nrows, 1)))),
            ImageHDU(name='data', data=np.float32(data.reshape((nrows, 1)))),
            ImageHDU(name='dq', data=np.int32(dq.reshape((nrows, 1)))),
            ImageHDU(name='stat', data=np.float32(stat.reshape((nrows, 1)))),
            ImageHDU(name='origin', data=np.int32(origin.reshape((nrows, 1)))),
        ]
        if weight is not None:
            hdulist.append(
                ImageHDU(name='weight',
                         data=np.float32(weight.reshape((nrows, 1)))))
        hdu = pyfits.HDUList(hdulist)
        hdu[1].header['BUNIT'] = "{}".format(wcs)
        hdu[2].header['BUNIT'] = "{}".format(wcs)
        hdu[3].header['BUNIT'] = "{}".format(wave)
        hdu[4].header['BUNIT'] = "{}".format(unit_data)
        hdu[6].header['BUNIT'] = "{}".format(unit_data**2)

    else:
        cols = [
            Column(name='xpos', format='1E', unit="{}".format(wcs),
                   array=np.float32(xpos)),
            Column(name='ypos', format='1E', unit="{}".format(wcs),
                   array=np.float32(ypos)),
            Column(name='lambda', format='1E', unit="{}".format(wave),
                   array=lbda),
            Column(name='data', format='1E', unit="{}".format(unit_data),
                   array=np.float32(data)),
            Column(name='dq', format='1J', array=np.int32(dq)),
            Column(name='stat', format='1E', unit="{}".format(unit_data**2),
                   array=np.float32(stat)),
            Column(name='origin', format='1J', array=np.int32(origin)),
        ]

        if weight is not None:
            cols.append(Column(name='weight', format='1E',
                               array=np.float32(weight)))
        coltab = pyfits.ColDefs(cols)
        tbhdu = pyfits.TableHDU(pyfits.FITS_rec.from_columns(coltab))
        hdu = pyfits.HDUList([prihdu, tbhdu])

    hdu.writeto(filename, clobber=True, output_verify='fix')

    warnings.simplefilter("default")


class PixTable(object):

    """PixTable class.

    This class manages input/output for MUSE pixel table files

    Parameters
    ----------
    filename : string
               The FITS file name. None by default.

    Attributes
    ----------
    filename       : string
                     The FITS file name. None if any.
    primary_header : pyfits.Header
                     The primary header.
    nrows          : integer
                     Number of rows.
    nifu           : integer
                     Number of merged IFUs that went into this pixel table.
    skysub         : boolean
                     If True, this pixel table was sky-subtracted.
    fluxcal        : boolean
                     If True, this pixel table was flux-calibrated.
    wcs            : astropy.units
                     Type of spatial coordinates of this pixel table
                     (u.pix, u.deg or u.rad)
    wave           : astropy.units
                     Type of spectral coordinates of this pixel table
    ima            : boolean
                     If True, pixtable is saved as multi-extension FITS image
                     instead of FITS binary table.
    """

    def __init__(self, filename, xpos=None, ypos=None, lbda=None, data=None,
                 dq=None, stat=None, origin=None, weight=None,
                 primary_header=None, save_as_ima=True, wcs=u.pix,
                 wave=u.angstrom, unit_data=u.count):
        """Create a PixTable object.

        Parameters
        ----------
        filename : string
                   The FITS file name. None by default.

        The FITS file is opened with memory mapping.
        Just the primary header and table dimensions are loaded.
        Methods get_xpos, get_ypos, get_lambda, get_data, get_dq
        ,get_stat and get_origin must be used to get columns data.
        """
        self._logger = logging.getLogger(__name__)
        self.filename = filename
        self.wcs = wcs
        self.wave = wave
        self.ima = save_as_ima

        self.xpos = None
        self.ypos = None
        self.lbda = None
        self.data = None
        self.stat = None
        self.dq = None
        self.origin = None
        self.weight = None
        self.nrows = 0
        self.nifu = 0
        self.skysub = False
        self.fluxcal = False
        self.unit_data = unit_data
        self.xc = 0.0
        self.yc = 0.0

        if filename is not None:
            try:
                self.hdulist = pyfits.open(self.filename, memmap=1)
            except IOError:
                raise IOError('file %s not found' % filename)

            self.primary_header = self.hdulist[0].header
            self.nrows = self.hdulist[1].header["NAXIS2"]
            self.ima = (self.hdulist[1].header['XTENSION'] == 'IMAGE')

            if self.ima:
                self.wcs = u.Unit(self.hdulist['xpos'].header['BUNIT'])
                self.wave = u.Unit(self.hdulist['lambda'].header['BUNIT'])
                self.unit_data = u.Unit(self.hdulist['data'].header['BUNIT'])
            else:
                self.wcs = u.Unit(self.hdulist[1].header['TUNIT1'])
                self.wave = u.Unit(self.hdulist[1].header['TUNIT3'])
                self.unit_data = u.Unit(self.hdulist[1].header['TUNIT4'])
        else:
            self.hdulist = None
            if (xpos is None or ypos is None or lbda is None or
                    data is None or dq is None or stat is None or
                    origin is None or primary_header is None):
                self.primary_header = pyfits.Header()
                self.nrows = 0
            else:
                self.primary_header = primary_header
                xpos = np.array(xpos)
                ypos = np.array(ypos)
                lbda = np.array(lbda)
                data = np.array(data)
                stat = np.array(stat)
                dq = np.array(dq)
                origin = np.array(origin)
                self.nrows = xpos.shape[0]
                if ypos.shape[0] != self.nrows or\
                   lbda.shape[0] != self.nrows or\
                   data.shape[0] != self.nrows or\
                   stat.shape[0] != self.nrows or\
                   dq.shape[0] != self.nrows or\
                   origin.shape[0] != self.nrows:
                    raise IOError('input data with different dimensions')
                else:
                    self.xpos = xpos
                    self.ypos = ypos
                    self.lbda = lbda
                    self.data = data
                    self.stat = stat
                    self.dq = dq
                    self.origin = origin
                if weight is None or weight.shape[0] == self.nrows:
                    self.weight = weight
                else:
                    raise IOError('input data with different dimensions')

        if self.nrows != 0:
            # Merged IFUs that went into this pixel tables
            try:
                self.nifu = \
                    self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE MERGED")
            except:
                self.nifu = 1
            # sky subtraction
            try:
                self.skysub = \
                    self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE SKYSUB")
            except:
                self.skysub = False
            # flux calibration
            try:
                self.fluxcal = self.get_keywords("HIERARCH ESO DRS MUSE "
                                                 "PIXTABLE FLUXCAL")
            except:
                self.fluxcal = False

            try:
                # center in degrees
                cunit = u.Unit(self.get_keywords("CUNIT1"))
                self.xc = (self.primary_header['RA'] * cunit).to(u.deg).value
                self.yc = (self.primary_header['DEC'] * cunit).to(u.deg).value
            except:
                try:
                    # center in pixels
                    self.xc = self.primary_header['RA']
                    self.yc = self.primary_header['DEC']
                except:
                    pass

    def copy(self):
        """Copy PixTable object in a new one and returns it."""
        result = PixTable(self.filename)
        result.wcs = self.wcs
        result.wave = self.wave
        result.unit_data = self.unit_data
        result.ima = self.ima

        if self.xpos is not None:
            result.xpos = self.xpos.__copy__()
        if self.ypos is not None:
            result.ypos = self.ypos.__copy__()
        if self.lbda is not None:
            result.lbda = self.lbda.__copy__()
        if self.data is not None:
            result.data = self.data.__copy__()
        if self.stat is not None:
            result.stat = self.stat.__copy__()
        if self.dq is not None:
            result.dq = self.dq.__copy__()
        if self.origin is not None:
            result.origin = self.origin.__copy__()
        if self.weight is not None:
            result.weight = self.weight.__copy__()

        result.nrows = self.nrows
        result.nifu = self.nifu
        result.skysub = self.skysub
        result.fluxcal = self.fluxcal

        result.primary_header = pyfits.Header(self.primary_header)

        result.xc = self.xc
        result.yc = self.yc

        return result

    def info(self):
        """Print information."""
        msg = "%i merged IFUs went into this pixel table" % self.nifu
        self._logger.info(msg)
        if self.skysub:
            msg = "This pixel table was sky-subtracted"
            self._logger.info(msg)
        if self.fluxcal:
            msg = "This pixel table was flux-calibrated"
            self._logger.info(msg)
        msg = '%s (%s)' % (
            self.primary_header["HIERARCH ESO DRS MUSE PIXTABLE WCS"],
            self.primary_header.comments["HIERARCH ESO DRS MUSE PIXTABLE WCS"])
        self._logger.info(msg)
        try:
            msg = self.hdulist.info()
            self._logger.info(msg)
        except:
            msg = 'No\tName\tType\tDim'
            self._logger.info(msg)
            msg = '0\tPRIMARY\tcard\t()'
            self._logger.info(msg)
            # print "1\t\tTABLE\t(%iR,%iC)" % (self.nrows,self.ncols)

    def write(self, filename, save_as_ima=True):
        """Save the object in a FITS file.

        Parameters
        ----------
        filename    : string
                      The FITS filename.
        save_as_ima : bool
                      If True, pixtable is saved as multi-extension FITS image
                      instead of FITS binary table.
        """
        write(filename, self.get_xpos(), self.get_ypos(),
              self.get_lambda(), self.get_data(), self.get_dq(),
              self.get_stat(), self.get_origin(), self.get_weight(),
              self.primary_header, save_as_ima, self.wcs, self.wave,
              self.unit_data)

        self.filename = filename
        self.ima = save_as_ima

    def get_column(self, name, ksel=None):
        """Load a column and return it.

        Parameters
        ----------
        name : string or attribute
               Name of the column.
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        attr_name = 'lbda' if name == 'lambda' else name
        attr = getattr(self, attr_name)
        if attr is not None:
            if ksel is None:
                return attr
            else:
                return attr[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        column = self.hdulist[name].data[:, 0]
                    else:
                        column = self.hdulist[1].data.field(name)
                else:
                    if self.ima:
                        column = self.hdulist[name].data[ksel, 0][0]
                    else:
                        column = self.hdulist[1].data.field(name)[ksel]
                return column

    def set_column(self, name, data, ksel=None):
        """Set a column (or a part of it).

        Parameters
        ----------
        name : string or attribute
               Name of the column.
        data : numpy.array
               data values
        ksel : output of np.where
               Elements depending on a condition.
        """
        attr_name = 'lbda' if name == 'lambda' else name
        data = np.asarray(data)
        if ksel is None:
            assert data.shape[0] == self.nrows, 'Wrong dimension number'
            setattr(self, attr_name, data)
        else:
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, getattr(self, 'get_' + name)())
            attr = getattr(self, attr_name)
            attr[ksel] = data

    def get_xpos(self, ksel=None, unit=None):
        """Load the xpos column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('xpos', ksel=ksel)
        else:
            return (self.get_column('xpos', ksel=ksel) * self.wcs).to(unit).value

    def set_xpos(self, xpos, ksel=None, unit=None):
        """Set xpos column (or a part of it).

        Parameters
        ----------
        xpos : numpy.array
               xpos values
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               unit of the xpos column in input.
        """
        if unit is not None:
            xpos = (xpos * unit).to(self.wcs).value
        self.set_column('xpos', xpos, ksel=ksel)
        self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X LOW']\
            = float(self.xpos.min())
        self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X HIGH']\
            = float(self.xpos.max())

    def get_ypos(self, ksel=None, unit=None):
        """Load the ypos column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('ypos', ksel=ksel)
        else:
            return (self.get_column('ypos', ksel=ksel) * self.wcs).to(unit).value

    def set_ypos(self, ypos, ksel=None, unit=None):
        """Set ypos column (or a part of it).

        Parameters
        ----------
        ypos : numpy.array
               ypos values
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               unit of the ypos column in input.
        """
        if unit is not None:
            ypos = (ypos * unit).to(self.wcs).value
        self.set_column('ypos', ypos, ksel=ksel)
        self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y LOW']\
            = float(self.ypos.min())
        self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y HIGH']\
            = float(self.ypos.max())

    def get_lambda(self, ksel=None, unit=None):
        """Load the lambda column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('lambda', ksel=ksel)
        else:
            return (self.get_column('lambda', ksel=ksel) * self.wave).to(unit).value

    def set_lambda(self, lbda, ksel=None, unit=None):
        """Set lambda column (or a part of it).

        Parameters
        ----------
        lbda : numpy.array
               lbda values
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               unit of the lambda column in input.
        """
        if unit is not None:
            lbda = (lbda * unit).to(self.wave).value
        self.set_column('lambda', lbda, ksel=ksel)
        self.primary_header['HIERARCH ESO DRS MUSE '
                            'PIXTABLE LIMITS LAMBDA LOW']\
            = float(self.lbda.min())
        self.primary_header['HIERARCH ESO DRS MUSE '
                            'PIXTABLE LIMITS LAMBDA HIGH']\
            = float(self.lbda.max())

    def get_data(self, ksel=None, unit=None):
        """Load the data column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('data', ksel=ksel)
        else:
            return (self.get_column('data', ksel=ksel) * self.unit_data).to(unit).value

    def set_data(self, data, ksel=None, unit=None):
        """Set data column (or a part of it).

        Parameters
        ----------
        data : numpy.array
               data values
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               unit of the data column in input.
        """
        if unit is not None:
            data = (data * unit).to(self.unit_data).value
        self.set_column('data', data, ksel=ksel)

    def get_stat(self, ksel=None, unit=None):
        """Load the stat column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('stat', ksel=ksel)
        else:
            return (self.get_column('stat', ksel=ksel) * (self.unit_data**2)).to(unit).value

    def set_stat(self, stat, ksel=None, unit=None):
        """Set stat column (or a part of it).

        Parameters
        ----------
        stat : numpy.array
               stat values
        ksel : output of np.where
               Elements depending on a condition.
        unit : astropy.units
               unit of the stat column in input.
        """
        if unit is not None:
            stat = (stat * unit).to(self.unit_data**2).value
        self.set_column('stat', stat, ksel=ksel)

    def get_dq(self, ksel=None):
        """Load the dq column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        return self.get_column('dq', ksel=ksel)

    def set_dq(self, dq, ksel=None):
        """Set dq column (or a part of it).

        Parameters
        ----------
        dq   : numpy.array
               dq values
        ksel : output of np.where
               Elements depending on a condition.
        """
        self.set_column('dq', dq, ksel=ksel)

    def get_origin(self, ksel=None):
        """Load the origin column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        return self.get_column('origin', ksel=ksel)

    def set_origin(self, origin, ksel=None):
        """Set origin column (or a part of it).

        Parameters
        ----------
        origin : numpy.array
                 origin values
        ksel   : output of np.where
                 Elements depending on a condition.
        """
        self.set_column('origin', origin, ksel=ksel)
        hdr = self.primary_header
        ifu = self.origin2ifu(self.origin)
        sli = self.origin2slice(self.origin)
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU LOW'] = int(ifu.min())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU HIGH'] = int(ifu.max())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS SLICE LOW'] = int(sli.min())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS SLICE HIGH'] = \
            int(sli.max())

        # merged pixtable
        if self.nifu > 1:
            hdr["HIERARCH ESO DRS MUSE PIXTABLE MERGED"] = len(np.unique(ifu))

    def get_weight(self, ksel=None):
        """Load the weight column and return it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        try:
            wght = self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE WEIGHTED")
        except KeyError:
            wght = False
        return self.get_column('weight', ksel=ksel) if wght else None

    def set_weight(self, weight, ksel=None):
        """Set weight column (or a part of it).

        Parameters
        ----------
        weight : numpy.array
                 weight values
        ksel   : output of np.where
                 Elements depending on a condition.
        """
        self.set_column('weight', weight, ksel=ksel)

    def get_exp(self):
        """Load the exposure numbers and return it as a column.

        Returns
        -------
        out : numpy.memmap
        """
        getk = self.get_keywords
        try:
            nexp = getk("HIERARCH ESO DRS MUSE PIXTABLE COMBINED")
            exp = np.empty(shape=self.nrows)
            for i in range(1, nexp + 1):
                first = getk("HIERARCH ESO DRS MUSE PIXTABLE EXP%i FIRST" % i)
                last = getk("HIERARCH ESO DRS MUSE PIXTABLE EXP%i LAST" % i)
                exp[first:last + 1] = i
        except:
            exp = None
        return exp

    def select_lambda(self, lbda, unit=u.angstrom):
        """Return a mask corresponding to the given wavelength range

        Parameters
        ----------
        lbda     : (float, float)
                   (min, max) wavelength range in angstrom.
        unit : astropy.units
               Unit of the wavelengths in input.

        Returns
        -------
        out : array of booleans
              mask
        """
        arr = self.get_lambda()
        mask = np.zeros(self.nrows, dtype=bool)
        if numexpr:
            for l1, l2 in lbda:
                l1 = (l1 * unit).to(self.wave).value
                l2 = (l2 * unit).to(self.wave).value
                mask |= numexpr.evaluate('(arr >= l1) & (arr < l2)')
        else:
            for l1, l2 in lbda:
                l1 = (l1 * unit).to(self.wave).value
                l2 = (l2 * unit).to(self.wave).value
                mask |= (arr >= l1) & (arr < l2)
        return mask

    def select_stacks(self, stacks, origin=None):
        """Return a mask corresponding to given stacks.

        Parameters
        ----------
        stacks     : list of integers
                     Sracks numbers (1,2,3 or 4)
        Returns
        -------
        out : array of booleans
              mask
        """
        from ..MUSE import Slicer
        assert min(stacks) > 0
        assert max(stacks) < 5
        sl = sorted([Slicer.sky2ccd(i) for st in stacks
                     for i in range(1 + 12 * (st - 1), 12 * st - 1)])
        self._logger.debug('Extract stack %s -> slices %s', stacks, sl)
        return self.select_slices(sl, origin=origin)

    def select_slices(self, slices, origin=None):
        """Return a mask corresponding to given slices.

        Parameters
        ----------
        slices       : list of integers
                       Slice number on the CCD.

        Returns
        -------
        out : array of booleans
              mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_sli = self.origin2slice(col_origin)
        if numexpr:
            mask = np.zeros(self.nrows, dtype=bool)
            for s in slices:
                mask |= numexpr.evaluate('col_sli == s')
            return mask
        else:
            return np.in1d(col_sli, slices)

    def select_ifus(self, ifus, origin=None):
        """Return a mask corresponding to given ifus.

        Parameters
        ----------
        ifu      : int or list
                   IFU number.

        Returns
        -------
        out : array of booleans
              mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_ifu = self.origin2ifu(col_origin)
        if numexpr:
            mask = np.zeros(self.nrows, dtype=bool)
            for ifu in ifus:
                mask |= numexpr.evaluate('col_ifu == ifu')
            return mask
        else:
            return np.in1d(col_ifu, ifus)

    def select_exp(self, exp, col_exp):
        """Return a mask corresponding to given exposure numbers.

        Parameters
        ----------
        exp      : list of integers
                   List of exposure numbers

        Returns
        -------
        out : array of booleans
              mask
        """
        mask = np.zeros(self.nrows, dtype=bool)
        if numexpr:
            for iexp in exp:
                mask |= numexpr.evaluate('col_exp == iexp')
        else:
            for iexp in exp:
                mask |= (col_exp == iexp)
        return mask

    def select_xpix(self, xpix, origin=None):
        """Return a mask corresponding to given detector pixels.

        Parameters
        ----------
        xpix     : list
                   [(min, max)] pixel range along the X axis

        Returns
        -------
        out : array of booleans
              mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_xpix = self.origin2xpix(col_origin)
        if hasattr(xpix, '__iter__'):
            mask = np.zeros(self.nrows, dtype=bool)
            if numexpr:
                for x1, x2 in xpix:
                    mask |= numexpr.evaluate('(col_xpix >= x1) & '
                                             '(col_xpix < x2)')
            else:
                for x1, x2 in xpix:
                    mask |= (col_xpix >= x1) & (col_xpix < x2)
        else:
            x1, x2 = xpix
            if numexpr:
                mask = numexpr.evaluate('(col_xpix >= x1) & (col_xpix < x2)')
            else:
                mask = (col_xpix >= x1) & (col_xpix < x2)
        return mask

    def select_ypix(self, ypix, origin=None):
        """Return a mask corresponding to given detector pixels.

        Parameters
        ----------
        ypix     : list
                   [(min, max)] pixel range along the Y axis

        Returns
        -------
        out : array of booleans
              mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_ypix = self.origin2ypix(col_origin)
        if hasattr(ypix, '__iter__'):
            mask = np.zeros(self.nrows, dtype=bool)
            if numexpr:
                for y1, y2 in ypix:
                    mask |= numexpr.evaluate('(col_ypix >= y1) & '
                                             '(col_ypix < y2)')
            else:
                for y1, y2 in ypix:
                    mask |= (col_ypix >= y1) & (col_ypix < y2)
        else:
            y1, y2 = ypix
            if numexpr:
                mask = numexpr.evaluate('(col_ypix >= y1) & (col_ypix < y2)')
            else:
                mask = (col_ypix >= y1) & (col_ypix < y2)
        return mask

    def select_sky(self, sky):
        """Return a mask corresponding to the given aperture on the sky (center, size and shape)

        Parameters
        ----------
        sky      : (float, float, float, char)
                   (y, x, size, shape) extract an aperture on the sky,
                   defined by a center (y, x) in degrees/pixel,
                   a shape ('C' for circular, 'S' for square)
                   and size (radius or half side length) in arcsec/pixels.

        Returns
        -------
        out : array of booleans
              mask
        """
        xpos, ypos = self.get_pos_sky()
        mask = np.zeros(self.nrows, dtype=bool)
        if numexpr:
            pi = np.pi  # NOQA
            for y0, x0, size, shape in sky:
                if shape == 'C':
                    if self.wcs == u.deg:
                        mask |= numexpr.evaluate(
                            '(((xpos - x0) * 3600 * cos(y0 * pi / 180.)) ** 2 '
                            '+ ((ypos - y0) * 3600) ** 2) < size ** 2')
                    elif self.wcs == u.rad:
                        mask |= numexpr.evaluate(
                            '(((xpos - x0) * 3600 * 180 / pi * cos(y0)) ** 2 '
                            '+ ((ypos - y0) * 3600 * 180 / pi)** 2) < size ** 2')
                    else:
                        mask |= numexpr.evaluate(
                            '((xpos - x0) ** 2 + (ypos - y0) ** 2) < size ** 2')
                elif shape == 'S':
                    if self.wcs == u.deg:
                        mask |= numexpr.evaluate(
                            '(abs((xpos - x0) * 3600 * cos(y0 * pi / 180.)) < size) '
                            '& (abs((ypos - y0) * 3600) < size)')
                    elif self.wcs == u.rad:
                        mask |= numexpr.evaluate(
                            '(abs((xpos - x0) * 3600 * 180 / pi * cos(y0)) < size) '
                            '& (abs((ypos - y0) * 3600 * 180 / pi) < size)')
                    else:
                        mask |= numexpr.evaluate(
                            '(abs(xpos - x0) < size) & (abs(ypos - y0) < size)')
                else:
                    raise ValueError('Unknown shape parameter')
        else:
            for y0, x0, size, shape in sky:
                if shape == 'C':
                    if self.wcs == u.deg:
                        mask |= (((xpos - x0) * 3600
                                  * np.cos(y0 * np.pi / 180.)) ** 2
                                 + ((ypos - y0) * 3600) ** 2) \
                            < size ** 2
                    elif self.wcs == u.rad:
                        mask |= (((xpos - x0) * 3600 * 180 / np.pi
                                  * np.cos(y0)) ** 2
                                 + ((ypos - y0) * 3600 * 180 / np.pi)
                                 ** 2) < size ** 2
                    else:
                        mask |= ((xpos - x0) ** 2
                                 + (ypos - y0) ** 2) < size ** 2
                elif shape == 'S':
                    if self.wcs == u.deg:
                        mask |= (np.abs((xpos - x0) * 3600
                                        * np.cos(y0 * np.pi / 180.)) < size) \
                            & (np.abs((ypos - y0) * 3600) < size)
                    elif self.wcs == u.rad:
                        mask |= (np.abs((xpos - x0) * 3600 * 180
                                        / np.pi * np.cos(y0)) < size) \
                            & (np.abs((ypos - y0) * 3600 * 180 / np.pi) < size)
                    else:
                        mask |= (np.abs(xpos - x0) < size) \
                            & (np.abs(ypos - y0) < size)
                else:
                    raise ValueError('Unknown shape parameter')
        return mask

    def extract_from_mask(self, mask):
        """Return a new pixtable extracted with the given mask.

        Parameters
        ----------
        mask : numpy.ndarray
               Mask (array of boolean).

        Returns
        -------
        out : PixTable

        """
        ksel = np.where(mask)
        nrows = len(ksel[0])
        if nrows == 0:
            return None

        hdr = self.primary_header.copy()

        # xpos
        xpos = self.get_xpos(ksel)
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X LOW'] = float(xpos.min())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X HIGH'] = float(xpos.max())

        # ypos
        ypos = self.get_ypos(ksel)
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y LOW'] = float(ypos.min())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y HIGH'] = float(ypos.max())

        # lambda
        lbda = self.get_lambda(ksel)
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS LAMBDA LOW'] = \
            float(lbda.min())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS LAMBDA HIGH'] = \
            float(lbda.max())

        # origin
        origin = self.get_origin(ksel)
        ifu = self.origin2ifu(origin)
        sl = self.origin2slice(origin)
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU LOW'] = int(ifu.min())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU HIGH'] = int(ifu.max())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS SLICE LOW'] = int(sl.min())
        hdr['HIERARCH ESO DRS MUSE PIXTABLE LIMITS SLICE HIGH'] = int(sl.max())

        # merged pixtable
        if self.nifu > 1:
            hdr["HIERARCH ESO DRS MUSE PIXTABLE MERGED"] = len(np.unique(ifu))

        # combined exposures
        selfexp = self.get_exp()
        if selfexp is not None:
            newexp = selfexp[ksel]
            numbers_exp = np.unique(newexp)
            hdr["HIERARCH ESO DRS MUSE PIXTABLE COMBINED"] = len(numbers_exp)
            for iexp, i in zip(numbers_exp, range(1, len(numbers_exp) + 1)):
                k = np.where(newexp == iexp)
                hdr["HIERARCH ESO DRS MUSE PIXTABLE EXP%i FIRST" % i] = k[0][0]
                hdr["HIERARCH ESO DRS MUSE PIXTABLE EXP%i LAST" % i] = k[0][-1]
            for i in range(len(numbers_exp) + 1,
                           self.get_keywords("HIERARCH ESO DRS MUSE "
                                             "PIXTABLE COMBINED") + 1):
                del hdr["HIERARCH ESO DRS MUSE PIXTABLE EXP%i FIRST" % i]
                del hdr["HIERARCH ESO DRS MUSE PIXTABLE EXP%i LAST" % i]

        # return sub pixtable
        data = self.get_data(ksel)
        stat = self.get_stat(ksel)
        dq = self.get_dq(ksel)
        weight = self.get_weight(ksel)
        return PixTable(None, xpos, ypos, lbda, data, dq, stat, origin,
                        weight, hdr, self.ima, self.wcs, self.wave,
                        unit_data=self.unit_data)

    def extract(self, filename=None, sky=None, lbda=None, ifu=None, sl=None,
                xpix=None, ypix=None, exp=None, stack=None, method='and'):
        """Extracts a subset of a pixtable using the following criteria:

        - aperture on the sky (center, size and shape)
        - wavelength range
        - IFU number
        - slice number
        - detector pixels
        - exposure numbers
        - stack number

        The arguments can be either single value or a list of values to select
        multiple regions.

        Parameters
        ----------
        filename : string
                   The FITS filename used to saves the resulted object.
        sky      : (float, float, float, char)
                   (y, x, size, shape) extract an aperture on the sky,
                   defined by a center (y, x) in degrees/pixel,
                   a shape ('C' for circular, 'S' for square)
                   and size (radius or half side length) in arcsec/pixels.
        lbda     : (float, float)
                   (min, max) wavelength range in angstrom.
        ifu      : int or list
                   IFU number.
        sl       : int or list
                   Slice number on the CCD.
        xpix     : (int, int) or list
                   (min, max) pixel range along the X axis
        ypix     : (int, int) or list
                   (min, max) pixel range along the Y axis
        exp      : list of integers
                   List of exposure numbers

        Returns
        -------
        out : PixTable
        """
        if self.nrows == 0:
            return None

        if isinstance(sky, tuple):
            sky = [sky]
        if isinstance(lbda, tuple):
            lbda = [lbda]
        if isinstance(ifu, (int, float)):
            ifu = [ifu]
        if isinstance(sl, (int, float)):
            sl = [sl]
        if isinstance(stack, (int, float)):
            stack = [stack]

        if method == 'and':
            kmask = np.ones(self.nrows, dtype=bool)
            lfunc = np.logical_and
        elif method == 'or':
            kmask = np.zeros(self.nrows, dtype=bool)
            lfunc = np.logical_or

        # Do the selection on the sky
        if sky is not None:
            lfunc(kmask, self.select_sky(sky), out=kmask)

        # Do the selection on wavelengths
        if lbda is not None:
            lfunc(kmask, self.select_lambda(lbda, unit=u.angstrom), out=kmask)

        # Do the selection on the origin column
        if (ifu is not None) or (sl is not None) or (stack is not None) or \
                (xpix is not None) or (ypix is not None):
            origin = self.get_origin()
            if sl is not None:
                lfunc(kmask, self.select_slices(sl, origin=origin), out=kmask)
            if stack is not None:
                lfunc(kmask, self.select_stacks(stack, origin=origin),
                      out=kmask)
            if ifu is not None:
                lfunc(kmask, self.select_ifus(ifu, origin=origin), out=kmask)
            if xpix is not None:
                lfunc(kmask, self.select_xpix(xpix, origin=origin), out=kmask)
            if ypix is not None:
                lfunc(kmask, self.select_ypix(ypix, origin=origin), out=kmask)
            origin = None

        # Do the selection on the exposure numbers
        if exp is not None:
            col_exp = self.get_exp()
            if col_exp is not None:
                lfunc(kmask, self.select_exp(exp, col_exp), out=kmask)

        # Compute the new pixtable
        pix = self.extract_from_mask(kmask)
        if filename is not None:
            pix.filename = filename
            pix.write(filename)
        return pix

    def origin2ifu(self, origin):
        """Converts the origin value and returns the ifu number.

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : integer
        """
        return ((origin >> 6) & 0x1f).astype(np.uint8)

    def origin2slice(self, origin):
        """Converts the origin value and returns the slice number.

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : integer
        """
        return (origin & 0x3f).astype(np.uint8)

    def origin2ypix(self, origin):
        """Converts the origin value and returns the y coordinates.

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : float
        """
        return (((origin >> 11) & 0x1fff) - 1).astype(np.uint16)

    def origin2xoffset(self, origin, ifu=None, sli=None):
        """Converts the origin value and returns the x coordinates offset.

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : float
        """
        col_ifu = ifu if ifu is not None else self.origin2ifu(origin)
        col_slice = sli if sli is not None else self.origin2slice(origin)
        key = "HIERARCH ESO DRS MUSE PIXTABLE EXP0 IFU%02d SLICE%02d XOFFSET"

        if isinstance(origin, np.ndarray):
            ifus = np.unique(col_ifu)
            slices = np.unique(col_slice)
            offsets = np.zeros((ifus.max() + 1, slices.max() + 1),
                               dtype=np.int32)
            for ifu in ifus:
                for sl in slices:
                    offsets[ifu, sl] = self.get_keywords(key % (ifu, sl))

            xoffset = offsets[col_ifu, col_slice]
        else:
            xoffset = self.get_keywords(key % (col_ifu, col_slice))
        return xoffset

    def origin2xpix(self, origin, ifu=None, sli=None):
        """Converts the origin value and returns the x coordinates.

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : float
        """
        return (self.origin2xoffset(origin, ifu=ifu, sli=sli) +
                ((origin >> 24) & 0x7f) - 1).astype(np.uint16)

    def origin2coords(self, origin):
        """Converts the origin value and returns (ifu, slice, ypix, xpix).

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : (integer, integer, float, float)
        """
        ifu, sli = self.origin2ifu(origin), self.origin2slice(origin)
        return (ifu, sli, self.origin2ypix(origin),
                self.origin2xpix(origin, ifu=ifu, sli=sli))

    def _get_pos_sky(self, xpos, ypos):
        try:
            spheric = (self.get_keywords(
                "HIERARCH ESO DRS MUSE PIXTABLE WCS")[0:9] == 'projected')
        except:
            spheric = False
        if spheric:  # spheric coordinates
            phi = xpos
            theta = ypos + np.pi / 2
            dp = self.yc * np.pi / 180
            ra = np.arctan2(np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.cos(dp) +
                            np.cos(theta) * np.sin(dp) * np.cos(phi)) * 180 / np.pi
            xpos_sky = self.xc + ra
            ypos_sky = np.arcsin(np.sin(theta) * np.sin(dp) -
                                 np.cos(theta) * np.cos(dp) * np.cos(phi)) * 180 / np.pi
        else:
            if self.wcs == u.deg:
                dp = self.yc * np.pi / 180
                xpos_sky = self.xc + xpos / np.cos(dp)
                ypos_sky = self.yc + ypos
            elif self.wcs == u.rad:
                dp = self.yc * np.pi / 180
                xpos_sky = self.xc + xpos * 180 / np.pi / np.cos(dp)
                ypos_sky = self.yc + ypos * 180 / np.pi
            else:
                xpos_sky = self.xc + xpos
                ypos_sky = self.yc + ypos
        return xpos_sky, ypos_sky

    def _get_pos_sky_numexpr(self, xpos, ypos):
        try:
            spheric = (self.get_keywords(
                "HIERARCH ESO DRS MUSE PIXTABLE WCS")[0:9] == 'projected')
        except:
            spheric = False
        pi = np.pi  # NOQA
        xc = self.xc  # NOQA
        yc = self.yc  # NOQA
        if spheric:  # spheric coordinates
            phi = xpos  # NOQA
            theta = numexpr.evaluate("ypos + pi/2")
            dp = numexpr.evaluate("yc * pi / 180")
            ra = numexpr.evaluate("arctan2(cos(theta) * sin(phi), sin(theta) * cos(dp) + cos(theta) * sin(dp) * cos(phi)) * 180 / pi")
            xpos_sky = numexpr.evaluate("xc + ra")
            ypos_sky = numexpr.evaluate("arcsin(sin(theta) * sin(dp) - cos(theta) * cos(dp) * cos(phi)) * 180 / pi")
        else:
            if self.wcs == u.deg:
                dp = numexpr.evaluate("yc * pi / 180")
                xpos_sky = numexpr.evaluate("xc + xpos / cos(dp)")
                ypos_sky = numexpr.evaluate("yc + ypos")
            elif self.wcs == u.rad:
                dp = numexpr.evaluate("yc * pi / 180")
                xpos_sky = numexpr.evaluate("xc + xpos * 180 / pi / cos(dp)")
                ypos_sky = numexpr.evaluate("yc + ypos * 180 / pi")
            else:
                xpos_sky = numexpr.evaluate("xc + xpos")
                ypos_sky = numexpr.evaluate("yc + ypos")
        return xpos_sky, ypos_sky

    def get_pos_sky(self, xpos=None, ypos=None):
        """Returns the absolute position on the sky in degrees/pixel.

        Parameters
        ----------
        xpos : numpy.array
               xpos values
        ypos : numpy.array
               ypos values

        Returns
        -------
        xpos_sky, ypos_sky : numpy.array, numpy.array
        """
        if xpos is None:
            xpos = self.get_xpos()
        if ypos is None:
            ypos = self.get_ypos()
        if numexpr:
            return self._get_pos_sky_numexpr(xpos, ypos)
        else:
            return self._get_pos_sky(xpos, ypos)

    def get_slices(self, verbose=True):
        """Returns slices dictionary.

        Parameters
        ----------
        verbose : boolean
                  If True, progression is printed.

        Returns
        -------
        out : dict
        """
        col_origin = self.get_origin()
        col_xpos = self.get_xpos()
        col_ypos = self.get_ypos()

        ifupix, slicepix, ypix, xpix = self.origin2coords(col_origin)
        ifus = np.unique(ifupix)
        slices = np.unique(slicepix)

        # build the slicelist
        slicelist = np.array(list(itertools.product(ifus, slices)))

        # compute mean sky position of each slice
        skypos = []
        for ifu, sl in slicelist:
            k = ((ifupix == ifu) & (slicepix == sl))
            skypos.append((col_xpos[k].mean(), col_ypos[k].mean()))
        skypos = np.array(skypos)

        slices = {'list': slicelist, 'skypos': skypos,
                  'ifupix': ifupix, 'slicepix': slicepix,
                  'xpix': xpix, 'ypix': ypix}

        if verbose:
            msg = '%d slices found, structure returned in \
                   slices dictionary ' % len(slicelist)
            self._logger.info(msg)

        return slices

    def get_keywords(self, key):
        """Returns the keyword value corresponding to key.

        Parameters
        ----------
        key : string
              Keyword.

        Returns
        -------
        out : float
        """
        try:
            return self.primary_header[key]
        except KeyError:
            # HIERARCH ESO PRO MUSE has been renamed into HIERARCH ESO DRS MUSE
            # in recent versions of the DRS.
            if key.startswith('HIERARCH ESO PRO MUSE'):
                alternate_key = key.replace('HIERARCH ESO PRO MUSE',
                                            'HIERARCH ESO DRS MUSE')
            elif key.startswith('HIERARCH ESO DRS MUSE'):
                alternate_key = key.replace('HIERARCH ESO DRS MUSE',
                                            'HIERARCH ESO PRO MUSE')
            else:
                raise
            return self.primary_header[alternate_key]

#     def reconstruct_sky_image(self, lbda=None, step=None):
#         """Reconstructs the image on the sky from the pixtable.
#
#         Parameters
#         ----------
#         lbda : (float,float)
#                (min, max) wavelength range in Angstrom.
#                If None, the image is reconstructed for all wavelengths.
#         step : (float,float)
#                Pixel steps of the final image
#                (in arcsec if the coordinates of this pixel table
#                are world coordinates on the sky ).
#                If None, the value corresponding to the keyword
#                "HIERARCH ESO INS PIXSCALE" is used.
#
#         Returns
#         -------
#         out : :class:`mpdaf.obj.Image`
#         """
#         # TODO replace by DRS
#         # step in arcsec
#
#         if step is None:
#             step = self.get_keywords('HIERARCH ESO OCS IPS PIXSCALE')
#             if step <= 0:
#                 raise ValueError('INS PIXSCALE not valid')
#             xstep = step
#             ystep = step
#         else:
#             ystep, xstep = step
#
#         col_dq = self.get_dq()
#         if lbda is None:
#             ksel = np.where((col_dq == 0))
#         else:
#             l1, l2 = lbda
#             l1 = l1 * (u.angstrom).to(self.wave)
#             l2 = l2 * (u.angstrom).to(self.wave)
#             col_lambda = self.get_lambda()
#             ksel = np.where((col_dq == 0) & (col_lambda > l1) &
#                             (col_lambda < l2))
#             del col_lambda
#         del col_dq
#
#         x = self.get_xpos(ksel) # deg ???
#         y = self.get_ypos(ksel)
#         data = self.get_data(ksel)
#
#         xmin = np.min(x)
#         xmax = np.max(x)
#         ymin = np.min(y)
#         ymax = np.max(y)
#
#         if self.wcs == u.deg:  # arcsec to deg
#             xstep /= (-3600. * np.cos((ymin + ymax) * np.pi / 180. / 2.))
#             ystep /= 3600.
#         elif self.wcs == u.rad:  # arcsec to rad
#             xstep /= (-3600. * 180. / np.pi * np.cos((ymin + ymax) / 2.))
#             ystep /= (3600. * 180. / np.pi)
#         else:  # pix
#             pass
#
#         nx = 1 + int((xmin - xmax) / xstep)
#         grid_x = np.arange(nx) * xstep + xmax
#         ny = 1 + int((ymax - ymin) / ystep)
#         grid_y = np.arange(ny) * ystep + ymin
#         shape = (ny, nx)
#
#         points = np.empty((len(ksel[0]), 2), dtype=float)
#         points[:, 0] = y
#         points[:, 1] = x
#
#         new_data = interpolate.griddata(points, data,
#                                         np.meshgrid(grid_y, grid_x),
#                                         method='linear').T
#
#         wcs = WCS(crpix=(1.0, 1.0), crval=(ymin, xmax),
#                   cdelt=(ystep, xstep), shape=shape)
#         ima = Image(data=new_data, wcs=wcs, unit=self.data_unit)
#         return ima

    def reconstruct_det_image(self, xstart=None, ystart=None,
                              xstop=None, ystop=None):
        """Reconstructs the image on the detector from the pixtable. The
        pixtable must concerns only one IFU, otherwise an exception is raised.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`
        """
        if self.nrows == 0:
            return None

        if self.nifu != 1:
            raise ValueError('Pixtable contains multiple IFU')

        col_data = self.get_data()
        col_origin = self.get_origin()

        ifu = np.empty(self.nrows, dtype='uint16')
        sl = np.empty(self.nrows, dtype='uint16')
        xpix = np.empty(self.nrows, dtype='uint16')
        ypix = np.empty(self.nrows, dtype='uint16')

        ifu, sl, ypix, xpix = self.origin2coords(col_origin)
        if len(np.unique(ifu)) != 1:
            raise ValueError('Pixtable contains multiple IFU')

        if xstart is None:
            xstart = xpix.min()
        if xstop is None:
            xstop = xpix.max()
        if ystart is None:
            ystart = ypix.min()
        if ystop is None:
            ystop = ypix.max()
        # xstart, xstop = xpix.min(), xpix.max()
        # ystart, ystop = ypix.min(), ypix.max()
        image = np.zeros((ystop - ystart + 1,
                          xstop - xstart + 1), dtype='float') * np.NaN
        image[ypix - ystart, xpix - xstart] = col_data

        wcs = WCS(crval=(ystart, xstart))
        return Image(data=image, wcs=wcs, unit=self.unit_data, copy=False)

    def reconstruct_det_waveimage(self):
        """Reconstructs an image of wavelength values on the detector from the
        pixtable. The pixtable must concerns only one IFU, otherwise an
        exception is raised.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`
        """
        if self.nrows == 0:
            return None

        if self.nifu != 1:
            raise ValueError('Pixtable contains multiple IFU')

        col_origin = self.get_origin()
        col_lambdas = self.get_lambda()

        ifu = np.empty(self.nrows, dtype='uint16')
        sl = np.empty(self.nrows, dtype='uint16')
        xpix = np.empty(self.nrows, dtype='uint16')
        ypix = np.empty(self.nrows, dtype='uint16')

        ifu, sl, ypix, xpix = self.origin2coords(col_origin)
        if len(np.unique(ifu)) != 1:
            raise ValueError('Pixtable contains multiple IFU')

        xstart, xstop = xpix.min(), xpix.max()
        ystart, ystop = ypix.min(), ypix.max()
        image = np.zeros((ystop - ystart + 1, xstop - xstart + 1),
                         dtype='float')
        image[ypix - ystart, xpix - xstart] = col_lambdas

        wcs = WCS(crval=(ystart, xstart))

        return Image(data=image, wcs=wcs, unit=self.wave, copy=False)

    def mask_column(self, maskfile=None, verbose=True):
        """Computes the mask column corresponding to a mask file.

        Parameters
        ----------
        maskfile : string
                   Path to a FITS image file with WCS information, used to mask
                   out bright continuum objects present in the FoV. Values must
                   be 0 for the background and >0 for objects.
        verbose : boolean
                  If True, progression is printed.

        Returns
        -------
        out : :class:`mpdaf.drs.PixTableMask`
        """

        mask = np.zeros(self.nrows, dtype=bool)
        if maskfile is None:
            return mask

        pos = np.array(self.get_pos_sky()[::-1]).T
        xpos_sky = pos[:, 1]
        ypos_sky = pos[:, 0]

        ima_mask = Image(maskfile)

        data = ima_mask.data.data
        label = ndimage.measurements.label(data)[0]
        ulabel = np.unique(label)
        ulabel = ulabel[ulabel > 0]
        nlabel = len(ulabel)
        msg = 'masking object %i/%i %g<x<%g %g<y<%g (%i pixels)'

        for i in ulabel:
            try:
                ksel = np.where(label == i)
                item = (slice(min(ksel[0]), max(ksel[0]) + 1, None),
                        slice(min(ksel[1]), max(ksel[1]) + 1, None))
                wcs = ima_mask.wcs[item]
                coord = wcs.get_range(unit=u.deg)
                step = wcs.get_step(unit=u.deg)
                y0, x0 = coord.min(axis=0) - step / 2
                y1, x1 = coord.max(axis=0) + step / 2
                ksel = np.where((xpos_sky > x0) & (xpos_sky < x1) &
                                (ypos_sky > y0) & (ypos_sky < y1))
                if verbose:
                    self._logger.info(msg, i, nlabel, x0, x1, y0, y1,
                                      len(ksel[0]))
                if len(ksel[0]) != 0:
                    pix = ima_mask.wcs.sky2pix(pos[ksel], nearest=True, unit=u.deg)
                    mask[ksel] |= (data[pix[:, 0], pix[:, 1]] != 0)
            except Exception:
                self._logger.warning('masking object %i failed', i)

        return PixTableMask(maskfile=maskfile, maskcol=mask,
                            pixtable=self.filename)

    def sky_ref(self, pixmask=None, dlbda=1.0, nmax=2, nclip=5.0, nstop=2):
        """Computes the reference sky spectrum using sigma clipped median.

        Algorithm from Kurt Soto (kurt.soto@phys.ethz.ch)

        Parameters
        ----------
        pixmask  : :class:`mpdaf.drs.PixTableMask`
                   column corresponding to a mask file
                   (previously computed by mask_column)
        dlbda    : double
                   wavelength step in angstrom
        nmax     : integer
                   maximum number of clipping iterations
        nclip    : float or (float,float)
                   Number of sigma at which to clip.
                   Single clipping parameter or lower/upper clipping parameters
        nstop    : integer
                   If the number of not rejected pixels is less
                   than this number, the clipping iterations stop.

        Returns
        -------
        out : :class:`mpdaf.obj.Spectrum`
        """
        # mask
        if pixmask is None:
            maskfile = ''
            mask = np.zeros(self.nrows, dtype=bool)
        else:
            maskfile = os.path.basename(pixmask.maskfile)
            mask = pixmask.maskcol

        # sigma clipped parameters
        if is_int(nclip) or is_float(nclip):
            nclip_low = nclip
            nclip_up = nclip
        else:
            nclip_low = nclip[0]
            nclip_up = nclip[1]
        # wavelength step
        lbda = self.get_lambda(unit=u.angstrom)
        lmin = np.min(lbda) - dlbda / 2.0
        lmax = np.max(lbda) + dlbda / 2.0
        n = (int)((lmax - lmin) / dlbda)

        # load the library, using numpy mechanisms
        path = os.path.dirname(__file__)[:-4]
        libCmethods = np.ctypeslib.load_library("libCmethods", path)
        # define argument types
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')
        # setup argument types
        libCmethods.mpdaf_sky_ref.argtypes = \
            [array_1d_double, array_1d_double, array_1d_int, ctypes.c_int,
             ctypes.c_double, ctypes.c_double, ctypes.c_int,
             ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
             array_1d_double]

        # setup return type
        libCmethods.mpdaf_sky_ref.restype = None

        data = self.get_data()
        data = data.astype(np.float64)
        lbda = lbda.astype(np.float64)
        mask = mask.astype(np.int32)
        result = np.empty(n, dtype=np.float64)
        # run C method
        libCmethods.mpdaf_sky_ref(data, lbda, mask, data.shape[0],
                                  np.float64(lmin),
                                  np.float64(dlbda), n, nmax,
                                  np.float64(nclip_low),
                                  np.float64(nclip_up), nstop, result)
        wave = WaveCoord(crpix=1.0, cdelt=dlbda, crval=np.min(lbda),
                         cunit=u.angstrom, shape=n)

        spe = Spectrum(data=result, wave=wave, unit=self.unit_data, copy=False)
        add_mpdaf_method_keywords(spe.primary_header,
                                  "drs.pixtable.sky_ref",
                                  ['pixtable', 'mask', 'dlbda', 'nmax',
                                   'nclip_low', 'nclip_up', 'nstop'],
                                  [os.path.basename(self.filename), maskfile,
                                   dlbda, nmax, nclip_low, nclip_up, nstop],
                                  ['pixtable',
                                   'file to mask out all bright objects',
                                   'wavelength step',
                                   'max number of clippinp iterations',
                                   'lower clipping parameter',
                                   'upper clipping parameter',
                                   'clipping minimum number'])
        return spe

    def subtract_slice_median(self, skyref, pixmask):
        """Computes the median value for all pairs (slice, quadrant)
        and subtracts this factor to each pixel
        to bring all slices to the same median value.

        pix(x,y,lbda) += < skyref(lbda) - pix(x,y,lbda) >_slice

        Parameters
        ----------
        skyref  : :class:`mpdaf.obj.Spectrum`
                  Reference sky spectrum
        pixmask : :class:`mpdaf.drs.PixTableMask`
                  column corresponding to a mask file
                  (previously computed by mask_column)
        Returns
        -------
        out : :class:`mpdaf.drs.PixTableAutoCalib`
        """

        origin = self.get_origin()
        ifu = self.origin2ifu(origin)
        sli = self.origin2slice(origin)
        xpix = self.origin2xpix(origin)
        ypix = self.origin2ypix(origin)

        # mask
        if pixmask is None:
            maskfile = ''
            maskcol = np.zeros(self.nrows, dtype=bool)
        else:
            maskfile = os.path.basename(pixmask.maskfile)
            maskcol = pixmask.maskcol

        # load the library, using numpy mechanisms
        path = os.path.dirname(__file__)[:-4]
        libCmethods = np.ctypeslib.load_library("libCmethods", path)

        # define argument types
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')

        # setup the return types and argument types
        libCmethods.mpdaf_slice_median.restype = None
        libCmethods.mpdaf_slice_median.argtypes = \
            [array_1d_double, array_1d_double, array_1d_double, array_1d_int,
             array_1d_int, array_1d_int, array_1d_double, array_1d_double,
             ctypes.c_int, array_1d_int, array_1d_double, array_1d_double,
             ctypes.c_int, array_1d_int, array_1d_int, ctypes.c_int]

        data = self.get_data()
        ifu = ifu.astype(np.int32)
        sli = sli.astype(np.int32)
        data = data.astype(np.float64)
        lbda = self.get_lambda()
        lbda = lbda.astype(np.float64)
        mask = maskcol.astype(np.int32)
        skyref_flux = (skyref.data.data.astype(np.float64) * skyref.unit).to(self.unit_data).value
        skyref_lbda = skyref.wave.coord(unit=self.wave)
        skyref_n = skyref.shape
        xpix = xpix.astype(np.int32)
        ypix = ypix.astype(np.int32)

        result = np.empty_like(data, dtype=np.float64)
        stat_result = np.empty_like(data, dtype=np.float64)
        corr = np.ones(24 * 48 * 4, dtype=np.float64)  # zeros
        npts = np.zeros(24 * 48 * 4, dtype=np.int32)

        libCmethods.mpdaf_slice_median(
            result, stat_result, corr, npts, ifu, sli, data, lbda,
            data.shape[0], mask, skyref_flux, skyref_lbda, skyref_n, xpix, ypix, 1)

        # set pixtable data
        self.set_data(result)

        if skyref.filename is None:
            skyref_file = ''
        else:
            skyref_file = os.path.basename(skyref.filename)

        # store parameters of the method in FITS keywords
        add_mpdaf_method_keywords(self.primary_header,
                                  "drs.pixtable.subtract_slice_median",
                                  ['mask', 'skyref'],
                                  [maskfile, skyref_file],
                                  ['file to mask out all bright objects',
                                   'reference sky spectrum'])

        # autocalib file
        autocalib = PixTableAutoCalib(
            method='drs.pixtable.subtract_slice_median',
            maskfile=maskfile, skyref=skyref_file,
            pixtable=os.path.basename(self.filename),
            ifu=np.ravel(np.swapaxes(np.resize(np.arange(1, 25), (48 * 4, 24)),
                                     0, 1)),
            sli=np.ravel(np.resize(np.arange(1, 49, 0.25).astype(np.int),
                                   (4, 24, 48))),
            quad=np.ravel(np.resize(np.arange(1, 5), (24 * 48, 4))),
            npts=npts, corr=corr)

        self._logger.info('pixtable %s updated',
                          os.path.basename(self.filename))

        # close libray
        # import _ctypes
        # _ctypes.dlclose(libCmethods._handle)
        # libCmethods._handle = None
        # libCmethods._name = None
        # libCmethods._FuncPtr = None
        # del libCmethods
        return autocalib

    def divide_slice_median(self, skyref, pixmask):
        """Computes the median value for all pairs (slices,
        quadrant) and divides each pixel
        by the corresponding factor to bring all slices
        to the same median value.
        pix(x,y,lbda) /= < pix(x,y,lbda) / skyref(lbda) >_slice_quadrant

        Algorithm from Kurt Soto (kurt.soto@phys.ethz.ch)

        Parameters
        ----------
        skyref  : :class:`mpdaf.obj.Spectrum`
                  Reference sky spectrum
        pixmask : :class:`mpdaf.drs.PixTableMask`
                  column corresponding to a mask file
                  (previously computed by mask_column)
        Returns
        -------
        out : :class:`mpdaf.drs.PixTableAutoCalib`
        """

        origin = self.get_origin()
        ifu = self.origin2ifu(origin)
        sli = self.origin2slice(origin)
        xpix = self.origin2xpix(origin)
        ypix = self.origin2ypix(origin)

        # mask
        if pixmask is None:
            maskfile = ''
            maskcol = np.zeros(self.nrows, dtype=bool)
        else:
            maskfile = os.path.basename(pixmask.maskfile)
            maskcol = pixmask.maskcol

        # load the library, using numpy mechanisms
        path = os.path.dirname(__file__)[:-4]
        libCmethods = np.ctypeslib.load_library("libCmethods", path)

        # define argument types
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')

        # setup the return types and argument types
        libCmethods.mpdaf_slice_median.restype = None
        libCmethods.mpdaf_slice_median.argtypes = \
            [array_1d_double, array_1d_double, array_1d_double, array_1d_int,
             array_1d_int, array_1d_int, array_1d_double, array_1d_double,
             ctypes.c_int, array_1d_int, array_1d_double, array_1d_double,
             ctypes.c_int, array_1d_int, array_1d_int, ctypes.c_int]

        data = self.get_data()
        ifu = ifu.astype(np.int32)
        sli = sli.astype(np.int32)
        data = data.astype(np.float64)
        lbda = self.get_lambda()
        lbda = lbda.astype(np.float64)
        mask = maskcol.astype(np.int32)
        skyref_flux = (skyref.data.data.astype(np.float64) * skyref.unit).to(self.unit_data).value
        skyref_lbda = skyref.wave.coord(unit=self.wave)
        skyref_n = skyref.shape
        xpix = xpix.astype(np.int32)
        ypix = ypix.astype(np.int32)

        result = np.empty_like(data, dtype=np.float64)
        result_stat = np.empty_like(data, dtype=np.float64)
        corr = np.ones(24 * 48 * 4, dtype=np.float64)  # zeros
        npts = np.zeros(24 * 48 * 4, dtype=np.int32)

        libCmethods.mpdaf_slice_median(
            result, result_stat, corr, npts, ifu, sli, data, lbda,
            data.shape[0], mask, skyref_flux, skyref_lbda, skyref_n,
            xpix, ypix, 0)

        # set pixtable data
        self.set_data(result)
        self.set_stat(result_stat)

        if skyref.filename is None:
            skyref_file = ''
        else:
            skyref_file = os.path.basename(skyref.filename)

        # store parameters of the method in FITS keywords
        add_mpdaf_method_keywords(self.primary_header,
                                  "drs.pixtable.divide_slice_median",
                                  ['mask', 'skyref'],
                                  [maskfile, skyref_file],
                                  ['file to mask out all bright objects',
                                   'reference sky spectrum'])

        # autocalib file
        autocalib = PixTableAutoCalib(
            method='drs.pixtable.divide_slice_median',
            maskfile=maskfile, skyref=skyref_file,
            pixtable=os.path.basename(self.filename),
            ifu=np.ravel(np.swapaxes(np.resize(np.arange(1, 25), (48 * 4, 24)),
                                     0, 1)),
            sli=np.ravel(np.resize(np.arange(1, 49, 0.25).astype(np.int),
                                   (4, 24, 48))),
            quad=np.ravel(np.resize(np.arange(1, 5), (24 * 48, 4))),
            npts=npts, corr=corr)

        self._logger.info('pixtable %s updated',
                          os.path.basename(self.filename))

        # close libray
        # import _ctypes
        # _ctypes.dlclose(libCmethods._handle)
        # libCmethods._handle = None
        # libCmethods._name = None
        # libCmethods._FuncPtr = None
        # del libCmethods
        return autocalib
