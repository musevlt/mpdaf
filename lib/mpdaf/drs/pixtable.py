"""pixtable.py Manages MUSE pixel table files."""

import datetime
import logging
import os.path
import numpy as np
import warnings
from astropy.io import fits as pyfits
from scipy import interpolate, ndimage
import string

from ..obj import Image, Spectrum, WaveCoord, WCS
from ..obj.objs import is_float, is_int
from ..tools.fits import add_mpdaf_method_keywords

try:
    import numexpr
except:
    numexpr = False


class PixTableMask(object):

    """PixTableMask class.

    This class manages input/output for MUSE pixel table files

    Parameters
    ----------
    filename : string
               The mask file name.
    mask     : array of boolean
               pixtable column corresponding to the mask

    Attributes
    ----------
    filename : string
               The mask file name.
    mask     : array of boolean
               pixtable column corresponding to the mask
    """

    def __init__(self, maskfile, maskcol):
        """creates a PixTable object.

        Parameters
        ----------
        filename : string
                   The mask file name.
        mask     : array of boolean
                   pixtable column corresponding to the mask
        """
        self.maskfile = maskfile
        self.maskcol = maskcol


def write(filename, xpos, ypos, lbda, data, dq, stat, origin, weight=None,
          primary_header=None, save_as_ima=True, wcs='pix', wave='Angstrom'):
    """Saves the object in a FITS file.

    Parameters
    ----------
    filename    : string
                  The FITS filename.
    save_as_ima : bool
                  If True, pixtable is saved as multi-extension FITS
    """
    logger = logging.getLogger('mpdaf corelib')
    pyfits.conf.extension_name_case_sensitive = True

    prihdu = pyfits.PrimaryHDU()
    warnings.simplefilter("ignore")
    if primary_header is not None:
        for card in primary_header.cards:
            try:
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
                d = {'class': 'pixtable', 'method': 'write'}
                logger.warning("%s keyword not written", card.keyword,
                               extra=d)
                pass
    prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
    prihdu.header['author'] = ('MPDAF', 'origin of the file')

    if save_as_ima:
        hdulist = [prihdu]
        nrows = xpos.shape[0]
        hdulist.append(pyfits.ImageHDU(name='xpos',
                                       data=np.float32(xpos.reshape((nrows, 1)))))
        hdulist.append(pyfits.ImageHDU(name='ypos',
                                       data=np.float32(ypos.reshape((nrows, 1)))))
        hdulist.append(pyfits.ImageHDU(name='lambda',
                                       data=np.float32(lbda.reshape((nrows, 1)))))
        hdulist.append(pyfits.ImageHDU(name='data',
                                       data=np.float32(data.reshape((nrows, 1)))))
        hdulist.append(pyfits.ImageHDU(name='dq',
                                       data=np.int32(dq.reshape((nrows, 1)))))
        hdulist.append(pyfits.ImageHDU(name='stat',
                                       data=np.float32(stat.reshape((nrows, 1)))))
        hdulist.append(pyfits.ImageHDU(name='origin',
                                       data=np.int32(origin.reshape((nrows,
                                                                     1)))))
        if weight is not None:
            hdulist.append(pyfits.ImageHDU(name='weight',
                                           data=np.float32(weight.reshape((nrows, 1)))))
        hdu = pyfits.HDUList(hdulist)
        hdu[1].header['BUNIT'] = wcs
        hdu[2].header['BUNIT'] = wcs
        hdu[3].header['BUNIT'] = wave
        hdu[4].header['BUNIT'] = 'count'
        hdu[6].header['BUNIT'] = 'count**2'
        if weight is not None:
            hdu[8].header['BUNIT'] = 'count'
        hdu.writeto(filename, clobber=True, output_verify='fix')
    else:
        cols = []
        cols.append(pyfits.Column(name='xpos', format='1E', unit=wcs,
                                  array=np.float32(xpos)))
        cols.append(pyfits.Column(name='ypos', format='1E', unit=wcs,
                                  array=np.float32(ypos)))
        cols.append(pyfits.Column(name='lambda', format='1E',
                                  unit='Angstrom', array=lbda))
        cols.append(pyfits.Column(name='data', format='1E', unit='count',
                                  array=np.float32(data)))
        cols.append(pyfits.Column(name='dq', format='1J', array=np.int32(dq)))
        cols.append(pyfits.Column(name='stat', format='1E', unit='count**2',
                                  array=np.float32(stat)))
        cols.append(pyfits.Column(name='origin', format='1J', array=np.int32(origin)))
        if weight is not None:
            cols.append(pyfits.Column(name='weight', format='1E',
                                      unit='count', array=np.float32(weight)))
        coltab = pyfits.ColDefs(cols)
        #tbhdu = pyfits.new_table(coltab)
        tbhdu = pyfits.TableHDU(pyfits.FITS_rec.from_columns(coltab))
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True, output_verify='fix')

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
    wcs            : string
                     Type of coordinates of this pixel table
                     ('pix', 'deg' or 'rad')
    ima            : boolean
                     If True, pixtable is saved as multi-extension FITS image
                     instead of FITS binary table.
    """

    def __init__(self, filename, xpos=None, ypos=None, lbda=None, data=None,
                 dq=None, stat=None, origin=None, weight=None,
                 primary_header=None, save_as_ima=True, wcs='pix',
                 wave='Angstrom'):
        """Creates a PixTable object.

        Parameters
        ----------
        filename : string
                   The FITS file name. None by default.

        The FITS file is opened with memory mapping.
        Just the primary header and table dimensions are loaded.
        Methods get_xpos, get_ypos, get_lambda, get_data, get_dq
        ,get_stat and get_origin must be used to get columns data.
        """
        self.logger = logging.getLogger('mpdaf corelib')
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

        if xpos is None and filename is not None:
            try:
                self.hdulist = pyfits.open(self.filename, memmap=1)
                self.primary_header = self.hdulist[0].header
                self.nrows = self.hdulist[1].header["NAXIS2"]
                self.ima = (self.hdulist[1].header['XTENSION'] == 'IMAGE')

                if self.ima:
                    self.wcs = self.hdulist['xpos'].header['BUNIT']
                    self.wave = self.hdulist['lambda'].header['BUNIT']
                else:
                    self.wcs = self.hdulist[1].header['TUNIT1']
                    self.wave = self.hdulist[1].header['TUNIT3']
            except IOError:
                raise IOError('file %s not found' % filename)
        else:
            self.hdulist = None
            if xpos is None or ypos is None or lbda is None or data is None or dq is None or stat is None or origin is None or primary_header is None:
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

            # center in degrees
            try:
                cunit = self.get_keywords("CUNIT1")
            except:
                cunit = 'pix'
            if cunit == 'rad':
                self.xc = self.primary_header['RA'] * 180 / np.pi
                self.yc = self.primary_header['DEC'] * 180 / np.pi
            elif cunit == 'deg':
                self.xc = self.primary_header['RA']
                self.yc = self.primary_header['DEC']
            else:
                self.xc = 0.0
                self.yc = 0.0

    def copy(self):
        """Copies PixTable object in a new one and returns it."""
        result = PixTable()
        result.filename = self.filename
        result.nrows = self.nrows
        result.nifu = self.nifu
        result.skysub = self.skysub
        result.fluxcal = self.fuxcal
        result.wcs = self.wcs
        result.wave = self.wave
        result.ima = self.ima
        result.primary_header = pyfits.Header(self.primary_header)
        result.xpos = self.xpos
        result.ypos = self.ypos
        result.lbda = self.lbda
        result.data = self.data
        result.stat = self.stat
        result.dq = self.dq
        result.origin = self.origin
        result.weight = self.weight
        return result

    def info(self):
        """Prints information."""
        d = {'class': 'PixTable', 'method': 'info'}
        msg = "%i merged IFUs went into this pixel table" % self.nifu
        self.logger.info(msg, extra=d)
        if self.skysub:
            msg = "This pixel table was sky-subtracted"
            self.logger.info(msg, extra=d)
        if self.fluxcal:
            msg = "This pixel table was flux-calibrated"
            self.logger.info(msg, extra=d)
        try:
            msg = '%s (%s)' % (self.primary_header["HIERARCH ESO "
                                                   "DRS MUSE PIXTABLE WCS"],
                               self.primary_header.comments["HIERARCH ESO "
                                                            "DRS MUSE PIXTABLE WCS"])
            self.logger.info(msg, extra=d)
        except:
            try:
                msg = '%s (%s)' % (self.primary_header["HIERARCH ESO "
                                                       "PRO MUSE PIXTABLE WCS"],
                                   self.primary_header.comments["HIERARCH ESO "
                                                                "PRO MUSE PIXTABLE WCS"])
                self.logger.info(msg, extra=d)
            except:
                pass
        try:
            msg = self.hdulist.info()
            self.logger.info(msg, extra=d)
        except:
            msg = 'No\tName\tType\tDim'
            self.logger.info(msg, extra=d)
            msg = '0\tPRIMARY\tcard\t()'
            self.logger.info(msg, extra=d)
            # print "1\t\tTABLE\t(%iR,%iC)" % (self.nrows,self.ncols)

    def write(self, filename, save_as_ima=True):
        """Saves the object in a FITS file.

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
              self.primary_header, save_as_ima, self.wcs, self.wave)

        self.filename = filename
        self.ima = save_as_ima

    def get_xpos(self, ksel=None):
        """Loads the xpos column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.xpos is not None:
            if ksel is None:
                return self.xpos
            else:
                return self.xpos[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        xpos = self.hdulist['xpos'].data[:, 0]
                    else:
                        xpos = self.hdulist[1].data.field('xpos')
                else:
                    if self.ima:
                        xpos = self.hdulist['xpos'].data[ksel, 0][0]
                    else:
                        xpos = self.hdulist[1].data.field('xpos')[ksel]
                return xpos

    def set_xpos(self, xpos, ksel=None):
        """Sets xpos column (or a part of it).

        Parameters
        ----------
        xpos : numpy.array
               xpos values
        ksel : output of np.where
               Elements depending on a condition.
        """
        xpos = np.array(xpos)
        if ksel is None:
            if xpos.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.xpos = xpos
        else:
            if self.xpos is None:
                self.xpos = self.get_xpos()
            self.xpos[ksel] = xpos
        try:
            self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X LOW']\
                = float(self.xpos.min())
            self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X HIGH']\
                = float(self.xpos.max())
        except:
            self.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X LOW']\
                = float(self.xpos.min())
            self.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X HIGH']\
                = float(self.xpos.max())

    def get_ypos(self, ksel=None):
        """Loads the ypos column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.ypos is not None:
            if ksel is None:
                return self.ypos
            else:
                return self.ypos[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        ypos = self.hdulist['ypos'].data[:, 0]
                    else:
                        ypos = self.hdulist[1].data.field('ypos')
                else:
                    if self.ima:
                        ypos = self.hdulist['ypos'].data[ksel, 0][0]
                    else:
                        ypos = self.hdulist[1].data.field('ypos')[ksel]
                return ypos

    def set_ypos(self, ypos, ksel=None):
        """Sets ypos column (or a part of it).

        Parameters
        ----------
        ypos : numpy.array
               ypos values
        ksel : output of np.where
               Elements depending on a condition.
        """
        ypos = np.array(ypos)
        if ksel is None:
            if ypos.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.ypos = ypos
        else:
            if self.ypos is None:
                self.ypos = self.get_ypos()
            self.ypos[ksel] = ypos
        try:
            self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y LOW']\
                = float(self.ypos.min())
            self.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y HIGH']\
                = float(self.ypos.max())
        except:
            self.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y LOW']\
                = float(self.ypos.min())
            self.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y HIGH']\
                = float(self.ypos.max())

    def get_lambda(self, ksel=None):
        """Loads the lambda column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.lbda is not None:
            if ksel is None:
                return self.lbda
            else:
                return self.lbda[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        lbda = self.hdulist['lambda'].data[:, 0]
                    else:
                        lbda = self.hdulist[1].data.field('lambda')
                else:
                    if self.ima:
                        lbda = self.hdulist['lambda'].data[ksel, 0][0]
                    else:
                        lbda = self.hdulist[1].data.field('lambda')[ksel]
                return lbda

    def set_lambda(self, lbda, ksel=None):
        """Sets lambda column (or a part of it).

        Parameters
        ----------
        lbda : numpy.array
               lbda values
        ksel : output of np.where
               Elements depending on a condition.
        """
        lbda = np.array(lbda)
        if ksel is None:
            if lbda.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.lbda = lbda
        else:
            if self.lbda is None:
                self.lbda = self.get_lbda()
            self.lbda[ksel] = lbda
        try:
            self.primary_header['HIERARCH ESO DRS MUSE '
                                'PIXTABLE LIMITS LAMBDA LOW']\
                = float(self.lbda.min())
            self.primary_header['HIERARCH ESO DRS MUSE '
                                'PIXTABLE LIMITS LAMBDA HIGH']\
                = float(self.lbda.max())
        except:
            self.primary_header['HIERARCH ESO PRO MUSE '
                                'PIXTABLE LIMITS LAMBDA LOW']\
                = float(self.lbda.min())
            self.primary_header['HIERARCH ESO PRO MUSE '
                                'PIXTABLE LIMITS LAMBDA HIGH']\
                = float(self.lbda.max())

    def get_data(self, ksel=None):
        """Loads the data column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.data is not None:
            if ksel is None:
                return self.data
            else:
                return self.data[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        data = self.hdulist['data'].data[:, 0]
                    else:
                        data = self.hdulist[1].data.field('data')
                else:
                    if self.ima:
                        data = self.hdulist['data'].data[ksel, 0][0]
                    else:
                        data = self.hdulist[1].data.field('data')[ksel]
                return data

    def set_data(self, data, ksel=None):
        """Sets data column (or a part of it).

        Parameters
        ----------
        data : numpy.array
               data values
        ksel : output of np.where
               Elements depending on a condition.
        """
        data = np.array(data)
        if ksel is None:
            if data.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.data = data
        else:
            if self.data is None:
                self.data = self.get_data()
            self.data[ksel] = data

    def get_stat(self, ksel=None):
        """Loads the stat column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.stat is not None:
            if ksel is None:
                return self.stat
            else:
                return self.stat[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        stat = self.hdulist['stat'].data[:, 0]
                    else:
                        stat = self.hdulist[1].data.field('stat')
                else:
                    if self.ima:
                        stat = self.hdulist['stat'].data[ksel, 0][0]
                    else:
                        stat = self.hdulist[1].data.field('stat')[ksel]
                return stat

    def set_stat(self, stat, ksel=None):
        """Sets stat column (or a part of it).

        Parameters
        ----------
        stat : numpy.array
               stat values
        ksel : output of np.where
               Elements depending on a condition.
        """
        stat = np.array(stat)
        if ksel is None:
            if stat.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.stat = stat
        else:
            if self.stat is None:
                self.stat = self.get_stat()
            self.stat[ksel] = stat

    def get_dq(self, ksel=None):
        """Loads the dq column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.dq is not None:
            if ksel is None:
                return self.dq
            else:
                return self.dq[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        dq = self.hdulist['dq'].data[:, 0]
                    else:
                        dq = self.hdulist[1].data.field('dq')
                else:
                    if self.ima:
                        dq = self.hdulist['dq'].data[ksel, 0][0]
                    else:
                        dq = self.hdulist[1].data.field('dq')[ksel]
                return dq

    def set_dq(self, dq, ksel=None):
        """Sets dq column (or a part of it).

        Parameters
        ----------
        dq   : numpy.array
               dq values
        ksel : output of np.where
               Elements depending on a condition.
        """
        dq = np.array(dq)
        if ksel is None:
            if dq.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.dq = dq
        else:
            if self.dq is None:
                self.dq = self.get_dq()
            self.dq[ksel] = dq

    def get_origin(self, ksel=None):
        """Loads the origin column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.origin is not None:
            if ksel is None:
                return self.origin
            else:
                return self.origin[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        origin = self.hdulist['origin'].data[:, 0]
                    else:
                        origin = self.hdulist[1].data.field('origin')
                else:
                    if self.ima:
                        origin = self.hdulist['origin'].data[ksel, 0][0]
                    else:
                        origin = self.hdulist[1].data.field('origin')[ksel]
                return origin

    def set_origin(self, origin, ksel=None):
        """Sets origin column (or a part of it).

        Parameters
        ----------
        origin : numpy.array
                 origin values
        ksel   : output of np.where
                 Elements depending on a condition.
        """
        origin = np.array(origin)
        if ksel is None:
            if origin.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.origin = origin
        else:
            if self.origin is None:
                self.origin = self.get_origin()
            self.origin[ksel] = origin
        try:
            self.primary_header['HIERARCH ESO DRS MUSE '
                                'PIXTABLE LIMITS IFU LOW'] = \
                int(self.origin2ifu(self.origin).min())
            self.primary_header['HIERARCH ESO DRS MUSE '
                                'PIXTABLE LIMITS IFU HIGH'] = \
                int(self.origin2ifu(self.origin).max())
            self.primary_header['HIERARCH ESO DRS MUSE '
                                'PIXTABLE LIMITS SLICE LOW'] = \
                int(self.origin2slice(self.origin).min())
            self.primary_header['HIERARCH ESO DRS MUSE '
                                'PIXTABLE LIMITS SLICE HIGH'] = \
                int(self.origin2slice(self.origin).max())
        except:
            self.primary_header['HIERARCH ESO PRO MUSE '
                                'PIXTABLE LIMITS IFU LOW'] = \
                int(self.origin2ifu(self.origin).min())
            self.primary_header['HIERARCH ESO PRO MUSE '
                                'PIXTABLE LIMITS IFU HIGH'] = \
                int(self.origin2ifu(self.origin).max())
            self.primary_header['HIERARCH ESO PRO MUSE '
                                'PIXTABLE LIMITS SLICE LOW'] = \
                int(self.origin2slice(self.origin).min())
            self.primary_header['HIERARCH ESO PRO MUSE '
                                'PIXTABLE LIMITS SLICE HIGH'] = \
                int(self.origin2slice(self.origin).max())

        # merged pixtable
        if self.nifu > 1:
            try:
                self.primary_header["HIERARCH ESO DRS MUSE PIXTABLE MERGED"]\
                    = len(np.unique(self.origin2ifu(self.origin)))
            except:
                self.primary_header["HIERARCH ESO PRO MUSE PIXTABLE MERGED"]\
                    = len(np.unique(self.origin2ifu(self.origin)))

    def get_weight(self, ksel=None):
        """Loads the weight column and returns it.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        if self.weight is not None:
            if ksel is None:
                return self.weight
            else:
                return self.weight[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                try:
                    if self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE WEIGHTED"):
                        if ksel is None:
                            if self.ima:
                                weight = self.hdulist['weight'].data[:, 0]
                            else:
                                weight = self.hdulist[1].data.field('weight')
                        else:
                            if self.ima:
                                weight = self.hdulist['weight'].data[ksel, 0][0]
                            else:
                                weight = self.hdulist[1].data.field('weight')[ksel]
                except:
                    weight = None
                return weight

    def set_weight(self, weight, ksel=None):
        """Sets weight column (or a part of it).

        Parameters
        ----------
        weight : numpy.array
                 weight values
        ksel   : output of np.where
                 Elements depending on a condition.
        """
        weight = np.array(weight)
        if ksel is None:
            if weight.shape[0] != self.nrows:
                raise IOError('Wrong dimension number')
            else:
                self.weight = weight
        else:
            if self.weight is None:
                self.weight = self.get_weight()
            self.weight[ksel] = weight

    def get_exp(self):
        """Loads the exposure numbers and returns it as a column.

        Returns
        -------
        out : numpy.memmap
        """
        try:
            nexp = self.get_keywords("HIERARCH ESO DRS MUSE "
                                     "PIXTABLE COMBINED")
            exp = np.empty(shape=(self.nrows))
            for i in range(1, nexp + 1):
                first = self.get_keywords("HIERARCH ESO DRS MUSE "
                                          "PIXTABLE EXP%i FIRST" % i)
                last = self.get_keywords("HIERARCH ESO DRS MUSE "
                                         "PIXTABLE EXP%i LAST" % i)
                exp[first:last + 1] = i
        except:
            exp = None
        return exp

    def _extract(self, filename=None, sky=None, lbda=None, ifu=None,
                 sl=None, xpix=None, ypix=None, exp=None):
        # type of coordinates
        primary_header = self.primary_header.copy()
        if self.nrows == 0:
            return None

        # To start select the whole pixtable
        kmask = np.ones(self.nrows).astype('bool')

        # Do the selection on the sky
        if sky is not None:
            xpos, ypos = self.get_pos_sky(self.get_xpos(), self.get_ypos())
            if (isinstance(sky, tuple)):
                sky = [sky]
            mask = np.zeros(self.nrows).astype('bool')
            for y0, x0, size, shape in sky:
                if shape == 'C':
                    if self.wcs == 'deg':
                        mask |= (((xpos - x0) * 3600
                                  * np.cos(y0 * np.pi / 180.)) ** 2
                                 + ((ypos - y0) * 3600) ** 2) \
                            < size ** 2
                    elif self.wcs == 'rad':
                        mask |= (((xpos - x0) * 3600 * 180 / np.pi
                                  * np.cos(y0)) ** 2
                                 + ((ypos - y0) * 3600 * 180 / np.pi)
                                 ** 2) < size ** 2
                    else:
                        mask |= ((xpos - x0) ** 2
                                 + (ypos - y0) ** 2) < size ** 2
                elif shape == 'S':
                    if self.wcs == 'deg':
                        mask |= (np.abs((xpos - x0) * 3600
                                        * np.cos(y0 * np.pi / 180.))
                                 < size) \
                            & (np.abs((ypos - y0) * 3600)
                               < size)
                    elif self.wcs == 'rad':
                        mask |= (np.abs((xpos - x0) * 3600 * 180
                                        / np.pi * np.cos(y0)) < size) \
                            & (np.abs((ypos - y0) * 3600 * 180
                                      / np.pi) < size)
                    else:
                        mask |= (np.abs(xpos - x0) < size) \
                            & (np.abs(ypos - y0) < size)
                else:
                    raise ValueError('Unknown shape parameter')
            kmask &= mask
            del mask
            del xpos
            del ypos

        # Do the selection on wavelengths
        if lbda is not None:
            col_lambda = self.get_lambda()
            if (isinstance(lbda, tuple)):
                lbda = [lbda]
            mask = np.zeros(self.nrows).astype('bool')
            for l1, l2 in lbda:
                mask |= (col_lambda >= l1) & (col_lambda < l2)
            kmask &= mask
            del mask
            del col_lambda

        # Do the selection on the origin column
        if (ifu is not None) or (sl is not None) or \
                (xpix is not None) or (ypix is not None):
            col_origin = self.get_origin()
            if sl is not None:
                if hasattr(sl, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for s in sl:
                        mask |= (self.origin2slice(col_origin) == s)
                    kmask &= mask
                    del mask
                else:
                    kmask &= (self.origin2slice(col_origin) == sl)
            if ifu is not None:
                if hasattr(ifu, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for i in ifu:
                        mask |= (self.origin2ifu(col_origin) == i)
                    kmask &= mask
                    del mask
                else:
                    kmask &= (self.origin2ifu(col_origin) == ifu)
            if xpix is not None:
                col_xpix = self.origin2xpix(col_origin)
                if hasattr(xpix, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for x1, x2 in xpix:
                        mask |= (col_xpix >= x1) & (col_xpix < x2)
                    kmask &= mask
                    del mask
                else:
                    x1, x2 = xpix
                    kmask &= (col_xpix >= x1) & (col_xpix < x2)
                del col_xpix
            if ypix is not None:
                col_ypix = self.origin2ypix(col_origin)
                if hasattr(ypix, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for y1, y2 in ypix:
                        mask |= (col_ypix >= y1) & (col_ypix < y2)
                    kmask &= mask
                    del mask
                else:
                    y1, y2 = ypix
                    kmask &= (col_ypix >= y1) & (col_ypix < y2)
                del col_ypix
            del col_origin

        # Do the selection on the exposure numbers
        if exp is not None:
            col_exp = self.get_exp()
            if col_exp is not None:
                mask = np.zeros(self.nrows).astype('bool')
                for iexp in exp:
                    mask |= (col_exp == iexp)
                kmask &= mask
                del mask
                del col_exp

        # Compute the new pixtable
        ksel = np.where(kmask)
        del kmask
        nrows = len(ksel[0])
        if nrows == 0:
            return None
        # xpos
        xpos = self.get_xpos(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X LOW'] = \
                float(xpos.min())
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X HIGH'] = \
                float(xpos.max())
        except:
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X LOW'] = \
                float(xpos.min())
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X HIGH'] = \
                float(xpos.max())
        # ypos
        ypos = self.get_ypos(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y LOW'] = \
                float(ypos.min())
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y HIGH'] = \
                float(ypos.max())
        except:
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y LOW'] = \
                float(ypos.min())
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y HIGH'] = \
                float(ypos.max())
        # lambda
        lbda = self.get_lambda(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS LAMBDA LOW'] = float(lbda.min())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS LAMBDA HIGH'] = float(lbda.max())
        except:
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS LAMBDA LOW'] = float(lbda.min())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS LAMBDA HIGH'] = float(lbda.max())
        # data
        data = self.get_data(ksel)
        # variance
        stat = self.get_stat(ksel)
        # pixel quality
        dq = self.get_dq(ksel)
        # origin
        origin = self.get_origin(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS IFU LOW'] = \
                int(self.origin2ifu(origin).min())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS IFU HIGH'] = \
                int(self.origin2ifu(origin).max())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS SLICE LOW'] = \
                int(self.origin2slice(origin).min())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS SLICE HIGH'] = \
                int(self.origin2slice(origin).max())
        except:
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS IFU LOW'] = \
                int(self.origin2ifu(origin).min())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS IFU HIGH'] = \
                int(self.origin2ifu(origin).max())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS SLICE LOW'] = \
                int(self.origin2slice(origin).min())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS SLICE HIGH'] = \
                int(self.origin2slice(origin).max())
        # merged pixtable
        if self.nifu > 1:
            try:
                primary_header["HIERARCH ESO DRS MUSE PIXTABLE MERGED"] = \
                    len(np.unique(self.origin2ifu(origin)))
            except:
                primary_header["HIERARCH ESO PRO MUSE PIXTABLE MERGED"] = \
                    len(np.unique(self.origin2ifu(origin)))

        # weight
        weight = self.get_weight(ksel)

        # combined exposures
        selfexp = self.get_exp()
        if selfexp is not None:
            newexp = selfexp[ksel]
            numbers_exp = np.unique(newexp)
            try:
                primary_header["HIERARCH ESO DRS MUSE PIXTABLE COMBINED"] = \
                    len(numbers_exp)
                for iexp, i in zip(numbers_exp, range(1,
                                                      len(numbers_exp) + 1)):
                    ksel = np.where(newexp == iexp)
                    primary_header["HIERARCH ESO DRS MUSE "
                                   "PIXTABLE EXP%i FIRST" % i] = ksel[0][0]
                    primary_header["HIERARCH ESO DRS MUSE "
                                   "PIXTABLE EXP%i LAST" % i] = ksel[0][-1]
                for i in range(len(numbers_exp) + 1,
                               self.get_keywords("HIERARCH ESO DRS MUSE "
                                                 "PIXTABLE COMBINED") + 1):
                    del primary_header["HIERARCH ESO DRS MUSE "
                                       "PIXTABLE EXP%i FIRST" % i]
                    del primary_header["HIERARCH ESO DRS MUSE "
                                       "PIXTABLE EXP%i LAST" % i]
            except:
                primary_header["HIERARCH ESO PRO MUSE PIXTABLE COMBINED"] = \
                    len(numbers_exp)
                for iexp, i in zip(numbers_exp,
                                   range(1, len(numbers_exp) + 1)):
                    ksel = np.where(newexp == iexp)
                    primary_header["HIERARCH ESO PRO MUSE "
                                   "PIXTABLE EXP%i FIRST" % i] = ksel[0][0]
                    primary_header["HIERARCH ESO PRO MUSE "
                                   "PIXTABLE EXP%i LAST" % i] = ksel[0][-1]
                for i in range(len(numbers_exp) + 1,
                               self.get_keywords("HIERARCH ESO PRO MUSE '\
                               PIXTABLE COMBINED") + 1):
                    del primary_header["HIERARCH ESO PRO MUSE "
                                       "PIXTABLE EXP%i FIRST" % i]
                    del primary_header["HIERARCH ESO PRO MUSE "
                                       "PIXTABLE EXP%i LAST" % i]

        # return sub pixtable
        if filename is None:
            return PixTable(None, xpos, ypos, lbda, data, dq, stat, origin,
                            weight, primary_header, self.ima, self.wcs, self.wave)
        else:  # write the result in a new file
            write(filename, xpos, ypos, lbda, data, dq, stat, origin,
                  weight, primary_header, self.ima, self.wcs, self.wave)
            return PixTable(filename)

    def _extract_numexpr(self, filename=None, sky=None, lbda=None, ifu=None,
                         sl=None, xpix=None, ypix=None, exp=None):
        # type of coordinates
        primary_header = self.primary_header.copy()
        if self.nrows == 0:
            return None

        # To start select the whole pixtable
        kmask = np.ones(self.nrows).astype('bool')

        # Do the selection on the sky
        if sky is not None:
            xpos, ypos = self.get_pos_sky()
            pi = np.pi
            if (isinstance(sky, tuple)):
                sky = [sky]
            mask = np.zeros(self.nrows).astype('bool')
            for y0, x0, size, shape in sky:
                if shape == 'C':
                    if self.wcs == 'deg':
                        mask |= numexpr.evaluate('(((xpos - x0) * 3600 * cos(y0 * pi / 180.)) ** 2 + ((ypos - y0) * 3600) ** 2) < size ** 2')
                    elif self.wcs == 'rad':
                        mask |= numexpr.evaluate('(((xpos - x0) * 3600 * 180 / pi * cos(y0)) ** 2 + ((ypos - y0) * 3600 * 180 / pi)** 2) < size ** 2')
                    else:
                        mask |= numexpr.evaluate('((xpos - x0) ** 2 + (ypos - y0) ** 2) < size ** 2')
                elif shape == 'S':
                    if self.wcs == 'deg':
                        mask |= numexpr.evaluate('(abs((xpos - x0) * 3600 * cos(y0 * pi / 180.)) < size) & (abs((ypos - y0) * 3600) < size)')
                    elif self.wcs == 'rad':
                        mask |= numexpr.evaluate('(abs((xpos - x0) * 3600 * 180 / pi * cos(y0)) < size) & (abs((ypos - y0) * 3600 * 180 / pi) < size)')
                    else:
                        mask |= numexpr.evaluate('(abs(xpos - x0) < size) & (abs(ypos - y0) < size)')
                else:
                    raise ValueError('Unknown shape parameter')
            kmask &= mask
            del mask
            del xpos
            del ypos

        # Do the selection on wavelengths
        if lbda is not None:
            col_lambda = self.get_lambda()
            if (isinstance(lbda, tuple)):
                lbda = [lbda]
            mask = np.zeros(self.nrows).astype('bool')
            for l1, l2 in lbda:
                mask |= numexpr.evaluate('(col_lambda >= l1) & (col_lambda < l2)')
            kmask &= mask
            del mask
            del col_lambda

        # Do the selection on the origin column
        if (ifu is not None) or (sl is not None) or \
                (xpix is not None) or (ypix is not None):
            col_origin = self.get_origin()
            if sl is not None:
                if hasattr(sl, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    sli = self.origin2slice(col_origin)
                    for s in sl:
                        mask |= numexpr.evaluate('sli == s')
                    kmask &= mask
                    del mask, sli
                else:
                    sli = self.origin2slice(col_origin)
                    kmask &= numexpr.evaluate('sli == sl')
                    del sli
            if ifu is not None:
                if hasattr(ifu, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    _i = self.origin2ifu(col_origin)
                    for i in ifu:
                        mask |= numexpr.evaluate('_i == i')
                    kmask &= mask
                    del mask, _i
                else:
                    _i = self.origin2ifu(col_origin)
                    kmask &= numexpr.evaluate('_i == ifu')
                    del _i
            if xpix is not None:
                col_xpix = self.origin2xpix(col_origin)
                if hasattr(xpix, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for x1, x2 in xpix:
                        mask |= numexpr.evaluate('(col_xpix >= x1) & (col_xpix < x2)')
                    kmask &= mask
                    del mask
                else:
                    x1, x2 = xpix
                    kmask &= numexpr.evaluate('(col_xpix >= x1) & (col_xpix < x2)')
                del col_xpix
            if ypix is not None:
                col_ypix = self.origin2ypix(col_origin)
                if hasattr(ypix, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for y1, y2 in ypix:
                        mask |= numexpr.evaluate('(col_ypix >= y1) & (col_ypix < y2)')
                    kmask &= mask
                    del mask
                else:
                    y1, y2 = ypix
                    kmask &= numexpr.evaluate('(col_ypix >= y1) & (col_ypix < y2)')
                del col_ypix
            del col_origin

        # Do the selection on the exposure numbers
        if exp is not None:
            col_exp = self.get_exp()
            if col_exp is not None:
                mask = np.zeros(self.nrows).astype('bool')
                for iexp in exp:
                    mask |= numexpr.evaluate('col_exp == iexp')
                kmask &= mask
                del mask
                del col_exp

        # Compute the new pixtable
        ksel = np.where(kmask)
        del kmask
        nrows = len(ksel[0])
        if nrows == 0:
            return None
        # xpos
        xpos = self.get_xpos(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X LOW'] = \
                float(xpos.min())
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X HIGH'] = \
                float(xpos.max())
        except:
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X LOW'] = \
                float(xpos.min())
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X HIGH'] = \
                float(xpos.max())
        # ypos
        ypos = self.get_ypos(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y LOW'] = \
                float(ypos.min())
            primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y HIGH'] = \
                float(ypos.max())
        except:
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y LOW'] = \
                float(ypos.min())
            primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y HIGH'] = \
                float(ypos.max())
        # lambda
        lbda = self.get_lambda(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS LAMBDA LOW'] = float(lbda.min())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS LAMBDA HIGH'] = float(lbda.max())
        except:
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS LAMBDA LOW'] = float(lbda.min())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS LAMBDA HIGH'] = float(lbda.max())
        # data
        data = self.get_data(ksel)
        # variance
        stat = self.get_stat(ksel)
        # pixel quality
        dq = self.get_dq(ksel)
        # origin
        origin = self.get_origin(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS IFU LOW'] = \
                int(self.origin2ifu(origin).min())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS IFU HIGH'] = \
                int(self.origin2ifu(origin).max())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS SLICE LOW'] = \
                int(self.origin2slice(origin).min())
            primary_header['HIERARCH ESO DRS MUSE '
                           'PIXTABLE LIMITS SLICE HIGH'] = \
                int(self.origin2slice(origin).max())
        except:
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS IFU LOW'] = \
                int(self.origin2ifu(origin).min())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS IFU HIGH'] = \
                int(self.origin2ifu(origin).max())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS SLICE LOW'] = \
                int(self.origin2slice(origin).min())
            primary_header['HIERARCH ESO PRO MUSE '
                           'PIXTABLE LIMITS SLICE HIGH'] = \
                int(self.origin2slice(origin).max())
        # merged pixtable
        if self.nifu > 1:
            try:
                primary_header["HIERARCH ESO DRS MUSE PIXTABLE MERGED"] = \
                    len(np.unique(self.origin2ifu(origin)))
            except:
                primary_header["HIERARCH ESO PRO MUSE PIXTABLE MERGED"] = \
                    len(np.unique(self.origin2ifu(origin)))

        # weight
        weight = self.get_weight(ksel)

        # combined exposures
        selfexp = self.get_exp()
        if selfexp is not None:
            newexp = selfexp[ksel]
            numbers_exp = np.unique(newexp)
            try:
                primary_header["HIERARCH ESO DRS MUSE PIXTABLE COMBINED"] = \
                    len(numbers_exp)
                for iexp, i in zip(numbers_exp, range(1,
                                                      len(numbers_exp) + 1)):
                    ksel = np.where(newexp == iexp)
                    primary_header["HIERARCH ESO DRS MUSE "
                                   "PIXTABLE EXP%i FIRST" % i] = ksel[0][0]
                    primary_header["HIERARCH ESO DRS MUSE "
                                   "PIXTABLE EXP%i LAST" % i] = ksel[0][-1]
                for i in range(len(numbers_exp) + 1,
                               self.get_keywords("HIERARCH ESO DRS MUSE "
                                                 "PIXTABLE COMBINED") + 1):
                    del primary_header["HIERARCH ESO DRS MUSE "
                                       "PIXTABLE EXP%i FIRST" % i]
                    del primary_header["HIERARCH ESO DRS MUSE "
                                       "PIXTABLE EXP%i LAST" % i]
            except:
                primary_header["HIERARCH ESO PRO MUSE PIXTABLE COMBINED"] = \
                    len(numbers_exp)
                for iexp, i in zip(numbers_exp,
                                   range(1, len(numbers_exp) + 1)):
                    ksel = np.where(newexp == iexp)
                    primary_header["HIERARCH ESO PRO MUSE "
                                   "PIXTABLE EXP%i FIRST" % i] = ksel[0][0]
                    primary_header["HIERARCH ESO PRO MUSE "
                                   "PIXTABLE EXP%i LAST" % i] = ksel[0][-1]
                for i in range(len(numbers_exp) + 1,
                               self.get_keywords("HIERARCH ESO PRO MUSE '\
                               PIXTABLE COMBINED") + 1):
                    del primary_header["HIERARCH ESO PRO MUSE "
                                       "PIXTABLE EXP%i FIRST" % i]
                    del primary_header["HIERARCH ESO PRO MUSE "
                                       "PIXTABLE EXP%i LAST" % i]

        # write the result in a new file
        if filename is None:
            return PixTable(None, xpos, ypos, lbda, data, dq, stat, origin,
                            weight, primary_header, self.ima, self.wcs, self.wave)
        else:
            write(filename, xpos, ypos, lbda, data, dq, stat, origin,
                  weight, primary_header, self.ima, self.wcs, self.wave)
            return PixTable(filename)

    def extract(self, filename=None, sky=None, lbda=None, ifu=None,
                sl=None, xpix=None, ypix=None, exp=None):
        """Extracts a subset of a pixtable using the following criteria:

        - aperture on the sky (center, size and shape)

        - wavelength range

        - IFU number

        - slice number

        - detector pixels

        - exposure numbers

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
                   (min, max) wavelength range in wavelength unit.
        ifu      : int
                   IFU number.
        sl       : int
                   Slice number on the CCD.
        xpix     : (int, int)
                   (min, max) pixel range along the X axis
        ypix     : (int, int)
                   (min, max) pixel range along the Y axis
        exp      : list of integers
                   List of exposure numbers

        Returns
        -------
        out : PixTable
        """
        if numexpr:
            return self._extract(filename, sky, lbda, ifu, sl,
                                 xpix, ypix, exp)
        else:
            return self._extract_numexpr(filename, sky, lbda, ifu,
                                         sl, xpix, ypix, exp)

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

    def origin2xoffset(self, origin):
        """Converts the origin value and returns the x coordinates offset.

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : float
        """
        col_ifu = self.origin2ifu(origin)
        col_slice = self.origin2slice(origin)
        if isinstance(origin, np.ndarray):
            xoffset = np.zeros_like(origin)
            for ifu in np.unique(col_ifu):
                for sl in np.unique(col_slice):
                    value = self.get_keywords('HIERARCH ESO DRS MUSE '
                                              'PIXTABLE EXP0 IFU%02d '
                                              'SLICE%02d XOFFSET'
                                              % (ifu, sl))
                    xoffset[np.where((col_ifu == ifu)
                                     & (col_slice == sl))] = value
        else:
            xoffset = self.get_keywords("HIERARCH ESO DRS MUSE "
                                        "PIXTABLE EXP0 IFU%02d "
                                        "SLICE%02d XOFFSET" % (col_ifu, col_slice))
        return xoffset

    def origin2xpix(self, origin):
        """Converts the origin value and returns the x coordinates.

        Parameters
        ----------
        origin : integer
                 Origin value.

        Returns
        -------
        out : float
        """
        return (self.origin2xoffset(origin) + ((origin >> 24) & 0x7f) - 1).astype(np.uint16)

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
        return (self.origin2ifu(origin), self.origin2slice(origin),
                self.origin2ypix(origin), self.origin2xpix(origin))

    def _get_pos_sky(self, xpos, ypos):
        try:
            spheric = (self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE WCS")[0:9] == 'projected')
        except:
            spheric = False
        if spheric:  # spheric coordinates
            phi = xpos
            theta = ypos + np.pi / 2
            dp = self.yc * np.pi / 180
            ra = np.arctan2(np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.cos(dp) + np.cos(theta) * np.sin(dp) * np.cos(phi)) * 180 / np.pi
            xpos_sky = self.xc + ra
            ypos_sky = np.arcsin(np.sin(theta) * np.sin(dp) - np.cos(theta) * np.cos(dp) * np.cos(phi)) * 180 / np.pi
        else:
            if self.wcs == 'deg':
                dp = self.yc * np.pi / 180
                xpos_sky = self.xc + xpos / np.cos(dp)
                ypos_sky = self.yc + ypos
            elif self.wcs == 'rad':
                dp = self.yc * np.pi / 180
                xpos_sky = self.xc + xpos * 180 / np.pi / np.cos(dp)
                ypos_sky = self.yc + ypos * 180 / np.pi
            else:
                xpos_sky = self.xc + xpos
                ypos_sky = self.yc + ypos
        return xpos_sky, ypos_sky

    def _get_pos_sky_numexpr(self, xpos, ypos):
        try:
            spheric = (self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE WCS")[0:9] == 'projected')
        except:
            spheric = False
        if spheric:  # spheric coordinates
            phi = xpos
            pi = np.pi
            theta = numexpr.evaluate("ypos + pi/2")
            yc = self.yc
            dp = numexpr.evaluate("yc * pi / 180")
            ra = numexpr.evaluate("arctan2(cos(theta) * sin(phi), sin(theta) * cos(dp) + cos(theta) * sin(dp) * cos(phi)) * 180 / pi")
            xc = self.xc
            xpos_sky = numexpr.evaluate("xc + ra")
            ypos_sky = numexpr.evaluate("arcsin(sin(theta) * sin(dp) - cos(theta) * cos(dp) * cos(phi)) * 180 / pi")
        else:
            if self.wcs == 'deg':
                yc = self.yc
                pi = np.pi
                dp = numexpr.evaluate("yc * pi / 180")
                xc = self.xc
                xpos_sky = numexpr.evaluate("xc + xpos / cos(dp)")
                ypos_sky = numexpr.evaluate("yc + ypos")
            elif self.wcs == 'rad':
                yc = self.yc
                pi = np.pi
                dp = numexpr.evaluate("yc * pi / 180")
                xc = self.xc
                xpos_sky = numexpr.evaluate("xc + xpos * 180 / pi / cos(dp)")
                ypos_sky = numexpr.evaluate("yc + ypos * 180 / pi")
            else:
                xc = self.xc
                yc = self.yc
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

        # build the slicelist
        slicelist = []
        for ifu in np.unique(ifupix):
            for sl in np.unique(slicepix):
                slicelist.append((ifu, sl))
        nslice = len(slicelist)
        slicelist = np.array(slicelist)

        # compute mean sky position of each slice
        skypos = []
        for ifu, sl in slicelist:
            k = np.where((ifupix == ifu) & (slicepix == sl))
            skypos.append((col_xpos[k].mean(), col_ypos[k].mean()))
        skypos = np.array(skypos)

        slices = {'list': slicelist, 'skypos': skypos,
                  'ifupix': ifupix, 'slicepix': slicepix,
                  'xpix': xpix, 'ypix': ypix}

        if verbose:
            d = {'class': 'PixTable', 'method': 'get_slices'}
            msg = '%d slices found, structure returned in \
                   slices dictionary ' % nslice
            self.logger.info(msg, extra=d)

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
        # HIERARCH ESO PRO MUSE has been renamed into HIERARCH ESO DRS MUSE
        # in recent versions of the DRS.
        if key.startswith('HIERARCH ESO PRO MUSE'):
            alternate_key = key.replace('HIERARCH ESO PRO MUSE',
                                        'HIERARCH ESO DRS MUSE')
        elif key.startswith('HIERARCH ESO DRS MUSE'):
            alternate_key = key.replace('HIERARCH ESO DRS MUSE',
                                        'HIERARCH ESO PRO MUSE')
        else:
            alternate_key = key
        try:
            return self.primary_header[key]
        except:
            return self.primary_header[alternate_key]

    def reconstruct_sky_image(self, lbda=None, step=None):
        """Reconstructs the image on the sky from the pixtable.

        Parameters
        ----------
        lbda : (float,float)
               (min, max) wavelength range in Angstrom.
               If None, the image is reconstructed for all wavelengths.
        step : (float,float)
               Pixel steps of the final image
               (in arcsec if the coordinates of this pixel table
               are world coordinates on the sky ).
               If None, the value corresponding to the keyword
               "HIERARCH ESO INS PIXSCALE" is used.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`
        """
        # TODO replace by DRS
        # step in arcsec

        if step is None:
            step = self.get_keywords('HIERARCH ESO OCS IPS PIXSCALE')
            if step <= 0:
                raise ValueError('INS PIXSCALE not valid')
            xstep = step
            ystep = step
        else:
            ystep, xstep = step

        col_dq = self.get_dq()
        if lbda is None:
            ksel = np.where((col_dq == 0))
        else:
            l1, l2 = lbda
            col_lambda = self.get_lambda()
            ksel = np.where((col_dq == 0) & (col_lambda > l1)
                            & (col_lambda < l2))
            del col_lambda
        del col_dq

        x = self.get_xpos(ksel)
        y = self.get_ypos(ksel)
        data = self.get_data(ksel)

        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)

        if self.wcs == "deg":  # arcsec to deg
            xstep /= (-3600. * np.cos((ymin + ymax) * np.pi / 180. / 2.))
            ystep /= 3600.
        elif self.wcs == "rad":  # arcsec to rad
            xstep /= (-3600. * 180. / np.pi * np.cos((ymin + ymax) / 2.))
            ystep /= (3600. * 180. / np.pi)
        else:  # pix
            pass

        nx = 1 + int((xmin - xmax) / xstep)
        grid_x = np.arange(nx) * xstep + xmax
        ny = 1 + int((ymax - ymin) / ystep)
        grid_y = np.arange(ny) * ystep + ymin
        shape = (ny, nx)

        points = np.empty((len(ksel[0]), 2), dtype=float)
        points[:, 0] = y
        points[:, 1] = x

        new_data = interpolate.griddata(points, data,
                                        np.meshgrid(grid_y, grid_x),
                                        method='linear').T

        wcs = WCS(crpix=(1.0, 1.0), crval=(ymin, xmax),
                  cdelt=(ystep, xstep), shape=shape)
        ima = Image(data=new_data, wcs=wcs)
        return ima

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
        return Image(shape=(image.shape), data=image, wcs=wcs)

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

        return Image(shape=(image.shape), data=image, wcs=wcs)

    def mask_column(self, maskfile=None, verbose=True):
        """Computes the mask column correcponding to a mask file

        Parameters
        ----------
        maskfile : string
                   mask file to mask out all bright
                   continuum objects present in the FoV
        verbose : boolean
                  If True, progression is printed.

        Returns
        -------
        out : :class:`mpdaf.drs.PixTableMask`
        """
        d = {'class': 'PixTable', 'method': 'mask_column'}

        mask = np.zeros(self.nrows, dtype=np.bool)
        if maskfile is None:
            return mask

        pos = np.array(self.get_pos_sky()[::-1]).T
        xpos_sky = pos[:, 1]
        ypos_sky = pos[:, 0]

        ima_mask = Image(maskfile)
        label = ndimage.measurements.label(ima_mask.data.data)[0]
        ulabel = np.unique(label)
        ulabel = ulabel[ulabel > 0]

        for i in ulabel:
            try:
                ksel = np.where(label == i)
                item = (slice(min(ksel[0]), max(ksel[0]) + 1, None),
                        slice(min(ksel[1]), max(ksel[1]) + 1, None))
                wcs = ima_mask.wcs[item]
                coord = wcs.get_range()
                step = wcs.get_step()
                y0, x0 = coord.min(axis=0) - step / 2
                y1, x1 = coord.max(axis=0) + step / 2
                ksel = np.where((xpos_sky > x0) & (xpos_sky < x1) &
                                (ypos_sky > y0) & (ypos_sky < y1))
                if verbose:
                    msg = 'masking object %i %g<x<%g %g<y<%g (%i pixels)' % (
                        i, x0, x1, y0, y1, len(ksel[0]))
                    self.logger.info(msg, extra=d)
                if len(ksel[0]) != 0:
                    pix = ima_mask.wcs.sky2pix(pos[ksel], nearest=True)
                    mask[ksel] |= (ima_mask.data.data[pix[:, 0], pix[:, 1]] != 0)
            except Exception as e:
                self.logger.warning('masking object %i failed', i, extra=d)

        return PixTableMask(maskfile, mask)

    def old_subtract_slice_median(self, mask=None, norm='sky'):
        """Computes the median value for all slices and applies in place a
        subtractive correction to each slice to bring all slices to the same
        median value.

        pix(x,y,lbda) -= < pix(x,y,lbda) >_slice (+ < pix(x,y,lbda) >_all)

        Parameters
        ----------
        mask : :class:`mpdaf.drs.PixTableMask`
               column corresponding to a mask file
               (previously computed by mask_column)
        norm : string
               Option for sky subtraction 'sky' or 'zero'
        """
        d = {'class': 'PixTable', 'method': 'old_subtract_slice_median'}

        origin = self.get_origin()
        ifu = self.origin2ifu(origin)
        sli = self.origin2slice(origin)

        if mask is None:
            maskfile = ''
            mask = np.zeros(self.nrows).astype('bool')
        else:
            maskfile = os.path.basename(mask.maskfile)
            mask = mask.maskcol

        import mpdaf
        import ctypes

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods",
                                                mpdaf.__path__[0])

        # define argument types
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')

        # setup the return types and argument types
        libCmethods.mpdaf_old_subtract_slice_median.restype = None
        libCmethods.mpdaf_old_subtract_slice_median.argtypes = \
            [array_1d_double, array_1d_double, array_1d_int,
             array_1d_int, array_1d_int, array_1d_double,
             array_1d_double, ctypes.c_int, array_1d_int, ctypes.c_int]

        data = self.get_data()
        ifu = ifu.astype(np.int32)
        sli = sli.astype(np.int32)
        data = data.astype(np.float64)
        lbda = self.get_lambda()
        lbda = lbda.astype(np.float64)
        mask = mask.astype(np.int32)
        if norm == 'sky':
            skysub = 1
        else:
            skysub = 0
        skysub = np.int32(skysub)

        result = np.empty_like(data).astype(np.float64)
        corr = np.empty(24 * 48).astype(np.float64)
        npts = np.empty(24 * 48).astype(np.int32)

        libCmethods.mpdaf_old_subtract_slice_median(result, corr, npts,
                                                    ifu, sli, data, lbda,
                                                    data.shape[0], mask,
                                                    skysub)

        # set pixtable data
        self.set_data(result)

        # write autocalib file
        filename = 'autocalib_' + \
            string.replace(os.path.basename(self.filename), '.fits', '.txt')

        header = 'ifu number| slice number | number of remaining pixels | correction value\n'
        header += 'drs.pixtable.old_subtract_slice_median\n'
        header += '   file to mask out all bright objects: %s\n' % maskfile
        header += '   option for sky subtraction: %s\n' % norm

        np.savetxt(filename,
                   zip(np.ravel(np.swapaxes(np.resize(np.arange(1, 25), (48, 24)), 0, 1)),
                       np.ravel(np.resize(np.arange(1, 49), (24, 48))),
                       npts, corr),
                   fmt='%02d %02d %d %g', header=header)

        # add keywords
        add_mpdaf_method_keywords(self.primary_header,
                                  "drs.pixtable.old_subtract_slice_median",
                                  ['mask', 'norm'],
                                  [maskfile, norm],
                                  ['file to mask out all bright objects',
                                   'Option for sky subtraction'])

        self.logger.info('pixtable %s updated' % os.path.basename(self.filename), extra=d)
        self.logger.info('%s saved' % filename, extra=d)

        # close libray
        #import _ctypes
        #_ctypes.dlclose(libCmethods._handle)
        #libCmethods._handle = None
        #libCmethods._name = None
        #libCmethods._FuncPtr = None
        del libCmethods

    def sky_ref(self, mask=None, dlbda=1.0, nmax=2, nclip=5.0, nstop=2):
        """Computes the reference sky spectrum using sigma clipped median.

        Parameters
        ----------
        mask  : :class:`mpdaf.drs.PixTableMask`
                column corresponding to a mask file
                (previously computed by mask_column)
        dlbda : double
                   wavelength step
        nmax  : integer
                maximum number of clipping iterations
        nclip : float or (float,float)
                Number of sigma at which to clip.
                Single clipping parameter or lower/upper clipping parameters
        nstop : integer
                If the number of not rejected pixels is less
                than this number, the clipping iterations stop.

        Returns
        -------
        out : :class:`mpdaf.obj.Spectrum`
        """
        # mask
        if mask is None:
            maskfile = ''
            mask = np.zeros(self.nrows).astype('bool')
        else:
            maskfile = os.path.basename(mask.maskfile)
            mask = mask.maskcol

        # sigma clipped parameters
        import mpdaf
        import ctypes
        if is_int(nclip) or is_float(nclip):
            nclip_low = nclip
            nclip_up = nclip
        else:
            nclip_low = nclip[0]
            nclip_up = nclip[1]
        # wavelength step
        lbda = self.get_lambda()
        lmin = np.min(lbda) - dlbda / 2.0
        lmax = np.max(lbda) + dlbda / 2.0
        n = (int)((lmax - lmin) / dlbda)

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods",
                                                mpdaf.__path__[0])
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
        result = np.empty(n).astype(np.float64)
        # run C method
        libCmethods.mpdaf_sky_ref(data, lbda, mask, data.shape[0],
                                  np.float64(lmin),
                                  np.float64(dlbda), n, nmax,
                                  np.float64(nclip_low),
                                  np.float64(nclip_up), nstop, result)
        wave = WaveCoord(crpix=1.0, cdelt=dlbda, crval=np.min(lbda),
                         cunit='Angstrom', shape=n)

        spe = Spectrum(shape=n, data=result, wave=wave)
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

    def subtract_slice_median(self, skyref, mask):
        """Computes the median value for all slices
        and subtracts this factor to each pixel
        to bring all slices to the same median value.

        pix(x,y,lbda) += < skyref(lbda) - pix(x,y,lbda) >_slice

        Parameters
        ----------
        skyref : string
                 FITS file containig the reference sky spectrum
        mask   : :class:`mpdaf.drs.PixTableMask`
                 column corresponding to a mask file
                 (previously computed by mask_column)
        """
        d = {'class': 'PixTable', 'method': 'subtract_slice_median'}

        origin = self.get_origin()
        ifu = self.origin2ifu(origin)
        sli = self.origin2slice(origin)

        spe_skyref = Spectrum(skyref)

        # mask
        if mask is None:
            maskfile = ''
            maskcol = np.zeros(self.nrows).astype('bool')
        else:
            maskfile = os.path.basename(mask.maskfile)
            maskcol = mask.maskcol

        import mpdaf
        import ctypes
        #global libCmethods

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods",
                                                mpdaf.__path__[0])

        # define argument types
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')

        # setup the return types and argument types
        libCmethods.mpdaf_subtract_slice_median.restype = None
        libCmethods.mpdaf_subtract_slice_median.argtypes = \
            [array_1d_double, array_1d_double, array_1d_int,
             array_1d_int, array_1d_int, array_1d_double,
             array_1d_double, ctypes.c_int, array_1d_int, array_1d_double,
             array_1d_double, ctypes.c_int]

        data = self.get_data()
        ifu = ifu.astype(np.int32)
        sli = sli.astype(np.int32)
        data = data.astype(np.float64)
        lbda = self.get_lambda()
        lbda = lbda.astype(np.float64)
        mask = maskcol.astype(np.int32)
        skyref_flux = spe_skyref.data.data.astype(np.float64)
        skyref_lbda = spe_skyref.wave.coord()
        skyref_n = spe_skyref.shape

        result = np.empty_like(data).astype(np.float64)
        corr = np.empty(24 * 48).astype(np.float64)
        npts = np.empty(24 * 48).astype(np.int32)

        libCmethods.mpdaf_subtract_slice_median(result, corr, npts, ifu, sli, data, lbda,
                                                data.shape[0], mask, skyref_flux,
                                                skyref_lbda, skyref_n)

        # set pixtable data
        self.set_data(result)

        # write autocalib file
        filename = 'autocalib_' + \
            string.replace(os.path.basename(self.filename), '.fits', '.txt')

        header = 'ifu number| slice number | number of remaining pixels | correction value\n'
        header += 'drs.pixtable.subtract_slice_median\n'
        header += '   file to mask out all bright objects: %s\n' % maskfile
        header += '   reference sky spectrum: %s\n' % os.path.basename(skyref)

        np.savetxt(filename,
                   zip(np.ravel(np.swapaxes(np.resize(np.arange(1, 25), (48, 24)), 0, 1)),
                       np.ravel(np.resize(np.arange(1, 49), (24, 48))),
                       npts, corr),
                   fmt='%02d %02d %d %g', header=header)

        # store parameters of the method in FITS keywords
        add_mpdaf_method_keywords(self.primary_header,
                                  "drs.pixtable.subtract_slice_median",
                                  ['mask', 'skyref'],
                                  [maskfile, os.path.basename(skyref)],
                                  ['file to mask out all bright objects',
                                   'reference sky spectrum'])

        self.logger.info('pixtable %s updated' % os.path.basename(self.filename), extra=d)
        self.logger.info('%s saved' % filename, extra=d)

        # close libray
        #import _ctypes
        #_ctypes.dlclose(libCmethods._handle)
        #libCmethods._handle = None
        #libCmethods._name = None
        #libCmethods._FuncPtr = None
        del libCmethods

    def divide_slice_median(self, skyref, mask):
        """Computes the median value for all slices and divides each pixel
        by the corresponding factor to bring all slices
        to the same median value.
        pix(x,y,lbda) /= < pix(x,y,lbda) / skyref(lbda) >_slice

        Parameters
        ----------
        skyref : string
                 FITS file containig the reference sky spectrum
        mask   : :class:`mpdaf.drs.PixTableMask`
                 column corresponding to a mask file
                 (previously computed by mask_column)
        """
        d = {'class': 'PixTable', 'method': 'divide_slice_median'}

        origin = self.get_origin()
        ifu = self.origin2ifu(origin)
        sli = self.origin2slice(origin)

        spe_skyref = Spectrum(skyref)

        # mask
        if mask is None:
            maskfile = ''
            maskcol = np.zeros(self.nrows).astype('bool')
        else:
            maskfile = os.path.basename(mask.maskfile)
            maskcol = mask.maskcol

        import mpdaf
        import ctypes
        #global libCmethods

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods",
                                                mpdaf.__path__[0])

        # define argument types
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')

        # setup the return types and argument types
        libCmethods.mpdaf_divide_slice_median.restype = None
        libCmethods.mpdaf_divide_slice_median.argtypes = \
            [array_1d_double, array_1d_double, array_1d_double, array_1d_int,
             array_1d_int, array_1d_int, array_1d_double, array_1d_double,
             array_1d_double, ctypes.c_int, array_1d_int, array_1d_double,
             array_1d_double, ctypes.c_int]

        data = self.get_data()
        ifu = ifu.astype(np.int32)
        sli = sli.astype(np.int32)
        data = data.astype(np.float64)
        lbda = self.get_lambda()
        lbda = lbda.astype(np.float64)
        stat = self.get_stat()
        stat = stat.astype(np.float64)
        mask = maskcol.astype(np.int32)
        skyref_flux = spe_skyref.data.data.astype(np.float64)
        skyref_lbda = spe_skyref.wave.coord()
        skyref_n = spe_skyref.shape

        result = np.empty_like(data).astype(np.float64)
        result_stat = np.empty_like(stat).astype(np.float64)
        corr = np.empty(24 * 48).astype(np.float64)
        npts = np.empty(24 * 48).astype(np.int32)

        libCmethods.mpdaf_divide_slice_median(result, result_stat, corr, npts,
                                              ifu, sli, data, stat, lbda,
                                              data.shape[0], mask, skyref_flux,
                                              skyref_lbda, skyref_n)

        # set pixtable data
        self.set_data(result)
        self.set_stat(result_stat)

        # store parameters of the method in FITS keywords
        add_mpdaf_method_keywords(self.primary_header,
                                  "drs.pixtable.divide_slice_median",
                                  ['mask', 'skyref'],
                                  [maskfile, os.path.basename(skyref)],
                                  ['file to mask out all bright objects',
                                   'reference sky spectrum'])

        # write autocalib file
        filename = 'autocalib_' + \
            string.replace(os.path.basename(self.filename), '.fits', '.txt')

        header = 'ifu number| slice number | number of remaining pixels | correction value\n'
        header += 'drs.pixtable.divide_slice_median\n'
        header += '   file to mask out all bright objects: %s\n' % maskfile
        header += '   reference sky spectrum: %s\n' % os.path.basename(skyref)

        np.savetxt(filename,
                   zip(np.ravel(np.swapaxes(np.resize(np.arange(1, 25), (48, 24)), 0, 1)),
                       np.ravel(np.resize(np.arange(1, 49), (24, 48))),
                       npts, corr),
                   fmt='%02d %02d %d %g', header=header)

        self.logger.info('pixtable %s updated' % os.path.basename(self.filename), extra=d)
        self.logger.info('%s saved' % filename, extra=d)

        # close libray
        #import _ctypes
        #_ctypes.dlclose(libCmethods._handle)
        #libCmethods._handle = None
        #libCmethods._name = None
        #libCmethods._FuncPtr = None
        del libCmethods
