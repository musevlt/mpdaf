""" pixtable.py Manages MUSE pixel table files
"""
from mpdaf.obj import Image
from mpdaf.obj import WCS
import numpy as np
try:
    from astropy.io import fits as pyfits
except:
    import pyfits
import datetime
import tempfile
import os
import shutil
import warnings
import logging


FORMAT = "WARNING mpdaf corelib %(class)s.%(method)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('mpdaf corelib')


def write(filename, xpos, ypos, lbda, data, dq, stat, origin, weight=None, \
          primary_header=None, save_as_ima=True, wcs='pix'):
    """Saves the object in a FITS file.
    
    Parameters
    ----------
    filename    : string
                  The FITS filename.
    save_as_ima : bool
                  If True, pixtable is saved as multi-extension FITS
    """
    try:
        # remove in pyfits version >3.0
        pyfits.setExtensionNameCaseSensitive()
    except:
        pyfits.EXTENSION_NAME_CASE_SENSITIVE.set(True)
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
                logger.warning("%s keyword not written", card.keyword, \
                               extra=d)
                pass
    prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
    prihdu.header['author'] = ('MPDAF', 'origin of the file')
    warnings.simplefilter("default")
    if save_as_ima:
        hdulist = [prihdu]
        nrows = xpos.shape[0]
        hdulist.append(pyfits.ImageHDU(name='xpos', \
                                       data=xpos.reshape((nrows, 1))))
        hdulist.append(pyfits.ImageHDU(name='ypos', \
                                       data=ypos.reshape((nrows, 1))))
        hdulist.append(pyfits.ImageHDU(name='lambda', \
                                       data=lbda.reshape((nrows, 1))))
        hdulist.append(pyfits.ImageHDU(name='data', \
                                       data=data.reshape((nrows, 1))))
        hdulist.append(pyfits.ImageHDU(name='dq', \
                                       data=np.int32(dq.reshape((nrows, 1)))))
        hdulist.append(pyfits.ImageHDU(name='stat', \
                                       data=stat.reshape((nrows, 1))))
        hdulist.append(pyfits.ImageHDU(name='origin', \
                                       data=np.int32(origin.reshape((nrows, \
                                                                     1)))))
        if weight is not None:
            hdulist.append(pyfits.ImageHDU(name='weight', \
                                           data=weight.reshape((nrows, 1))))
        hdu = pyfits.HDUList(hdulist)
        hdu[1].header['BUNIT'] = wcs
        hdu[2].header['BUNIT'] = wcs
        hdu[3].header['BUNIT'] = 'Angstrom'
        hdu[4].header['BUNIT'] = 'count'
        hdu[6].header['BUNIT'] = 'count**2'
        if weight is not None:
            hdu[8].header['BUNIT'] = 'count'
        hdu.writeto(filename, clobber=True, output_verify='fix')
    else:
        cols = []
        cols.append(pyfits.Column(name='xpos', format='1E', unit=wcs, \
                                      array=xpos))
        cols.append(pyfits.Column(name='ypos', format='1E', unit=wcs, \
                                      array=ypos))
        cols.append(pyfits.Column(name='lambda', format='1E', \
                                  unit='Angstrom', array=lbda))
        cols.append(pyfits.Column(name='data', format='1E', unit='count', \
                                  array=data))
        cols.append(pyfits.Column(name='dq', format='1J', array=dq))
        cols.append(pyfits.Column(name='stat', format='1E', unit='count**2', \
                                  array=stat))
        cols.append(pyfits.Column(name='origin', format='1J', array=origin))
        if weight is not None:
            cols.append(pyfits.Column(name='weight', format='1E', \
                                      unit='count', array=weight))
        coltab = pyfits.ColDefs(cols)
        #tbhdu = pyfits.new_table(coltab)
        tbhdu = pyfits.TableHDU(pyfits.FITS_rec.from_columns(coltab))
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True, output_verify='fix')


class PixTable(object):
    """PixTable class

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

    def __init__(self, filename):
        """creates a PixTable object

        Parameters
        ----------
        filename : string
                   The FITS file name. None by default.

        The FITS file is opened with memory mapping.
        Just the primary header and table dimensions are loaded.
        Methods get_xpos, get_ypos, get_lambda, get_data, get_dq
        ,get_stat and get_origin must be used to get columns data.
        """
        self.filename = filename
        self.nrows = 0
        self.nifu = 0
        self.skysub = False
        self.fluxcal = False
        self.wcs = 'pix'
        self.ima = True

        if filename != None:
            try:
                hdulist = pyfits.open(self.filename, memmap=1)
                self.primary_header = hdulist[0].header
                self.nrows = hdulist[1].header["NAXIS2"]
                self.ima = (hdulist[1].header['XTENSION'] == 'IMAGE')
                hdulist.close()

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
                    self.fluxcal = self.get_keywords("HIERARCH ESO DRS MUSE "\
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
                
                if self.ima:
                    self.wcs = hdulist[1].header['BUNIT']
                else:
                    self.wcs = hdulist[1].header['TUNIT1']
            except IOError:
                raise IOError('file %s not found' % filename)
        else:
            self.primary_header = pyfits.Header()

    def __del__(self):
        """Removes temporary files used for memory mapping.
        """
        try:
            if os.path.basename(self.filename) in \
            os.listdir(tempfile.gettempdir()):
                os.remove(self.filename)
        except:
            pass

    def copy(self):
        """Copies PixTable object in a new one and returns it.
        """
        result = PixTable()
        result.filename = self.filename

        result.nrows = self.nrows
        result.nifu = self.nifu
        result.skysub = self.skysub
        result.fluxcal = self.fuxcal
        result.wcs = self.wcs
        result.ima = self.ima
        result.primary_header = pyfits.Header(self.primary_header)
        return result

    def info(self):
        """Prints information.
        """
        print "%i merged IFUs went into this pixel table" % self.nifu
        if self.skysub:
            "This pixel table was sky-subtracted"
        if self.fluxcal:
            "This pixel table was flux-calibrated"
        try:
            print '%s (%s)' % (self.primary_header["HIERARCH ESO "\
                                                   "DRS MUSE PIXTABLE WCS"], \
                               self.primary_header.comments["HIERARCH ESO "\
                                                   "DRS MUSE PIXTABLE WCS"])
        except:
            try:
                print '%s (%s)' % (self.primary_header["HIERARCH ESO "\
                                                       "PRO MUSE PIXTABLE WCS"],\
                                    self.primary_header.comments["HIERARCH ESO "\
                                                        "PRO MUSE PIXTABLE WCS"])
            except:
                pass
        if self.filename != None:
            hdulist = pyfits.open(self.filename, memmap=1)
            print hdulist.info()
            hdulist.close()
        else:
            print 'No\tName\tType\tDim'
            print '0\tPRIMARY\tcard\t()'
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
        if self.ima == save_as_ima:
            shutil.copy(self.filename, filename)
        else:
            write(filename, self.get_xpos(), self.get_ypos(), \
                  self.get_lambda(), self.get_data(), self.get_dq(), \
                  self.get_stat(), self.get_origin(), self.get_weight(), \
                  self.primary_header, save_as_ima, self.wcs)
        if os.path.basename(self.filename) \
        in os.listdir(tempfile.gettempdir()):
            os.remove(self.filename)
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
        hdulist = pyfits.open(self.filename, memmap=1)
        if ksel is None:
            if self.ima:
                xpos = hdulist['xpos'].data[:, 0]
            else:
                xpos = hdulist[1].data.field('xpos')
        else:
            if self.ima:
                xpos = hdulist['xpos'].data[ksel, 0][0]
            else:
                xpos = hdulist[1].data.field('xpos')[ksel]
        hdulist.close()
        return xpos

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
        hdulist = pyfits.open(self.filename, memmap=1)
        if ksel is None:
            if self.ima:
                ypos = hdulist['ypos'].data[:, 0]
            else:
                ypos = hdulist[1].data.field('ypos')
        else:
            if self.ima:
                ypos = hdulist['ypos'].data[ksel, 0][0]
            else:
                ypos = hdulist[1].data.field('ypos')[ksel]
        hdulist.close()
        return ypos
    
    def get_xpos_sky(self, ksel=None):
        """Returns the x absolute position on the sky in degrees/pixel.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.
               
        Returns
        -------
        out : numpy.array
        """
        xpos = self.get_xpos(ksel)
        ypos = self.get_ypos(ksel)
        try:
            spheric = (self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE WCS")[0:9] == 'projected')
        except:
            spheric = False
        if spheric: #spheric coordinates
            phi = xpos
            theta = ypos + np.pi/2
            dp = self.yc * np.pi / 180
            ra = np.arctan2(np.cos(theta) * np.sin(phi), \
                      np.sin(theta) * np.cos(dp) + np.cos(theta) * np.sin(dp) * np.cos(phi)) * 180 / np.pi
            xpos_sky = self.xc + ra
        else:
            if self.wcs == 'deg':
                dp = self.yc * np.pi / 180
                xpos_sky = self.xc + xpos / np.cos(dp)
            elif self.wcs == 'rad':
                dp = self.yc * np.pi / 180
                xpos_sky = self.xc + xpos * 180 / np.pi / np.cos(dp)
            else:
                xpos_sky = self.xc + xpos       
        return xpos_sky
    
    def get_ypos_sky(self, ksel=None):
        """Returns the y absolute position on the sky in degrees/pixel.

        Parameters
        ----------
        ksel : output of np.where
               Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        try:
            spheric = (self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE WCS")[0:9] == 'projected')
        except:
            spheric = False
        if spheric: #spheric coordinates
            phi = self.get_xpos(ksel)
            theta = self.get_ypos(ksel) + np.pi/2
            dp = self.yc * np.pi / 180
            ypos_sky = np.arcsin(np.sin(theta) * np.sin(dp) - np.cos(theta) * np.cos(dp) * np.cos(phi)) * 180 / np.pi
        else:
            ypos = self.get_ypos(ksel)
            if self.wcs == 'rad':
                ypos_sky = self.yc + ypos * 180 / np.pi
            else:
                ypos_sky = self.yc + ypos
        return ypos_sky

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
        hdulist = pyfits.open(self.filename, memmap=1)
        if ksel is None:
            if self.ima:
                lbda = hdulist['lambda'].data[:, 0]
            else:
                lbda = hdulist[1].data.field('lambda')
        else:
            if self.ima:
                lbda = hdulist['lambda'].data[ksel, 0][0]
            else:
                lbda = hdulist[1].data.field('lambda')[ksel]
        hdulist.close()
        return lbda

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
        hdulist = pyfits.open(self.filename, memmap=1)
        if ksel is None:
            if self.ima:
                data = hdulist['data'].data[:, 0]
            else:
                data = hdulist[1].data.field('data')
        else:
            if self.ima:
                data = hdulist['data'].data[ksel, 0][0]
            else:
                data = hdulist[1].data.field('data')[ksel]
        hdulist.close()
        return data

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
        hdulist = pyfits.open(self.filename, memmap=1)
        if ksel is None:
            if self.ima:
                stat = hdulist['stat'].data[:, 0]
            else:
                stat = hdulist[1].data.field('stat')
        else:
            if self.ima:
                stat = hdulist['stat'].data[ksel, 0][0]
            else:
                stat = hdulist[1].data.field('stat')[ksel]
        hdulist.close()
        return stat

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
        hdulist = pyfits.open(self.filename, memmap=1)
        if ksel is None:
            if self.ima:
                dq = hdulist['dq'].data[:, 0]
            else:
                dq = hdulist[1].data.field('dq')
        else:
            if self.ima:
                dq = hdulist['dq'].data[ksel, 0][0]
            else:
                dq = hdulist[1].data.field('dq')[ksel]
        hdulist.close()
        return dq

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
        hdulist = pyfits.open(self.filename, memmap=1)
        if ksel is None:
            if self.ima:
                origin = hdulist['origin'].data[:, 0]
            else:
                origin = hdulist[1].data.field('origin')
        else:
            if self.ima:
                origin = hdulist['origin'].data[ksel, 0][0]
            else:
                origin = hdulist[1].data.field('origin')[ksel]
        hdulist.close()
        return origin

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
        try:
            if self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE WEIGHTED"):
                hdulist = pyfits.open(self.filename, memmap=1)
                if ksel is None:
                    if self.ima:
                        weight = hdulist['weight'].data[:, 0]
                    else:
                        weight = hdulist[1].data.field('weight')
                else:
                    if self.ima:
                        weight = hdulist['weight'].data[ksel, 0][0]
                    else:
                        weight = hdulist[1].data.field('weight')[ksel]
                hdulist.close()
        except:
            weight = None
        return weight

    def get_exp(self):
        """Loads the exposure numbers and returns it as a column.

        Returns
        -------
        out : numpy.memmap
        """
        try:
            nexp = self.get_keywords("HIERARCH ESO DRS MUSE "\
                                     "PIXTABLE COMBINED")
            exp = np.empty(shape=(self.nrows))
            for i in range(1, nexp + 1):
                first = self.get_keywords("HIERARCH ESO DRS MUSE "\
                                          "PIXTABLE EXP%i FIRST" % i)
                last = self.get_keywords("HIERARCH ESO DRS MUSE "\
                                         "PIXTABLE EXP%i LAST" % i)
                exp[first:last + 1] = i
        except:
            exp = None
        return exp

    def extract(self, filename=None, sky=None, lbda=None, ifu=None, \
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
                   defined by a center (y, x),
                   a shape ('C' for circular, 'S' for square)
                   and size (radius or half side length).
        lbda     : (float, float)
                   (min, max) wavelength range in Angstrom.
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
        # type of coordinates
        primary_header = self.primary_header.copy()
        if self.nrows == 0:
            return None

        # To start select the whole pixtable
        kmask = np.ones(self.nrows).astype('bool')

        # Do the selection on the sky
        if sky is not None:
            xpos = self.get_xpos_sky()
            ypos = self.get_ypos_sky()
            if (isinstance(sky, tuple)):
                sky = [sky]
            mask = np.zeros(self.nrows).astype('bool')
            for y0, x0, size, shape in sky:
                if shape == 'C':
                    if self.wcs == 'deg':
                        mask |= (((xpos - x0) * 3600 \
                                  * np.cos(y0 * np.pi / 180.)) ** 2 \
                                + ((ypos - y0) * 3600) ** 2) \
                                < size ** 2
                    elif self.wcs == 'rad':
                        mask |= (((xpos - x0) * 3600 * 180 / np.pi \
                                  * np.cos(y0)) ** 2 \
                                + ((ypos - y0) * 3600 * 180 / np.pi)\
                                 ** 2) < size ** 2
                    else:
                        mask |= ((xpos - x0) ** 2 \
                                 + (ypos - y0) ** 2) < size ** 2
                elif shape == 'S':
                    if self.wcs == 'deg':
                        mask |= (np.abs((xpos - x0) * 3600 \
                                * np.cos(y0 * np.pi / 180.)) \
                                 < size) \
                                & (np.abs((ypos - y0) * 3600) \
                                 < size)
                    elif self.wcs == 'rad':
                        mask |= (np.abs((xpos - x0) * 3600 * 180 \
                                / np.pi * np.cos(y0)) < size) \
                                & (np.abs((ypos - y0) * 3600 * 180 \
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
                    kmask &= (self.origin2slice(col_origin) == slice)
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
        #lambda
        lbda = self.get_lambda(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE '\
                           'PIXTABLE LIMITS LAMBDA LOW'] = float(lbda.min())
            primary_header['HIERARCH ESO DRS MUSE '\
                            'PIXTABLE LIMITS LAMBDA HIGH'] = float(lbda.max())
        except:
            primary_header['HIERARCH ESO PRO MUSE '\
                            'PIXTABLE LIMITS LAMBDA LOW'] = float(lbda.min())
            primary_header['HIERARCH ESO PRO MUSE '\
                            'PIXTABLE LIMITS LAMBDA HIGH'] = float(lbda.max())
        #data
        data = self.get_data(ksel)
        #variance
        stat = self.get_stat(ksel)
        # pixel quality
        dq = self.get_dq(ksel)
        # origin
        origin = self.get_origin(ksel)
        try:
            primary_header['HIERARCH ESO DRS MUSE '\
                            'PIXTABLE LIMITS IFU LOW'] = \
                            int(self.origin2ifu(origin).min())
            primary_header['HIERARCH ESO DRS MUSE '\
                            'PIXTABLE LIMITS IFU HIGH'] = \
                            int(self.origin2ifu(origin).max())
            primary_header['HIERARCH ESO DRS MUSE '\
                            'PIXTABLE LIMITS SLICE LOW'] = \
                            int(self.origin2slice(origin).min())
            primary_header['HIERARCH ESO DRS MUSE '\
                            'PIXTABLE LIMITS SLICE HIGH'] = \
                            int(self.origin2slice(origin).max())
        except:
            primary_header['HIERARCH ESO PRO MUSE '\
                            'PIXTABLE LIMITS IFU LOW'] = \
                            int(self.origin2ifu(origin).min())
            primary_header['HIERARCH ESO PRO MUSE '\
                            'PIXTABLE LIMITS IFU HIGH'] = \
                            int(self.origin2ifu(origin).max())
            primary_header['HIERARCH ESO PRO MUSE '\
                            'PIXTABLE LIMITS SLICE LOW'] = \
                            int(self.origin2slice(origin).min())
            primary_header['HIERARCH ESO PRO MUSE '\
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
                for iexp, i in zip(numbers_exp, range(1, \
                                                      len(numbers_exp) + 1)):
                    ksel = np.where(newexp == iexp)
                    primary_header["HIERARCH ESO DRS MUSE "\
                                    "PIXTABLE EXP%i FIRST" % i] = ksel[0][0]
                    primary_header["HIERARCH ESO DRS MUSE "\
                                    "PIXTABLE EXP%i LAST" % i] = ksel[0][-1]
                for i in range(len(numbers_exp) + 1, \
                               self.get_keywords("HIERARCH ESO DRS MUSE "\
                                                 "PIXTABLE COMBINED") + 1):
                    del primary_header["HIERARCH ESO DRS MUSE "\
                                       "PIXTABLE EXP%i FIRST" % i]
                    del primary_header["HIERARCH ESO DRS MUSE "\
                                       "PIXTABLE EXP%i LAST" % i]
            except:
                primary_header["HIERARCH ESO PRO MUSE PIXTABLE COMBINED"] = \
                len(numbers_exp)
                for iexp, i in zip(numbers_exp, \
                                   range(1, len(numbers_exp) + 1)):
                    ksel = np.where(newexp == iexp)
                    primary_header["HIERARCH ESO PRO MUSE "\
                                    "PIXTABLE EXP%i FIRST" % i] = ksel[0][0]
                    primary_header["HIERARCH ESO PRO MUSE "\
                                    "PIXTABLE EXP%i LAST" % i] = ksel[0][-1]
                for i in range(len(numbers_exp) + 1, \
                               self.get_keywords("HIERARCH ESO PRO MUSE '\
                               PIXTABLE COMBINED") + 1):
                    del primary_header["HIERARCH ESO PRO MUSE "\
                                       "PIXTABLE EXP%i FIRST" % i]
                    del primary_header["HIERARCH ESO PRO MUSE "\
                                       "PIXTABLE EXP%i LAST" % i]

        # write the result in a new file
        if filename is None:
            (fd, filename) = tempfile.mkstemp(prefix='mpdaf')
            os.close(fd)

        write(filename, xpos, ypos, lbda, data, dq, stat, origin,\
               weight, primary_header, self.ima, self.wcs)
        return PixTable(filename)

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
        return (origin >> 6) & 0x1f

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
        return origin & 0x3f

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
        return ((origin >> 11) & 0x1fff) - 1

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
                    value = self.get_keywords('HIERARCH ESO DRS MUSE '\
                                              'PIXTABLE EXP0 IFU%02d '\
                                              'SLICE%02d XOFFSET' \
                                              % (ifu, sl))
                    xoffset[np.where((col_ifu == ifu) \
                                     & (col_slice == sl))] = value
        else:
            xoffset = self.get_keywords("HIERARCH ESO DRS MUSE "\
                                        "PIXTABLE EXP0 IFU%02d "\
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
        return self.origin2xoffset(origin) + ((origin >> 24) & 0x7f) - 1

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
            print('%d slices found, structure returned '\
                  'in slices dictionary ' % nslice)

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
            alternate_key = key.replace('HIERARCH ESO PRO MUSE', \
                                        'HIERARCH ESO DRS MUSE')
        elif key.startswith('HIERARCH ESO DRS MUSE'):
            alternate_key = key.replace('HIERARCH ESO DRS MUSE', \
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
        from scipy import interpolate

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
            ksel = np.where((col_dq == 0) & (col_lambda > l1) \
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

        if self.wcs == "deg": #arcsec to deg
            xstep /= (-3600. * np.cos((ymin + ymax) * np.pi / 180. / 2.))
            ystep /= 3600.
        elif self.wcs == "deg": #arcsec to rad
            xstep /= (-3600. * 180. / np.pi * np.cos((ymin + ymax) / 2.))
            ystep /= (3600. * 180. / np.pi)
        else: #pix
            pass

        nx = 1 + int((xmin - xmax) / xstep)
        grid_x = np.arange(nx) * xstep + xmax
        ny = 1 + int((ymax - ymin) / ystep)
        grid_y = np.arange(ny) * ystep + ymin
        shape = (ny, nx)

        points = np.empty((len(ksel[0]), 2), dtype=float)
        points[:, 0] = y
        points[:, 1] = x

        new_data = interpolate.griddata(points, data, \
                                       np.meshgrid(grid_y, grid_x), \
                                       method='linear').T

        from mpdaf.obj import Image, WCS
        wcs = WCS(crpix=(1.0, 1.0), crval=(ymin, xmax), \
                  cdelt=(ystep, xstep), shape=shape)
        ima = Image(data=new_data, wcs=wcs)
        return ima

    def reconstruct_det_image(self, xstart=None, ystart=None, \
                              xstop=None, ystop=None):
        """Reconstructs the image on the detector from the pixtable.
        The pixtable must concerns only one IFU,
        otherwise an exception is raised.

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
        image = np.zeros((ystop - ystart + 1, \
                          xstop - xstart + 1), dtype='float') * np.NaN
        image[ypix - ystart, xpix - xstart] = col_data

        wcs = WCS(crval=(ystart, xstart))
        return Image(shape=(image.shape), data=image, wcs=wcs)

    def reconstruct_det_waveimage(self):
        """Reconstructs an image of wavelength values
        on the detector from the pixtable.
        The pixtable must concerns only one IFU,
        otherwise an exception is raised.

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
        image = np.zeros((ystop - ystart + 1, xstop - xstart + 1), \
                         dtype='float')
        image[ypix - ystart, xpix - xstart] = col_lambdas

        wcs = WCS(crval=(ystart, xstart))

        return Image(shape=(image.shape), data=image, wcs=wcs)
