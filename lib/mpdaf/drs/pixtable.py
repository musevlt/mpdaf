""" pixtable.py Manages MUSE pixel table files
"""
from mpdaf.obj import Image
from mpdaf.obj import WCS
import numpy as np
import pyfits
import datetime
import tempfile
import os

class PixTable(object):
    """PixTable class

    This class manages input/output for MUSE pixel table files
    
    :param filename: The FITS file name. None by default.
    :type filename: string.

    Attributes
    ----------
    filename : string
    The FITS file name. None if any.

    primary_header: pyfits.CardList
    The primary header

    nrows: integer
    Number of rows

    ncols: integer
    Number of columns

    """

    def __init__(self, filename=None):
        """creates a PixTable object

        Parameters
        ----------
        filename : string
        The FITS file name. None by default.

        Notes
        -----
        filename=None creates an empty object

        The FITS file is opened with memory mapping.
        Just the primary header and table dimensions are loaded.
        Methods get_xpos, get_ypos, get_lambda, get_data, get_dq
        ,get_stat and get_origin must be used to get columns data.
        """
        self.filename = filename
        self.nrows = 0
        self.ncols = 0

        # name of memory-mapped files
        self.__xpos = None
        self.__ypos = None
        self.__lbda = None
        self.__data = None
        self.__dq = None
        self.__stat = None
        self.__origin = None

        if filename!=None:
            try:
                hdulist = pyfits.open(self.filename,memmap=1)
                self.primary_header = hdulist[0].header.ascard
                self.nrows = hdulist[1].header["NAXIS2"]
                self.ncols = hdulist[1].header["TFIELDS"]
                hdulist.close()
            except IOError:
                print 'IOError: file %s not found' % `filename`
                print
                self.filename = None
                self.primary_header = None
        else:
            self.primary_header = pyfits.CardList()

    def __del__(self):
        """Removes temporary files used for memory mapping.
        """
        try:
            if self.__xpos != None:
                os.remove(self.__xpos)
            if self.__ypos != None:
                os.remove(self.__ypos)
            if self.__lbda != None:
                os.remove(self.__lbda)
            if self.__data != None:
                os.remove(self.__data)
            if self.__dq != None:
                os.remove(self.__dq)
            if self.__stat != None:
                os.remove(self.__stat)
            if self.__origin != None:
                os.remove(self.__origin)
        except:
            pass

    def copy(self):
        """Copies PixTable object in a new one and returns it.
        """
        result = PixTable()
        result.filename = self.filename

        result.nrows = self.nrows
        result.ncols = self.ncols
        result.primary_header = pyfits.CardList(self.primary_header)
        #xpos
        (fd,result.__xpos) = tempfile.mkstemp(prefix='mpdaf')
        selfxpos=self.get_xpos()
        xpos = np.memmap(result.__xpos,dtype="float32",shape=(self.nrows))
        xpos[:] = selfxpos[:]
        del xpos, selfxpos
        os.close(fd)
        #ypos
        (fd,result.__ypos) = tempfile.mkstemp(prefix='mpdaf')
        selfypos=self.get_ypos()
        ypos = np.memmap(result.__ypos,dtype="float32",shape=(self.nrows))
        ypos[:] = selfypos[:]
        del ypos, selfypos
        os.close(fd)
        #lambda
        (fd,result.__lbda) = tempfile.mkstemp(prefix='mpdaf')
        selflbda=self.get_lambda()
        lbda = np.memmap(result.__lbda,dtype="float32",shape=(self.nrows))
        lbda[:] = selflbda[:]
        del lbda, selflbda
        os.close(fd)
        #data
        (fd,result.__data) = tempfile.mkstemp(prefix='mpdaf')
        selfdata=self.get_data()
        data = np.memmap(result.__data,dtype="float32",shape=(self.nrows))
        data[:] = selfdata[:]
        del data, selfdata
        os.close(fd)
        #variance
        (fd,result.__stat) = tempfile.mkstemp(prefix='mpdaf')
        selfstat=self.get_stat()
        stat = np.memmap(result.__stat,dtype="float32",shape=(self.nrows))
        stat[:] = selfstat[:]
        del stat, selfstat
        os.close(fd)
        # pixel quality
        (fd,result.__dq) = tempfile.mkstemp(prefix='mpdaf')
        selfdq = self.get_dq()
        dq = np.memmap(result.__dq,dtype="uint32",shape=(self.nrows))
        dq[:] = selfdq[:]
        del dq, selfdq
        os.close(fd)
        # origin
        (fd,result.__origin) = tempfile.mkstemp(prefix='mpdaf')
        selforigin = self.get_origin()
        origin = np.memmap(result.__origin,dtype="uint32",shape=(self.nrows))
        origin[:] = selforigin[:]
        del origin, selforigin
        os.close(fd)
        return result

    def info(self):
        """Prints information.
        """
        if self.filename != None:
            hdulist = pyfits.open(self.filename,memmap=1)
            print hdulist.info()
            hdulist.close()
        else:
            print 'No\tName\tType\tDim'
            print '0\tPRIMARY\tcard\t()'
            print "1\t\tTABLE\t(%iR,%iC)" % (self.nrows,self.ncols)

    def get_xpos(self):
        """Loads the xpos column and returns it.
        
        :rtype: numpy.memmap
        """
        if self.__xpos != None:
            xpos = np.memmap(self.__xpos,dtype="float32",shape=(self.nrows))
            return xpos
        else:
            if self.filename == None:
                print 'format error: empty XPOS column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.__xpos) = tempfile.mkstemp(prefix='mpdaf')
                data_xpos = hdulist[1].data.field('xpos')
                xpos = np.memmap(self.__xpos,dtype="float32",shape=(self.nrows))
                xpos[:] = data_xpos[:]
                hdulist.close()
                os.close(fd)
                return xpos

    def get_ypos(self):
        """Loads the ypos column and returns it.
        
        :rtype: numpy.memmap
        """
        if self.__ypos != None:
            ypos = np.memmap(self.__ypos,dtype="float32",shape=(self.nrows))
            return ypos
        else:
            if self.filename == None:
                print 'format error: empty YPOS column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.__ypos) = tempfile.mkstemp(prefix='mpdaf')
                data_ypos = hdulist[1].data.field('ypos')
                ypos = np.memmap(self.__ypos,dtype="float32",shape=(self.nrows))
                ypos[:] = data_ypos[:]
                hdulist.close()
                os.close(fd)
                return ypos

    def get_lambda(self):
        """Loads the lambda column and returns it.
        
        :rtype: numpy.memmap
        """
        if self.__lbda != None:
            lbda = np.memmap(self.__lbda,dtype="float32",shape=(self.nrows))
            return lbda
        else:
            if self.filename == None:
                print 'format error: empty YLAMBDA column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.__lbda) = tempfile.mkstemp(prefix='mpdaf')
                data_lbda = hdulist[1].data.field('lambda')
                lbda = np.memmap(self.__lbda,dtype="float32",shape=(self.nrows))
                lbda[:] = data_lbda[:]
                hdulist.close()
                os.close(fd)
                return lbda

    def get_data(self):
        """Loads the data column and returns it.
        
        :rtype: numpy.memmap
        """
        if self.__data != None:
            data = np.memmap(self.__data,dtype="float32",shape=(self.nrows))
            return data
        else:
            if self.filename == None:
                print 'format error: empty DATA column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.__data) = tempfile.mkstemp(prefix='mpdaf')
                data_data = hdulist[1].data.field('data')
                data = np.memmap(self.__data,dtype="float32",shape=(self.nrows))
                data[:] = data_data[:]
                hdulist.close()
                os.close(fd)
                return data

    def get_stat(self):
        """Loads the stat column and returns it.
        
        :rtype: numpy.memmap
        """
        if self.__stat != None:
            stat = np.memmap(self.__stat,dtype="float32",shape=(self.nrows))
            return stat
        else:
            if self.filename == None:
                print 'format error: empty STAT column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.__stat) = tempfile.mkstemp(prefix='mpdaf')
                data_stat = hdulist[1].data.field('stat')
                stat = np.memmap(self.__stat,dtype="float32",shape=(self.nrows))
                stat[:] = data_stat[:]
                hdulist.close()
                os.close(fd)
                return stat

    def get_dq(self):
        """Loads the dq column and returns it.
        
        :rtype: numpy.memmap
        """
        if self.__dq != None:
            dq = np.memmap(self.__dq,dtype="uint32",shape=(self.nrows))
            return dq
        else:
            if self.filename == None:
                print 'format error: empty DQ column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.__dq) = tempfile.mkstemp(prefix='mpdaf')
                data_dq = hdulist[1].data.field('dq')
                dq = np.memmap(self.__dq,dtype="uint32",shape=(self.nrows))
                dq[:] = data_dq[:]
                hdulist.close()
                os.close(fd)
                return dq

    def get_origin(self):
        """Loads the origin column and returns it.
        
        :rtype: numpy.memmap
        """
        if self.__origin != None:
            origin = np.memmap(self.__origin,dtype="uint32",shape=(self.nrows))
            return origin
        else:
            if self.filename == None:
                print 'format error: empty ORIGIN column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.__origin) = tempfile.mkstemp(prefix='mpdaf')
                data_origin = hdulist[1].data.field('origin')
                origin = np.memmap(self.__origin,dtype="uint32",shape=(self.nrows))
                origin[:] = data_origin[:]
                hdulist.close()
                os.close(fd)
                return origin

    def write(self,filename):
        """Saves the object in a FITS file.
        
        :param filename: The FITS filename.
        :type filename: string
        """
        prihdu = pyfits.PrimaryHDU()
        if self.primary_header is not None:
            for card in self.primary_header:
                try:
                    prihdu.header.update(card.key, card.value, card.comment)
                except ValueError:
                    if isinstance(card.value,str):
                        n = 80 - len(card.key) - 14
                        s = card.value[0:n]
                        prihdu.header.update('hierarch %s' %card.key, s, card.comment)
                    else:
                        prihdu.header.update('hierarch %s' %card.key, card.value, card.comment)
                except:
                    pass
        prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
        prihdu.header.update('author', 'MPDAF', 'origin of the file')
        cols = []
        cols.append(pyfits.Column(name='xpos', format='1E',unit='deg', array=self.get_xpos()))
        cols.append(pyfits.Column(name='ypos', format='1E',unit='deg', array=self.get_ypos()))
        cols.append(pyfits.Column(name='lambda', format='1E',unit='Angstrom', array=self.get_lambda()))
        cols.append(pyfits.Column(name='data', format='1E',unit='count', array=self.get_data()))
        cols.append(pyfits.Column(name='dq', format='1J',unit='None', array=self.get_dq()))
        cols.append(pyfits.Column(name='stat', format='1E',unit='None', array=self.get_stat()))
        cols.append(pyfits.Column(name='origin', format='1J',unit='count**2', array=self.get_origin()))
        coltab = pyfits.ColDefs(cols)
        tbhdu = pyfits.new_table(coltab)
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True)
        # update attributes
        self.filename = filename

    def extract(self, sky=None, lbda=None, ifu=None, slice=None, xpix=None, ypix=None):
        """Extracts a subset of a pixtable using the following criteria:
        
        - aperture on the sky (center, size and shape)
        
        - wavelength range
        
        - IFU number
        
        - slice number
        
        - detector pixels
        
        The arguments can be either single value or a list of values to select
        multiple regions.

        
        :param sky: (y, x, size, shape) extract an aperture on the sky, defined by a center (y, x), a shape ('C' for circular, 'S' for square) and size (radius or half side length).
        :type sky: (float, float, float, char)

        :param lbda: (min, max) wavelength range in Angstrom.
        :type lbda: (float, float)

        :param ifu: IFU number.
        :type ifu: int

        :param slice: Slice number on the CCD.
        :type slice: int

        :param xpix: (min, max) pixel range along the X axis
        :type xpix: (int, int)

        :param ypix: (min, max) pixel range along the Y axis
        :type ypix: (int, int)
        
        :rtype: PixTable
        """

        # First create an empty pixtable
        ptab = PixTable()
        ptab.primary_header = self.primary_header.copy()
        ptab.ncols = self.ncols
        if self.nrows == 0:
            return ptab

        # To start select the whole pixtable
        kmask = np.ones(self.nrows).astype('bool')

        # Do the selection on the sky
        if sky is not None:
            col_xpos = self.get_xpos()
            col_ypos = self.get_ypos()
            if (isinstance(sky, tuple)):
                sky = [sky]
            mask = np.zeros(self.nrows).astype('bool')
            for y0,x0,size,shape in sky:
                if shape == 'C':
                    mask |= (((col_xpos-x0)*np.cos(y0))**2 + (col_ypos-y0)**2) < size**2
                elif shape == 'S':
                    mask |= (np.abs((col_xpos-x0)*np.cos(y0)) < size) & (np.abs(col_ypos-y0) < size)
                else:
                    raise ValueError, 'Unknown shape parameter'
            kmask &= mask
            del mask
            del col_xpos
            del col_ypos

        # Do the selection on wavelengths
        if lbda is not None:
            col_lambda = self.get_lambda()
            if (isinstance(lbda, tuple)):
                lbda = [lbda]
            mask = np.zeros(self.nrows).astype('bool')
            for l1,l2 in lbda:
                mask |= (col_lambda>=l1) & (col_lambda<l2)
            kmask &= mask
            del mask
            del col_lambda

        # Do the selection on the origin column
        if (ifu is not None) or (slice is not None) or (xpix is not None) or (ypix is not None):
            col_origin = self.get_origin()
            if slice is not None:
                if hasattr(slice, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for s in slice:
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
                    for x1,x2 in xpix:
                         mask |= (col_xpix>=x1) & (col_xpix<x2)
                    kmask &= mask
                    del mask
                else:
                    x1,x2 = xpix
                    kmask &= (col_xpix>=x1) & (col_xpix<x2)
                del col_xpix
            if ypix is not None:
                col_ypix = self.origin2ypix(col_origin)
                if hasattr(ypix, '__iter__'):
                    mask = np.zeros(self.nrows).astype('bool')
                    for y1,y2 in ypix:
                         mask |= (col_ypix>=y1) & (col_ypix<y2)
                    kmask &= mask
                    del mask
                else:
                    y1,y2 = ypix
                    kmask &= (col_ypix>=y1) & (col_ypix<y2)
                del col_ypix
            del col_origin

        # Compute the new pixtable
        ksel = np.where(kmask)
        del kmask
        ptab.nrows = len(ksel[0])
        if ptab.nrows == 0:
            return ptab
        #xpos
        (fd,ptab.__xpos) = tempfile.mkstemp(prefix='mpdaf')
        xpos = np.memmap(ptab.__xpos,dtype="float32",shape=(ptab.nrows))
        selfxpos=self.get_xpos()
        xpos[:] = selfxpos[ksel]
        try:
            x_low = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X LOW']
            x_high = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS X HIGH']
        except:
            x_low = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X LOW']
            x_high = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS X HIGH']
        x_low.value = float(xpos.min())
        x_high.value = float(xpos.max())
        del xpos,selfxpos
        os.close(fd)
        #ypos
        (fd,ptab.__ypos) = tempfile.mkstemp(prefix='mpdaf')
        ypos = np.memmap(ptab.__ypos,dtype="float32",shape=(ptab.nrows))
        selfypos=self.get_ypos()
        ypos[:] = selfypos[ksel]
        try:
            y_low = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y LOW']
            y_high = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS Y HIGH']
        except:
            y_low = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y LOW']
            y_high = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS Y HIGH']
        y_low.value = float(ypos.min())
        y_high.value = float(ypos.max())
        del ypos,selfypos
        os.close(fd)
        #lambda
        (fd,ptab.__lbda) = tempfile.mkstemp(prefix='mpdaf')
        lbda = np.memmap(ptab.__lbda,dtype="float32",shape=(ptab.nrows))
        selflbda=self.get_lambda()
        lbda[:] = selflbda[ksel]
        try:
            lbda_low = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS LAMBDA LOW']
            lbda_high = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS LAMBDA HIGH']
        except:
            lbda_low = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS LAMBDA LOW']
            lbda_high = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS LAMBDA HIGH']
        lbda_low.value = float(lbda.min())
        lbda_high.value = float(lbda.max())
        del lbda,selflbda
        os.close(fd)
        #data
        (fd,ptab.__data) = tempfile.mkstemp(prefix='mpdaf')
        selfdata = self.get_data()
        data = np.memmap(ptab.__data,dtype="float32",shape=(ptab.nrows))
        data[:] = selfdata[ksel]
        del data,selfdata
        os.close(fd)
        #variance
        (fd,ptab.__stat) = tempfile.mkstemp(prefix='mpdaf')
        selfstat=self.get_stat()
        stat = np.memmap(ptab.__stat,dtype="float32",shape=(ptab.nrows))
        stat[:] = selfstat[ksel]
        del stat,selfstat
        os.close(fd)
        # pixel quality
        (fd,ptab.__dq) = tempfile.mkstemp(prefix='mpdaf')
        selfdq = self.get_dq()
        dq = np.memmap(ptab.__dq,dtype="uint32",shape=(ptab.nrows))
        dq[:] = selfdq[ksel]
        del dq,selfdq
        os.close(fd)
        # origin
        (fd,ptab.__origin) = tempfile.mkstemp(prefix='mpdaf')
        selforigin = self.get_origin()
        origin = np.memmap(ptab.__origin,dtype="uint32",shape=(ptab.nrows))
        origin[:] = selforigin[ksel]
        try:
            ifu_low = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU LOW']
            ifu_high = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU HIGH']
            slice_low = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS SLICE LOW']
            slice_high = ptab.primary_header['HIERARCH ESO DRS MUSE PIXTABLE LIMITS SLICE HIGH']
        except:
            ifu_low = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS IFU LOW']
            ifu_high = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS IFU HIGH']
            slice_low = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS SLICE LOW']
            slice_high = ptab.primary_header['HIERARCH ESO PRO MUSE PIXTABLE LIMITS SLICE HIGH']
        ifu_low.value = int(self.origin2ifu(origin).min())
        ifu_high.value = int(self.origin2ifu(origin).max())
        slice_low.value = int(self.origin2slice(origin).min())
        slice_high.value = int(self.origin2slice(origin).max())
        del origin,selforigin
        os.close(fd)

        return ptab

    def origin2ifu(self, origin):
        """Converts the origin value and returns the ifu number.

        :param origin: Origin value.
        :type origin: integer
        
        :rtype: int
        """
        return (origin >> 6) & 0x1f

    def origin2slice(self, origin):
        """Converts the origin value and returns the slice number.

        :param origin: Origin value.
        :type origin: integer
        
        :rtype: int
        """
        return origin & 0x3f

    def origin2ypix(self, origin):
        """Converts the origin value and returns the y coordinates.

        :param origin: Origin value.
        :type origin: integer
        
        :rtype: float
        """
        return ((origin >> 11) & 0x1fff) - 1

    def origin2xoffset(self, origin):
        """Converts the origin value and returns the x coordinates offset.
        
        :param origin: Origin value.
        :type origin: integer
        
        :rtype: float
        """
        col_ifu = self.origin2ifu(origin)
        col_slice = self.origin2slice(origin)
        if isinstance(origin, np.ndarray):
            xoffset = np.zeros_like(origin)
            for ifu in np.unique(col_ifu):
                for slice in np.unique(col_slice):
                    value = self.get_keywords('HIERARCH ESO DRS MUSE PIXTABLE EXP0 IFU%02d SLICE%02d XOFFSET' % (ifu, slice))
                    xoffset[np.where((col_ifu == ifu) & (col_slice == slice))] = value
        else:
            xoffset = self.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE EXP0 IFU%02d SLICE%02d XOFFSET" % (ifu, slice))
        return xoffset

    def origin2xpix(self, origin):
        """Converts the origin value and returns the x coordinates.
       
        :param origin: Origin value.
        :type origin: integer
        
        :rtype: float
        """
        return self.origin2xoffset(origin) + ((origin >> 24) & 0x7f) - 1

    def origin2coords(self, origin):
        """Converts the origin value and returns (ifu, slice, ypix, xpix).

        :param origin: Origin value.
        :type origin: integer
        
        :rtype: (integer, integer, float, float)
        """
        return (self.origin2ifu(origin), self.origin2slice(origin),
                self.origin2ypix(origin), self.origin2xpix(origin))

    def get_slices(self):
        """Returns slices dictionary.
        
        :rtype: dict
        """
        col_origin = self.get_origin()
        col_xpos = self.get_xpos()
        col_ypos = self.get_ypos()

        ifupix,slicepix,ypix,xpix = self.origin2coords(col_origin)

        # build the slicelist
        slicelist = []
        for ifu in np.unique(ifupix):
            for sl in np.unique(slicepix):
                slicelist.append((ifu,sl))
        nslice = len(slicelist)
        slicelist = np.array(slicelist)

        # compute mean sky position of each slice
        skypos = []
        for ifu,sl in slicelist:
            k = np.where((ifupix == ifu) & (slicepix == sl))
            skypos.append((col_xpos[k].mean(), col_ypos[k].mean()))
        skypos = np.array(skypos)

        slices = {'list':slicelist, 'skypos':skypos, 'ifupix':ifupix, 'slicepix':slicepix,
                       'xpix':xpix, 'ypix':ypix}

        print('%d slices found, structure returned in slices dictionary '%(nslice))

        return slices

    def get_keywords(self,key):
        """Returns the keyword value corresponding to key.
        
        :param key: Keyword.
        :type key: string
        
        :rtype: float
        """
        # HIERARCH ESO PRO MUSE has been renamed into HIERARCH ESO DRS MUSE
        # in recent versions of the DRS. Try with the
        if key.startswith('HIERARCH ESO PRO MUSE'):
            alternate_key = key.replace('HIERARCH ESO PRO MUSE', 'HIERARCH ESO DRS MUSE')
        elif key.startswith('HIERARCH ESO DRS MUSE'):
            alternate_key = key.replace('HIERARCH ESO DRS MUSE', 'HIERARCH ESO PRO MUSE')
        else:
            alternate_key = key
        try:
            return self.primary_header[key].value
        except:
            return self.primary_header[alternate_key].value
    
    def reconstruct_sky_image(self, lbda=None, step=None):
        """Reconstructs the image on the sky from the pixtable.
        
        :param lbda: (min, max) wavelength range in Angstrom. If None, the image is reconstructed for all wavelengths.
        :type lbda: (float,float)
        :param step: pixel step of the final image in arcsec. If None, the value corresponding to the keyword "HIERARCH ESO INS PIXSCALE" is used.
        :type step: (float,float)
        
        :rtype: :class:`mpdaf.obj.Image`
        """
        #TODO replace by DRS
        #step in arcsec
        from scipy import interpolate
        
        if step is None:
            step = self.get_keywords('HIERARCH ESO INS PIXSCALE')
            if step <= 0 :
                raise ValueError, 'INS PIXSCALE not valid'
            xstep = step
            ystep = step
        else:
            ystep,xstep = step    
        
        col_dq = self.get_dq()
        if lbda is None:
            ksel = np.where((col_dq==0))
        else:
            l1,l2 = lbda
            col_lambda = self.get_lambda()
            ksel = np.where((col_dq==0) & (col_lambda>l1) & (col_lambda<l2))
            del col_lambda
        del col_dq
        
        x = self.get_xpos()[ksel]
        y = self.get_ypos()[ksel]
        data = self.get_data()[ksel]
        
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        
        xstep /= (-3600.*np.cos((ymin+ymax)/2.))
        ystep /= 3600.
        
        nx = 1 + int( (xmin - xmax) / xstep )
        grid_x = np.arange(nx) * xstep + xmax
        ny = 1 + int( (ymax - ymin) / ystep )
        grid_y = np.arange(ny) * ystep + ymin
        shape = (ny,nx)
          
        points = np.empty((len(ksel[0]),2),dtype=float)
        points[:,0] = self.get_ypos()[ksel]
        points[:,1] = self.get_xpos()[ksel]
        data = self.get_data()[ksel]

        new_data= interpolate.griddata(points, data, np.meshgrid(grid_y,grid_x), method='linear').T

        from mpdaf.obj import Image,WCS
        wcs = WCS(crpix=(1.0,1.0),crval=(ymin,xmax),cdelt=(ystep,xstep),shape=shape)
        ima = Image(data=new_data,wcs=wcs)
        return ima

    def reconstruct_det_image(self):
        """Reconstructs the image on the detector from the pixtable.
        The pixtable must concerns only one IFU, otherwise an exception is raised.
        
        :rtype: :class:`mpdaf.obj.Image`
        """
        if self.nrows == 0:
            return None

        col_data = self.get_data()
        col_origin = self.get_origin()

        ifu = np.empty(self.nrows, dtype='uint16')
        slice = np.empty(self.nrows, dtype='uint16')
        xpix = np.empty(self.nrows, dtype='uint16')
        ypix = np.empty(self.nrows, dtype='uint16')

        ifu,slice,ypix,xpix = self.origin2coords(col_origin)
        if len(np.unique(ifu)) != 1:
            raise ValueError, 'Pixtable contains multiple IFU'

        xstart, xstop = xpix.min(), xpix.max()
        ystart, ystop = ypix.min(), ypix.max()
        image = np.zeros((ystop - ystart + 1, xstop - xstart + 1), dtype='float')
        image[ypix - ystart, xpix - xstart] = col_data

        wcs = WCS(crval=(ystart, xstart))

        return Image(shape=(image.shape), data=image, wcs=wcs)
