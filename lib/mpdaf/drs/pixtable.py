""" pixtable.py Manages MUSE pixel table files
"""
import numpy as np
import pyfits
import datetime
import tempfile
import os

class PixTable(object):
    """PixTable class

    This class manages input/output for MUSE pixel table files

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

    xpos: string
    name of memory-mapped files used for accessing pixel position on the x-axis (in deg)

    ypos: string
    name of memory-mapped files used for accessing pixel position on the y-axis (in deg)

    lbda: string
    name of memory-mapped files used for accessing wavelength value (in Angstrom)

    data: string
    name of memory-mapped files used for accessing pixel values (in e-)

    dq: string
    name of memory-mapped files used for accessing bad pixel status as defined by Euro3D

    stat: string
    name of memory-mapped files used for accessing variance

    origin: string
    name of memory-mapped files used for accessing an encoded value of IFU and slice number

    Public methods
    --------------
    Creation: init, copy

    Info: info

    Save: write

    Get: get_xpos, get_ypos, get_lambda, get_data, get_dq, get_stat, get_origin

    Other: extract, origin2ifu, origin2slice, origin2ypix, origin2xoffset, origin2xpix, origin2coords, get_slices
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
        self.xpos = None
        self.ypos = None
        self.lbda = None
        self.data = None
        self.dq = None
        self.stat = None
        self.origin = None

        if filename!=None:
            try:
                hdulist = pyfits.open(self.filename,memmap=1)
                self.primary_header = hdulist[0].header.ascardlist()
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
        """removes temporary files used for memory mapping"""
        try:
            if self.xpos != None:
                os.remove(self.xpos)
            if self.ypos != None:
                os.remove(self.ypos)
            if self.lbda != None:
                os.remove(self.lbda)
            if self.data != None:
                os.remove(self.data)
            if self.dq != None:
                os.remove(self.dq)
            if self.stat != None:
                os.remove(self.stat)
            if self.origin != None:
                os.remove(self.origin)
        except:
            pass

    def copy(self):
        """copies PixTable object in a new one and returns it"""
        result = PixTable()
        result.filename = self.filename

        result.nrows = self.nrows
        result.ncols = self.ncols
        result.primary_header = pyfits.CardList(self.primary_header)
        #xpos
        (fd,result.xpos) = tempfile.mkstemp(prefix='mpdaf')
        selfxpos=self.get_xpos()
        xpos = np.memmap(result.xpos,dtype="float32",shape=(self.nrows))
        xpos[:] = selfxpos[:]
        del xpos, selfxpos
        os.close(fd)
        #ypos
        (fd,result.ypos) = tempfile.mkstemp(prefix='mpdaf')
        selfypos=self.get_ypos()
        ypos = np.memmap(result.ypos,dtype="float32",shape=(self.nrows))
        ypos[:] = selfypos[:]
        del ypos, selfypos
        os.close(fd)
        #lambda
        (fd,result.lbda) = tempfile.mkstemp(prefix='mpdaf')
        selflbda=self.get_lambda()
        lbda = np.memmap(result.lbda,dtype="float32",shape=(self.nrows))
        lbda[:] = selflbda[:]
        del lbda, selflbda
        os.close(fd)
        #data
        (fd,result.data) = tempfile.mkstemp(prefix='mpdaf')
        selfdata=self.get_data()
        data = np.memmap(result.data,dtype="float32",shape=(self.nrows))
        data[:] = selfdata[:]
        del data, selfdata
        os.close(fd)
        #variance
        (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
        selfstat=self.get_stat()
        stat = np.memmap(result.stat,dtype="float32",shape=(self.nrows))
        stat[:] = selfstat[:]
        del stat, selfstat
        os.close(fd)
        # pixel quality
        (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
        selfdq = self.get_dq()
        dq = np.memmap(result.dq,dtype="uint32",shape=(self.nrows))
        dq[:] = selfdq[:]
        del dq, selfdq
        os.close(fd)
        # origin
        (fd,result.origin) = tempfile.mkstemp(prefix='mpdaf')
        selforigin = self.get_origin()
        origin = np.memmap(result.origin,dtype="uint32",shape=(self.nrows))
        origin[:] = selforigin[:]
        del origin, selforigin
        os.close(fd)
        return result

    def info(self):
        """prints information"""
        if self.filename != None:
            hdulist = pyfits.open(self.filename,memmap=1)
            print hdulist.info()
            hdulist.close()
        else:
            print 'No\tName\tType\tDim'
            print '0\tPRIMARY\tcard\t()'
            print "1\t\tTABLE\t(%iR,%iC)" % (self.nrows,self.ncols)

    def get_xpos(self):
        """loads the xpos column and returns it"""
        if self.xpos != None:
            xpos = np.memmap(self.xpos,dtype="float32",shape=(self.nrows))
            return xpos
        else:
            if self.filename == None:
                print 'format error: empty XPOS column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.xpos) = tempfile.mkstemp(prefix='mpdaf')
                data_xpos = hdulist[1].data.field('xpos')
                xpos = np.memmap(self.xpos,dtype="float32",shape=(self.nrows))
                xpos[:] = data_xpos[:]
                hdulist.close()
                os.close(fd)
                return xpos

    def get_ypos(self):
        """loads the ypos column and returns it"""
        if self.ypos != None:
            ypos = np.memmap(self.ypos,dtype="float32",shape=(self.nrows))
            return ypos
        else:
            if self.filename == None:
                print 'format error: empty YPOS column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.ypos) = tempfile.mkstemp(prefix='mpdaf')
                data_ypos = hdulist[1].data.field('ypos')
                ypos = np.memmap(self.ypos,dtype="float32",shape=(self.nrows))
                ypos[:] = data_ypos[:]
                hdulist.close()
                os.close(fd)
                return ypos

    def get_lambda(self):
        """loads the lambda column and returns it"""
        if self.lbda != None:
            lbda = np.memmap(self.lbda,dtype="float32",shape=(self.nrows))
            return lbda
        else:
            if self.filename == None:
                print 'format error: empty YLAMBDA column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.lbda) = tempfile.mkstemp(prefix='mpdaf')
                data_lbda = hdulist[1].data.field('lambda')
                lbda = np.memmap(self.lbda,dtype="float32",shape=(self.nrows))
                lbda[:] = data_lbda[:]
                hdulist.close()
                os.close(fd)
                return lbda

    def get_data(self):
        """loads the data column and returns it"""
        if self.data != None:
            data = np.memmap(self.data,dtype="float32",shape=(self.nrows))
            return data
        else:
            if self.filename == None:
                print 'format error: empty DATA column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.data) = tempfile.mkstemp(prefix='mpdaf')
                data_data = hdulist[1].data.field('data')
                data = np.memmap(self.data,dtype="float32",shape=(self.nrows))
                data[:] = data_data[:]
                hdulist.close()
                os.close(fd)
                return data

    def get_stat(self):
        """loads the stat column and returns it"""
        if self.stat != None:
            stat = np.memmap(self.stat,dtype="float32",shape=(self.nrows))
            return stat
        else:
            if self.filename == None:
                print 'format error: empty STAT column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.stat) = tempfile.mkstemp(prefix='mpdaf')
                data_stat = hdulist[1].data.field('stat')
                stat = np.memmap(self.stat,dtype="float32",shape=(self.nrows))
                stat[:] = data_stat[:]
                hdulist.close()
                os.close(fd)
                return stat

    def get_dq(self):
        """loads the dq column and returns it"""
        if self.dq != None:
            dq = np.memmap(self.dq,dtype="uint32",shape=(self.nrows))
            return dq
        else:
            if self.filename == None:
                print 'format error: empty DQ column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.dq) = tempfile.mkstemp(prefix='mpdaf')
                data_dq = hdulist[1].data.field('dq')
                dq = np.memmap(self.dq,dtype="uint32",shape=(self.nrows))
                dq[:] = data_dq[:]
                hdulist.close()
                os.close(fd)
                return dq

    def get_origin(self):
        """loads the origin column and returns it"""
        if self.origin != None:
            origin = np.memmap(self.origin,dtype="uint32",shape=(self.nrows))
            return origin
        else:
            if self.filename == None:
                print 'format error: empty ORIGIN column'
                print
                return None
            else:
                hdulist = pyfits.open(self.filename,memmap=1)
                (fd,self.origin) = tempfile.mkstemp(prefix='mpdaf')
                data_origin = hdulist[1].data.field('origin')
                origin = np.memmap(self.origin,dtype="uint32",shape=(self.nrows))
                origin[:] = data_origin[:]
                hdulist.close()
                os.close(fd)
                return origin

    def write(self,filename):
        """ saves the object in a FITS file
        Parameters
        ----------
        filename : string
        The FITS filename
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

    def extract(self, center, size=None, lbda=None, shape='C'):
        """ extracts a spatial aperture and a wavelength range from a PixTable,
        aperture is define as center,size [if size=None the full field is used]
        size = radius (for circular aperture) or half side length (for square aperture)
        wavelength range = (l1,l2) if None the full wavelength range is used

        Parameters
        ----------
        center: (float,float)
        (x,y) center coordinate in deg

        size: float
        size in deg

        lbda: (float,float)
        (min, max) wavelength range in Angstrom

        shape: char
        'C' for circular aperture, 'S' for square aperture
        """
        x0,y0 = center
        ptab = PixTable()
        ptab.primary_header = pyfits.CardList(self.primary_header)
        ptab.ncols = self.ncols
        if self.nrows != 0:
            col_xpos = self.get_xpos()
            col_ypos = self.get_ypos()

            if lbda is None:
                if size is None:
                    return self
                elif shape == 'C':
                    ksel = np.where(((col_xpos-x0)**2 + (col_ypos-y0)**2)<size**2)
                elif shape == 'S':
                    ksel = np.where((np.abs(col_xpos-x0)<size) & (np.abs(col_ypos-y0)<size))
                else:
                    raise ValueError, 'Unknown shape parameter'
            else:
                l1,l2 = lbda
                col_lambda = self.get_lambda()
                if size is None:
                    ksel = np.where((col_lambda>l1) & (col_lambda<l2))
                else:
                    if shape == 'C':
                        ksel = np.where((((col_xpos-x0)**2 + (col_ypos-y0)**2)<size**2) &
                                        (col_lambda>l1) & (col_lambda<l2))
                    elif shape == 'S':
                        ksel = np.where((np.abs(col_xpos-x0)<size) & (np.abs(col_ypos-y0)<size) &
                                        (col_lambda>l1) & (col_lambda<l2))
                    else:
                        raise ValueError, 'Unknown shape parameter'
                del col_lambda
            npts = len(ksel[0])
            if npts == 0:
                raise ValueError, 'Empty selection'
            ptab.nrows = npts
            #xpos
            (fd,ptab.xpos) = tempfile.mkstemp(prefix='mpdaf')
            xpos = np.memmap(ptab.xpos,dtype="float32",shape=(npts))
            xpos[:] = col_xpos[ksel]
            del xpos,col_xpos
            os.close(fd)
            #ypos
            (fd,ptab.ypos) = tempfile.mkstemp(prefix='mpdaf')
            ypos = np.memmap(ptab.ypos,dtype="float32",shape=(npts))
            ypos[:] = col_ypos[ksel]
            del ypos,col_ypos
            os.close(fd)
            #lambda
            (fd,ptab.lbda) = tempfile.mkstemp(prefix='mpdaf')
            lbda = np.memmap(ptab.lbda,dtype="float32",shape=(npts))
            selflbda=self.get_lambda()
            lbda[:] = selflbda[ksel]
            del lbda,selflbda
            os.close(fd)
            #data
            (fd,ptab.data) = tempfile.mkstemp(prefix='mpdaf')
            selfdata=self.get_data()
            data = np.memmap(ptab.data,dtype="float32",shape=(npts))
            data[:] = selfdata[ksel]
            del data,selfdata
            os.close(fd)
            #variance
            (fd,ptab.stat) = tempfile.mkstemp(prefix='mpdaf')
            selfstat=self.get_stat()
            stat = np.memmap(ptab.stat,dtype="float32",shape=(npts))
            stat[:] = selfstat[ksel]
            del stat,selfstat
            os.close(fd)
            # pixel quality
            (fd,ptab.dq) = tempfile.mkstemp(prefix='mpdaf')
            selfdq = self.get_dq()
            dq = np.memmap(ptab.dq,dtype="uint32",shape=(npts))
            dq[:] = selfdq[ksel]
            del dq,selfdq
            os.close(fd)
            # origin
            (fd,ptab.origin) = tempfile.mkstemp(prefix='mpdaf')
            selforigin = self.get_origin()
            origin = np.memmap(ptab.origin,dtype="uint32",shape=(npts))
            origin[:] = selforigin[ksel]
            del origin,selforigin
            os.close(fd)
        return ptab

    def origin2ifu(self, origin):
        """ converts the origin value and returns the ifu number

        Parameters
        ----------
        origin: integer
        origin value
        """
        return (origin >> 6) & 0x1f

    def origin2slice(self, origin):
        """ converts the origin value and returns the slice number

        Parameters
        ----------
        origin: integer
        origin value
        """
        return origin & 0x3f

    def origin2ypix(self, origin):
        """ converts the origin value and returns the y coordinates

        Parameters
        ----------
        origin: integer
        origin value
        """
        return ((origin >> 11) & 0x1fff) - 1

    def origin2xoffset(self, origin):
        """ converts the origin value and returns the x coordinates offset
        Parameters
        ----------
        origin: integer
        origin value
        """
        col_ifu = self.origin2ifu(origin)
        col_slice = self.origin2slice(origin)
        if isinstance(origin, np.ndarray):
            xoffset = np.zeros_like(origin)
            for ifu in np.unique(col_ifu):
                for slice in np.unique(col_slice):
                    value = self.primary_header["ESO PRO MUSE PIXTABLE EXP0 IFU%02d SLICE%02d XOFFSET" % (ifu, slice)].value
                    xoffset[np.where((col_ifu == ifu) & (col_slice == slice))] = value
        else:
            xoffset = self.primary_header["ESO PRO MUSE PIXTABLE EXP0 IFU%02d SLICE%02d XOFFSET" % (ifu, slice)].value
        return xoffset

    def origin2xpix(self, origin):
        """ converts the origin value and returns the x coordinates offset
        Parameters
        ----------
        origin: integer
        origin value
        """
        return self.origin2xoffset(origin) + ((origin >> 24) & 0x7f) - 1

    def origin2coords(self, origin):
        """ converts the origin value and returns (ifu, slice, ypix, xpix)

        Parameters
        ----------
        origin: integer
        origin value
        """
        return (self.origin2ifu(origin), self.origin2slice(origin),
                self.origin2ypix(origin), self.origin2xpix(origin))

    def get_slices(self):
        '''returns slices dictionary'''
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
        return self.primary_header[key].value
