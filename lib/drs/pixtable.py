""" pixtable.py Manages MUSE pixel table files
"""
import numpy as np
import pyfits
import datetime

class PixTable(object):

    def __init__(self, filename=None):
        self.filename = filename
        self.formats = dict()
        self.units = dict()
        self.data = dict()
        self.nrows = 0
        self.ncols = 0
        if filename!=None:
            try:
                hdulist = pyfits.open(self.filename,memmap=1)
                self.primary_header = hdulist[0].header.ascardlist()
                self.nrows = hdulist[1].header["NAXIS2"]
                self.ncols = hdulist[1].header["TFIELDS"]
                for col in hdulist[1].columns:
                    #ptab.data[col.name] = f[1].data.field(col.name)
                    self.units[col.name] = col.unit
                    self.formats[col.name] = col.format
                hdulist.close()
            except IOError:
                print 'IOError: file %s not found' % `filename`
                print
                self.filename = None
                self.primary_header = None
        else:
            self.primary_header = pyfits.CardList()

    def copy(self):
        """copies PixTable object in a new one and returns it"""
        result = PixTable()
        result.filename = self.filename

        result.nrows = self.nrows
        result.ncols = self.ncols
        result.primary_header = pyfits.CardList(self.primary_header)

        for key,value in self.formats.items():
            result.formats[key] = value
        for key,value in self.units.items():
            result.units[key] = value
        for key,value in self.data.items():
            result.data[key] = value.__copy__()
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

    def get_data(self):
        """opens the FITS file with memory mapping, loads the data table and returns it"""
        if len(self.data) != 0:
            return self.data
        else:
            hdulist = pyfits.open(self.filename,memmap=1)
            data = dict()
            for col in hdulist[1].columns:
                data[col.name] = hdulist[1].data.field(col.name)
            hdulist.close()
            return data

    def write(self,filename):
        prihdu = pyfits.PrimaryHDU()
        if self.primary_header is not None:
            for card in self.primary_header:
                try:
                    prihdu.header.update(card.key, card.value, card.comment)
                except ValueError:
                    prihdu.header.update('hierarch %s' %card.key, card.value, card.comment)
                except:
                    pass
        prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
        prihdu.header.update('author', 'MPDAF', 'origin of the file')
        cols = []
        data = self.get_data()
        for key in data.keys():
            cols.append(pyfits.Column(name=key, format=self.formats[key],
                                      unit=self.units[key], array=data[key]))
        coltab = pyfits.ColDefs(cols)
        tbhdu = pyfits.new_table(coltab)
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True)
        # update attributes
        self.filename = filename
        self.data = dict()

    def extract(self, center, radius, lbda=None):
        x0,y0 = center
        ptab = PixTable()
        ptab.primary_header = pyfits.CardList(self.primary_header)
        ptab.formats = self.formats
        ptab.units = self.units
        ptab.ncols = self.ncols

        if len(self.data) != 0:
            #data in memory
            if lbda is None:
                ksel = np.where(((self.data['xpos']-x0)**2 + (self.data['ypos']-y0)**2)<radius**2)
            else:
                l1,l2 = lbda
                ksel = np.where((((self.data['xpos']-x0)**2 + (self.data['ypos']-y0)**2)<radius**2) &
                         (self.data['lambda']>l1) & (self.data['lambda']<l2))
            npts = len(ksel[0])
            if npts == 0:
                raise ValueError, 'Empty selection'
            ptab.nrows = npts
            for key in self.data.keys():
                ptab.data[key] = self.data[key][ksel]
        else:
            #open file with memory mapping
            hdulist = pyfits.open(self.filename,memmap=1)
            col_xpos = hdulist[1].data.field('xpos')
            col_ypos = hdulist[1].data.field('ypos')
            if lbda is None:
                ksel = np.where(((col_xpos-x0)**2 + (col_ypos-y0)**2)<radius**2)
            else:
                l1,l2 = lbda
                col_lambda = hdulist[1].data.field('lambda')
                ksel = np.where((((col_xpos-x0)**2 + (col_ypos-y0)**2)<radius**2) &
                         (col_lambda>l1) & (col_lambda<l2))
                del col_lambda
            del col_xpos,col_ypos
            npts = len(ksel[0])
            if npts == 0:
                raise ValueError, 'Empty selection'
            ptab.nrows = npts
            for key in self.formats.keys():
                ptab.data[key] = hdulist[1].data.field(key)[ksel]
            hdulist.close()
        return ptab

    # Convert the origin value to (ifu, slice, y, x)
    def origin2coords(self, origin):
        slice = origin & 0x3f
        ifu = (origin >> 6) & 0x1f
        y = ((origin >> 11) & 0x1fff) - 1
        xslice = ((origin >> 24) & 0x7f) - 1
        xoffset = self.primary_header["ESO PRO MUSE PIXTABLE IFU%02d SLICE%02d XOFFSET" % (ifu, slice)].value
        x = xoffset + xslice
        return (ifu, slice, y, x)

    def get_slices(self):
        #returns slices dictionary
        xpix = np.zeros(self.nrows, dtype='int')
        ypix = np.zeros(self.nrows, dtype='int')
        ifupix = np.zeros(self.nrows, dtype='int')
        slicepix = np.zeros(self.nrows, dtype='int')

        col_origin = None
        col_xpos = None
        col_ypos = None

        # get ifu,slice number and pixel coord
        if len(self.data) != 0:
            #data in memory
            col_origin = self.data['origin']
            col_xpos = self.data['xpos']
            col_ypos =   self.data['ypos']
        else:
            #open file with memory mapping
            hdulist = pyfits.open(self.filename,memmap=1)
            col_origin = hdulist[1].data.field('origin')
            col_xpos = hdulist[1].data.field('xpos')
            col_ypos = hdulist[1].data.field('ypos')
            hdulist.close()

        for i,orig in enumerate(col_origin):
            ifupix[i],slicepix[i],ypix[i],xpix[i] = self.origin2coords(orig)

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

        #del working array
        del col_origin, col_xpos, col_ypos

        slices = {'list':slicelist, 'skypos':skypos, 'ifupix':ifupix, 'slicepix':slicepix,
                       'xpix':xpix, 'ypix':ypix}

        print('%d slices found, stucture returned in slices dictionary '%(nslice))

        return slices

