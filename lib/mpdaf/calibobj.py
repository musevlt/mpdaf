""" calibobj.py Manages calibration FITS files 
type MASTER_BIAS MASTER_DARK MASTER_FLAT OBJECT_RESAMPLED
"""
import numpy as np
import pyfits
import pylab
import datetime

class CalibFile(object):
    """CalibFile class
    
    This class manages inout/output for the calibration files

    Attributes
    ----------
    filename : string
    The FITS file name. None if any.
    
    primary_header: pyfits.CardList
    The primary header
    
    data: array
    Array containing the image data

    dq: array
    Array containing the data quality

    stat: array
    Array containing the statistics
    
    nx: integer
    Dimension of the data/dq/stat arry along the x-axis
    
    ny: integer
    Dimension of the data/dq/stat arry along the y-axis

    Methods
    -------
    Creation: init, copy
    
    Info: info
    
    save: write
    
    get: get_data, get_dq, get_stat
    """    
    
    def __init__(self, filename=None):
        """creates a CalibFile object
        
        Parameters
        ----------   
        filename : string
        The FITS file name. None by default.

        Notes
        -----
        filename=None creates an empty object

        The FITS file is opened with memory mapping.
        Just the primary header and array dimensions are loaded.
        Methods get_data, get_dq and get_stat must be used to get array extensions.
        """
        self.filename = filename
        self.data = None
        self.dq = None
        self.stat = None
        self.nx = 0
        self.ny = 0
        if filename!=None:
            try:
                hdulist = pyfits.open(self.filename,memmap=1)
                self.primary_header = hdulist[0].header.ascardlist()
                try:
                    self.nx = hdulist["DATA"].header["NAXIS1"]
                    self.ny = hdulist["DATA"].header["NAXIS2"]
                except:
                    print 'format error: no image DATA extension'
                    print
                try:
                    naxis1 = hdulist["DQ"].header["NAXIS1"]
                    naxis2 = hdulist["DQ"].header["NAXIS2"]
                except:
                    print 'format error: no image DQ extension'
                    print
                if naxis1!=self.nx and naxis2!=self.ny:
                    print 'format error: DATA and DQ with different sizes'
                    print
                try:
                    naxis1 = hdulist["STAT"].header["NAXIS1"]
                    naxis2 = hdulist["STAT"].header["NAXIS2"]
                except:
                    print 'format error: no image STAT extension'
                    print
                if naxis1!=self.nx and naxis2!=self.ny:
                    print 'format error: DATA and STAT with different sizes'
                    print       
                hdulist.close()
            except IOError:
                print 'IOError: file %s not found' % `filename`
                print
                self.filename = None
                self.primary_header = None
        else:
            self.primary_header = pyfits.CardList()
            
    def copy(self):
        """copies CalibFile object in a new one and returns it"""
        result = CalibFile()
        result.filename = self.filename
        result.primary_header = pyfits.CardList(self.primary_header)
        if self.data != None:
            result.data = self.data.__copy__()
        if self.dq != None:
            result.dq = self.dq.__copy__()
        if self.stat != None:
            result.stat = self.stat.__copy__()
        result.nx = self.nx
        result.ny = self.ny
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
            print "1\tDATA\timage\t(%i,%i)" % (self.nx,self.ny)
            print "2\tDQ\timage\t(%i,%i)" % (self.nx,self.ny)
            print "3\tSTAT\timage\t(%i,%i)" % (self.nx,self.ny)
    
    def get_data(self):
        """opens the FITS file with memory mapping, loads the data array and returns it"""
        if self.data != None:
            return self.data
        else:
            hdulist = pyfits.open(self.filename,memmap=1)
            data = hdulist["DATA"].data
            return data
    
    def get_dq(self):
        """opens the FITS file with memory mapping, loads the dq array and returns it"""
        if self.dq != None:
            return self.dq
        else:
            hdulist = pyfits.open(self.filename,memmap=1)
            data = hdulist["DQ"].data
            return data
        
    def get_stat(self):
        """opens the FITS file with memory mapping, loads the stat array and returns it"""
        if self.stat != None:
            return self.stat
        else:
            hdulist = pyfits.open(self.filename,memmap=1)
            data = hdulist["STAT"].data
            return data

    def write(self,filename):
        """ saves the object in a FITS file
        Parameters
        ----------
        filename : string
        The FITS filename
        """
        # create primary header
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
        hdulist = [prihdu]
        datahdu = pyfits.ImageHDU(name='DATA', data=self.get_data())
        hdulist.append(datahdu)
        dqhdu = pyfits.ImageHDU(name='DQ', data=self.get_dq())
        hdulist.append(dqhdu)
        stathdu = pyfits.ImageHDU(name='STAT', data=self.get_stat())
        hdulist.append(stathdu)
        # save to disk 
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True)
        # update attributes
        self.filename = filename
    