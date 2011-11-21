""" calibobj.py Manages calibration FITS files 
type MASTER_BIAS MASTER_DARK MASTER_FLAT OBJECT_RESAMPLED
"""
import numpy as np
import pyfits
import pylab
import datetime
import os
import tempfile
import multiprocessing
import sys

class CalibFile(object):
    """CalibFile class
    
    This class manages input/output for the calibration files

    Attributes
    ----------
    filename : string
    The FITS file name. None if any.
    
    primary_header: pyfits.CardList
    The primary header
    
    data: float array
    name of memory-mapped files used for accessing pixel values

    dq: integer array
    name of memory-mapped files used for accessing bad pixel status as defined by Euro3D

    stat: float array
    name of memory-mapped files used for accessing variance
    
    nx: integer
    Dimension of the data/dq/stat arrays along the x-axis
    
    ny: integer
    Dimension of the data/dq/stat arrays along the y-axis

    Public methods
    --------------
    Creation: init, copy
    
    Info: info
    
    Save: write
    
    Get: get_data, get_dq, get_stat
    
    Arithmetic: + - * /
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
            
    def __del__(self):
        """removes temporary files used for memory mapping"""
        if self.data != None:
            os.remove(self.data)
        if self.dq != None:
            os.remove(self.dq)
        if self.stat != None:
            os.remove(self.stat)
            
    def copy(self):
        """copies CalibFile object in a new one and returns it"""
        result = CalibFile()
        result.filename = self.filename
        result.primary_header = pyfits.CardList(self.primary_header)
        result.nx = self.nx
        result.ny = self.ny
        #data
        (fd,result.data) = tempfile.mkstemp(prefix='mpdaf')
        selfdata=self.get_data()
        data = np.memmap(result.data,dtype="float32",shape=(self.ny,self.nx))
        data[:] = selfdata[:]
        #variance
        (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
        selfstat=self.get_stat()
        stat = np.memmap(result.stat,dtype="float32",shape=(self.ny,self.nx))
        stat[:] = selfstat[:]
        # pixel quality
        (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
        selfdq = self.get_dq()
        dq = np.memmap(result.dq,dtype="int32",shape=(self.ny,self.nx))
        dq[:] = selfdq[:]  
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
        if self.filename == None:
            if self.data == None:
                print 'format error: empty DATA extension'
                print
                return None
            else:
                data = np.memmap(self.data,dtype="float32",shape=(self.ny,self.nx))
                return data
        else:
            hdulist = pyfits.open(self.filename,memmap=1)
            data = hdulist["DATA"].data
            hdulist.close()
            return data
    
    def get_dq(self):
        """opens the FITS file with memory mapping, loads the dq array and returns it"""
        if self.filename == None:
            if self.dq == None:
                print 'format error: empty DQ extension'
                print
                return None
            else:
                dq = np.memmap(self.dq,dtype="int32",shape=(self.ny,self.nx))
                return dq
        else:
            hdulist = pyfits.open(self.filename,memmap=1)
            data = hdulist["DQ"].data
            hdulist.close()
            return data
        
    def get_stat(self):
        """opens the FITS file with memory mapping, loads the stat array and returns it"""
        if self.filename == None:
            if self.stat == None:
                print 'format error: empty STAT extension'
                print
                return None
            else:
                stat = np.memmap(self.stat,dtype="float32",shape=(self.ny,self.nx))
                return stat
        else:
            hdulist = pyfits.open(self.filename,memmap=1)
            data = hdulist["STAT"].data
            hdulist.close()
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
        if self.data != None:
            os.remove(self.data)
        if self.dq != None:
            os.remove(self.dq)
        if self.stat != None:
            os.remove(self.stat)
        self.data = None
        self.dq = None
        self.stat = None
      
      
    def __add__(self,other):
        """adds either a number or a CalibFile object"""
        result = CalibFile()             
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        (fd,result.data) = tempfile.mkstemp(prefix='mpdaf')
        (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
        (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
        if isinstance(other,CalibFile):
            #sum data values
            newdata = np.ndarray.__add__(self.get_data(),other.get_data())
            data = np.memmap(result.data,dtype="float32",shape=(self.ny,self.nx))
            data[:] = newdata[:]
            # sum variance
            newstat = np.ndarray.__add__(self.get_stat(),other.get_stat())
            stat = np.memmap(result.stat,dtype="float32",shape=(self.ny,self.nx))
            stat[:] = newstat[:]
            # pixel quality
            newdq = np.logical_or(self.get_dq(),other.get_dq())
            dq = np.memmap(result.dq,dtype="int32",shape=(self.ny,self.nx))
            dq[:] = newdq[:]
        else:
            # sum data values
            newdata = np.ndarray.__add__(self.get_data(),other)
            data = np.memmap(result.data,dtype="float32",shape=(self.ny,self.nx))
            data[:] = newdata[:]
            #variance
            selfstat=self.get_stat()
            stat = np.memmap(result.stat,dtype="float32",shape=(self.ny,self.nx))
            stat[:] = selfstat[:]
            # pixel quality
            selfdq = self.get_dq()
            dq = np.memmap(result.dq,dtype="int32",shape=(self.ny,self.nx))
            dq[:] = selfdq[:]
        return result
        
    def __iadd__(self,other):
        """adds either a number or a CalibFile object"""
        if self.filename != None:
            return self.__add__(other)
        else:
            if isinstance(other,CalibFile):
                #sum data values
                newdata = np.ndarray.__add__(self.get_data(),other.get_data())
                data = np.memmap(self.data,dtype="float32",shape=(self.ny,self.nx))
                data[:] = newdata[:]
                # sum variance
                newstat = np.ndarray.__add__(self.get_stat(),other.get_stat())
                stat = np.memmap(self.stat,dtype="float32",shape=(self.ny,self.nx))
                stat[:] = newstat[:]
                # pixel quality
                newdq = np.logical_or(self.get_dq(),other.get_dq())
                dq = np.memmap(self.dq,dtype="int32",shape=(self.ny,self.nx))
                dq[:] = newdq[:]
            else:
                # sum data values
                newdata = np.ndarray.__add__(self.get_data(),other)
                data = np.memmap(self.data,dtype="float32",shape=(self.ny,self.nx))
                data[:] = newdata[:]
            return self
            
    def __sub__(self,other):
        """subtracts either a number or a CalibFile object"""
        result = CalibFile()             
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        (fd,result.data) = tempfile.mkstemp(prefix='mpdaf')
        (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
        (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
        if isinstance(other,CalibFile):
            #sum data values
            newdata = np.ndarray.__sub__(self.get_data(),other.get_data())
            data = np.memmap(result.data,dtype="float32",shape=(self.ny,self.nx))
            data[:] = newdata[:]
            # sum variance
            newstat = np.ndarray.__add__(self.get_stat(),other.get_stat())
            stat = np.memmap(result.stat,dtype="float32",shape=(self.ny,self.nx))
            stat[:] = newstat[:]
            # pixel quality
            newdq = np.logical_or(self.get_dq(),other.get_dq())
            dq = np.memmap(result.dq,dtype="int32",shape=(self.ny,self.nx))
            dq[:] = newdq[:]
        else:
            # sum data values
            newdata = np.ndarray.__sub__(self.get_data(),other)
            data = np.memmap(result.data,dtype="float32",shape=(self.ny,self.nx))
            data[:] = newdata[:]
            #variance
            selfstat=self.get_stat()
            stat = np.memmap(result.stat,dtype="float32",shape=(self.ny,self.nx))
            stat[:] = selfstat[:]
            # pixel quality
            selfdq = self.get_dq()
            dq = np.memmap(result.dq,dtype="int32",shape=(self.ny,self.nx))
            dq[:] = selfdq[:]
        return result
    
    def __isub__(self,other):
        """subtracts either a number or a CalibFile object"""
        if self.filename != None:
            return self.__sub__(other)
        else:
            if isinstance(other,CalibFile):
                #sum data values
                newdata = np.ndarray.__sub__(self.get_data(),other.get_data())
                data = np.memmap(self.data,dtype="float32",shape=(self.ny,self.nx))
                data[:] = newdata[:]
                # sum variance
                newstat = np.ndarray.__add__(self.get_stat(),other.get_stat())
                stat = np.memmap(self.stat,dtype="float32",shape=(self.ny,self.nx))
                stat[:] = newstat[:]
                # pixel quality
                newdq = np.logical_or(self.get_dq(),other.get_dq())
                dq = np.memmap(self.dq,dtype="int32",shape=(self.ny,self.nx))
                dq[:] = newdq[:]
            else:
                # sum data values
                newdata = np.ndarray.__sub__(self.get_data(),other)
                data = np.memmap(self.data,dtype="float32",shape=(self.ny,self.nx))
                data[:] = newdata[:]
            return self
    
    
    def __mul__(self,other):
        """multiplies by a number"""
        if isinstance(other,CalibFile):
            print 'unsupported operand type * and / for CalibFile'
            print
            return None
        else:
            result = CalibFile()             
            result.primary_header = self.primary_header
            result.nx = self.nx
            result.ny = self.ny
            (fd,result.data) = tempfile.mkstemp(prefix='mpdaf')
            (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
            (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
            # sum data values
            newdata = np.ndarray.__mul__(self.get_data(),other)
            data = np.memmap(result.data,dtype="float32",shape=(self.ny,self.nx))
            data[:] = newdata[:]
            #variance
            newstat = np.ndarray.__mul__(self.get_stat(),other*other)
            stat = np.memmap(result.stat,dtype="float32",shape=(self.ny,self.nx))
            stat[:] = selfstat[:]
            # pixel quality
            selfdq = self.get_dq()
            dq = np.memmap(result.dq,dtype="int32",shape=(self.ny,self.nx))
            dq[:] = selfdq[:]
            return result
        
    
    def __imul__(self,other):
        """multiplies by a number"""
        if self.filename != None:
            return self.__mul__(other)
        else:
            if isinstance(other,CalibFile):
                print 'unsupported operand type * and / for CalibFile'
                print
                return None
            else:
                # sum data values
                newdata = np.ndarray.__mul__(self.get_data(),other)
                data = np.memmap(self.data,dtype="float32",shape=(self.ny,self.nx))
                data[:] = newdata[:]
                #variance
                newstat = np.ndarray.__mul__(self.get_stat(),other*other)
                stat = np.memmap(self.stat,dtype="float32",shape=(self.ny,self.nx))
                stat[:] = selfstat[:]
            return self   
    
    
    def __div__(self,other): 
        """divides by a number"""
        return self.__mul__(1./other)
    
    
    def __idiv__(self,other): 
        """divides by a number"""
        return self.__imul__(1./other)
 
STR_FUNCTIONS = { 'CalibFile.__mul__' : CalibFile.__mul__,
                  'CalibFile.__imul__' : CalibFile.__imul__,
                  'CalibFile.__div__' : CalibFile.__div__,
                  'CalibFile.__idiv__' : CalibFile.__idiv__,
                  'CalibFile.__sub__' : CalibFile.__sub__,
                  'CalibFile.__isub__' : CalibFile.__isub__,
                  'CalibFile.__add__' : CalibFile.__add__,
                  'CalibFile.__iadd__' : CalibFile.__iadd__,
                  }       
        
class CalibDir(object):
    """CalibDir class
    
    This class manages input/output for a repository containing calibration files (one per ifu)

    Attributes
    ----------
    dirname : string
    The repository name. None if any.
    This repository must contain files labeled <type>_<ifu id>.fits
    
    type : string
    Type of calibration files that appears in filenames (<type>_<ifu id>.fits)
    
    files : dict
    List of files (ifu id,CalibFile)

    Public methods
    --------------
    Creation: init, copy
    
    Info: info
    
    Save: write
    
    Get: get_data, get_dq, get_stat
    
    Arithmetic: + - * /
    """  
          
    def __init__(self, type, dirname = None):
        """creates a CalibDir object
        
        Parameters
        ---------- 
        type : string
        Type of calibration files that appears in filenames (<type>_<ifu id>.fits)
          
        dirname : string
        The repository name. None if any.
        This repository must contain files labeled <type>_<ifu id>.fits
        """
        self.dirname = dirname
        self.type = type
        self.files = dict()
        if dirname != None:
            for i in range(24):
                ifu = i+1
                filename = "%s/%s_%02d.fits" %(dirname,type,ifu)
                if os.path.exists(filename):
                    fileobj = CalibFile(filename)
                    self.files[ifu] = fileobj
        
                
    def copy(self):
        """copies Calibdir object in a new one and returns it"""
        result = CalibDir(self.dirname,self.type)
        if self.dirname == None:
            for ifu,fileobj in self.files.items():
                files[ifu] = fileobj.copy()
        return result
                
    def info(self):
        """prints information"""
        print '%i %s files' %(len(self.files),self.type)
        for ifu,fileobj in self.files.items():
            print 'ifu %i' %ifu
            fileobj.info()
            
    def _check(self,other):
        """checks that other CalibDir contains the same ifu that the current object"""
        if self.type != other.type:
            print 'error: objects with different types'
            print
            return False
        if len(self.files) != len(other.files) or len(set(self.files)-set(other.files)) != 0:
            print 'error: objects that contains different ifu data'
            print
            return False   
        else:
            return True
            
    def __mul__(self,other):
        """multiplies by a number"""
        if isinstance(other,DirFile):
            print 'unsupported operand type * and / for CalibFile'
            print
            return None
        else:
            return self._mp_operator(other,'CalibFile.__mul__')
        
    def __imul__(self,other):
        """multiplies by a number"""
        if isinstance(other,DirFile):
            print 'unsupported operand type * and / for CalibFile'
            print
            return None
        else:
            return self._mp_operator(other,'CalibFile.__imul__')


    def __div__(self,other):
        """divides by a number"""
        if isinstance(other,DirFile):
            print 'unsupported operand type * and / for CalibFile'
            print
            return None
        else:
            return self._mp_operator(other,'CalibFile.__div__')
        
    def __idiv__(self,other):
        """divides by a number"""
        if isinstance(other,DirFile):
            print 'unsupported operand type * and / for CalibFile'
            print
            return None
        else:
            return self._mp_operator(other,'CalibFile.__idiv__')


    def __sub__(self,other):
        """subtracts either a number or a CalibFile object"""
        if self._check(other):
            return self._mp_operator(other,'CalibFile.__sub__')
        else:
            return None
        
    def __isub__(self,other):
        """subtracts either a number or a CalibFile object"""
        if self._check(other):
            return self._mp_operator(other,'CalibFile.__isub__')
        else:
            return None


    def __add__(self,other):
        """adds either a number or a CalibFile object"""
        if self._check(other):
            return self._mp_operator(other,'CalibFile.__add__')
        else:
            return None
        
    def __iadd__(self,other):
        """adds either a number or a CalibFile object"""
        if self._check(other):
            return self._mp_operator(other,'CalibFile.__iadd__')
        else:
            return None
        
    def _mp_operator(self,other,funcname):
        #multiprocessing function
        cpu_count = multiprocessing.cpu_count()
        result = CalibDir(self.type)
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
        if isinstance(other,CalibDir):
            for k in self.files.keys():
                processlist.append([funcname,k,self.files[k],other.files[k]])
        else:
            for k in self.files.keys():
                processlist.append([funcname,k,self.files[k],other])
        processresult = pool.map(_process_operator,processlist)
        for k,out in processresult:
            result.files[k] = out
        sys.stdout.write('\r                        \n')
        return result
    
    def write(self,dirname):
        """writes files in dirname"""
        if self.dirname == None:
            for ifu,fileobj in self.files.items():
                filename = "%s/%s_%02d.fits" %(dirname,self.type,ifu)
                fileobj.write(filename)
        self.dirname = dirname
        
    def __len__(self):
        """returns the number of files"""
        return len(self.files)

    def __getitem__(self, key):
        """returns the CalibFile object

        Parameters
        ----------
        key : integer
        Ifu id
        """
        extname = "CHAN%02d" %key
        if self.files.has_key(key):
            return self.files[key]
        else:
            print 'invalid key'
            print
            return None

    def __setitem__(self,key,value):
        """sets the corresponding CalibFile with value

        Parameters
        ----------
        key : integer
        Ifu id

        value: CalibFile
        CalibFile object
        """
        if isinstance(value,CalibFile):
            self.files[key] = value
        else:
            print 'format error: %s incompatible with CalibFile' %type(value)
            print
            return None
    
def _process_operator(arglist):
    #decorator used to define arithmetic functions with a RawFits object
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    f1 = arglist[2]
    f2 = arglist[3]
    out = function(f1,f2)
    sys.stdout.write(".")
    sys.stdout.flush()
    return (k,out)
