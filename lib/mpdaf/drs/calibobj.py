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
    """CalibFile class manages input/output for the calibration files.
    
    :param filename: The FITS file name. None by default (in this case, an empty object is created).
        
            The FITS file is opened with memory mapping.
            
            Just the primary header and array dimensions are loaded.
            
            Methods get_data, get_dq and get_stat must be used to get array extensions.
    :type filename: string

    Attributes
    ----------
    filename : string
    The FITS file name. None if any.
    
    primary_header: pyfits.CardList
    The primary header
    
    data: string
    name of memory-mapped files used for accessing pixel values

    dq: string
    name of memory-mapped files used for accessing bad pixel status as defined by Euro3D

    stat: string
    name of memory-mapped files used for accessing variance
    
    nx: integer
    Dimension of the data/dq/stat arrays along the x-axis
    
    ny: integer
    Dimension of the data/dq/stat arrays along the y-axis
    """    
    
    def __init__(self, filename=None):
        """creates a CalibFile object
        
        :param filename: The FITS file name. None by default (in this case, an empty object is created).
        
            The FITS file is opened with memory mapping.
            
            Just the primary header and array dimensions are loaded.
            
            Methods get_data, get_dq and get_stat must be used to get array extensions.
        :type filename: string
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
                self.primary_header = hdulist[0].header.ascard
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
        try:
            if self.data != None:
                os.remove(self.data)
            if self.dq != None:
                os.remove(self.dq)
            if self.stat != None:
                os.remove(self.stat)
        except:
            pass
            
    def copy(self):
        """Returns a new copy of a CalibFile object.
        """
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
        os.close(fd)
        #variance
        (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
        selfstat=self.get_stat()
        stat = np.memmap(result.stat,dtype="float32",shape=(self.ny,self.nx))
        stat[:] = selfstat[:]
        os.close(fd)
        # pixel quality
        (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
        selfdq = self.get_dq()
        dq = np.memmap(result.dq,dtype="int32",shape=(self.ny,self.nx))
        dq[:] = selfdq[:]
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
            print "1\tDATA\timage\t(%i,%i)" % (self.nx,self.ny)
            print "2\tDQ\timage\t(%i,%i)" % (self.nx,self.ny)
            print "3\tSTAT\timage\t(%i,%i)" % (self.nx,self.ny)
    
    def get_data(self):
        """Opens the FITS file with memory mapping, loads the data array and returns it.
        """
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
        """Opens the FITS file with memory mapping, loads the dq array and returns it.
        """
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
        """Opens the FITS file with memory mapping, loads the stat array and returns it.
        """
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
        """Saves the object in a FITS file.
        
        :param filename: the FITS filename.
        :type filename: string 
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
        """Adds either a number or a CalibFile object.
        """
        result = CalibFile()             
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        (fd,result.data) = tempfile.mkstemp(prefix='mpdaf')
        os.close(fd)
        (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
        os.close(fd)
        (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
        os.close(fd)
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
        if self.data == None:
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
        """Subtracts either a number or a CalibFile object.
        """
        result = CalibFile()             
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        (fd,result.data) = tempfile.mkstemp(prefix='mpdaf')
        os.close(fd)
        (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
        os.close(fd)
        (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
        os.close(fd)
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
        if self.data == None:
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
        """Multiplies by a number.
        """
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
            os.close(fd)
            (fd,result.dq) = tempfile.mkstemp(prefix='mpdaf')
            os.close(fd)
            (fd,result.stat) = tempfile.mkstemp(prefix='mpdaf')
            os.close(fd)
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
        if self.data == None:
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
        """Divides by a number.
        """
        return self.__mul__(1./other)
    
    
    def __idiv__(self,other):
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
    """CalibDir class manages input/output for a repository containing calibration files (one per ifu).

    :param type: Type of calibration files that appears in filenames (<type>_<ifu id>.fits)
    :type type: string  
    :param dirname: The repository name.
        
            This repository must contain files labeled <type>_<ifu id>.fits
    :type dirname: string

    Attributes
    ----------
    dirname : string
    The repository name. None if any.
    This repository must contain files labeled <type>_<ifu id>.fits
    
    type : string
    Type of calibration files that appears in filenames (<type>_<ifu id>.fits)
    
    files : dict
    List of files (ifu id,CalibFile)
    
    progress: bool
    If True, progress of multiprocessing tasks are displayed. True by default.
    """  
          
    def __init__(self, type, dirname = None):
        """Creates a CalibDir object.
        
        :param type: Type of calibration files that appears in filenames (<type>_<ifu id>.fits)
        :type type: string
        :param dirname: The repository name.
        
            This repository must contain files labeled <type>_<ifu id>.fits
        :type dirname: string
        """
        self.dirname = dirname
        self.progress = True
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
        """Returns a new copy of a CalibDir object.
        """
        result = CalibDir(self.type,self.dirname)
        if self.dirname == None:
            for ifu,fileobj in self.files.items():
                files[ifu] = fileobj.copy()
        return result
                
    def info(self):
        """Prints information.
        """
        print '%i %s files' %(len(self.files),self.type)
        for ifu,fileobj in self.files.items():
            print 'ifu %i' %ifu
            fileobj.info()
            
    def _check(self,other):
        """Checks that other CalibDir contains the same ifu that the current object.
        """
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

    def __add__(self,other):
        """Adds either a number or a CalibFile object.
        """
        return self._mp_operator(other,_add_calib_files,_add_calib)
    
    def __sub__(self,other):
        """Adds either a number or a CalibFile object.
        """
        return self._mp_operator(other,_sub_calib_files,_sub_calib)
    
    def __mul__(self,other):
        """Multiplies by a number.
        """
        if isinstance(other,CalibFile):
            print 'unsupported operand type * and / for CalibFile'
            print
            return None
        else:
             return self._mp_operator(other,None,_mul_calib)
         
    def __div__(self,other): 
        """Divides by a number.
        """
        return self.__mul__(1./other)
        
    def _mp_operator(self,other,funcfile,funcnumber):
        if isinstance(other,CalibDir):
            if self._check(other):
                cpu_count = multiprocessing.cpu_count()
                result = CalibDir(self.type)
                pool = multiprocessing.Pool(processes = cpu_count)
                processlist = list()
                for k in self.files.keys():
                    processlist.append([k,self.files[k].nx,self.files[k].ny,self.files[k].filename,self.files[k].data,self.files[k].dq,self.files[k].stat,other.files[k].filename,other.files[k].data,other.files[k].dq,other.files[k].stat, self.progress])
                processresult = pool.map(funcfile,processlist)
                for k,data,dq,stat in processresult:
                    out = CalibFile()
                    out.primary_header = pyfits.CardList(self.files[k].primary_header)
                    out.nx = self.files[k].nx
                    out.ny = self.files[k].ny
                    #data
                    (fd,out.data) = tempfile.mkstemp(prefix='mpdaf')
                    rdata = np.memmap(out.data,dtype="float32",shape=(out.ny,out.nx))
                    rdata[:] = data[:]
                    os.close(fd)
                    #variance
                    (fd,out.stat) = tempfile.mkstemp(prefix='mpdaf')
                    rstat = np.memmap(out.stat,dtype="float32",shape=(out.ny,out.nx))
                    if stat!= None:
                        rstat[:] = stat[:]
                    else:
                        rstat[:] = self.files[k].get_stat()[:]
                    os.close(fd)
                    # pixel quality
                    (fd,out.dq) = tempfile.mkstemp(prefix='mpdaf')
                    rdq = np.memmap(out.dq,dtype="int32",shape=(out.ny,out.nx))
                    if dq!= None:
                        rdq[:] = dq[:]  
                    else:
                        rdq[:] = self.files[k].get_dq()[:]
                    result.files[k] = out
                    os.close(fd)
            else:
                return None
        else:
            cpu_count = multiprocessing.cpu_count()
            result = CalibDir(self.type)
            pool = multiprocessing.Pool(processes = cpu_count)
            processlist = list()
            for k in self.files.keys():
                processlist.append([k,self.files[k].nx,self.files[k].ny,self.files[k].filename,self.files[k].data,self.files[k].dq,self.files[k].stat,other,self.progress])
            processresult = pool.map(funcnumber,processlist)
            for k,data,dq,stat in processresult:
                out = CalibFile()
                out.primary_header = pyfits.CardList(self.files[k].primary_header)
                out.nx = self.files[k].nx
                out.ny = self.files[k].ny
                #data
                (fd,out.data) = tempfile.mkstemp(prefix='mpdaf')
                rdata = np.memmap(out.data,dtype="float32",shape=(out.ny,out.nx))
                rdata[:] = data[:]
                os.close(fd)
                #variance
                (fd,out.stat) = tempfile.mkstemp(prefix='mpdaf')
                rstat = np.memmap(out.stat,dtype="float32",shape=(out.ny,out.nx))
                os.close(fd)
                if stat!= None:
                    rstat[:] = stat[:]
                else:
                    rstat[:] = self.files[k].get_stat()[:]
                # pixel quality
                (fd,out.dq) = tempfile.mkstemp(prefix='mpdaf')
                rdq = np.memmap(out.dq,dtype="int32",shape=(out.ny,out.nx))
                if dq!= None:
                    rdq[:] = dq[:]  
                else:
                    rdq[:] = self.files[k].get_dq()[:]
                result.files[k] = out
                os.close(fd)
        if self.progress:
            sys.stdout.write('\r                        \n')
        return result
    
    def write(self,dirname):
        """Writes files in self.dirname.
        """
        if self.dirname == None:
            for ifu,fileobj in self.files.items():
                filename = "%s/%s_%02d.fits" %(dirname,self.type,ifu)
                fileobj.write(filename)
        self.dirname = dirname
        
    def __len__(self):
        """Returns the number of files.
        """
        return len(self.files)

    def __getitem__(self, key):
        """Returns the CalibFile object.

        :param key: Ifu id.
        :type key: integer
        """
        extname = "CHAN%02d" %key
        if key in self.files:
            return self.files[key]
        else:
            print 'invalid key'
            print
            return None

    def __setitem__(self,key,value):
        """Sets the corresponding CalibFile with value.

        :param key: Ifu id.
        :type key: integer
        :param value: CalibFile object.
        :type value: CalibFile
        """
        if isinstance(value,CalibFile):
            self.files[key] = value
        else:
            print 'format error: %s incompatible with CalibFile' %type(value)
            print
            return None
        

def _add_calib_files(arglist):
    """adds CalibFile extensions"""
    k = arglist[0]
    nx = arglist[1]
    ny = arglist[2]
    filename1 = arglist[3]
    filedata1 = arglist[4]
    filedq1 = arglist[5]
    filestat1 = arglist[6]
    filename2 = arglist[7]
    filedata2 = arglist[8]
    filedq2 = arglist[9]
    filestat2 = arglist[10]
    progress = arglist[11]
    if filename1 == None:
        if filedata1 == None or filestat1 == None or filedq1 == None:
            print 'format error: empty extension'
            print
            return None
        else:
            data1 = np.memmap(filedata1,dtype="float32",shape=(ny,nx))
            stat1 = np.memmap(filestat1,dtype="float32",shape=(ny,nx))
            dq1 = np.memmap(filedq1,dtype="int32",shape=(ny,nx))
    else:
        hdulist = pyfits.open(filename1,memmap=1)
        data1 = hdulist["DATA"].data
        stat1 = hdulist["STAT"].data
        dq1 = hdulist["DQ"].data
        hdulist.close()
    if filename2 == None:
        if filedata2 == None or filestat2 == None or filedq2 == None:
            print 'format error: empty extension'
            print
            return None
        else:
            data2 = np.memmap(filedata2,dtype="float32",shape=(ny,nx))
            stat2 = np.memmap(filestat2,dtype="float32",shape=(ny,nx))
            dq2 = np.memmap(filedq2,dtype="int32",shape=(ny,nx))
    else:
        hdulist = pyfits.open(filename1,memmap=1)
        data2 = hdulist["DATA"].data
        stat2 = hdulist["STAT"].data
        dq2 = hdulist["DQ"].data
        hdulist.close()
    #sum data values
    newdata = np.ndarray.__add__(data1,data2) 
    # sum variance
    newstat = np.ndarray.__add__(stat1,stat2)
    # pixel quality
    newdq = np.logical_or(dq1,dq2)
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return(k,newdata,newdq,newstat)

def _add_calib(arglist):
    """adds CalibFile extensions"""
    k = arglist[0]
    nx = arglist[1]
    ny = arglist[2]
    filename1 = arglist[3]
    filedata1 = arglist[4]
    other = arglist[7]
    progress = arglist[8]
    if filename1 == None:
        if filedata1 == None:
            print 'format error: empty extension'
            print
            return None
        else:
            data1 = np.memmap(filedata1,dtype="float32",shape=(ny,nx))
    else:
        hdulist = pyfits.open(filename1,memmap=1)
        data1 = hdulist["DATA"].data
        hdulist.close()
    #sum data values
    newdata = np.ndarray.__add__(data1,other) 
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return(k,newdata,None,None)

def _sub_calib_files(arglist):
    """subtracts CalibFile extensions"""
    k = arglist[0]
    nx = arglist[1]
    ny = arglist[2]
    filename1 = arglist[3]
    filedata1 = arglist[4]
    filedq1 = arglist[5]
    filestat1 = arglist[6]
    filename2 = arglist[7]
    filedata2 = arglist[8]
    filedq2 = arglist[9]
    filestat2 = arglist[10]
    progress = arglist[11]
    if filename1 == None:
        if filedata1 == None or filestat1 == None or filedq1 == None:
            print 'format error: empty extension'
            print
            return None
        else:
            data1 = np.memmap(filedata1,dtype="float32",shape=(ny,nx))
            stat1 = np.memmap(filestat1,dtype="float32",shape=(ny,nx))
            dq1 = np.memmap(filedq1,dtype="int32",shape=(ny,nx))
    else:
        hdulist = pyfits.open(filename1,memmap=1)
        data1 = hdulist["DATA"].data
        stat1 = hdulist["STAT"].data
        dq1 = hdulist["DQ"].data
        hdulist.close()
    if filename2 == None:
        if filedata2 == None or filestat2 == None or filedq2 == None:
            print 'format error: empty extension'
            print
            return None
        else:
            data2 = np.memmap(filedata2,dtype="float32",shape=(ny,nx))
            stat2 = np.memmap(filestat2,dtype="float32",shape=(ny,nx))
            dq2 = np.memmap(filedq2,dtype="int32",shape=(ny,nx))
    else:
        hdulist = pyfits.open(filename1,memmap=1)
        data2 = hdulist["DATA"].data
        stat2 = hdulist["STAT"].data
        dq2 = hdulist["DQ"].data
        hdulist.close()
    #sum data values
    newdata = np.ndarray.__sub__(data1,data2) 
    # sum variance
    newstat = np.ndarray.__add__(stat1,stat2)
    # pixel quality
    newdq = np.logical_or(dq1,dq2)
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return(k,newdata,newdq,newstat)

def _sub_calib(arglist):
    """subtracts CalibFile extensions"""
    k = arglist[0]
    nx = arglist[1]
    ny = arglist[2]
    filename1 = arglist[3]
    filedata1 = arglist[4]
    other = arglist[7]
    progress = arglist[8]
    if filename1 == None:
        if filedata1 == None:
            print 'format error: empty extension'
            print
            return None
        else:
            data1 = np.memmap(filedata1,dtype="float32",shape=(ny,nx))
    else:
        hdulist = pyfits.open(filename1,memmap=1)
        data1 = hdulist["DATA"].data
        hdulist.close()
    #sum data values
    newdata = np.ndarray.__sub__(data1,other) 
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return(k,newdata,None,None)

def _mul_calib(arglist):
    """subtracts CalibFile extensions"""
    k = arglist[0]
    nx = arglist[1]
    ny = arglist[2]
    filename1 = arglist[3]
    filedata1 = arglist[4]
    filestat1 = arglist[6]
    other = arglist[7]
    progress = arglist[8]
    if filename1 == None:
        if filedata1 == None or filestat1 == None:
            print 'format error: empty extension'
            print
            return None
        else:
            data1 = np.memmap(filedata1,dtype="float32",shape=(ny,nx))
            stat1 = np.memmap(filestat1,dtype="float32",shape=(ny,nx))
    else:
        hdulist = pyfits.open(filename1,memmap=1)
        data1 = hdulist["DATA"].data
        stat1 = hdulist["STAT"].data
        hdulist.close()
    #sum data values
    newdata = np.ndarray.__mul__(data1,other) 
    newstat = np.ndarray.__mul__(stat1,other*other)
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return(k,newdata,None,newstat)
