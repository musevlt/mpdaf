""" rawobj.py Manages raw FITS file"""
import numpy as np
import pyfits
import pylab
import multiprocessing
import datetime
import sys
#import matplotlib as mpl
#import matplotlib.pyplot as plt


class Channel(object):
    
    """Channel class
    
    Channel object corresponds to an extension of a raw FITS file

    Attributes
    ----------
    extname : string
    The extension name

    header: pyfits.CardList
    The extension header

    data: array
    Array containing the pixel values of the image extension
    
    nx,ny : integers
    Lengths of data in X and Y

    mask : boolean mask
    Arrays that contents TRUE for overscanned pixels, FALSE for the others

    Public methods
    --------------
    Creation: init, copy
    
    Arithmetic: + - * / pow sqrt
    
    Info: get_nx, get_ny
    
    DRS: trimmed, overscan, trimmed_image
    """

    def __init__(self, extname,filename=None ):
        """creates a Channel object
        
        Parameters
        ----------
        extname : string
        The extension name
    
        filename : string
        The raw FITS file name. None by default.

        nx,ny : integers
        Lengths of data in X and Y

        mask : boolean mask
        Arrays that contents TRUE for overscanned pixels, FALSE for the others
        """
        self.extname = extname
        if filename!=None:
            hdulist = pyfits.open(filename,memmap=1)
            self.header = hdulist[extname].header.ascardlist()
            self.nx = hdulist[extname].header["NAXIS1"]
            self.ny = hdulist[extname].header["NAXIS2"]
            data = hdulist[extname].data
            self.data = np.ndarray(np.shape(data))
            self.data [:] = data[:]
            hdulist.close()
        else:
            self.header = pyfits.CardList()
            self.nx = 0
            self.ny = 0
            self.data = None
        self.mask = self._init_mask()


    def _init_mask(self):
        """creates mask that invalidates over scanned pixels"""
        m = np.ones((self.ny,self.nx), dtype = int)
        try:
            nx_data = self.header["NAXIS1"].value # length of data in X
            ny_data = self.header["NAXIS2"].value # length of data in Y
            nx_data2 = self.header["ESO DET CHIP NX"].value # Physical active pixels in X
            ny_data2 = self.header["ESO DET CHIP NY"].value # Physical active pixels in Y
            m = np.ones((self.ny,self.nx), dtype = int)

            for i in range(4):
                try:
                    n = i+1
                    key = "ESO DET OUT%i" % n
                    nx = self.header["%s NX" % key].value # Output data pixels in X
                    ny = self.header["%s NY" % key].value # Output data pixels in Y
                    prscx = self.header["%s PRSCX" % key].value # Output prescan pixels in X
                    prscy = self.header["%s PRSCY" % key].value # Output prescan pixels in Y
                    x = self.header["%s X" % key].value # X location of output
                    y = self.header["%s Y"% key].value # Y location of output
                    if x < nx_data2/2:
                        i1 = x - 1 + prscx
                        i2 = i1 + nx
                    else:
                        i2 = nx_data - prscx
                        i1 = i2 - nx
                    if y < ny_data2/2:
                        j1 = y -1 + prscy
                        j2 = j1 + ny
                    else:
                        j2 = ny_data  - prscy
                        j1 = j2 - ny
                    m[j1:j2,i1:i2] *= 0
                except:
                    break
        except:
            pass
        mask = np.ma.make_mask(m)
        return mask


    def copy(self):
        """copies Channel object in a new one and returns it"""
        result = Channel(self.extname,None)
        result.header = pyfits.CardList(self.header)
        if self.data != None:
            result.data = self.data.__copy__()
        result.nx = self.nx
        result.ny = self.ny
        result.mask = self.mask.__copy__()
        return result


    def _decorator(function):
        # decorator used to define arithmetic functions
        def _wrapper(self,other):
            if isinstance(other,Channel):
                if self.extname!=other.extname:
                    print 'Error: operations on channel extensions with different names'
                    print
                    return None
                result = Channel(self.extname)             
                result.header = self.header
                result.nx = self.nx
                result.ny = self.ny
                result.mask = self.mask
                result.data = function(self.data,other.data)
                if isinstance(result.data,np.ma.core.MaskedArray):
                    result.data = result.data.data
                return result
            else:
                result = Channel(self.extname)             
                result.header = self.header
                result.nx = self.nx
                result.ny = self.ny
                result.mask = self.mask
                result.data = function(self.data,other)
                if isinstance(result.data,np.ma.core.MaskedArray):
                    result.data = result.data.data
                return result
        return _wrapper

    def _idecorator(function):
        # decorator used to define in-place arithmetic functions
        def _wrapper(self,other):
            if isinstance(other,Channel):
                if self.extname!=other.extname:
                    print 'Error: operations on channel extensions with different names'
                    print
                    return None
                result = Channel(self.extname)
                result.header = self.header
                result.nx = self.nx
                result.ny = self.ny
                result.mask = self.mask
                result.data = function(self.data,other.data)
                return result
            else:
                result = Channel(self.extname)
                result.header = self.header
                result.nx = self.nx
                result.ny = self.ny
                result.mask = self.mask
                result.data = function(self.data,other)
                return result
        return _wrapper

    @_decorator
    def __mul__(self,other):
        """multiplies either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__mul__(self,other)
        else:
            return np.ndarray.__mul__(self,other)

    @_idecorator
    def __imul__(self,other):
        """multiplies either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__mul__(self,other)
        else:
            return np.ndarray.__mul__(self,other)


    @_decorator
    def __div__(self,other):
        """divides either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__div__(self,other)
        else:
            return np.ndarray.__div__(self,other)

    @_idecorator
    def __idiv__(self,other):
        """divides either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__div__(self,other)
        else:
            return np.ndarray.__div__(self,other)


    @_decorator
    def __sub__(self,other):
        """subtracts either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__sub__(self,other)
        else:
            return np.ndarray.__sub__(self,other)


    @_idecorator
    def __isub__(self,other):
        """subtracts either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__sub__(self,other)
        else:
            return np.ndarray.__sub__(self,other)


    @_decorator
    def __add__(self,other):
        """adds either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__add__(self,other)
        else:
            return np.ndarray.__add__(self,other)


    @_idecorator
    def __iadd__(self,other):
        """adds either a number or a Channel object"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__add__(self,other)
        else:
            return np.ndarray.__add__(self,other)


    @_decorator
    def __pow__(self,other):
        """computes the power exponent"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__pow__(self,other)
        else:
            return np.ndarray.__pow__(self,other)

    @_idecorator
    def __ipow__(self,other):
        """computes the power exponent"""
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__pow__(self,other)
        else:
            return np.ndarray.__pow__(self,other)


    def sqrt(self):
        """computes the power exponent"""
        result = Channel(self.extname)
        result.header = self.header
        result.nx = self.nx
        result.ny = self.ny
        result.mask = self.mask
        result.data = np.sqrt(self.data)
        if isinstance(result.data,np.ma.core.MaskedArray):
            result.data = result.data.data
        return result


    def trimmed(self):
        """returns a Channel object containing only reference to the valid pixels"""
        #x = np.ma.MaskedArray(self.data, mask=self.mask)
        #return x

        result = Channel(self.extname)
        result.header = self.header
        result.nx = self.nx
        result.ny = self.ny
        result.mask = self.mask
        result.data = np.ma.MaskedArray(self.data, mask=self.mask, copy=True)
        return result


    def overscan(self):
        """returns a Channel object containing only reference to the overscanned pixels"""
        #x = np.ma.MaskedArray(self.data, mask=np.logical_not(self.mask))
        #return x
        result = Channel(self.extname)
        result.header = self.header
        result.nx = self.nx
        result.ny = self.ny
        result.mask = self.mask
        result.data = np.ma.MaskedArray(self.data, mask=np.logical_not(self.mask), copy=True)
        #result.data = np.ma.MaskedArray(self.data, mask=np.logical_not(self.mask))
        return result


    def trimmed_image(self):
        """removes over scanned pixels from data"""
        nx_data = self.header["ESO DET CHIP NX"].value # Physical active pixels in X
        ny_data = self.header["ESO DET CHIP NY"].value # Physical active pixels in Y
        if isinstance(self.data,np.ma.core.MaskedArray):
            x = np.ma.MaskedArray(self.data.data, mask=self.mask)
        else:
            x = np.ma.MaskedArray(self.data, mask=self.mask)
        data = np.ma.compressed(x)
        data = np.reshape(data,(ny_data,nx_data))
        return data


STR_FUNCTIONS = { 'Channel.__mul__' : Channel.__mul__,
                  'Channel.__imul__' : Channel.__imul__,
                  'Channel.__div__' : Channel.__div__,
                  'Channel.__idiv__' : Channel.__idiv__,
                  'Channel.__sub__' : Channel.__sub__,
                  'Channel.__isub__' : Channel.__isub__,
                  'Channel.__add__' : Channel.__add__,
                  'Channel.__iadd__' : Channel.__iadd__,
                  'Channel.__pow__' : Channel.__pow__,
                  'Channel.__ipow__' : Channel.__ipow__,
                  'Channel.sqrt' : Channel.sqrt,
                  'Channel.trimmed' : Channel.trimmed,
                  'Channel.overscan' : Channel.overscan,
                  }    


class RawFile(object):
    """RawFile class
    
    This class manages input/output for raw FITS file

    Attributes
    ----------
    filename : string
    The raw FITS file name. None if any.

    channels: dict
    List of extension (extname,Channel)

    primary_header: pyfits.CardList
    The primary header

    nx,ny : integers
    Lengths of data in X and Y

    next: interger
    Number of extensions

    Methods
    -------
    Creation: init,copy
    
    Arithmetic: + - * /
    
    Info: info, len
    
    save: write
    
    get: [], get_channel
    """
    
    def __init__(self, filename=None):
        """creates a RawFile object
        
        Parameters
        ----------   
        filename : string
        The raw FITS file name. None by default.
    
        channels: dict
        List of extension (extname,Channel)

        primary_header: pyfits.CardList
        The primary header

        nx,ny : integers
        Lengths of data in X and Y

        next: interger
        Number of extensions

        Notes
        -----
        filename=None creates an empty object

        The FITS file is opened with memory mapping.
        Just the primary header and the list of extension name are loaded.
        Method get_channel(extname) returns the corresponding channel
        Operator [extnumber] loads and returns the corresponding channel.
        """
        self.filename = filename
        self.channels = dict()
        self.nx = 0
        self.ny = 0
        self.next = 0
        if filename!=None:
            try:
                hdulist = pyfits.open(self.filename,memmap=1)
                self.primary_header = hdulist[0].header.ascardlist()
                n = 1
                while True:
                    try:
                        extname = hdulist[n].header["EXTNAME"]
                        exttype = hdulist[n].header["XTENSION"]
                        if exttype=='IMAGE':
                            nx = hdulist[n].header["NAXIS1"]
                            ny = hdulist[n].header["NAXIS2"]
                            if self.nx == 0:
                                self.nx = nx
                                self.ny = ny
                            if nx!=self.nx and ny!=self.ny:
                                print 'format error: image extensions with different sizes'
                                print
                                return None
                            self.channels[extname] = None
                        n = n+1
                    except:
                        break
                    self.next = n-1
                    hdulist.close()
            except IOError:
                print 'IOError: file %s not found' % `filename`
                print
                self.filename = None
                self.primary_header = None
        else:
            self.filename = None
            self.primary_header = pyfits.CardList()

    def copy(self):
        """copies RawFile object in a new one and returns it"""
        result = RawFile(self.filename)
        if result.filename == None:
            result.primary_header = pyfits.CardList(self.primary_header)
            result.nx = self.nx
            result.ny = self.ny
            result.next = self.next
            for name,chan in self.channels.items():
                if chan != None:
                    result.channels[name] = chan.copy()
                else:
                    result.channels[name] = None
        return result

    def info(self):
        """prints information"""
        if self.filename != None:
            print self.filename
        else:
            print 'NoName'
        print 'Nb extensions:\t%i (loaded:%i)'% (self.next,len(self.channels))
        print 'format:\t(%i,%i)'% (self.nx,self.ny)

    def get_channel(self,extname):
        """returns a Channel object
        
        Parameters
        ----------
        extname : string
        The extension name
        """
        if self.channels[extname] != None:
            return self.channels[extname]
        else:
            chan = Channel(extname,self.filename)
            return chan

    def __len__(self):
        """returns the number of extensions"""
        return self.next

    def __getitem__(self, key):
        """loads the Channel object if relevant
        and returns it

        Parameters
        ----------
        key : integer
        The extension number
        """
        extname = "CHAN%02d" %key
        if self.channels[extname] == None:
            self.channels[extname] = Channel(extname,self.filename)
        return self.channels[extname]

    def __setitem__(self,key,value):
        """sets the corresponding channel with value

        Parameters
        ----------
        key : integer
        The extension number

        value: Channel or array
        Channel object or image
        """
        extname = "CHAN%02d" %key
        if isinstance(value,Channel):
            if value.nx == self.nx and value.ny == self.ny:
                self.channels[extname] = value
            else:
                print 'format error: set an image extension with different sizes'
                print
                return None
        elif isinstance(value,np.ndarray):
            if np.shape(value) == (self.ny,self.nx):
                chan = Channel(extname)
                chan.data = value
                chan.nx = self.nx
                chan.ny = self.ny
                self.channels[extname] = chan
            else:
                print 'format error: set an image extension with bad dimensions'
                print
                return None
        else:
            print 'format error: %s incompatible with an image extension' %type(value)
            print
            return None


    def __mul__(self,other):
        """multiplies either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__mul__')

    def __imul__(self,other):
        """multiplies either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__imul__')

    def __div__(self,other):
        """divides either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__div__')

    def __idiv__(self,other):
        """divides either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__idiv__')

    def __sub__(self,other):
        """subtracts either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__sub__')

    def __isub__(self,other):
        """subtracts either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__isub__')

    def __add__(self,other):
        """adds either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__add__')

    def __iadd__(self,other):
        """adds either a number or a RawFits object"""
        return self._mp_operator(other,'Channel.__iadd__')

    def __pow__(self,other):
        """computes the power exponent"""
        return self._mp_operator(other,'Channel.__pow__')

    def __ipow__(self,other):
        """computes the power exponent"""
        return self._mp_operator(other,'Channel.__ipow__')

    def _mp_operator(self,other,funcname):
        #multiprocessing function
        cpu_count = multiprocessing.cpu_count()
        result = RawFile()
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        result.next = self.next
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
        if self.channels is not None:
            for k in self.channels.keys():
                processlist.append([funcname,k,self,other])
            if isinstance(other,RawFile):
                processresult = pool.map(_process_operator,processlist)
            else:
                processresult = pool.map(_process_operator2,processlist)
            for k,out in processresult:
                result.channels[k] = out
            sys.stdout.write('\r                        \n')
        return result

    def sqrt(self):
        cpu_count = multiprocessing.cpu_count()
        result = RawFile()
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        result.next = self.next
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
        if self.channels is not None:
            for k in self.channels.keys():
                processlist.append(['Channel.sqrt',k,self])
            processresult = pool.map(_process_operator3,processlist)
            for k,out in processresult:
                result.channels[k] = out
            sys.stdout.write('\r                        \n')
        return result

    def trimmed(self):
        cpu_count = multiprocessing.cpu_count()
        result = RawFile()
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        result.next = self.next
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
        if self.channels is not None:
            for k in self.channels.keys():
                processlist.append(['Channel.trimmed',k,self])
            processresult = pool.map(_process_operator3,processlist)
            for k,out in processresult:
                result.channels[k] = out
            sys.stdout.write('\r                        \n')
        return result

    def overscan(self):
        cpu_count = multiprocessing.cpu_count()
        result = RawFile()
        result.primary_header = self.primary_header
        result.nx = self.nx
        result.ny = self.ny
        result.next = self.next
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
        if self.channels is not None:
            for k in self.channels.keys():
                processlist.append(['Channel.overscan',k,self])
            processresult = pool.map(_process_operator3,processlist)
            for k,out in processresult:
                result.channels[k] = out
            sys.stdout.write('\r                        \n')
        return result

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
        if self.channels is not None:
            for name in self.channels.keys():
                chan = self.get_channel(name)
                try:
                    if isinstance(chan.data,np.ma.core.MaskedArray):
                        dhdu = pyfits.ImageHDU(name=name, data=chan.data.data)
                    else:
                        dhdu = pyfits.ImageHDU(name=name, data=chan.data)
                    if chan.header is not None:
                        for card in chan.header:
                            try:
                                dhdu.header.update(card.key, card.value, card.comment)
                            except ValueError:
                                dhdu.header.update('hierarch %s' %card.key, card.value, card.comment)
                            except:
                                pass
                    hdulist.append(dhdu)
                except:
                    pass
        # save to disk 
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True)
        # update attributes
        self.filename = filename
        for name,chan in self.channels.items():
            del chan
            self.channels[name] = None


def _process_operator(arglist):
    #decorator used to define arithmetic functions with a RawFits object
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    obj = arglist[2]
    other = arglist[3]
    v = obj.get_channel(k)
    try:
        v2 = other.get_channel(k)
    except:
        print 'Error: operations on raw files with different extensions'
        print
        return
    out = function(v,v2)
    sys.stdout.write(".")
    sys.stdout.flush()
    return (k,out)

def _process_operator2(arglist):
    #decorator used to define arithmetic functions with a number
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    obj = arglist[2]
    other = arglist[3]
    v = obj.get_channel(k)
    out = function(v,other)
    sys.stdout.write(".")
    sys.stdout.flush()
    return (k,out)

def _process_operator3(arglist):
    #decorator used to define sqrt/trimmed
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    obj = arglist[2]
    v = obj.get_channel(k)
    out = function(v)
    sys.stdout.write(".")
    sys.stdout.flush()
    return (k,out)

