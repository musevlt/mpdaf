""" rawobj.py Manages raw FITS file"""
import numpy as np
import pyfits
import multiprocessing
import datetime
import sys
#import matplotlib as mpl
import matplotlib.pyplot as plt
from mpdaf import obj
import warnings

class Channel(object):
    
    """Channel object corresponds to an extension of a raw FITS file.
    
    :param extname: The extension name.
    :type extname: string
    :param filename: The raw FITS file name.
    :type filename: string

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
    """

    def __init__(self, extname=None,filename=None, data=None ):
        """Creates a Channel object.
        
        :param extname: The extension name.
        :type extname: string
        :param filename: The raw FITS file name.
        :type filename: string
        """
        self.extname = extname
        if filename!=None:
            hdulist = pyfits.open(filename,memmap=1)
            self.header = hdulist[extname].header.ascard
            self.nx = hdulist[extname].header["NAXIS1"]
            self.ny = hdulist[extname].header["NAXIS2"]
            try:
                data = hdulist[extname].data
                self.data = np.ndarray(np.shape(data))
                self.data [:] = data[:]
            except:
                self.data = None
            hdulist.close()
        elif data is not None:
            self.header = pyfits.CardList()
            shape = np.shape(data)
            self.data = np.ndarray(shape)
            self.data [:] = data[:]
            self.nx = shape[1]
            self.ny = shape[0]
        else:
            self.header = pyfits.CardList()
            self.nx = 0
            self.ny = 0
            self.data = None
        self.mask = self._init_mask()


    def _init_mask(self):
        """Creates mask that invalidates over scanned pixels.
        """
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
        """Returns a copy of the Channel object.
        """
        result = Channel(self.extname)
        result.header = pyfits.CardList(self.header)
        try:
            result.data = self.data.__copy__()
        except:
            result.data = None
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
        """Multiplies either a number or a Channel object.
        """
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__mul__(self,other)
        else:
            return np.ndarray.__mul__(self,other)

    @_idecorator
    def __imul__(self,other):
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__mul__(self,other)
        else:
            return np.ndarray.__mul__(self,other)


    @_decorator
    def __div__(self,other):
        """Divides either a number or a Channel object.
        """
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__div__(self,other)
        else:
            return np.ndarray.__div__(self,other)

    @_idecorator
    def __idiv__(self,other):
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__div__(self,other)
        else:
            return np.ndarray.__div__(self,other)


    @_decorator
    def __sub__(self,other):
        """Subtracts either a number or a Channel object.
        """
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__sub__(self,other)
        else:
            return np.ndarray.__sub__(self,other)


    @_idecorator
    def __isub__(self,other):
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__sub__(self,other)
        else:
            return np.ndarray.__sub__(self,other)


    @_decorator
    def __add__(self,other):
        """Adds either a number or a Channel object.
        """
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__add__(self,other)
        else:
            return np.ndarray.__add__(self,other)


    @_idecorator
    def __iadd__(self,other):
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__add__(self,other)
        else:
            return np.ndarray.__add__(self,other)


    @_decorator
    def __pow__(self,other):
        """Computes the power exponent.
        """
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__pow__(self,other)
        else:
            return np.ndarray.__pow__(self,other)

    @_idecorator
    def __ipow__(self,other):
        if isinstance(self,np.ma.core.MaskedArray):
            return np.ma.MaskedArray.__pow__(self,other)
        else:
            return np.ndarray.__pow__(self,other)


    def sqrt(self):
        """Computes the positive square-root.
        """
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
        """Returns a Channel object containing only reference to the valid pixels.
        
        :rtype: :class:`mpdaf.drs.Channel`
        """
        result = Channel(self.extname)
        result.header = self.header
        result.nx = self.nx
        result.ny = self.ny
        result.mask = self.mask
        result.data = np.ma.MaskedArray(self.data, mask=self.mask, copy=True)
        return result

    def overscan(self):
        """Returns a Channel object containing only reference to the overscanned pixels.
        
        :rtype: :class:`mpdaf.drs.Channel`
        """
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
    
    
    def get_image(self, det_out = None):
        """Returns an Image object.
        
        :param det_out: number of output detector. If None, all image is returned.
        :type det_out: integer in [1,4]
        
        :rtype: :class:`mpdaf.obj.Image`
        """
        wcs = obj.WCS(pyfits.Header(self.header))
        ima = obj.Image(wcs=wcs, data=self.data.__copy__())
        
        if det_out is not None:
            nx_data = self.header["NAXIS1"].value # length of data in X
            ny_data = self.header["NAXIS2"].value # length of data in Y
            nx_data2 = self.header["ESO DET CHIP NX"].value # Physical active pixels in X
            ny_data2 = self.header["ESO DET CHIP NY"].value # Physical active pixels in Y
            key = "ESO DET OUT%i" % det_out
            nx = self.header["%s NX" % key].value # Output data pixels in X
            ny = self.header["%s NY" % key].value # Output data pixels in Y
            prscx = self.header["%s PRSCX" % key].value # Output prescan pixels in X
            prscy = self.header["%s PRSCY" % key].value # Output prescan pixels in Y
            x = self.header["%s X" % key].value # X location of output
            y = self.header["%s Y"% key].value # Y location of output
            if x < nx_data2/2:
                i1 = x - 1
                i2 = i1 + nx  + 2*prscx
            else:
                i2 = nx_data
                i1 = i2 - nx - 2*prscx
            if y < ny_data2/2:
                j1 = y -1
                j2 = j1 + ny + 2*prscy
            else:
                j2 = ny_data
                j1 = j2 - ny - 2*prscy
            ima = ima[j1:j2,i1:i2]         
        
        return ima

    def get_trimmed_image(self, det_out = None, bias_substract = False, bias = False):
        """Returns an Image object without over scanned pixels.
        
        :param det_out: number of output detector. If None, all image is returned.
        :type det_out: integer in [1,4]
        
        :param bias_substract: If True, median value of the overscanned pixels is substracted
        :type bias_substract: boolean
        
        :param bias: If True, median value of the overscanned pixels is subtracted
        :type bias: boolean
        
        :rtype: :class:`mpdaf.obj.Image`
        """
        nx_data2 = self.header["ESO DET CHIP NX"].value # Physical active pixels in X
        ny_data2= self.header["ESO DET CHIP NY"].value # Physical active pixels in Y
        if isinstance(self.data,np.ma.core.MaskedArray):
            work = np.ma.MaskedArray(self.data.data.__copy__(), mask=self.mask)
        else:
            work = np.ma.MaskedArray(self.data.__copy__(), mask=self.mask)
        
        if bias_substract:
            warnings.warn("get_trimmed_image: bias_substract parameter will be replaced by bias parameter", DeprecationWarning)
            bias = True
            
        if bias:
            
            ksel = np.where(self.mask==True)
            nx_data = self.header["NAXIS1"].value # length of data in X
            ny_data = self.header["NAXIS2"].value # length of data in Y
            
            if det_out is None:
                for det in range(1,5):
                    key = "ESO DET OUT%i" % det
                    nx = self.header["%s NX" % key].value # Output data pixels in X
                    ny = self.header["%s NY" % key].value # Output data pixels in Y
                    prscx = self.header["%s PRSCX" % key].value # Output prescan pixels in X
                    prscy = self.header["%s PRSCY" % key].value # Output prescan pixels in Y
                    x = self.header["%s X" % key].value # X location of output
                    y = self.header["%s Y"% key].value # Y location of output
                    if x < nx_data2/2:
                        i1 = x - 1
                        i2 = i1 + nx + 2*prscx
                    else:
                        i2 = nx_data
                        i1 = i2 - nx - 2*prscx
                    if y < ny_data2/2:
                        j1 = y -1
                        j2 = j1 + ny + 2*prscy
                    else:
                        j2 = ny_data
                        j1 = j2 - ny - 2*prscy
                        
                    ksel = np.where(self.mask[j1:j2,i1:i2]==True)
                    bias_level = np.median((work.data[j1:j2,i1:i2])[ksel]) 
                    work[j1:j2,i1:i2] -= bias_level
            else:
                key = "ESO DET OUT%i" % det_out
                nx = self.header["%s NX" % key].value # Output data pixels in X
                ny = self.header["%s NY" % key].value # Output data pixels in Y
                prscx = self.header["%s PRSCX" % key].value # Output prescan pixels in X
                prscy = self.header["%s PRSCY" % key].value # Output prescan pixels in Y
                x = self.header["%s X" % key].value # X location of output
                y = self.header["%s Y"% key].value # Y location of output
                if x < nx_data2/2:
                    i1 = x - 1
                    i2 = i1 + nx + 2*prscx
                else:
                    i2 = nx_data
                    i1 = i2 - nx - 2*prscx
                if y < ny_data2/2:
                    j1 = y -1
                    j2 = j1 + ny + 2*prscy
                else:
                    j2 = ny_data
                    j1 = j2 - ny - 2*prscy
                        
                ksel = np.where(self.mask[j1:j2,i1:i2]==True)
                bias_level = np.median(work.data[j1:j2,i1:i2][ksel]) 
                work [j1:j2,i1:i2] -= bias_level
                
        data = np.ma.compressed(work)
        data = np.reshape(data,(ny_data2,nx_data2))
        wcs = obj.WCS(crpix=(1.0,1.0), shape=(ny_data2,nx_data2))
        ima = obj.Image(wcs=wcs, data=data)
        
        if det_out is not None:
            nx_data = self.header["NAXIS1"].value # length of data in X
            ny_data = self.header["NAXIS2"].value # length of data in Y
            key = "ESO DET OUT%i" % det_out
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
                j2 = ny_data - prscy
                j1 = j2 - ny
            ima = ima[j1:j2,i1:i2]  
        
        return ima
    
    def get_image_mask_overscan(self, det_out = None):
        """Returns an Image object in which overscanned pixels are masked.
        
        :param det_out: number of output detector. If None, all image is returned.
        :type det_out: integer in [1,4]
        
        :rtype: :class:`mpdaf.obj.Image`
        """
        wcs = obj.WCS(pyfits.Header(self.header))
        ima = obj.Image(wcs=wcs, data=self.data)
        ima.data = np.ma.MaskedArray(self.data.__copy__(), mask=self.mask, copy=True)
        
        if det_out is not None:
            nx_data = self.header["NAXIS1"].value # length of data in X
            ny_data = self.header["NAXIS2"].value # length of data in Y
            nx_data2 = self.header["ESO DET CHIP NX"].value # Physical active pixels in X
            ny_data2 = self.header["ESO DET CHIP NY"].value # Physical active pixels in Y
            key = "ESO DET OUT%i" % det_out
            nx = self.header["%s NX" % key].value # Output data pixels in X
            ny = self.header["%s NY" % key].value # Output data pixels in Y
            prscx = self.header["%s PRSCX" % key].value # Output prescan pixels in X
            prscy = self.header["%s PRSCY" % key].value # Output prescan pixels in Y
            x = self.header["%s X" % key].value # X location of output
            y = self.header["%s Y"% key].value # Y location of output
            if x < nx_data2/2:
                i1 = x - 1
                i2 = i1 + nx + 2*prscx
            else:
                i2 = nx_data
                i1 = i2 - nx - 2*prscx
            if y < ny_data2/2:
                j1 = y -1
                j2 = j1 + ny + 2*prscy
            else:
                j2 = ny_data
                j1 = j2 - ny - 2*prscy
            ima = ima[j1:j2,i1:i2] 
        
        return ima
    
    def get_image_just_overscan(self, det_out = None):
        """Returns an Image object in which only overscanned pixels are not masked.
        
        :param det_out: number of output detector. If None, all image is returned.
        :type det_out: integer in [1,4]
        
        :rtype: :class:`mpdaf.obj.Image`
        """
        wcs = obj.WCS(pyfits.Header(self.header))
        ima = obj.Image(wcs=wcs, data=self.data)
        ima.data = np.ma.MaskedArray(self.data.__copy__(), mask=np.logical_not(self.mask), copy=True)
        
        if det_out is not None:
            nx_data = self.header["NAXIS1"].value # length of data in X
            ny_data = self.header["NAXIS2"].value # length of data in Y
            nx_data2 = self.header["ESO DET CHIP NX"].value # Physical active pixels in X
            ny_data2 = self.header["ESO DET CHIP NY"].value # Physical active pixels in Y
            key = "ESO DET OUT%i" % det_out
            nx = self.header["%s NX" % key].value # Output data pixels in X
            ny = self.header["%s NY" % key].value # Output data pixels in Y
            prscx = self.header["%s PRSCX" % key].value # Output prescan pixels in X
            prscy = self.header["%s PRSCY" % key].value # Output prescan pixels in Y
            x = self.header["%s X" % key].value # X location of output
            y = self.header["%s Y"% key].value # Y location of output
            if x < nx_data2/2:
                i1 = x - 1
                i2 = i1 + nx + 2*prscx
            else:
                i2 = nx_data
                i1 = i2 - nx - 2*prscx
            if y < ny_data2/2:
                j1 = y -1
                j2 = j1 + ny + 2*prscy
            else:
                j2 = ny_data
                j1 = j2 - ny - 2*prscy
            ima = ima[j1:j2,i1:i2] 
        
        return ima
    


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


def Channel_median(channels):
    result = Channel(channels[0].extname)
    result.header = channels[0].header
    result.nx = channels[0].nx
    result.ny = channels[0].ny
    result.mask = channels[0].mask
    result.data = np.empty_like(channels[0].data)
    arrays = []
    for chan in channels:
        arrays.append(chan.data)
        result.mask += chan.mask
    arrays = np.array(arrays, dtype=np.int16)
    result.data = np.median(arrays,axis=0)
    if isinstance(result.data,np.ma.core.MaskedArray):
        result.data = result.data.data
    return result

class RawFile(object):
    """
    RawFile class manages input/output for raw FITS file.
    
    :param filename: The raw FITS file name. filename=None creates an empty object.
        
            The FITS file is opened with memory mapping.
            
            Just the primary header and the list of extension name are loaded.
            
            Method get_channel(extname) returns the corresponding channel
            
            Operator [extnumber] loads and returns the corresponding channel.
    :type filename: string

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

    next: integer
    Number of extensions
    
    progress: boolean
    If True, progress of multiprocessing tasks are displayed. True by default.
    """
    
    def __init__(self, filename=None):
        """Creates a RawFile object.
        
        :param filename: The raw FITS file name. filename=None creates an empty object.
        
            The FITS file is opened with memory mapping.
            
            Just the primary header and the list of extension name are loaded.
            
            Method get_channel(extname) returns the corresponding channel
            
            Operator [extnumber] loads and returns the corresponding channel.
        :type filename: string
        """
        self.filename = filename
        self.progress = True
        self.channels = dict()
        self.nx = 0
        self.ny = 0
        self.next = 0
        if filename!=None:
            try:
                hdulist = pyfits.open(self.filename,memmap=1)
                self.primary_header = hdulist[0].header.ascard
                n = 1
                while True:
                    try:
                        extname = hdulist[n].header["EXTNAME"]
                        exttype = hdulist[n].header["XTENSION"]
                        if exttype=='IMAGE' and hdulist[n].header["NAXIS"]!=0:
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
        """Returns a copy of the RawFile object.
        """
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
        """Prints information.
        """
        if self.filename != None:
            print self.filename
        else:
            print 'NoName'
        print 'Nb extensions:\t%i (loaded:%i %s)'% (self.next,len(self.channels),self.channels.keys())
        print 'format:\t(%i,%i)'% (self.nx,self.ny)
        
    def get_keywords(self,key):
        """Returns the keyword value.
        """
        return self.primary_header[key].value
        
    def get_channels_extname_list(self):
        """Returns the list of existing channels names.
        """
        return self.channels.keys()

    def get_channel(self,extname):
        """Returns a Channel object.
        
        :param extname: The extension name.
        :type extname: string
        :rtype: :class:`mpdaf.drs.Channel`
        """
        if self.channels[extname] != None:
            return self.channels[extname]
        else:
            chan = Channel(extname,self.filename)
            return chan

    def __len__(self):
        """Returns the number of extensions.
        """
        return self.next

    def __getitem__(self, key):
        """Loads the Channel object if relevant and returns it.

        :param key: The extension number.
        :type key: integer
        :rtype: :class:`mpdaf.drs.Channel`
        """
        extname = "CHAN%02d" %key
        if self.channels[extname] == None:
            self.channels[extname] = Channel(extname,self.filename)
        return self.channels[extname]

    def __setitem__(self,key,value):
        """Sets the corresponding channel.

        :param key: The extension number.
        :type key: integer
        :param value: Channel object or image
        :type value: `mpdaf.drs.Channel` or array
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
                
                if chan.header is not None:
                    chan.header.update(card.key, card.value, card.comment)
            else:
                print 'format error: set an image extension with bad dimensions'
                print
                return None
        else:
            print 'format error: %s incompatible with an image extension' %type(value)
            print
            return None


    def __mul__(self,other):
        """Multiplies either a number or a RawFits object.
        """
        return self._mp_operator(other,'Channel.__mul__')

    def __imul__(self,other):
        return self._mp_operator(other,'Channel.__imul__')

    def __div__(self,other):
        """Divides either a number or a RawFits object.
        """
        return self._mp_operator(other,'Channel.__div__')

    def __idiv__(self,other):
        return self._mp_operator(other,'Channel.__idiv__')

    def __sub__(self,other):
        """Subtracts either a number or a RawFits object.
        """
        return self._mp_operator(other,'Channel.__sub__')

    def __isub__(self,other):
        return self._mp_operator(other,'Channel.__isub__')

    def __add__(self,other):
        """Adds either a number or a RawFits object.
        """
        return self._mp_operator(other,'Channel.__add__')

    def __iadd__(self,other):
        return self._mp_operator(other,'Channel.__iadd__')

    def __pow__(self,other):
        """Computes the power exponent of each channel.
        """
        return self._mp_operator(other,'Channel.__pow__')

    def __ipow__(self,other):
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
                processlist.append([funcname,k,self,other,self.progress])
            if isinstance(other,RawFile):
                processresult = pool.map(_process_operator,processlist)
            else:
                processresult = pool.map(_process_operator2,processlist)
            for k,out in processresult:
                result.channels[k] = out
            if self.progress:
                sys.stdout.write('\r                        \n')
        return result

    def sqrt(self):
        """Compute the square root of each channel.
        """
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
                processlist.append(['Channel.sqrt',k,self,self.progress])
            processresult = pool.map(_process_operator3,processlist)
            for k,out in processresult:
                result.channels[k] = out
            if self.progress:
                sys.stdout.write('\r                        \n')
        return result

    def trimmed(self):
        """Returns a RawFile object containing only reference to the valid pixels.
        
        :rtype: :class:`mpdaf.drs.RawFile`
        """
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
                processlist.append(['Channel.trimmed',k,self,self.progress])
            processresult = pool.map(_process_operator3,processlist)
            for k,out in processresult:
                result.channels[k] = out
            if self.progress:
                sys.stdout.write('\r                        \n')
        return result

    def overscan(self):
        """Returns a RawFile object containing only reference to the overscanned pixels.
        
        :rtype: :class:`mpdaf.drs.RawFile`
        """
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
                processlist.append(['Channel.overscan',k,self,self.progress])
            processresult = pool.map(_process_operator3,processlist)
            for k,out in processresult:
                result.channels[k] = out
            if self.progress:
                sys.stdout.write('\r                        \n')
        return result

    def write(self,filename):
        """Saves the object in a FITS file.
        
        :param filename: The FITS filename.
        :type filename: string
        """
        # create primary header
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
                                if card.key != "EXTNAME":
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
        hdu.writeto(filename, clobber=True, output_verify='fix')
        # update attributes
        self.filename = filename
        for name,chan in self.channels.items():
            del chan
            self.channels[name] = None
            
    def plot(self, title=None, channels="all", area = None, scale='linear', vmin=None, vmax=None, zscale=False, colorbar=None, **kargs): 
        """Plots the image.
          :param title: Figure title (None by default).
          :type title: string
          :param channels: list of channel names. All by default.
          :type channels: list or 'all'
          :param area: list of pixels [pmin,pmax,qmin,qmax] to zoom.
          :type title: list
          :param scale: The stretch function to use for the scaling (default is 'linear').
          :type scale: linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'
          :param vmin: Minimum pixel value to use for the scaling.
  
           If None, vmin is set to min of data.
          :type vmin: float
          :param vmax: Maximum pixel value to use for the scaling.
        
           If None, vmax is set to max of data.
          :type vmax: float
          :param zscale: If true, vmin and vmax are computed using the IRAF zscale algorithm.
          :type zscale: bool
          :param colorbar: If 'h'/'v', a horizontal/vertical colorbar is added.
          :type colorbar: bool
          :param kargs: kargs can be used to set additional Artist properties.
          :type kargs: matplotlib.artist.Artist
        """
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        if title is not None:
            plt.title(title)
        if channels == "all":
            for name in self.channels.keys():
                chan = self.get_channel(name)
                ima = chan.get_trimmed_image(det_out = None, bias_substract = False, bias = False)
                if area is not None:
                    ima = ima[area[0]:area[1],area[2]:area[3]] 
                ichan = int(name[-2:])
                fig.add_subplot(4,6,ichan)
                ima.plot(None, scale, vmin, vmax, zscale, colorbar, **kargs)
                plt.axis('off')
                plt.text(ima.shape[0]-500, ima.shape[1]-500,'%i'%ichan,style='italic',bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
        else:
            nchan = len(channels)
            nrows = int(np.sqrt(nchan))
            if nchan%nrows == 0:
                ncols = nchan/nrows
            else:
                ncols = int(nchan/nrows) + 1
            for i, name in enumerate(channels):
                chan = self.get_channel(name)
                ima = chan.get_trimmed_image(det_out = None, bias_substract = False, bias = False)
                if area is not None:
                    ima = ima[area[0]:area[1],area[2]:area[3]] 
                ichan = int(name[-2:])
                fig.add_subplot(nrows,ncols,i+1)
                ima.plot(None, scale, vmin, vmax, zscale, colorbar, **kargs)
                plt.axis('off')
                plt.text(ima.shape[0]-500, ima.shape[1]-500,'%i'%ichan,style='italic',bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})

def _process_operator(arglist):
    #decorator used to define arithmetic functions with a RawFits object
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    obj = arglist[2]
    other = arglist[3]
    progress = arglist[4]
    v = obj.get_channel(k)
    try:
        v2 = other.get_channel(k)
    except:
        print 'Error: operations on raw files with different extensions'
        print
        return
    out = function(v,v2)
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return (k,out)

def _process_operator2(arglist):
    #decorator used to define arithmetic functions with a number
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    obj = arglist[2]
    other = arglist[3]
    progress = arglist[4]
    v = obj.get_channel(k)
    out = function(v,other)
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return (k,out)

def _process_operator3(arglist):
    #decorator used to define sqrt/trimmed
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    obj = arglist[2]
    progress = arglist[3]
    v = obj.get_channel(k)
    out = function(v)
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    return (k,out)


def _process_median(arglist):
    k = arglist[0]
    list_chan = arglist[1]
    out = Channel_median(list_chan)
    return (k,out)
    
    
def RawFile_median(RawList):
    cpu_count = multiprocessing.cpu_count()
    result = RawFile()
    result.primary_header = RawList[0].primary_header
    result.nx = RawList[0].nx
    result.ny = RawList[0].ny
    result.next = RawList[0].next
    pool = multiprocessing.Pool(processes = cpu_count)
    processlist = list()
    if RawList[0].channels is not None:
        for k in RawList[0].channels.keys():
            ChanList = []
            for raw in RawList:
                ChanList.append(raw.get_channel(k))
            #result.channels[k] = Channel_median(ChanList)
            processlist.append([k,ChanList])
        processresult = pool.map(_process_median,processlist)
        for k,out in processresult:
            result.channels[k] = out
    return result
