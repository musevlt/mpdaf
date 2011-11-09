""" rawobj.py Manages raw FITS file"""
import numpy as np
import pyfits
import pylab
import multiprocessing
import datetime
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
    
    has_scan: boolean
    Does the channel data contain over scanned pixels ?

    Methods
    -------
    Creation: init,copy
    
    Arithmetic: + - * /
    
    Info: get_nx, get_ny
    
    DRS: trimm_overscan
    """

    def __init__(self, extname,filename=None, has_scan=True ):
        """creates a Channel object
        
        Parameters
        ----------
        extname : string
        The extension name
    
        filename : string
        The raw FITS file name. None by default.
    
        has_scan: boolean
        Does the channel data contain over scanned pixels ? True by default.
        """
        self.extname = extname
        self.has_scan = has_scan
        if filename!=None:
            hdulist = pyfits.open(filename,memmap=1)
            self.header = hdulist[extname].header.ascardlist()
            self.data = hdulist[extname].data
            hdulist.close()
            if has_scan == False:
                self.remove_preoverscan()
        else:
            self.header = pyfits.CardList()
            self.data = None

    def copy(self):
        """copies Channel object in a new one and returns it"""
        result = Channel(self.extname,None,self.has_scan)
        result.header = pyfits.CardList(self.header)
        if self.data != None:
            result.data = self.data.__copy__()
        return result

    def trimm_overscan(self):
        """removes over scanned pixels from data"""
        if self.has_scan == True:
            try:
                nx_data = self.header["NAXIS1"].value # length of data in X
                ny_data = self.header["NAXIS2"].value # length of data in Y
                nx_data2 = self.header["ESO DET CHIP NX"].value # Physical active pixels in X
                ny_data2 = self.header["ESO DET CHIP NY"].value # Physical active pixels in Y
                data2 = np.zeros((ny_data2,nx_data2), dtype=float)
            
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
                            #corresponding slice in X for data2 (active pixels)
                            i1_data2 = x - 1
                            i2_data2 = i1_data2 + nx
                            #corresponding slice in X for data (real pixels)
                            i1_data = x - 1 + prscx  
                            i2_data = i1_data + nx
                        else:
                            #corresponding slice in X for data2 (active pixels)
                            i2_data2 = x
                            i1_data2 = i2_data2 - nx
                            #corresponding slice in X for data (real pixels)
                            i2_data = nx_data - prscx
                            i1_data = i2_data - nx
                        if y < ny_data2/2:
                            #corresponding slice in Y for data2 (active pixels)
                            j1_data2 = y -1
                            j2_data2 = j1_data2 + ny
                            #corresponding slice in Y for data (real pixels)
                            j1_data = y -1 + prscy
                            j2_data = j1_data + ny
                        else:
                            #corresponding slice in Y for data2 (active pixels)
                            j2_data2 = y
                            j1_data2 = j2_data2 - ny
                            #corresponding slice in Y for data (real pixels)
                            j2_data = ny_data  - prscy
                            j1_data = j2_data - ny
                        #copy
                        data2[j1_data2:j2_data2,i1_data2:i2_data2] = self.data[j1_data:j2_data,i1_data:i2_data]
                    except:
                        break
            except:
                data2 = None
            self.data = data2
            self.has_scan = False
            #return data2

    def get_nx(self):
        """returns NAXIS1 value"""
        try:
            nx_data = self.header["NAXIS1"].value # length of data in X
        except:
            nx_data = 0
        return nx_data

    def get_ny(self):
        """returns NAXIS2 value"""
        try:
            ny_data = self.header["NAXIS2"].value # length of data in Y
        except:
            ny_data = 0
        return ny_data

    def decorator(function):
        # decorator used to define arithmetic functions
        def _wrapper(self,other):
            if isinstance(other,Channel):
                if self.extname!=other.extname:
                    print 'Error: operations on channel extensions with different names'
                    print
                    return None
                if self.has_scan!=other.has_scan:
                    print 'Error: operations on channel with and without pre and over scans'
                    print
                    return None
                print self.extname
                result = Channel(self.extname)             
                result.header = self.header
                result.data = function(self.data,other.data)
                return result
            else:
                result = Channel(self.extname)             
                result.header = self.header
                result.data = function(self.data,other)
                return result
        return _wrapper


    @decorator
    def __mul__(self,other):
        """multiplies either a number or a Channel object"""
        return np.ndarray.__mul__(self,other)

    @decorator
    def __div__(self,other):
        """divides either a number or a Channel object"""
        return np.ndarray.__div__(self,other)

    @decorator
    def __sub__(self,other):
        """subtracts either a number or a Channel object"""
        return np.ndarray.__sub__(self,other)

    @decorator
    def __add__(self,other):
        """adds either a number or a Channel object"""
        return np.ndarray.__add__(self,other)



STR_FUNCTIONS = { 'Channel.__mul__' : Channel.__mul__,
                  'Channel.__div__' : Channel.__div__,
                  'Channel.__sub__' : Channel.__sub__,
                  'Channel.__add__' : Channel.__add__,
                  }    


class RawFile(object):
    """RawFile class
    
    This class manages input/output for raw FITS file

    Attributes
    ----------
    filename : string
    The raw FITS file name. None if any.
    
    has_scan: boolean
    Does the channels data contain overscan pixels ?

    channels: dict
    List of extension (extname,Channel)

    primary_header: pyfits.CardList
    The primary header

    Methods
    -------
    Creation: init,copy
    
    Arithmetic: + - * /
    
    Info: info
    
    save: write
    
    get: get_channel
    """
    
    def __init__(self, filename=None, has_scan=True):
        """creates a RawFile object
        
        Parameters
        ----------   
        filename : string
        The raw FITS file name. None by default.
    
        has_scan: boolean
        Does the channel data contain over scanned pixels ? True by default.

        Notes
        -----
        filename=None creates an empty object

        The FITS file is opened with memory mapping.
        Just the primary header and the list of extension name are loaded.
        Method get_channel(extname) must be used to create the corresponding Channel object.
        """
        self.filename = filename
        self.has_scan = has_scan
        self.channels = dict()
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
                            self.channels[extname] = None
                        n = n+1
                    except:
                        break                    
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
        result = RawFile(self.filename,self.has_scan)
        if result.filename==None:
            result.primary_header = pyfits.CardList(self.primary_header)
            for name,chan in self.channels.items():
                if chan != None:
                    result.channels[name] = chan.copy()
                else:
                    result.channels[name] = None
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
            n = 1
            for k,v in self.channels.items():
                print "%i\t%s\tchannel\t(%i,%i)" % (n,k,v.get_nx(),v.get_ny())
                n = n + 1

    def get_channel(self,extname, has_scan=True):
        """returns a Channel object
        
        Parameters
        ----------
        extname : string
        The extension name
    
        has_scan: boolean
        Does the channel data contain over scanned pixels ?
        """
        if self.channels[extname] != None:
            return self.channels[extname]
        else:
            chan = Channel(extname,self.filename,has_scan)
            return chan

    def __mul__(self,other):
        """multiplies either a number or a RawFits object"""
        return self.mp_operator(other,'Channel.__mul__')

    def __div__(self,other):
        """divides either a number or a RawFits object"""
        return self.mp_operator(other,'Channel.__div__')

    def __sub__(self,other):
        """subtracts either a number or a RawFits object"""
        return self.mp_operator(other,'Channel.__sub__')

    def __add__(self,other):
        """adds either a number or a RawFits object"""
        return self.mp_operator(other,'Channel.__add__')

    def mp_operator(self,other,funcname):
        #multiprocessing function
        cpu_count = multiprocessing.cpu_count()
        result = RawFile()
        result.primary_header = self.primary_header
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
        if self.channels is not None:
            for k in self.channels.keys():
                processlist.append([funcname,k,self,other])
            if isinstance(other,RawFile):
                processresult = pool.map(process_operator,processlist)
            else:
                processresult = pool.map(process_operator2,processlist)
            for k,out in processresult:
                result.channels[k] = out
        return result

    def write(self,filename):
        """ saves the object in a FITS file
        Parameters
        ----------
        filename : string
        The FITS filename
        """
        if self.has_scan == False:
            print 'Error: can not write raw file without pre and over scans'
            print
            return None
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


def process_operator(arglist):
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
    return (k,out)

def process_operator2(arglist):
    #decorator used to define arithmetic functions with a number
    function = STR_FUNCTIONS[arglist[0]]
    k = arglist[1]
    obj = arglist[2]
    other = arglist[3]
    v = obj.get_channel(k)
    out = function(v,other)
    return (k,out)
