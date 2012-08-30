""" cube.py manages Cube objects"""
import numpy as np
import pyfits
import datetime

from coords import WCS
from coords import WaveCoord
from objs import is_float
from objs import is_int

class iter_spe(object):
    def __init__(self,cube):
        self.cube = cube
        self.p = cube.shape[1]
        self.q = cube.shape[2]
        
    def next(self):
        """Returns the next spectrum."""
        if self.q == 0:
            self.p -= 1
            self.q = self.cube.shape[2]
        self.q -= 1
        if self.p == 0:
            raise StopIteration
        print self.p-1,self.q
        return self.cube[:,self.p-1,self.q]
    
    def __iter__(self):
        """Returns the iterator itself."""
        return self
        
class iter_ima(object): 
    
    def __init__(self,cube):
        self.cube = cube
        self.k = cube.shape[0]
        
    def next(self):
        """Returns the next image."""
        if self.k == 0:
            raise StopIteration
        self.k -= 1
        return self.cube[self.k,:,:]
    
    def __iter__(self):
        """Returns the iterator itself."""
        return self

class Cube(object):
    """This class manages Cube objects.
    
    :param filename: Possible FITS filename.
    :type filename: string
    :param ext: Number/name of the data extension or numbers/names of the data and variance extensions.
    :type ext: integer or (integer,integer) or string or (string,string)
    :param notnoise: True if the noise Variance cube is not read (if it exists).
  
           Use notnoise=True to create cube without variance extension.
    :type notnoise: boolean
    :param shape: Lengths of data in Z, Y and X. Python notation is used (nz,ny,nx). (101,101,101) by default.
    :type shape: integer or (integer,integer,integer)
    :param wcs: World coordinates.
    :type wcs: :class:`mpdaf.obj.WCS`
    :param wave: Wavelength coordinates.
    :type wave: :class:`mpdaf.obj.WaveCoord`
    :param unit: Possible data unit type. None by default.
    :type unit: string
    :param data: Array containing the pixel values of the cube. None by default.
    :type data: float array
    :param var: Array containing the variance. None by default.
    :type var: float array
    :param fscale: Flux scaling factor (1 by default).
    :type fscale: float

    Attributes
    ----------
    
    filename (string) : Possible FITS filename.

    unit (string) : Possible data unit type

    cards (pyfits.CardList) : Possible FITS header instance.

    data (masked array numpy.ma) : Array containing the pixel values of the cube.

    shape (array of 3 integers) : Lengths of data in Z and Y and X (python notation (nz,ny,nx)).

    var (array) : Array containing the variance.

    fscale (float) : Flux scaling factor (1 by default).

    wcs (:class:`mpdaf.obj.WCS`) : World coordinates.

    wave (:class:`mpdaf.obj.WaveCoord`) : Wavelength coordinates
    """
    
    def __init__(self, filename=None, ext = None, notnoise=False, shape=(101,101,101), wcs = None, wave = None, unit=None, data=None, var=None,fscale=1.0):
        """Creates a Cube object.
        
        :param filename: Possible FITS filename.
        :type filename: string
        :param ext: Number/name of the data extension or numbers/names of the data and variance extensions.
        :type ext: integer or (integer,integer) or string or (string,string)
        :param notnoise: True if the noise Variance cube is not read (if it exists).
  
           Use notnoise=True to create cube without variance extension.
        :type notnoise: boolean
        :param shape: Lengths of data in Z, Y and X. Python notation is used (nz,ny,nx). (101,101,101) by default.
        :type shape: integer or (integer,integer,integer)
        :param wcs: World coordinates.
        :type wcs: :class:`mpdaf.obj.WCS`
        :param wave: Wavelength coordinates.
        :type wave: :class:`mpdaf.obj.WaveCoord`
        :param unit: Possible data unit type. None by default.
        :type unit: string
        :param data: Array containing the pixel values of the cube. None by default.
        :type data: float array
        :param var: Array containing the variance. None by default.
        :type var: float array
        :param fscale: Flux scaling factor (1 by default).
        :type fscale: float
        """
        #possible FITS filename
        self.cube = True
        self.filename = filename
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1, we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 3:
                    raise IOError, 'Wrong dimension number: not a cube'
                self.unit = hdr.get('BUNIT', None)
                self.cards = hdr.ascard
                self.shape = np.array([hdr['NAXIS3'],hdr['NAXIS2'],hdr['NAXIS1']])
                self.data = np.array(f[0].data, dtype=float)
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                # WCS object from data header
                try:
                    self.wcs = WCS(hdr)
                except:
                    print "error: wcs not copied."
                    self.wcs = None
                #Wavelength coordinates
                if 'CDELT3' in hdr:
                    cdelt = hdr.get('CDELT3')
                elif 'CD3_3' in hdr:
                    cdelt = hdr.get('CD3_3')
                else:
                    cdelt = 1.0
                crpix = hdr.get('CRPIX3')
                crval = hdr.get('CRVAL3')
                cunit = hdr.get('CUNIT3','')
                self.wave = WaveCoord(crpix, cdelt, crval, cunit,self.shape[0])
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = np.array(f['DATA'].data, dtype=float)
                else:
                    if isinstance(ext,int) or isinstance(ext,str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = np.array(f[n].data, dtype=float)
                if h['NAXIS'] != 3:
                    raise IOError, 'Wrong dimension number in DATA extension'
                self.unit = h.get('BUNIT', None)
                self.cards = h.ascard
                self.shape= np.array([h['NAXIS3'],h['NAXIS2'],h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(h) # WCS object from data header
                except:
                    print "error: wcs not copied."
                    self.wcs = None
                #Wavelength coordinates
                if 'CDELT3' in h:
                    cdelt = h.get('CDELT3')
                elif 'CD3_3' in h:
                    cdelt = h.get('CD3_3')
                else:
                    cdelt = 1.0
                crpix = h.get('CRPIX3')
                crval = h.get('CRVAL3')
                cunit = h.get('CUNIT3','')
                self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape[0])
                self.var = None
                if not notnoise:
                    try:
                        if ext is None:
                            fstat = f['STAT']
                        else:
                            n = ext[1]
                            fstat = f[n]
                        if fstat.header['NAXIS'] != 3:
                            raise IOError, 'Wrong dimension number in variance extension'
                        if fstat.header['NAXIS1'] != self.shape[2] and fstat.header['NAXIS2'] != self.shape[1] and fstat.header['NAXIS3'] != self.shape[0]:
                            raise IOError, 'Number of points in STAT not equal to DATA'
                        self.var = np.array(fstat.data, dtype=float)
                    except:
                        self.var = None
                # DQ extension
                try:
                    mask = np.ma.make_mask(f['DQ'].data)
                    self.data = np.ma.MaskedArray(self.data, mask=mask)
                except:
                    pass
            f.close()
        else:
            #possible data unit type
            self.unit = unit
            # possible FITS header instance
            self.cards = pyfits.CardList()
            #data
            if is_int(shape):
                shape = (shape,shape,shape)
            elif len(shape) == 2:
                shape = (shape[0],shape[1],shape[1])
            elif len(shape) == 3:
                pass
            else:
                raise ValueError, 'dim with dimension > 3'
            if data is None:
                self.data = None
                self.shape = np.array(shape)
            else:
                self.data = np.array(data, dtype = float)
                try:
                    self.shape = np.array(data.shape)
                except:
                    self.shape = np.array(shape)

            if notnoise or var is None:
                self.var = None
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                self.wcs = wcs
                if wcs is not None:
                    self.wcs.wcs.naxis1 = self.shape[2]
                    self.wcs.wcs.naxis2 = self.shape[1]
                    if wcs.wcs.naxis1 !=0 and wcs.wcs.naxis2 != 0 and ( wcs.wcs.naxis1 != self.shape[2] or wcs.wcs.naxis2 != self.shape[1]):
                        print "warning: world coordinates and data have not the same dimensions. Shape of WCS object is modified."
            except :
                self.wcs = None
                print "error: world coordinates not copied."
            try:
                self.wave = wave
                if wave is not None:
                    if wave.shape is not None and wave.shape != self.shape[0]:
                        print "warning: wavelength coordinates and data have not the same dimensions. Shape of WaveCoord object is modified."
                    self.wave.shape = self.shape[0]
            except :
                self.wave = None
                print "error: wavelength solution not copied."
        #Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

    def copy(self):
        """Copies Cube object in a new one and returns it.
        """
        cub = Cube()
        cub.filename = self.filename
        cub.unit = self.unit
        cub.cards = pyfits.CardList(self.cards)
        cub.shape = self.shape.__copy__()
        try:
            cub.data = self.data.__copy__()
        except:
            cub.data = None
        try:
            cub.var = self.var.__copy__()
        except:
            cub.var = None
        cub.fscale = self.fscale
        try:
            cub.wcs = self.wcs.copy()
        except:
            cub.wcs = None
        try:
            cub.wave = self.wave.copy()
        except:
            cub.wave = None
        return cub

    def write(self,filename):
        """ Saves the cube in a FITS file.
        
        :param filename: The FITS filename.
        :type filename: string
        """

        #ToDo: pb with mask !!!!!!!!!!!!!!!!!

        # create primary header
        prihdu = pyfits.PrimaryHDU()

        #world coordinates
        wcs_cards = self.wcs.to_header().ascard

        if self.var is None: # write simple fits file without extension
            prihdu.data = self.data.data
            if self.cards is not None:
                for card in self.cards:
                    try:
                        prihdu.header.update(card.key, card.value, card.comment)
                    except:
                        pass
            prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
            prihdu.header.update('author', 'MPDAF', 'origin of the file')
            # add world coordinate
            for card in wcs_cards:
                prihdu.header.update(card.key, card.value, card.comment)
            prihdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
            prihdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
            prihdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
            prihdu.header.update('CTYPE3', 'LINEAR', 'world coordinate type')
            prihdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
            if self.unit is not None:
                prihdu.header.update('BUNIT', self.unit, 'data unit type')
            prihdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist = [prihdu]
        else: # write fits file with primary header and two extensions
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
            if self.cards is not None:
                for card in self.cards:
                    try:
                        tbhdu.header.update(card.key, card.value, card.comment)
                    except:
                        pass
            tbhdu.header.update('date', str(datetime.datetime.now()), 'creation date')
            tbhdu.header.update('author', 'MPDAF', 'origin of the file')
            # add world coordinate
            for card in wcs_cards:
                tbhdu.header.update(card.key, card.value, card.comment)
            tbhdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
            tbhdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
            tbhdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
            tbhdu.header.update('CTYPE3', 'LINEAR', 'world coordinate type')
            tbhdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
            if self.unit is not None:
                tbhdu.header.update('BUNIT', self.unit, 'data unit type')
            tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(tbhdu)
            # create spectrum STAT in second extension
            nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
            # add world coordinate
            for card in wcs_cards:
                nbhdu.header.update(card.key, card.value, card.comment)
            nbhdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
            nbhdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
            nbhdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
            nbhdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
#            if self.unit is not None:
#                nbhdu.header.update('UNIT', self.unit, 'data unit type')
#            nbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(nbhdu)
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True)

        self.filename = filename

    def info(self):
        """Prints information.
        """
        if self.filename is None:
            print '%i X %i X %i cube (no name)' %(self.shape[2],self.shape[1],self.shape[0])
        else:
            print '%i X %i X %i cube (%s)' %(self.shape[2],self.shape[1],self.shape[0],self.filename)
        data = '.data(%i,%i,%i)' %(self.shape[0],self.shape[1],self.shape[2])
        if self.data is None:
            data = 'no data'
        noise = '.var(%i,%i,%i)' %(self.shape[0],self.shape[1],self.shape[2])
        if self.var is None:
            noise = 'no noise'
        if self.unit is None:
            unit = 'no unit'
        else:
            unit = self.unit
        print '%s (%s) fscale=%g, %s' %(data,unit,self.fscale,noise)
        if self.wcs is None:
            print 'no world coordinates for spatial direction'
        else:
            self.wcs.info()
        if self.wave is None:
            print 'no world coordinates for spectral direction'
        else:
            self.wave.info()


    def __le__ (self, item):
        """Masks data array where greater than a given value.
        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item/self.fscale)
        return result

    def __lt__ (self, item):
        """Masks data array where greater or equal than a given value.
        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item/self.fscale)
        return result

    def __ge__ (self, item):
        """Masks data array where less than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item/self.fscale)
        return result

    def __gt__ (self, item):
        """Masks data array where less or equal than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item/self.fscale)
        return result

    def resize(self):
        """Resizes the cube to have a minimum number of masked values.
        """
        if self.data is not None:
            ksel = np.where(self.data.mask==False)
            try:
                item = (slice(ksel[0][0], ksel[0][-1]+1, None), slice(ksel[1][0], ksel[1][-1]+1, None),slice(ksel[2][0], ksel[2][-1]+1, None))
                self.data = self.data[item]
                if is_int(item[0]):
                    if is_int(item[1]):
                        self.shape = (1,1,data.shape[0])
                    elif is_int(item[2]):
                        self.shape = (1,data.shape[0],1)
                    else:
                        self.shape = (1,data.shape[0],data.shape[1])
                elif is_int(item[1]):
                    if is_int(item[2]):
                        self.shape = (data.shape[0],1,1)
                    else:
                        self.shape = (data.shape[0],1,data.shape[1])
                elif is_int(item[2]):
                        self.shape = (data.shape[0],data.shape[1],1)
                else:
                    self.shape = data.shape
                if self.var is not None:
                    self.var = self.var[item]
                try:
                    self.wcs = self.wcs[item[1],item[2]]
                except:
                    self.wcs = None
                    print "error: wcs not copied."
                try:
                    self.wave = self.wave[item[0]]
                except:
                    self.wave = None
                    print "error: wavelength solution not copied."
            except:
                pass

    def __add__(self, other):
        """Adds other

        cube1 + number = cube2 (cube2[k,p,q]=cube1[k,p,q]+number)

        cube1 + cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]+cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 + image = cube2 (cube2[k,p,q]=cube1[k,p,q]+image[p,q])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 + spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]+spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if is_float(other) or is_int(other):
            # cube + number = cube (add pixel per pixel)
            res = self.copy()
            res.data = self.data + (other/np.double(self.fscale))
            return res
        try:
            # cube1 + cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]+cube2[k,j,i])
            # dimensions must be the same
            # if not equal to None, world coordinates must be the same
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(shape=self.shape , fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spatial directions'
                        return None
                    res.data = self.data + (other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except:
            try:
                # cube1 + image = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                # the 2 first dimensions of cube1 must be equal to the image dimensions
                # if not equal to None, world coordinates in spatial directions must be the same
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for objects with different sizes'
                        return None
                    else:
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data + (other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except:
                try:
                    # cube1 + spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
                    # the last dimension of cube1 must be equal to the spectrum dimension
                    # if not equal to None, world coordinates in spectral direction must be the same
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data + (other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            if self.unit == other.unit:
                                res.unit = self.unit
                            return res
                except:
                    print 'Operation forbidden'
                    return None

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtracts other

        cube1 - number = cube2 (cube2[k,p,q]=cube1[k,p,q]-number)

        cube1 - cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]-cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 - image = cube2 (cube2[k,p,q]=cube1[k,p,q]-image[p,q])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 - spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]-spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if is_float(other) or is_int(other):
            #cube1 - number = cube2 (cube2[k,j,i]=cube1[k,j,i]-number)
            res = self.copy()
            res.data = self.data - (other/np.double(self.fscale))
            return res
        try:
            #cube1 - cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]-cube2[k,j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(shape=self.shape , fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spatial directions'
                        return None
                    res.data = self.data - (other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except:
            try:
                #cube1 - image = cube2 (cube2[k,j,i]=cube1[k,j,i]-image[j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data - (other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except:
                try:
                    #cube1 - spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]-spectrum[k])
                    #The last dimension of cube1 must be equal to the spectrum dimension.
                    #If not equal to None, world coordinates in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data - (other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            if self.unit == other.unit:
                                res.unit = self.unit
                            return res
                except:
                    print 'Operation forbidden'
                    return None

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if is_float(other) or is_int(other):
            res = self.copy()
            res.data = (other/np.double(self.fscale)) - self.data
            return res
        try:
            if other.cube:
                return other.__sub__(self)
        except:
            try:
                if other.image:
                    return other.__sub__(self)
            except:
                try:
                    if other.spectrum:
                        return other.__sub__(self)
                except:
                    print 'Operation forbidden'
                    return None

    def __mul__(self, other):
        """Multiplies by other

        cube1 * number = cube2 (cube2[k,p,q]=cube1[k,p,q]*number)

        cube1 * cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]*cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 * image = cube2 (cube2[k,p,q]=cube1[k,p,q]*image[p,q])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 * spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]*spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if is_float(other) or is_int(other):
            #cube1 * number = cube2 (cube2[k,j,i]=cube1[k,j,i]*number)
            res = self.copy()
            res.fscale *= other
            if res.var is not None:
                res.var *= other*other
            return res
        try:
            #cube1 * cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]*cube2[k,j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(shape=self.shape , fscale=self.fscale*other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spatial directions'
                        return None
                    res.data = self.data * other.data
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except:
            try:
                #cube1 * image = cube2 (cube2[k,j,i]=cube1[k,j,i]*image[j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale * other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data * other.data[np.newaxis,:,:]
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except:
                try:
                    #cube1 * spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]*spectrum[k])
                    #The last dimension of cube1 must be equal to the spectrum dimension.
                    #If not equal to None, world coordinates in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale*other.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data * other.data[:,np.newaxis,np.newaxis]
                            if self.unit == other.unit:
                                res.unit = self.unit
                            return res
                except:
                    print 'Operation forbidden'
                    return None

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """Divides by other

        cube1 / number = cube2 (cube2[k,p,q]=cube1[k,p,q]/number)

        cube1 / cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]/cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 / image = cube2 (cube2[k,p,q]=cube1[k,p,q]/image[p,q])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 / spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]/spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if is_float(other) or is_int(other):
            #cube1 / number = cube2 (cube2[k,j,i]=cube1[k,j,i]/number)
            res = self.copy()
            res.fscale /= other
            if res.var is not None:
                res.var /= other*other
            return res
        try:
            #cube1 / cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]/cube2[k,j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(shape=self.shape , fscale=self.fscale/other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise ValueError, 'Operation forbidden for cubes with different world coordinates in spatial directions'
                    res.data = self.data / other.data
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except:
            try:
                #cube1 / image = cube2 (cube2[k,j,i]=cube1[k,j,i]/image[j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale / other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data / other.data[np.newaxis,:,:]
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except:
                try:
                    #cube1 / spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]/spectrum[k])
                    #The last dimension of cube1 must be equal to the spectrum dimension.
                    #If not equal to None, world coordinates in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale/other.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data / other.data[:,np.newaxis,np.newaxis]
                            if self.unit == other.unit:
                                res.unit = self.unit
                            return res
                except:
                    print 'Operation forbidden'
                    return None

    def __rdiv__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if is_float(other) or is_int(other):
            #cube1 / number = cube2 (cube2[k,j,i]=cube1[k,j,i]/number)
            res = self.copy()
            res.fscale = other / res.fscale
            if res.var is not None:
                res.var = other*other / (res.var*res.var)
            return res
        try:
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    return other.__div__(self)
        except:
            try:
                if other.image:
                    return other.__div__(self)
            except:
                try:
                    if other.spectrum:
                       return other.__div__(self)
                except:
                    print 'Operation forbidden'
                    return None

    def __pow__(self, other):
        """Computes the power exponent.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        if is_float(other) or is_int(other):
            res.data = self.data**other
            res.fscale = res.fscale**other
            res.var = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def _sqrt(self):
        """Computes the positive square-root of data extension.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        self.data = np.sqrt(self.data)
        self.fscale = np.sqrt(self.fscale)
        self.var = None
        
    def sqrt(self):
        """Returns a cube containing the positive square-root of data extension.
        """
        res = self.copy()
        res._sqrt()
        return res

    def _abs(self):
        """Computes the absolute value of data extension.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        self.data = np.abs(self.data)
        self.fscale = np.abs(self.fscale)
        self.var = None

    def abs(self):
        """Returns a cube containing the absolute value of data extension.
        """
        res = self.copy()
        res._abs()
        return res

    def __getitem__(self,item):
        """Returns the corresponding object:
        cube[k,p,k] = value
        cube[k,:,:] = spectrum
        cube[:,p,q] = image
        cube[:,:,:] = sub-cube
        """
        if isinstance(item, tuple) and len(item)==3:
            data = self.data[item]
            if is_int(item[0]):
                if is_int(item[1]) and is_int(item[2]):
                    #return a float
                    return data
                else:
                    #return an image
                    from image import Image
                    if is_int(item[1]):
                        shape = (1,data.shape[0])
                    elif is_int(item[2]):
                        shape = (data.shape[0],1)
                    else:
                        shape = data.shape
                    var = None
                    if self.var is not None:
                        var = self.var[item]
                    try:
                        wcs = self.wcs[item[1],item[2]]
                    except:
                        wcs = None
                    res = Image(shape=shape, wcs = wcs, unit=self.unit, fscale=self.fscale)
                    res.data = data
                    res.var =var
                    return res
            elif is_int(item[1]) and is_int(item[2]):
                #return a spectrum
                from spectrum import Spectrum
                shape = data.shape[0]
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Spectrum(shape=shape, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
                res.var = var
                return res
            else:
                #return a cube
                if is_int(item[1]):
                    shape = (data.shape[0],1,data.shape[1])
                elif is_int(item[2]):
                    shape = (data.shape[0],data.shape[1],1)
                else:
                    shape = data.shape
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[1],item[2]]
                except:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Cube(shape=shape, wcs = wcs, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
                res.var = var
                return res
        else:
            raise ValueError, 'Operation forbidden'

    def get_lambda(self,lbda_min,lbda_max=None):
        """Returns the sub-cube corresponding to a wavelength range.

        :param lbda_min: Minimum wavelength.
        :type lbda_min: float

        :param lbda_max: Maximum wavelength.
        :type lbda_max: float
        """
        if lbda_max is None:
            lbda_max = lbda_min
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            pix_min = max(0,int(self.wave.pixel(lbda_min)))
            pix_max = min(self.shape[0],int(self.wave.pixel(lbda_max)) + 1)
            if (pix_min+1)==pix_max:
                return self.data[pix_min,:,:]
            else:
                return self[pix_min:pix_max,:,:]
            
    def get_step(self):
        """Returns the cube steps [dlbda,dy,dx].
        """
        step = np.empty(3)
        step[0] = self.wave.cdelt
        step[1:] = self.wcs.get_step()
        return step
    
    def get_range(self):
        """Returns [ [lbda_min,y_min,x_min], [lbda_max,y_max,x_max] ].
        """
        range = np.empty((2,3))
        range[:,0] = self.wave.get_range()
        range[:,1:] = self.wcs.get_range()
        return range
    
    def get_start(self):
        """Returns [lbda,y,x] corresponding to pixel (0,0,0).
        """
        start = np.empty(3)
        start[0] = self.wave.get_start()
        start[1:] = self.wcs.get_start()
        return start
    
    def get_end(self):
        """Returns [lbda,y,x] corresponding to pixel (-1,-1,-1).
        """
        end = np.empty(3)
        end[0] = self.wave.get_end()
        end[1:] = self.wcs.get_end()
        return end
    
    def get_rot(self):
        """Returns the rotation angle.
        """
        return self.wcs.get_rot()
        
            
    def __setitem__(self,key,value):
        """Sets the corresponding part of data.
        """
        #self.data[key] = value
        if self.data is None:
            raise ValueError, 'empty data array'
        try:
            self.data[key] = other/np.double(self.fscale)
        except:
            try:
                #other is an image
                if other.image:
                    if self.wcs is not None and other.wcs is not None and not self.wcs.isEqual(other.wcs):
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    self.data[key] = other.data*np.double(other.fscale/self.fscale)
            except:
                try:
                    #other is a spectrum
                    if other.spectrum:
                        if self.wave is not None and other.wave is not None and not self.wave.isEqual(other.wave):
                            print 'Operation forbidden for spectra with different world coordinates'
                            return None
                        self.data[key] = other.data*np.double(other.fscale/self.fscale)
                except:
                    print 'Operation forbidden'
                    return None
            
    def set_wcs(self, wcs, wave):
        """Sets the world coordinates.

        :param wcs: World coordinates.
        :type wcs: :class:`mpdaf.obj.WCS`

        :param wave: Wavelength coordinates.
        :type wave: :class:`mpdaf.obj.WaveCoord`
        """
        self.wcs = wcs
        self.wcs.wcs.naxis1 = self.shape[2]
        self.wcs.wcs.naxis2 = self.shape[1]
        if wcs.wcs.naxis1 !=0 and wcs.wcs.naxis2 != 0 and (wcs.wcs.naxis1 != self.shape[2] or wcs.wcs.naxis2 != self.shape[1]):
            print "warning: world coordinates and data have not the same dimensions."
        
        if wave.shape is not None and wave.shape != self.shape[0]:
            print "warning: wavelength coordinates and data have not the same dimensions."
        self.wave = wave
        self.wave.shape = self.shape[0]
            
    def set_var(self,var):
        """Sets the variance array.
        
        :param var: Input variance array. If None, variance is set with zeros.
        :type var: float array
        """
        if var is None:
            self.var = np.zeros((self.shape[0],self.shape[1], self.shape[2]))
        else:
            if self.shape[0] == np.shape(var)[0] and self.shape[1] == np.shape(var)[1] and self.shape[2] == np.shape(var)[2]:
                self.var = var
            else:
                raise ValueError, 'var and data have not the same dimensions.'
            
    def sum(self,axis=None):
        """Returns the sum over the given axis.
        
        :param axis: Axis or axes along which a sum is performed. 
        
                    The default (axis = None) is perform a sum over all the dimensions of the cube and returns a float.
                    
                    axis = 0  is perform a sum over the wavelength dimension and returns an image.
                    
                    axis = (1,2) is perform a sum over the (X,Y) axes and returns a spectrum.
                    
                    Other cases return None.
        :type axis: None or int or tuple of ints
        """
        if axis is None:
            return self.data.sum()    
        elif axis==0:
            #return an image
            from image import Image
            data = np.ma.sum(self.data, axis)
            res = Image(notnoise=True, shape=data.shape, wcs = self.wcs, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        elif axis==tuple([1,2]):
            #return a spectrum
            from spectrum import Spectrum
            data = np.ma.sum(self.data,axis=1).sum(axis=1)
            res = Spectrum(notnoise=True, shape=data.shape[0], wave = self.wave, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        else:
            return None
        
    def mean(self,axis=None):
        """ Returns the mean over the given axis.
        
        :param axis: Axis or axes along which a mean is performed. 
        
                    The default (axis = None) is perform a mean over all the dimensions of the cube and returns a float.
                    
                    axis = 0  is perform a mean over the wavelength dimension and returns an image.
                    
                    axis = (1,2) is perform a mean over the (X,Y) axes and returns a spectrum.
                    
                    Other cases return None.
        :type axis: None or int or tuple of ints
        """
        if axis is None:
            return self.data.mean()    
        elif axis==0:
            #return an image
            from image import Image
            data = np.ma.mean(self.data, axis)
            res = Image(notnoise=True, shape=data.shape, wcs = self.wcs, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        elif axis==tuple([1,2]):
            #return a spectrum
            from spectrum import Spectrum
            data = np.ma.mean(self.data, axis=1).mean(axis=1)
            res = Spectrum(notnoise=True, shape=data.shape[0], wave = self.wave, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        else:
            return None
