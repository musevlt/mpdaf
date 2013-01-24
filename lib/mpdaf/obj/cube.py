""" cube.py manages Cube objects"""
import numpy as np
import pyfits
import datetime
import multiprocessing
import types

from coords import WCS
from coords import WaveCoord
from objs import is_float
from objs import is_int

class iter_spe(object):
    def __init__(self, cube, index=False):
        self.cube = cube
        self.p = cube.shape[1]
        self.q = cube.shape[2]
        self.index = index
        
    def next(self):
        """Returns the next spectrum."""
        if self.q == 0:
            self.p -= 1
            self.q = self.cube.shape[2]
        self.q -= 1
        if self.p == 0:
            raise StopIteration
        if self.index is False:
            return self.cube[:,self.p-1,self.q]
        else:
            return (self.cube[:,self.p-1,self.q],(self.p-1,self.q))
    
    def __iter__(self):
        """Returns the iterator itself."""
        return self
    
        
class iter_ima(object): 
    
    def __init__(self, cube, index=False):
        self.cube = cube
        self.k = cube.shape[0]
        self.index = index
        
    def next(self):
        """Returns the next image."""
        if self.k == 0:
            raise StopIteration
        self.k -= 1
        if self.index is False:
            return self.cube[self.k,:,:]
        else:
            return (self.cube[self.k,:,:],self.k)
    
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
    :type notnoise: bool
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

    primary_header (pyfits.CardList) : Possible FITS primary header instance.

    data_header (pyfits.CardList) : Possible FITS data header instance.

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
        :type notnoise: bool
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
                self.primary_header = pyfits.CardList()
                self.data_header = hdr.ascard
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
                self.primary_header = hdr.ascard
                self.data_header = h.ascard
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
            self.data_header = pyfits.CardList()
            self.primary_header = pyfits.CardList()
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
                    if wcs.wcs.naxis1 !=0 and wcs.wcs.naxis2 != 0 and ( wcs.wcs.naxis1 != self.shape[2] or wcs.wcs.naxis2 != self.shape[1]):
                        print "warning: world coordinates and data have not the same dimensions. Shape of WCS object is modified."
                    self.wcs.wcs.naxis1 = self.shape[2]
                    self.wcs.wcs.naxis2 = self.shape[1]
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
        """Returns a new copy of a Cube object.
        """
        cub = Cube()
        cub.filename = self.filename
        cub.unit = self.unit
        cub.data_header = pyfits.CardList(self.data_header)
        cub.primary_header = pyfits.CardList(self.primary_header)
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
    
    def clone(self, var = False):
        """Returns a new cube of the same shape and coordinates, filled with zeros.
        
        :param var: Presence of the variance extension.
        :type var: bool
        """
        try:
            wcs=self.wcs.copy()
        except:
            wcs=None
        try:
            wave=self.wave.copy()
        except:
            wave=None
        if var is False:
            cube = Cube(wcs=wcs,wave=wave,data=np.zeros(shape=self.shape),unit=self.unit)
        else:
            cube = Cube(wcs=wcs,wave=wave,data=np.zeros(shape=self.shape),var=np.zeros(shape=self.shape),unit=self.unit)
        return cube
        

    def write(self,filename):
        """ Saves the cube in a FITS file.
        
        :param filename: The FITS filename.
        :type filename: string
        """

        # create primary header
        prihdu = pyfits.PrimaryHDU()
        for card in self.primary_header:
            try:
                prihdu.header.update(card.key, card.value, card.comment)
            except:
                try:
                    card.verify('fix')
                    prihdu.header.update(card.key, card.value, card.comment)
                except:
                    try:
                        if isinstance(card.value,str):
                            n = 80 - len(card.key) - 14
                            s = card.value[0:n]
                            prihdu.header.update('hierarch %s' %card.key, s, card.comment)
                        else:
                            prihdu.header.update('hierarch %s' %card.key, card.value, card.comment)
                    except:
                        print "warning: %s not copied in primary header"%card.key
                        pass
        prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
        prihdu.header.update('author', 'MPDAF', 'origin of the file')
        hdulist = [prihdu]
        
        #world coordinates
        wcs_cards = self.wcs.to_header().ascard

        # create spectrum DATA extension
        tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
        for card in self.data_header:
            try:
                
                if card.key != 'CD1_1' and card.key != 'CD1_2' and card.key != 'CD2_1' and card.key != 'CD2_2' and card.key != 'CDELT1' and card.key != 'CDELT2' and tbhdu.header.keys().count(card.key)==0:
                    tbhdu.header.update(card.key, card.value, card.comment)
            except:
                try:
                    card.verify('fix')
                    if card.key != 'CD1_1' and card.key != 'CD1_2' and card.key != 'CD2_1' and card.key != 'CD2_2' and card.key != 'CDELT1' and card.key != 'CDELT2' and tbhdu.header.keys().count(card.key)==0:
                        prihdu.header.update(card.key, card.value, card.comment)
                except:
                    print "warning: %s not copied in data header"%card.key
                    pass
        # add world coordinate
        cd = self.wcs.get_cd()
        tbhdu.header.update('CTYPE1', wcs_cards['CTYPE1'].value, wcs_cards['CTYPE1'].comment)
        tbhdu.header.update('CUNIT1', wcs_cards['CUNIT1'].value, wcs_cards['CUNIT1'].comment)
        tbhdu.header.update('CRVAL1', wcs_cards['CRVAL1'].value, wcs_cards['CRVAL1'].comment)
        tbhdu.header.update('CRPIX1', wcs_cards['CRPIX1'].value, wcs_cards['CRPIX1'].comment)
        tbhdu.header.update('CD1_1', cd[0,0], 'partial of first axis coordinate w.r.t. x ')
        tbhdu.header.update('CD1_2', cd[0,1], 'partial of first axis coordinate w.r.t. y')
        tbhdu.header.update('CTYPE2', wcs_cards['CTYPE2'].value, wcs_cards['CTYPE2'].comment)
        tbhdu.header.update('CUNIT2', wcs_cards['CUNIT2'].value, wcs_cards['CUNIT2'].comment)
        tbhdu.header.update('CRVAL2', wcs_cards['CRVAL2'].value, wcs_cards['CRVAL2'].comment)
        tbhdu.header.update('CRPIX2', wcs_cards['CRPIX2'].value, wcs_cards['CRPIX2'].comment)
        tbhdu.header.update('CD2_1', cd[1,0], 'partial of second axis coordinate w.r.t. x')
        tbhdu.header.update('CD2_2', cd[1,1], 'partial of second axis coordinate w.r.t. y')
        tbhdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
        tbhdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
        tbhdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
        tbhdu.header.update('CTYPE3', 'LINEAR', 'world coordinate type')
        tbhdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
        if self.unit is not None:
            tbhdu.header.update('BUNIT', self.unit, 'data unit type')
        tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
        hdulist.append(tbhdu)
        
        self.wcs = WCS(tbhdu.header)
        
        # create spectrum STAT extension
        if self.var is not None:
            nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
            # add world coordinate
#            for card in wcs_cards:
#                nbhdu.header.update(card.key, card.value, card.comment)
            nbhdu.header.update('CTYPE1', wcs_cards['CTYPE1'].value, wcs_cards['CTYPE1'].comment)
            nbhdu.header.update('CUNIT1', wcs_cards['CUNIT1'].value, wcs_cards['CUNIT1'].comment)
            nbhdu.header.update('CRVAL1', wcs_cards['CRVAL1'].value, wcs_cards['CRVAL1'].comment)
            nbhdu.header.update('CRPIX1', wcs_cards['CRPIX1'].value, wcs_cards['CRPIX1'].comment)
            nbhdu.header.update('CD1_1', cd[0,0], 'partial of first axis coordinate w.r.t. x ')
            nbhdu.header.update('CD1_2', cd[0,1], 'partial of first axis coordinate w.r.t. y')
            nbhdu.header.update('CTYPE2', wcs_cards['CTYPE2'].value, wcs_cards['CTYPE2'].comment)
            nbhdu.header.update('CUNIT2', wcs_cards['CUNIT2'].value, wcs_cards['CUNIT2'].comment)
            nbhdu.header.update('CRVAL2', wcs_cards['CRVAL2'].value, wcs_cards['CRVAL2'].comment)
            nbhdu.header.update('CRPIX2', wcs_cards['CRPIX2'].value, wcs_cards['CRPIX2'].comment)
            nbhdu.header.update('CD2_1', cd[1,0], 'partial of second axis coordinate w.r.t. x')
            nbhdu.header.update('CD2_2', cd[1,1], 'partial of second axis coordinate w.r.t. y')
            nbhdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
            nbhdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
            nbhdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
            nbhdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
            hdulist.append(nbhdu)
            
        # create DQ extension
        if np.ma.count_masked(self.data) != 0:
            dqhdu = pyfits.ImageHDU(name='DQ', data=np.uint8(self.data.mask))
            for card in wcs_cards:
                dqhdu.header.update(card.key, card.value, card.comment)
            hdulist.append(dqhdu)
            
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True)

        self.filename = filename

    def info(self):
        """Prints information.
        """
        if self.filename is None:
            print '%i X %i X %i cube (no name)' %(self.shape[0],self.shape[1],self.shape[2])
        else:
            print '%i X %i X %i cube (%s)' %(self.shape[0],self.shape[1],self.shape[2],self.filename)
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
                        self.shape = np.array((1,1,data.shape[0]))
                    elif is_int(item[2]):
                        self.shape = np.array((1,data.shape[0],1))
                    else:
                        self.shape = np.array((1,data.shape[0],data.shape[1]))
                elif is_int(item[1]):
                    if is_int(item[2]):
                        self.shape = np.array((data.shape[0],1,1))
                    else:
                        self.shape = np.array((data.shape[0],1,data.shape[1]))
                elif is_int(item[2]):
                        self.shape = np.array((data.shape[0],data.shape[1],1))
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
            
    def unmask(self):
        """Unmasks the cube (just invalid data (nan,inf) are masked).
        """
        self.data.mask = False
        self.data = np.ma.masked_invalid(self.data)
        
        
    def mask_variance(self, threshold):
        """Masks pixels with a variance upper than threshold value.

        :param threshold: Threshold value.
        :type threshold: float
        """
        if self.var is None:
            raise ValueError, 'Operation forbidden without variance extension.'
        else:
            ksel = np.where(self.var > threshold)
            self.data[ksel] = np.ma.masked  
            
    def mask_selection(self, ksel):
        """Masks pixels corresponding to the selection.
        
        :param ksel: elements depending on a condition (output of np.where)
        :type ksel: ndarray or tuple of ndarrays
        """
        self.data[ksel] = np.ma.masked

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
                    #coordinate
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
                    #data
                    res.data = self.data + (other.data*np.double(other.fscale/self.fscale))
                    #variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                    #unit
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
                        #coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        #data
                        res.data = self.data + (other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        #variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = np.ones(self.shape)*other.var[np.newaxis,:,:]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                        elif other.var is None:
                            res.var = self.var
                        else:
                            res.var = self.var + other.var[np.newaxis,:,:]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                        #unit
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
                            #coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            #data
                            res.data = self.data + (other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            #variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = np.ones(self.shape) * other.var[:,np.newaxis,np.newaxis]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                            elif other.var is None:
                                res.var = self.var
                            else:
                                res.var = self.var + other.var[:,np.newaxis,np.newaxis]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                            #unit
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
                    #coordinates
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
                    #data
                    res.data = self.data - (other.data*np.double(other.fscale/self.fscale))
                    #variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                    #unit
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
                        #coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        #data
                        res.data = self.data - (other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        #variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = np.ones(self.shape)*other.var[np.newaxis,:,:]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                        elif other.var is None:
                            res.var = self.var
                        else:
                            res.var = self.var + other.var[np.newaxis,:,:]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                        #unit
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
                            #coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            #data
                            res.data = self.data - (other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            #variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = np.ones(self.shape) * other.var[:,np.newaxis,np.newaxis]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                            elif other.var is None:
                                res.var = self.var
                            else:
                                res.var = self.var + other.var[:,np.newaxis,np.newaxis]*np.double(other.fscale*other.fscale/self.fscale/self.fscale)
                            #unit
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
                    #coordinates
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
                    #data
                    res.data = self.data * other.data
                    #variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var*self.data*self.data
                    elif other.var is None:
                        res.var = self.var*other.data*other.data
                    else:
                        res.var = other.var*self.data*self.data + self.var*other.data*other.data
                    #unit
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
                        #coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        #data
                        res.data = self.data * other.data[np.newaxis,:,:]
                        #variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var[np.newaxis,:,:]*self.data*self.data
                        elif other.var is None:
                            res.var = self.var*other.data[np.newaxis,:,:]*other.data[np.newaxis,:,:]
                        else:
                            res.var = other.var[np.newaxis,:,:]*self.data*self.data + self.var*other.data[np.newaxis,:,:]*other.data[np.newaxis,:,:]
                        #unit
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
                            #coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            #data
                            res.data = self.data * other.data[:,np.newaxis,np.newaxis]
                            #variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = other.var[:,np.newaxis,np.newaxis]*self.data*self.data
                            elif other.var is None:
                                res.var = self.var*other.data[:,np.newaxis,np.newaxis]*other.data[:,np.newaxis,np.newaxis]
                            else:
                                res.var = other.var[:,np.newaxis,np.newaxis]*self.data*self.data + self.var*other.data[:,np.newaxis,np.newaxis]*other.data[:,np.newaxis,np.newaxis]
                            #unit
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
                    #coordinates
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
                    #data
                    res.data = self.data / other.data
                    #variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var*self.data*self.data/(other.data**4)
                    elif other.var is None:
                        res.var = self.var*other.data*other.data/(other.data**4)
                    else:
                        res.var = (other.var*self.data*self.data + self.var*other.data*other.data)/(other.data**4)
                    #unit
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
                        #coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        #data
                        res.data = self.data / other.data[np.newaxis,:,:]
                        #variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var[np.newaxis,:,:]*self.data*self.data/(other.data[np.newaxis,:,:]**4)
                        elif other.var is None:
                            res.var = self.var*other.data[np.newaxis,:,:]*other.data[np.newaxis,:,:]/(other.data[np.newaxis,:,:]**4)
                        else:
                            res.var = (other.var[np.newaxis,:,:]*self.data*self.data + self.var*other.data[np.newaxis,:,:]*other.data[np.newaxis,:,:])/(other.data[np.newaxis,:,:]**4)
                        #unit
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
                            #coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            #data
                            res.data = self.data / other.data[:,np.newaxis,np.newaxis]
                            #variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = other.var[:,np.newaxis,np.newaxis]*self.data*self.data/(other.data[:,np.newaxis,np.newaxis]**4)
                            elif other.var is None:
                                res.var = self.var*other.data[:,np.newaxis,np.newaxis]*other.data[:,np.newaxis,np.newaxis]/(other.data[:,np.newaxis,np.newaxis]**4)
                            else:
                                res.var = (other.var[:,np.newaxis,np.newaxis]*self.data*self.data + self.var*other.data[:,np.newaxis,np.newaxis]*other.data[:,np.newaxis,np.newaxis])/(other.data[:,np.newaxis,np.newaxis]**4)
                            #unit
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
        if self.var is not None:
            self.var = 3*self.var*self.fscale**5/self.data**4
        self.data = np.sqrt(self.data)
        self.fscale = np.sqrt(self.fscale)
        
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
                    return data*self.fscale
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
                return self.data[pix_min,:,:]*self.fscale
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
        
            
    def __setitem__(self,key,other):
        """Sets the corresponding part of data.
        """
        #self.data[key] = value
        if self.data is None:
            raise ValueError, 'empty data array'
        try:
            self.data[key] = other/np.double(self.fscale)
        except:
            try:
                #other is a cube
                if other.cube:
                    try:
                        if self.wcs is not None and other.wcs is not None and (self.wcs.get_step()!=other.wcs.get_step()).any() \
                        and self.wave is not None and other.wave is not None and (self.wave.get_step()!=other.wave.get_step()):
                            print 'Warning: cubes with different steps'
                        self.data[key] = other.data*np.double(other.fscale/self.fscale)
                    except:
                        self.data[key] = other.data*np.double(other.fscale/self.fscale)
            except:
                try:
                    #other is an image
                    if other.image:
                        try:
                            if self.wcs is not None and other.wcs is not None and (self.wcs.get_step()!=other.wcs.get_step()).any():
                                print 'Warning: images with different steps'
                            self.data[key] = other.data*np.double(other.fscale/self.fscale)
                        except:
                            self.data[key] = other.data*np.double(other.fscale/self.fscale)
                except:
                    try:
                        #other is a spectrum
                        if other.spectrum:
                            if self.wave is not None and other.wave is not None and (self.wave.get_step()!=other.wave.get_step()):
                                print 'Warning: cubes with different steps'
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
            return self.data.sum()*self.fscale
        elif axis==0:
            #return an image
            from image import Image
            data = np.ma.sum(self.data, axis)
            if self.var is not None:
                var = np.sum(self.var,axis)
            else:
                var = None
            res = Image(shape=data.shape, wcs = self.wcs, unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        elif axis==tuple([1,2]):
            #return a spectrum
            from spectrum import Spectrum
            data = np.ma.sum(self.data,axis=1).sum(axis=1)
            if self.var is not None:
                var = np.ma.sum(self.var,axis=1).sum(axis=1)
            else:
                var = None
            res = Spectrum(shape=data.shape[0], wave = self.wave, unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
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
            return self.data.mean()*self.fscale
        elif axis==0:
            #return an image
            from image import Image
            data = np.ma.mean(self.data, axis)
            if self.var is not None:
                var = np.ma.mean(self.var, axis)
            else:
                var = None
            res = Image(shape=data.shape, wcs = self.wcs, unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        elif axis==tuple([1,2]):
            #return a spectrum
            from spectrum import Spectrum
            data = np.ma.mean(self.data, axis=1).mean(axis=1)
            if self.var is not None:
                var = np.ma.mean(self.var, axis=1).mean(axis=1)
            else:
                var = None
            res = Spectrum(notnoise=True, shape=data.shape[0], wave = self.wave, unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        else:
            return None
        
    def _rebin_factor_(self, factor):
        '''Shrinks the size of the cube by factor.
        New size is an integer multiple of the original size.
        
        Parameter
        ----------
        factor : (integer,integer,integer)
        Factor in z, y and x.
        Python notation: (nz,ny,nx)
        '''
        # new size is an integer multiple of the original size
        assert not np.sometrue(np.mod( self.shape[0], factor[0] ))
        assert not np.sometrue(np.mod( self.shape[1], factor[1] ))
        assert not np.sometrue(np.mod( self.shape[2], factor[2] ))
        #shape
        self.shape = np.array((self.shape[0]/factor[0],self.shape[1]/factor[1],self.shape[2]/factor[2]))
        #data
        self.data = self.data.reshape(self.shape[0],factor[0],self.shape[1],factor[1],self.shape[2],factor[2]).sum(1).sum(2).sum(3)/factor[0]/factor[1]/factor[2]
        #variance
        if self.var is not None:
            self.var = self.var.reshape(self.shape[0],factor[0],self.shape[1],factor[1],self.shape[2],factor[2]).sum(1).sum(2).sum(3)/factor[0]/factor[1]/factor[2]/factor[0]/factor[1]/factor[2]
        #coordinates
        cdelt = self.wcs.get_step()
        self.wcs = self.wcs.rebin_factor(factor[1:])
        crval = self.wave.coord()[0:factor[0]].sum()/factor[0]
        self.wave = WaveCoord(1, self.wave.cdelt*factor[0], crval, self.wave.cunit,self.shape[0])
        
    def _rebin_factor(self, factor, margin='center', flux=False):
        '''Shrinks the size of the cube by factor.
  
          :param factor: Factor in z, y and x. Python notation: (nz,ny,nx).
          :type factor: integer or (integer,integer,integer)
          :param flux: This parameters is used if new size is not an integer multiple of the original size.
          
              If Flux is False, the cube is truncated and the flux is not conserved.
              
              If Flux is True, margins are added to the cube to conserve the flux.
          :type flux: bool
          :param margin: This parameters is used if new size is not an integer multiple of the original size. 
  
            In 'center' case, cube is truncated/pixels are added on the left and on the right, on the bottom and of the top of the cube. 
        
            In 'origin'case, cube is truncated/pixels are added at the end along each direction
          :type margin: 'center' or 'origin'   
        '''
        if is_int(factor):
            factor = (factor,factor,factor)
        if factor[0]<1 or factor[0]>=self.shape[0] or factor[1]<1 or factor[1]>=self.shape[1] or factor[2]<1 or factor[2]>=self.shape[2]:
            raise ValueError, 'factor must be in ]1,shape['
            return None
        if not np.sometrue(np.mod( self.shape[0], factor[0] )) and not np.sometrue(np.mod( self.shape[1], factor[1] )) and not np.sometrue(np.mod( self.shape[2], factor[2] )):
            # new size is an integer multiple of the original size
            self._rebin_factor_(factor)
            return None
        else:
            factor = np.array(factor)
            newshape = self.shape/factor
            n = self.shape - newshape*factor
            
            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0]==1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0]/2
                    n0_right = self.shape[0] - n[0] + n0_left
            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1]==1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1]/2
                    n1_right = self.shape[1] - n[1] + n1_left
            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2]==1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2]/2
                    n2_right = self.shape[2] - n[2] + n2_left
            
            cub = self[n0_left:n0_right,n1_left:n1_right,n2_left:n2_right]
            cub._rebin_factor_(factor)
            
            if flux is False:
                self.shape = cub.shape
                self.data = cub.data
                self.var = cub.var
                self.wave = cub.wave
                self.wcs = cub.wcs
                return None
            else:
                newshape = cub.shape
                wave = cub.wave
                wcs = cub.wcs
                if n0_left != 0:
                    newshape[0] += 1
                    wave.crpix += 1
                    wave.shape += 1
                    l_left = 1
                else:
                    l_left = 0
                if n0_right != self.shape[0]:
                    newshape[0] += 1
                    l_right = newshape[0]-1
                else:
                    l_right  = newshape[0]
                    
                if n1_left != 0:
                    newshape[1] += 1
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    wcs.set_naxis2(wcs.wcs.naxis2 +1)
                    p_left = 1
                else:
                    p_left = 0
                if n1_right != self.shape[1]:
                    newshape[1] += 1
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    p_right = newshape[1]-1
                else:
                    p_right  = newshape[1]
                
                if n2_left != 0:
                    newshape[2] += 1
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    wcs.set_naxis1(wcs.wcs.naxis1 +1)
                    q_left = 1
                else:
                    q_left = 0
                if n2_right != self.shape[2]:
                    newshape[2] += 1
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    q_right = newshape[2]-1
                else:
                    q_right  = newshape[2]
                    
                data = np.empty(newshape)
                mask = np.empty(newshape,dtype=bool)
                data[l_left:l_right,p_left:p_right,q_left:q_right] = cub.data
                mask[l_left:l_right,p_left:p_right,q_left:q_right] = cub.data.mask
                
                if self.var is None:
                    var = None
                else:
                    var = np.empty(newshape)
                    var[l_left:l_right,p_left:p_right,q_left:q_right] = cub.var
                    
                F = factor[0] * factor[1] * factor[2]
                F2 = F * F
                
                if cub.shape[0] != newshape[0]:
                    d = self.data[n0_right:,n1_left:n1_right,n2_left:n2_rigth].sum(axis=0).reshape(cub.shape[1],factor[1],cub.shape[2],factor[2]).sum(1).sum(2) / F
                    data[-1,p_left:q_left,q_left:q_right] = d.data
                    mask[-1,p_left:q_left,q_left:q_right] = d.mask
                    if var is not None:
                        var[-1,p_left:q_left,q_left:q_right] = self.var[n0_right:,n1_left:n1_right,n2_left:n2_rigth].sum(axis=0).reshape(cub.shape[1],factor[1],cub.shape[2],factor[2]).sum(1).sum(2) / F2
                if l_left==1:
                    d = self.data[:n0_left,n1_left:n1_right,n2_left:n2_rigth].sum(axis=0).reshape(cub.shape[1],factor[1],cub.shape[2],factor[2]).sum(1).sum(2) / F
                    data[0,p_left:q_left,q_left:q_right] = d.data
                    mask[0,p_left:q_left,q_left:q_right] = d.mask
                    if var is not None:
                       var[0,p_left:q_left,q_left:q_right] = self.var[:n0_left,n1_left:n1_right,n2_left:n2_rigth].sum(axis=0).reshape(cub.shape[1],factor[1],cub.shape[2],factor[2]).sum(1).sum(2) / F2
                if cub.shape[1] != newshape[1]:
                    d = self.data[n0_left:n0_right,n1_right:,n2_left:n2_rigth].sum(axis=1).reshape(cub.shape[0],factor[0],cub.shape[2],factor[2]).sum(1).sum(2) / F
                    data[l_left:l_rigth,-1,q_left:q_right] = d.data
                    mask[l_left:l_rigth,-1,q_left:q_right] = d.mask
                    if var is not None:
                        var[l_left:l_rigth,-1,q_left:q_right] = self.var[n0_left:n0_right,n1_right:,n2_left:n2_rigth].sum(axis=1).reshape(cub.shape[0],factor[0],cub.shape[2],factor[2]).sum(1).sum(2) / F2
                if p_left==1:
                    d = self.data[n0_left:no_right,:n1_left,n2_left:n2_rigth].sum(axis=1).reshape(cub.shape[0],factor[0],cub.shape[2],factor[2]).sum(1).sum(2) / F
                    data[l_left:l_rigth,0,q_left:q_right] = d.data
                    mask[l_left:l_rigth,0,q_left:q_right] = d.mask
                    if var is not None:
                        var[l_left:l_rigth,0,q_left:q_right] = self.var[n0_left:no_right,:n1_left,n2_left:n2_rigth].sum(axis=1).reshape(cub.shape[0],factor[0],cub.shape[2],factor[2]).sum(1).sum(2) / F2
                    
                if cub.shape[2] != newshape[2]:
                    d = self.data[n0_left:n0_right,n1_left:n1_right:,n2_rigth:].sum(axis=2).reshape(cub.shape[0],factor[0],cub.shape[1],factor[1]).sum(1).sum(2) / F
                    data[l_left:l_rigth,p_left:p_right,-1] = d.data
                    mask[l_left:l_rigth,p_left:p_right,-1] = d.mask
                    if var is not None:
                        var[l_left:l_rigth,p_left:p_right,-1] = self.var[n0_left:n0_right,n1_left:n1_right:,n2_rigth:].sum(axis=2).reshape(cub.shape[0],factor[0],cub.shape[1],factor[1]).sum(1).sum(2) / F2
                if q_left==1:
                    d = self.data[n0_left:n0_right,n1_left:n1_right:,:n2_left].sum(axis=2).reshape(cub.shape[0],factor[0],cub.shape[1],factor[1]).sum(1).sum(2) / F
                    data[l_left:l_rigth,p_left:p_right,0] = d.data
                    mask[l_left:l_rigth,p_left:p_right,0] = d.mask
                    if var is not None:
                        var[l_left:l_rigth,p_left:p_right,0] = self.var[n0_left:n0_right,n1_left:n1_right:,:n2_left].sum(axis=2).reshape(cub.shape[0],factor[0],cub.shape[1],factor[1]).sum(1).sum(2) / F2
                    
                if l_left==1 and p_left==1 and q_left==1:
                    data[0,0,0] = self.data[:n0_left,:n1_left,:n2_left].sum()/ F
                    mask[0,0,0] = self.mask[:n0_left,:n1_left,:n2_left].any()
                    if var is not None:
                        var[0,0,0] = self.var[:n0_left,:n1_left,:n2_left].sum()/ F2
                if l_left==1 and p_right==(newshape[1]-1) and q_left==1:
                    data[0,-1,0] = self.data[:n0_left,n1_right:,:n2_left].sum()/ F
                    mask[0,-1,0] = self.mask[:n0_left,n1_right:,:n2_left].any()
                    if var is not None:
                        var[0,-1,0] = self.var[:n0_left,n1_right:,:n2_left].sum()/ F2
                if l_left==1 and p_right==(newshape[1]-1) and q_right==(newshape[2]-1):
                    data[0,-1,-1] = self.data[:n0_left,n1_right:,n2_right:].sum()/ F
                    mask[0,-1,-1] = self.mask[:n0_left,n1_right:,n2_right:].any()
                    if var is not None:
                        var[0,-1,-1] = self.var[:n0_left,n1_right:,n2_right:].sum()/ F2
                if l_left==1 and p_left==1 and q_right==(newshape[2]-1):
                    data[0,0,-1] = self.data[:n0_left,:n1_left,n2_right:].sum()/ F
                    mask[0,0,-1] = self.mask[:n0_left,:n1_left,n2_right:].any()
                    if var is not None:
                        var[0,0,-1] = self.var[:n0_left,:n1_left,n2_right:].sum()/ F2
                if l_left==(newshape[0]-1) and p_left==1 and q_left==1:
                    data[-1,0,0] = self.data[n0_right:,:n1_left,:n2_left].sum()/ F
                    mask[-1,0,0] = self.mask[n0_right:,:n1_left,:n2_left].any()
                    if var is not None:
                        var[-1,0,0] = self.var[n0_right:,:n1_left,:n2_left].sum()/ F2
                if l_left==(newshape[0]-1) and p_right==(newshape[1]-1) and q_left==1:
                    data[-1,-1,0] = self.data[n0_right:,n1_right:,:n2_left].sum()/ F
                    mask[-1,-1,0] = self.mask[n0_right:,n1_right:,:n2_left].any()
                    if var is not None:
                        var[-1,-1,0] = self.var[n0_right:,n1_right:,:n2_left].sum()/ F2
                if l_left==(newshape[0]-1) and p_right==(newshape[1]-1) and q_right==(newshape[2]-1):
                    data[-1,-1,-1] = self.data[n0_right:,n1_right:,n2_right:].sum()/ F
                    mask[-1,-1,-1] = self.mask[n0_right:,n1_right:,n2_right:].any()
                    if var is not None:
                        var[-1,-1,-1] = self.var[n0_right:,n1_right:,n2_right:].sum()/ F2
                if l_left==(newshape[0]-1) and p_left==1 and q_right==(newshape[2]-1):
                    data[-1,0,-1] = self.data[n0_right:,:n1_left,n2_right:].sum()/ F
                    mask[-1,0,-1] = self.mask[n0_right:,:n1_left,n2_right:].any()    
                    if var is not None:
                        var[-1,0,-1] = self.var[n0_right:,:n1_left,n2_right:].sum()/ F2
                    
                if p_left==1 and q_left==1:
                    d = self.data[n0_left:n0_right,:n1_left,:n2_left].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) / F
                    data[l_left:l_right,0,0] = d.data
                    mask[l_left:l_right,0,0] = d.mask
                    if var is not None:
                        var[l_left:l_right,0,0] = self.var[n0_left:n0_right,:n1_left,:n2_left].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) / F2
                if l_left==1 and p_left==1:
                    d = self.data[:n0_left,:n1_left,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F
                    data[0,0,q_left:q_right] = d.data
                    mask[0,0,q_left:q_right] = d.mask
                    if var is not None:
                        var[0,0,q_left:q_right] = self.var[:n0_left,:n1_left,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F2
                if l_left==1 and q_left==1:
                    d = self.data[:n0_left,n1_left:n1_right,:n2_left].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F
                    data[0,p_left:p_right,0] = d.data
                    mask[0,p_left:p_right,0] = d.mask
                    if var is not None:
                        var[0,p_left:p_right,0] = self.var[:n0_left,n1_left:n1_right,:n2_left].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F2
                    
                if p_left==1 and q_right==(newshape[2]-1):
                    d = self.data[n0_left:n0_right,:n1_left,n2_right:].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) / F
                    data[l_left:l_right,0,-1] = d.data
                    mask[l_left:l_right,0,-1] = d.mask
                    if var is not None:
                        var[l_left:l_right,0,-1] = self.var[n0_left:n0_right,:n1_left,n2_right:].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) / F2
                if l_left==1 and p_right==(newshape[1]-1):
                    d = self.data[:n0_left,n1_right:,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F
                    data[0,-1,q_left:q_right] = d.data
                    mask[0,-1,q_left:q_right] = d.mask
                    if var is not None:
                        var[0,-1,q_left:q_right] = self.var[:n0_left,n1_right:,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F2
                if l_left==1 and q_right==(newshape[2]-1):
                    d = self.data[:n0_left,n1_left:n1_right,n2_right:].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F
                    data[0,p_left:p_right,-1] = d.data
                    mask[0,p_left:p_right,-1] = d.mask
                    if var is not None:
                        var[0,p_left:p_right,-1] = self.var[:n0_left,n1_left:n1_right,n2_right:].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F2
                    
                if p_right==(newshape[1]-1) and q_left==1:
                    d = self.data[n0_left:n0_right,n1_right:,:n2_left].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) / F
                    data[l_left:l_right,-1,0] = d.data
                    mask[l_left:l_right,-1,0] = d.mask
                    if var is not None:
                        var[l_left:l_right,-1,0] = self.var[n0_left:n0_right,n1_right:,:n2_left].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) / F2
                if l_right==(newshape[0]-1) and p_left==1:
                    d = self.data[n0_right:,:n1_left,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F
                    data[-1,0,q_left:q_right] = d.data
                    mask[-1,0,q_left:q_right] = d.mask
                    if var is not None:
                        var[-1,0,q_left:q_right] = self.var[n0_right:,:n1_left,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F2
                if l_right==(newshape[0]-1) and q_left==1:
                    d = self.data[n0_right:,n1_left:n1_right,:n2_left].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F
                    data[-1,p_left:p_right,0] = d.data
                    mask[-1,p_left:p_right,0] = d.mask
                    if var is not None:
                        var[-1,p_left:p_right,0] = self.var[n0_right:,n1_left:n1_right,:n2_left].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F2
                
                if p_right==(newshape[1]-1) and q_right==(newshape[2]-1):
                    d = self.data[n0_left:n0_right,n1_right:,n2_right:].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) /F
                    data[l_left:l_right,-1,-1] = d.data
                    mask[l_left:l_right,-1,-1] = d.mask
                    if var is not None:
                        var[l_left:l_right,-1,-1] = self.var[n0_left:n0_right,n1_right:,n2_right:].sum(axis=2).sum(axis=1).reshape(cub.shape[0],factor[0]).sum(1) /F2
                if l_right==(newshape[0]-1) and p_right==(newshape[1]-1):
                    d = self.data[n0_right:,n1_right:,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F
                    data[-1,-1,q_left:q_right] = d.data
                    mask[-1,-1,q_left:q_right] = d.mask
                    if var is not None:
                        var[-1,-1,q_left:q_right] = self.var[n0_right:,n1_right:,n2_left:n2_right].sum(axis=0).sum(axis=0).reshape(cub.shape[2],factor[2]).sum(1) / F2
                if l_right==(newshape[0]-1) and q_right==(newshape[2]-1):
                    d = self.data[n0_right:,n1_left:n1_right,n2_right:].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F
                    data[-1,p_left:p_right,-1] = d.data
                    mask[-1,p_left:p_righ,-1] = d.mask
                    if var is not None:
                        var[-1,p_left:p_righ,-1] = self.var[n0_right:,n1_left:n1_right,n2_right:].sum(axis=2).sum(axis=0).reshape(cub.shape[1],factor[1]).sum(1) / F2
                
                self.shape = newshape
                self.wcs = wcs
                self.wave = wave
                self.data = np.ma.array(data, mask=mask)
                self.var = var
                return None
                
    def rebin_factor(self, factor, margin='center', flux=False):
        '''Shrinks the size of the cube by factor.
  
          :param factor: Factor in z, y and x. Python notation: (nz,ny,nx).
          :type factor: integer or (integer,integer,integer)
          :param flux: This parameters is used if new size is not an integer multiple of the original size.
          
              If Flux is False, the cube is truncated and the flux is not conserved.
              
              If Flux is True, margins are added to the cube to conserve the flux.
          :type flux: bool
          :param margin: This parameters is used if new size is not an integer multiple of the original size. 
  
            In 'center' case, cube is truncated/pixels are added on the left and on the right, on the bottom and of the top of the cube. 
        
            In 'origin'case, cube is truncated/pixels are added at the end along each direction
          :type margin: 'center' or 'origin'   
        '''
        res = self.copy()
        res._rebin_factor(factor, margin, flux)
        return res
    
    def _med_(self,k,p,q,kfactor,pfactor,qfactor):
        return np.ma.median(self.data[k*kfactor:(k+1)*kfactor,p*pfactor:(p+1)*pfactor,q*qfactor:(q+1)*qfactor])
    
    def _rebin_median_(self, factor):
        '''Shrinks the size of the cube by factor.
        New size is an integer multiple of the original size.
        
        Parameter
        ----------
        factor : (integer,integer,integer)
        Factor in z, y and x.
        Python notation: (nz,ny,nx)
        '''
        # new size is an integer multiple of the original size
        assert not np.sometrue(np.mod( self.shape[0], factor[0] ))
        assert not np.sometrue(np.mod( self.shape[1], factor[1] ))
        assert not np.sometrue(np.mod( self.shape[2], factor[2] ))
        #shape
        self.shape = np.array((self.shape[0]/factor[0],self.shape[1]/factor[1],self.shape[2]/factor[2]))
        #data
        grid = np.lib.index_tricks.nd_grid()
        g = grid[0:self.shape[0],0:self.shape[1],0:self.shape[2]]
        vfunc = np.vectorize(self._med_)
        data = vfunc(g[0],g[1],g[2],factor[0],factor[1],factor[2])
        mask = self.data.mask.reshape(self.shape[0],factor[0],self.shape[1],factor[1],self.shape[2],factor[2]).sum(1).sum(2).sum(3)
        self.data = np.ma.array(data, mask=mask)
        #variance
        self.var = None
        #coordinates
        cdelt = self.wcs.get_step()
        self.wcs = self.wcs.rebin_factor(factor[1:])
        crval = self.wave.coord()[0:factor[0]].sum()/factor[0]
        self.wave = WaveCoord(1, self.wave.cdelt*factor[0], crval, self.wave.cunit,self.shape[0])
 
    def rebin_median(self, factor, margin='center'):
        '''Shrinks the size of the cube by factor.
  
          :param factor: Factor in z, y and x. Python notation: (nz,ny,nx).
          :type factor: integer or (integer,integer,integer)
          :param margin: This parameters is used if new size is not an integer multiple of the original size. 
  
            In 'center' case, cube is truncated on the left and on the right, on the bottom and of the top of the cube. 
        
            In 'origin'case, cube is truncatedat the end along each direction
          :type margin: 'center' or 'origin'  
          :rtype: :class:`mpdaf.obj.Cube`
        '''
        if is_int(factor):
            factor = (factor,factor,factor)
        if factor[0]<1 or factor[0]>=self.shape[0] or factor[1]<1 or factor[1]>=self.shape[1] or factor[2]<1 or factor[2]>=self.shape[2]:
            raise ValueError, 'factor must be in ]1,shape['
            return None
        if not np.sometrue(np.mod( self.shape[0], factor[0] )) and not np.sometrue(np.mod( self.shape[1], factor[1] )) and not np.sometrue(np.mod( self.shape[2], factor[2] )):
            # new size is an integer multiple of the original size
            res = self.copy()
        else:
            factor = np.array(factor)
            newshape = self.shape/factor
            n = self.shape - newshape*factor
            
            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0]==1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0]/2
                    n0_right = self.shape[0] - n[0] + n0_left
            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1]==1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1]/2
                    n1_right = self.shape[1] - n[1] + n1_left
            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2]==1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2]/2
                    n2_right = self.shape[2] - n[2] + n2_left
            
            res = self[n0_left:n0_right,n1_left:n1_right,n2_left:n2_right]
            
        res._rebin_median_(factor)
        return res
          
    def loop_spe_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """loops over all spectra to apply a function/method.
        Returns the resulting cube.
        Multiprocessing is used.
        
        :param f: Spectrum method or function that the first argument is a spectrum object. 
        :type f: function or :class:`mpdaf.obj.Spectrum` method
        :param cpu: number of CPUs. It is also possible to set the mpdaf.CPU global variable.
        :type cpu: integer
        :param verbose: if True, progression is printed.
        :type verbose: boolean
        :param kargs: kargs can be used to set function arguments.
        
        :rtype: :class:`mpdaf.obj.Cube` if f returns :class:`mpdaf.obj.Spectrum`,
            
                :class:`mpdaf.obj.Image` if f returns a number,
                
                np.array(dtype=object) in others cases. 
        """
        from mpdaf import CPU
        if cpu is not None and cpu<multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU<multiprocessing.cpu_count():
            cpu_count = cpu
        else:
            cpu_count = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
           
        if isinstance (f, types.MethodType):
            f = f.__name__
        
        for sp,pos in iter_spe(self, index=True):
            processlist.append([sp,pos,f,kargs])
        num_tasks = len(processlist)
       
        processresult = pool.imap_unordered(_process_spe,processlist)
        pool.close()
        
        if verbose:
            print "loop_spe_multiprocessing (%s): %i tasks"%(f,num_tasks)
            import time
            import sys
            while (True):
                time.sleep(5)
                completed = processresult._index
                if completed == num_tasks: 
                    output = ""
                    sys.stdout.write("\r\x1b[K"+output.__str__())
                    break
                output = "\r Waiting for %i tasks to complete (%i%% done) ..." %(num_tasks-completed, float(completed)/float(num_tasks)*100.0)
                sys.stdout.write("\r\x1b[K"+output.__str__())
                sys.stdout.flush()
            
        init = True        
        for pos,out in processresult:
            if is_float(out) or is_int(out):
                # f returns a number -> iterator returns an image
                if init:
                    from image import Image
                    result = Image(wcs=self.wcs.copy(),data=np.zeros(shape=(self.shape[1],self.shape[2])),unit=self.unit)
                    init = False       
                p,q = pos
                result[p,q] = out
            else:         
                try:
                    if out.spectrum:
                    #f returns a spectrum -> iterator returns a cube
                        if init:
                            if self.var is None:
                                result = Cube(wcs=self.wcs.copy(),wave=out.wave.copy(),data=np.zeros(shape=(out.shape,self.shape[1],self.shape[2])),unit=self.unit)
                            else:
                                result = Cube(wcs=self.wcs.copy(),wave=out.wave.copy(),data=np.zeros(shape=(out.shape,self.shape[1],self.shape[2])),var=np.zeros(shape=(out.shape,self.shape[1],self.shape[2])),unit=self.unit)
                            init = False               
                        p,q = pos
                        result[:,p,q] = out
                        
                except:
                    #f returns dtype -> iterator returns an array of dtype
                    if init:
                        result = np.empty((self.shape[1],self.shape[2]),dtype=type(out))
                        init = False  
                    p,q = pos
                    result[p,q] = out
                
        return result    
        
    
    def loop_ima_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """loops over all images to apply a function/method.
        Returns the resulting cube.
        Multiprocessing is used.
        
        :param f: Image method or function that the first argument is a Image object. It should return an Image object. 
        :type f: function or :class:`mpdaf.obj.Image` method
        :param cpu: number of CPUs. It is also possible to set the mpdaf.CPU global variable.
        :type cpu: integer
        :param verbose: if True, progression is printed.
        :type verbose: boolean
        :param kargs: kargs can be used to set function arguments.
        :rtype: :class:`mpdaf.obj.Cube` if f returns :class:`mpdaf.obj.Image`,
        
                :class:`mpdaf.obj.Spectrum` if f returns a number,
                
                 np.array(dtype=object) in others cases.
        """
        from mpdaf import CPU
        if cpu is not None and cpu<multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU<multiprocessing.cpu_count():
            cpu_count = cpu
        else:
            cpu_count = multiprocessing.cpu_count() - 1
        
        pool = multiprocessing.Pool(processes = cpu_count)
        processlist = list()
           
        if isinstance (f, types.MethodType):
            f = f.__name__
        
        for ima,k in iter_ima(self, index=True):
            header = ima.wcs.to_header()
            processlist.append([ima,k,f,header,kargs])
        num_tasks = len(processlist)
       
        processresult = pool.imap_unordered(_process_ima,processlist)
        pool.close()
        
        if verbose:
            print "loop_ima_multiprocessing (%s): %i tasks"%(f,num_tasks)
            import time, sys
            while (True):
                time.sleep(5)
                completed = processresult._index
                if completed == num_tasks:
                    output = ""
                    sys.stdout.write("\r\x1b[K"+output.__str__())
                    break
                output = "\r Waiting for %i tasks to complete (%i%% done) ..." %(num_tasks-completed, float(completed)/float(num_tasks)*100.0)
                sys.stdout.write("\r\x1b[K"+output.__str__())
                sys.stdout.flush()
            
        init = True        
        for k,out,h in processresult:  
            if is_float(out) or is_int(out):
            # f returns a number -> iterator returns a spectrum
                if init:
                    from spectrum import Spectrum
                    result = Spectrum(wave=self.wave.copy(),data=np.zeros(shape=self.shape[0]),unit=self.unit)
                    init = False
                result[k] = out
            else:
                try:
                    if out.image:
                    #f returns an image -> iterator returns a cube
                        if init:
                            wcs = WCS(h)
                            wcs.set_naxis1(out.shape[1])
                            wcs.set_naxis2(out.shape[0])
                            if self.var is None:
                                result = Cube(wcs=wcs,wave=self.wave.copy(),data=np.zeros(shape=(self.shape[0],out.shape[0],out.shape[1])),unit=self.unit)
                            else:
                                result = Cube(wcs=wcs,wave=self.wave.copy(),data=np.zeros(shape=(self.shape[0],out.shape[0],out.shape[1])),var=np.zeros(shape=(self.shape[0],out.shape[0],out.shape[1])),unit=self.unit)
                            init = False
                        result[k,:,:] = out
                except:
                #f returns dtype -> iterator returns an array of dtype
                    if init:
                        result = np.empty(self.shape[0],dtype=type(out))
                        init = False
                    result[k] = out
                    
        return result
                
   
def _process_spe(arglist):
    try:
        obj = arglist[0]
        pos = arglist[1]
        f = arglist[2]
        kargs = arglist[3]
        if isinstance (f,types.FunctionType):
            obj_result = f(obj,**kargs)
        else:
            obj_result = getattr(obj, f)(**kargs)  
        
        return (pos,obj_result)
    except Exception as inst:
        raise type(inst) , str(inst) + '\n The error occurred for the spectrum [:,%i,%i]'%(pos[0],pos[1])

def _process_ima(arglist):
    try:
        obj = arglist[0]
        #bug multiprocessing & pywcs
        obj.wcs = WCS(arglist[3])
        obj.wcs.set_naxis1(obj.shape[1])
        obj.wcs.set_naxis2(obj.shape[0])
    
        pos = arglist[1]
        f = arglist[2]
        kargs = arglist[4]
        if isinstance (f,types.FunctionType):
            obj_result = f(obj,**kargs)
        else:
            obj_result = getattr(obj, f)(**kargs)  
        try:
            header = obj_result.wcs.to_header()
        except:
            header = None
    
        return (pos,obj_result,header)
    except Exception as inst:
        raise type(inst) , str(inst) + '\n The error occurred for the image [%i,:,:]'%pos
