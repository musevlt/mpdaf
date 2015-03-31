from astropy.io import fits as pyfits
from astropy.io.fits import Column, ImageHDU, Card
import datetime
import numpy as np
import warnings

class Line(object):
    """ This class describes a line source.
    
    Attributes
    ----------
    ID           : string
                   Name
    lbda_rest    : float
                   Rest wavelength (in Angstrom)
    lbda_obs     : float
                   Observed line wavelength (in Angstrom)
    err_lbda_obs : float
                   Observed wavelength error (in Angstrom)
    width        : float
                   Width of the line
    err_width    : float
                   Width error
    flux         : float
                   Flux value
    err_flux     : float
                   Flux error
    ew           : float
                   Equivalent width
    err_ew       : float
                   Equivalent width error
    name         : string
                   Name
    """
    
    def __init__(self, ID, lbda_rest, name, lbda_obs, err_lbda_obs, width, err_width, flux, err_flux, ew, err_ew):
        self.ID = ID
        self.lbda_rest = lbda_rest
        self.name = name
        self.lbda_obs = lbda_obs
        self.err_lbda_obs = err_lbda_obs
        self.width = width
        self.err_width = err_width
        self.flux = flux
        self.err_flux = err_flux
        self.ew = ew
        self.err_ew = err_ew

    def obs2rest(self, z):
        pass
    
    def rest2obs(self, z):
        pass


class Source(object):
    """This class describes an object.

    Attributes
    ----------
    ID      : integer
              ID of the source
    ra      : double
              Right ascension in degrees
    dec     : double
              Declination in degrees
    origin  : string
              Name of detector software which creates this object
    lines   : list of lines
              List of :class:`mpdaf.sdetect.Line`
    spe     : :class:`dict`
              Dictionary containing spectra.
              
              Keys gives the origin of the spectrum
              ('tot' for total spectrum, TBC).
              
              Values are :class:`mpdaf.obj.Spectrum` object
    ima     : :class:`dict`
              Dictionary containing images.
              
              Keys gives the filter ('white' for white image, TBC)
              
              Values are :class:`mpdaf.obj.Image` object
    cub     : :class:`mpdaf.obj.Cube`
              sub-data cube containing the object
    z       : float
              redshift
    errz    : float
              redshift error
    flag    : string
              Quality flag
    comment : string
              Users comments
              
    selfi: fwhmx, fwhmy, angle
    """

    def __init__(self, filename=None, ID=None, ra=None, dec=None, origin=None,
                 lines=[], spe={}, ima={}, cub=None, z=None, errz=None,
                 flag='', comment='', extras={}):
        """
    Parameters
    ----------
    ID      : integer
              ID of the source
    ra      : double
              Right ascension in degrees
    dec     : double
              Declination in degrees
    lines   : list of lines
              List of :class:`mpdaf.sdetect.Line`
    spe     : :class:`dict`
              Dictionary containing spectra.
              
              Keys gives the origin of the spectrum
              ('tot' for total spectrum, TBC).
              
              Values are :class:`mpdaf.obj.Spectrum` object
    ima     : :class:`dict`
              Dictionary containing images.
              
              Keys gives the filter ('white' for white image, TBC)
              
              Values are :class:`mpdaf.obj.Image` object
    cub     : :class:`mpdaf.obj.Cube`
              sub-data cube containing the object
    origin  : string
              Name of detector software which creates this object
    flag    : string
              Quality flag
    comment : string
              Users comments
    extra keywords : dict{key: value} or dict{key: (value, comment)}
        """
        if filename is not None:
            self.read(filename)
        else:
            if ID is None or ra is None or dec is None or origin is None:
                raise IOError('ID, ra, dec, origin are mandatory parameters to create a Source object') 
            self.ID = ID
            self.ra = ra
            self.dec = dec
            self.origin = origin
            self.lines = lines
            self.spe = spe
            self.ima = ima
            self.cub = cub
            self.flag = flag
            self.comment = comment
            self.z = z
            self.errz = errz
            self.extras = pyfits.Header()
            for key, value in extras.iteritems():
                self.extras[key] = value
            
            
            #self.filename = ''   
        
    def get_extra(self, key):
        return self.extras[key]
    
    #def set                                
        
    def write(self, filename):
        """
        """
        warnings.simplefilter("ignore")
        # create primary header
        prihdu = pyfits.PrimaryHDU()
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        prihdu.header['origin'] = (self.origin, 'detector software which creates this object')
        prihdu.header['flag'] = (self.flag, 'quality flag')
        prihdu.header['comment'] = (self.comment, 'user comment')
        prihdu.header['ra'] = (np.float32(self.ra), 'RA in degrees')
        prihdu.header['dec'] = (np.float32(self.dec), 'DEC in degrees')
        hdulist = [prihdu]

        #lines
        ID = np.int32([line.ID for line in self.lines])
        lbda_rest = np.float32([line.lbda_rest for line in self.lines])
        name = np.array([line.name for line in self.lines], dtype=np.dtype('a20'))
        lbda_obs = np.float32([line.lbda_obs for line in self.lines])
        err_lbda_obs = np.float32([line.err_lbda_obs for line in self.lines])
        width = np.float32([line.width for line in self.lines])
        err_width = np.float32([line.err_width for line in self.lines])
        flux = np.float32([line.flux for line in self.lines])
        err_flux = np.float32([line.err_flux for line in self.lines])
        ew = np.float32([line.ew for line in self.lines])
        err_ew = np.float32([line.err_ew for line in self.lines])
        
        cols = [Column(name='ID', format='1J', unit='', array=ID),
                Column(name='LBDA_REST', format='1E', unit='Angstrom',
                       array=lbda_rest),
                Column(name='NAME', format='A20', unit='', array=name),
                Column(name='LBDA_OBS', format='1E', unit='Angstrom',
                       array=lbda_obs),
                Column(name='ERR_LBDA_OBS', format='1E', unit='Angstrom',
                       array=err_lbda_obs),
                Column(name='WIDTH', format='1E', unit='Angstrom',
                       array=width),
                Column(name='ERR_WIDTH', format='1E', unit='Angstrom',
                       array=err_width),
                Column(name='FLUX', format='1E', unit='10**(-20)*erg/s/cm**2',
                       array=flux),
                Column(name='ERR_FLUX', format='1E', unit='10**(-20)*erg/s/cm**2',
                       array=err_flux),
                Column(name='EW', format='1E', unit='Angstrom', array=ew),
                Column(name='ERR_EW', format='1E', unit='Angstrom', array=err_ew),]

        coltab = pyfits.ColDefs(cols)
        tbhdu = pyfits.TableHDU(name='LINES', data=pyfits.FITS_rec.from_columns(coltab))
        hdulist.append(tbhdu)
        
        #spectra
        for key, spe in self.spe.iteritems():
            ext_name = 'SPE_%s_DATA'%key
            data_hdu = spe.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'SPE_%s_STAT'%key
            stat_hdu = spe.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)
            
        #images
        for key, ima in self.ima.iteritems():
            ext_name = 'IMA_%s_DATA'%key
            data_hdu = ima.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'IMA_%s_STAT'%key
            stat_hdu = ima.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)
        
        #cube
        if self.cub is not None:
            ext_name = 'CUBE_DATA'
            data_hdu = self.cub.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'CUBE_STAT'
            stat_hdu = self.cub.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)
            
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True, output_verify='fix')
        warnings.simplefilter("default")
        
        #self.filename = filename
    
    def read(self, filename):
        #self.filename = filename
        pass
    
        
    def info(self):
        """Prints information.
        """
        pass
    
    def plot(self, bima):
        """Draws ellipse contours on the background image specified by the user
        
        Parameters
        ----------
        bima : string
               Image FITS filename
        """
        pass
    
    def crackz(self):
        """Method to estimate the best redshift matching a list of emission lines
        
        Parameters
        ----------
        eml  : :class:`dict`
               Full catalog of lines to test redshift
        eml2 : :class:`dict`
               Smaller catalog containing only the brightest lines to test
              
        Returns
        -------
        out : (float, float, list<double>, list<string>)
              (redshift, redshift error, list of wavelengths, list of lines names)
        """
        pass