from astropy.io import fits as pyfits
from astropy.table import Table, Column

import datetime
import logging
import numpy as np
import os.path
import shutil
import warnings

from ..obj import Cube, Image, Spectrum
from ..obj.objs import is_int
from .catalog import Catalog

emlines = {1215.67: 'Lyalpha',
           1550.0: 'CIV',
           1909.0: 'CIII]',
           2326.0: 'CII',
           3726.032: '[OII]',
           3728.8149: '[OII]2',
           3798.6001: 'Htheta',
           3834.6599: 'Heta',
           3869.0: '[NeIII]3',
           3888.7: 'Hzeta',
           3967.0: '[NeIII]2',
           4102.0: 'Hdelta',
           4340.0: 'Hgamma',
           4861.3198: 'Hbeta',
           4959.0: '[OIII]',
           5007.0: '[OIII]2',
           6548.0: '[NII]',
           6562.7998: 'Halpha',
           6583.0: '[NII]',
           6716.0: '[SII]',
           6731.0: '[SII]2'}


def matchlines(nlines, wl, z, eml):
    """ try to match all the lines given : 
    for each line computes the distance in Angstroms to the closest line.
    Add the errors
    
    Algorithm from Johan Richard (johan.richard@univ-lyon1.fr)
     
     Parameters
     ----------
     nlines : integer
              Number of emission lines
     wl     : array<double>
              Table of wavelengths
     z      : double
              Redshift to test
     eml    : dict
              Full catalog of lines to test redshift
              key: wavelength, value: name
              
    Returns
    -------
    out : (array<double>, array<double>)
          (list of wavelengths, errors)
              
    """
    jfound = np.zeros(nlines, dtype=np.int)
    lbdas = np.array(eml.keys())
    error = 0
    for i in range(nlines):
        # finds closest emline to this line
        jfound[i] = np.argmin((wl[i] / (1 + z) - lbdas) ** 2.0)
        error += (wl[i] / (1 + z) - lbdas[jfound[i]]) ** 2.0
    error = np.sqrt(error / nlines)
    if((nlines >= 2)and(jfound[0] == jfound[1])):
        error = 15.
    return(error, jfound)


def crackz(nlines, wl, flux, eml):
    """Method to estimate the best redshift matching a list of emission lines
     
     Parameters
     ----------
     nlines : integer
              Number of emission lines
     wl     : array<double>
              Table of observed line wavelengths
     flux   : array<double>
              Table of line fluxes
     eml    : dict
              Full catalog of lines to test redshift
              
    Algorithm from Johan Richard (johan.richard@univ-lyon1.fr)
    
    Returns
    -------
    out : (float, float, integer, list<double>, list<double>, list<string>)
          (redshift, redshift error, list of wavelengths, list of fluxes, list of lines names)
    """
    errmin = 3.0
    zmin = 0.0
    zmax = 7.0
    if(nlines == 0):
        return -9999.0, -9999.0, 0, [], [], []
    if(nlines == 1):
        return -9999.0, -9999.0, 1, wl, flux, ["Lya or [OII]"]
    if(nlines > 1):
        found = 0
        lbdas = np.array(eml.keys())
        lnames = np.array(eml.values())
        for z in np.arange(zmin, zmax, 0.001):
            (error, jfound) = matchlines(nlines, wl, z, eml)
            if(error < errmin):
                errmin = error
                found = 1
                zfound = z
                jfinal = jfound.copy()
        if(found == 1):
            jfinal = np.array(jfinal).astype(int)
            return zfound, errmin / np.min(lbdas[jfinal]), nlines, \
                wl, flux, list(lnames[jfinal[0:nlines]])
        else:
            if(nlines > 3):
                # keep the three brightest
                ksel = np.argsort(flux)[-1:-4:-1]
                return crackz(3, wl[ksel], flux[ksel], eml)
            if(nlines == 3):
                # keep the two brightest
                ksel = np.argsort(flux)[-1:-3:-1]
                return crackz(2, wl[ksel], flux[ksel], eml)
            if(nlines == 2):
                # keep the brightest
                ksel = np.argsort(flux)[-1]
                return -9999.0, -9999.0, 1, [wl[ksel]], [flux[ksel]], \
                    ["Lya or [OII]"]

class Source(object):
    """This class describes an object.

    Attributes
    ----------
    header  : pyfits.Header
              FITS header instance
    lines   : astropy.Table
              List of lines 
    mag     : astropy.Table
              List of magnitudes
    z       : astropy.Table
              List of redshifts
    spectra : :class:`dict`
              Dictionary containing spectra.
              
              Keys gives the origin of the spectrum
              ('tot' for total spectrum, TBC).
              
              Values are :class:`mpdaf.obj.Spectrum` object
    images  : :class:`dict`
              Dictionary containing images.
              
              Keys gives the filter ('SRC_WHITE' for white image, TBC)
              
              Values are :class:`mpdaf.obj.Image` object
    cubes   : :class:`dict`
                  Dictionary containing small data cubes
                  
                  Keys gives a description of the cube
                  
                  Values are :class:`mpdaf.obj.Cube` objects
    """
    
    def __init__(self, header, lines=None, mag=None, z=None,
                 spectra=None, images=None, cubes=None):
        """
        """
        # FITS header
        if not ('RA' in header and 'DEC' in header 
                and 'ID' in header and 'CUBE' in header
                and 'ORIGIN' in header and 'ORIGIN_V' in header):
            raise IOError('ID, RA, DEC, ORIGIN, ORIGIN_V and CUBE are \
            mandatory parameters to create a Source object') 
        self.header = header
        # Table LINES
        self.lines = lines
        # Table MAG
        self.mag = mag
        # Table Z
        self.z = z
        # Dictionary SPECTRA
        if spectra is None:
            self.spectra = {}
        else:
            self.spectra = spectra
        # Dictionary IMAGES
        if images is None:
            self.images = {}
        else:
            self.images = images
        # Dictionary CUBES
        if cubes is None:
            self.cubes = {}
        else:
            self.cubes = cubes
        # logger
        self.logger = logging.getLogger('mpdaf corelib')

    @classmethod
    def from_data(cls, ID, ra, dec, origin, proba=None, confi=None, extras=None,
                 lines=None, mag=None, z=None,
                 spectra=None, images=None, cubes=None):
        """
        Parameters
        ----------
        ID      : integer
                  ID of the source
        ra      : double
                  Right ascension in degrees
        dec     : double
                  Declination in degrees
        origin  : tuple (string, string, string)
                  1- Name of the detector software which creates this object
                  2- Version of the detector software which creates this object
                  3- Name of the FITS data cube from which this object has been extracted.
        proba   : float
                  Detection probability
        confi   : integer
                  Expert confidence index
        extra   : dict{key: value} or dict{key: (value, comment)}
                  Extra keywords
        lines   : astropy.Table
                  List of lines
        mag     : astropy.Lines
                  List of magnitudes.
        z       : astropy.Table
                  List of redshifts
        spectra : :class:`dict`
                  Dictionary containing spectra.
              
                  Keys gives the origin of the spectrum
                  ('tot' for total spectrum, TBC).
              
                  Values are :class:`mpdaf.obj.Spectrum` object
        images  : :class:`dict`
                  Dictionary containing small images.
                
                  Keys gives the filter ('SRC_WHITE' for white image, TBC)
              
                  Values are :class:`mpdaf.obj.Image` object
        cubes   : :class:`dict`
                  Dictionary containing small data cubes
                  
                  Keys gives a description of the cube
                  
                  Values are :class:`mpdaf.obj.Cube` objects
        """
        header = pyfits.Header()
        header['ID'] = (ID, 'object ID')
        header['RA'] = (np.float32(ra), 'RA in degrees')
        header['DEC'] = (np.float32(dec), 'DEC in degrees')
        header['ORIGIN'] = (origin[0], 'detection software')
        header['ORIGIN_V'] = (origin[1], 'version of the detection software')
        header['CUBE'] = (os.path.basename(origin[2]), 'MUSE data cube')
        if proba is not None:
            header['DPROBA'] = (np.float32(proba), 'Detection probability')
        if confi is not None:
            header['CONFI'] = (confi, 'Confidence index')
        if extras is not None:
            for key, value in extras.iteritems():
                header[key] = value
            
        return cls(header, lines, mag, z, spectra, images, cubes)
       
            
    @classmethod
    def from_file(cls, filename):
        """
        Parameters
        ----------
        filename : string
                   FITS filename
        """
        hdulist = pyfits.open(filename)
        hdr = hdulist[0].header
        lines = None
        mag = None
        z = None
        spectra = {}
        images = {}
        cubes = {}
        for i in range(1, len(hdulist)):
            hdu = hdulist[i]
            extname = hdu.header['EXTNAME']
            #lines
            if extname == 'LINES':
                lines = Table(hdu.data)
            # mag
            if extname == 'MAG':
                mag = Table(hdu.data)
            # Z
            if extname == 'Z':
                z = Table(hdu.data)
            # spectra
            elif extname[:3] == 'SPE' and extname[-4:]=='DATA':
                spe_name = extname[4:-5]
                try:
                    ext_var = hdulist.index_of('SPE_'+spe_name+'_STAT')
                    ext = (i, ext_var)
                except:
                    ext = i
                spectra[spe_name] = Spectrum(filename, ext=ext)
            #images
            elif extname[:3] == 'IMA' and extname[-4:]=='DATA':
                ima_name = extname[4:-5]
                try:
                    ext_var = hdulist.index_of('IMA_'+ima_name+'_STAT')
                    ext = (i, ext_var)
                except:
                    ext = i
                images[ima_name] = Image(filename, ext=ext)
            elif extname[:3] == 'CUB' and extname[-4:]=='DATA':
                cub_name = extname[4:-5]
                try:
                    ext_var = hdulist.index_of('CUB_'+cub_name+'_STAT')
                    ext = (i, ext_var)
                except:
                    ext = i
                cubes[cub_name] = Cube(filename, ext=ext, ima=False)
        hdulist.close()
        return cls(hdr, lines, mag, z, spectra, images, cubes)
                               
        
    def write(self, filename):
        """writes source in a FITS file
        
        Parameters
        ----------
        filename : string
                   FITS filename
        """
        warnings.simplefilter("ignore")
        # create primary header
        prihdu = pyfits.PrimaryHDU(header=self.header)
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')

        hdulist = [prihdu]

        # lines
        if self.lines is not None:
            tbhdu = pyfits.BinTableHDU(name='LINES', data=np.array(self.lines))
            hdulist.append(tbhdu)
            
        # magnitudes
        if self.mag is not None:
            tbhdu = pyfits.BinTableHDU(name='MAG', data=np.array(self.mag))
            hdulist.append(tbhdu)
            
        # redshifts
        if self.z is not None:
            tbhdu = pyfits.BinTableHDU(name='Z', data=np.array(self.z))
            hdulist.append(tbhdu)
        
        #spectra
        for key, spe in self.spectra.iteritems():
            ext_name = 'SPE_%s_DATA'%key
            data_hdu = spe.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'SPE_%s_STAT'%key
            stat_hdu = spe.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)
            
        #images
        for key, ima in self.images.iteritems():
            ext_name = 'IMA_%s_DATA'%key
            data_hdu = ima.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'IMA_%s_STAT'%key
            stat_hdu = ima.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)
        
        #cubes
        for key, cub in self.cubes.iteritems():
            ext_name = 'CUB_%s_DATA'%key
            data_hdu = cub.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'CUB_%s_STAT'%key
            stat_hdu = cub.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)
            
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True, output_verify='fix')
        warnings.simplefilter("default")
        
    def info(self):
        """Prints information.
        """
        d = {'class': 'Source', 'method': 'info'}
        for l in repr(self.header).split('\n'):
            if l.split()[0] != 'SIMPLE' and l.split()[0] != 'BITPIX' and \
            l.split()[0] != 'NAXIS' and l.split()[0] != 'EXTEND' and \
            l.split()[0] != 'DATE' and l.split()[0] != 'AUTHOR':
                self.logger.info(l, extra=d)
        print '\n'
        for key, spe in self.spectra.iteritems():
            self.logger.info('spectra[\'%s\']'%key, extra=d)
            spe.info()
            print '\n'
        for key, ima in self.images.iteritems():
            self.logger.info('images[\'%s\']'%key, extra=d)
            ima.info()
            print '\n'
        for key, cub in self.cubes.iteritems():
            self.logger.info('cubes[\'%s\']'%key, extra=d)
            cub.info()
            print '\n'
        if self.lines is not None:
            self.logger.info('lines', extra=d)
            for l in self.lines.pformat():
                self.logger.info(l, extra=d)
            print '\n'
        if self.mag is not None:
            self.logger.info('magnitudes', extra=d)
            for l in self.mag.pformat():
                self.logger.info(l, extra=d)
            print '\n'
        if self.z is not None:
            self.logger.info('redshifts', extra=d)
            for l in self.z.pformat():
                self.logger.info(l, extra=d)
            print '\n'

    def __getattr__(self, item):
        """Maps values to attributes.
        """
        try:
            return self.header[item]
        except KeyError:
            raise AttributeError(item)
 
    def __setattr__(self, item, value):
        """Maps attributes to values.
        """
        if item=='header' or item=='logger' or \
           item=='lines' or item=='mag' or item=='z' or \
           item=='cubes' or item=='images'or item=='spectra':
            return dict.__setattr__(self, item, value)
        else:
            self.header[item] = value
            
    def add_comment(self, comment, author):
        i = 1
        while 'COMMENT%03d'%i in self.header:
            i += 1
        self.header['COMMENT%03d'%i] = (comment, '%s %s'%(author, str(datetime.date.today())))
    
    def remove_comment(self, ncomment):
        del self.header['COMMENT%03d'%ncomment]
        
    def add_image(self, image, name, size=None):
        """ Extracts a image centered on the source center
        and appends it to the image dictionary.
        
        Parameters
        ----------
        image : :class:`mpdaf.obj.Image`
                Input image MPDAF object
        name  : string
                Name used to distingish this image
        size  : float or (float, float)
                Size of the image in arcsec.
                If None, the size of the white image extension is taken if it exists.
        """
        if size is None:
            white_ima = self.images['SRC_WHITE']
            size = np.abs(white_ima.get_step() * white_ima.shape)*3600.0
        else:
            if is_int(size):
                size = (size, size)
        
        size = np.array(size)
        radius = size/2./3600.0
        radius_ra = radius[1] / np.cos(np.deg2rad(self.dec))
        ra_min = self.ra - radius_ra
        ra_max = self.ra + radius_ra
        dec_min = self.dec - radius[0]
        dec_max = self.dec + radius[0]
        subima = image.truncate(dec_min, dec_max, ra_min, ra_max, mask=False)
        self.images[name] = subima
        
    def add_z(self, desc, z, errz):
        """Adds a redshift value
        
        Parameters
        ----------
        desc : string
               Redshift description.
        z    : float
               Redshidt value.
        errz : float
               Redshift error.
        """
        if self.z is None:
            self.z = Table(names=['Z_DESC', 'Z', 'Z_ERR'],
                           rows=[[desc, z, errz]],
                           dtype=('S20', 'f6', 'f6'))
            self.z['Z'].format = '%.6f'
            self.z['Z_ERR'].format = '%.6f'
        else:
            if desc in self.z['Z_DESC']:
                self.z['Z'][self.z['Z_DESC']==desc] = z
                self.z['Z_ERR'][self.z['Z_DESC']==desc] = errz
            else:
                self.z.add_row([desc, z, errz])
                
    def add_mag(self, band, m, errm):
        """Adds a magnitude value
        
        Parameters
        ----------
        band : string
               Filter name.
        m    : float
               Magnitude value.
        errm : float
               Magnitude error.
        """
        if self.mag is None:
            self.mag = Table(names=['BAND', 'MAG', 'MAG_ERR'],
                           rows=[[band, m, errm]],
                           dtype=('S20', 'f6', 'f6'))
            self.mag['MAG'].format = '%.6f'
            self.mag['MAG_ERR'].format = '%.6f'
        else:
            if band in self.mag['BAND']:
                self.mag['MAG'][self.mag['BAND']==band] = m
                self.mag['MAG_ERR'][self.mag['BAND']==band] = errm
            else:
                self.mag.add_row([band, m, errm])
                

    def crack_z(self, eml=None, nlines=np.inf):
        """Method to estimate the best redshift matching a list of emission lines
     
         Parameters
         ----------
         eml    : dict{float: string}
                  Full catalog of lines to test redshift
                  Dictionary: key is the wavelength value in Angtsrom,
                  value is the name of the line.
                  if None, default catalog is used.
          nlines  : integer
                    estimated the redshift if the list of emission lines is inferior to this value
              
        Algorithm from Johan Richard (johan.richard@univ-lyon1.fr)
        """
        nline_max = nlines
        if eml is None:
            eml = emlines
            
        wl = np.array(self.lines['LBDA_OBS'])
        flux = np.array(self.lines['FLUX'])
        nlines = len(wl)
        
        z, errz, nlines, wl, flux, lnames = crackz(nlines, wl, flux, eml)
        
        if nlines > 0:
            if nlines < nline_max:
                #redshift
                self.add_z('EMI', z, errz)
                #line names
                if 'LINE' not in self.lines.colnames:
                    col = Column(data=None, name='LINE', dtype='S20', length=len(self.lines))
                    self.lines.add_column(col)
                for w, name in zip(wl, lnames):
                    self.lines['LINE'][self.lines['LBDA_OBS']==w] = name
        
class SourceList(list):
    """
        list< :class:`mpdaf.sdetect.Source` >
    """
    
    def write(self, name, path='.', overwrite=True):
        """ Creates a directory named name and saves all sources files and the catalog file in this folder.
        path/name.fits: catalog file
        path/name/nameNNNN.fits: source file (NNNN corresponds to the ID of the source)
        
        Parameters
        ----------
        name : string
               Name of the catalog
        path : string
               path where the catalog will be saved.
        overwrite : boolean
                    Overwrite the catalog if it already exists
        """
        if not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))

        path = os.path.normpath(path)
        
        path2 = path + '/' + name
        if not os.path.exists(path2):
            os.makedirs(path2)
        else:
            if overwrite:
                shutil.rmtree(path2)
                os.makedirs(path2)
                
        for source in self:
            source.write('%s/%s-%04d.fits'%(path2, name, source.ID))
         
        fcat = '%s/%s.fits'%(path, name)
        if overwrite and os.path.isfile(fcat) :
            os.remove(fcat)
        cat = Catalog.from_sources(self)
        cat.write(fcat)
        # For FITS tables, the maximum number of fields is 999
    