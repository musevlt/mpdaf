from astropy.io import fits as pyfits
from astropy.table import Table

import datetime
import numpy as np
import warnings
import logging
import os.path
import shutil

from ..obj import Cube, Image, Spectrum
from .catalog import Catalog

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
    cube    : :class:`mpdaf.obj.Cube`
              sub-data cube containing the object
    """
    
    def __init__(self, header, lines=None, mag=None, z=None,
                 spectra=None, images=None, cube=None):
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
        # CUBE
        self.cube = cube
        # logger
        self.logger = logging.getLogger('mpdaf corelib')

    @classmethod
    def from_data(cls, ID, ra, dec, origin, proba=None, confi=None, extras=None,
                 lines=None, mag=None, z=None,
                 spectra=None, images=None, cube=None):
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
        cube    : :class:`mpdaf.obj.Cube`
                  Small data cube containing the object
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
            
        return cls(header, lines, mag, z, spectra, images, cube)
            
            
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
        cube = None
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
            elif extname == 'CUBE_DATA':
                try:
                    ext_var = hdulist.index_of('CUBE_STAT')
                    ext = (i, ext_var)
                except:
                    ext = i
                cube = Cube(filename, ext=ext, ima=False)
        hdulist.close()
        return cls(hdr, lines, mag, z, spectra, images, cube)
                               
        
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
        
        #cube
        if self.cube is not None:
            ext_name = 'CUBE_DATA'
            data_hdu = self.cube.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'CUBE_STAT'
            stat_hdu = self.cube.get_stat_hdu(name=ext_name)
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
        if self.cube is not None:
            self.logger.info('cube', extra=d)
            self.cube.info()
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
           item=='cube' or item=='images'or item=='spectra':
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
        size  : float
                Size of the image in arcsec.
                If None, the size of the white image extension is taken if it exists.
        """
        if size is None:
            white_ima = self.images['SRC_WHITE']
            size = max(np.abs(white_ima.get_step() * white_ima.shape))
        else:
            size /= 3600.
        radius = size/2.
        ra_min = self.ra - radius
        ra_max = self.ra + radius
        dec_min = self.dec - radius
        dec_max = self.dec + radius
        subima = image.truncate(dec_min, dec_max, ra_min, ra_max, mask=False)
        self.images[name] = subima

class SourceList(list):
    """
        list< :class:`mpdaf.sdetect.Source` >
    """
    
    def write(self, name, overwrite=True):
        """ Creates a directory named name and saves all sources files and the catalog file in this folder.
        name/name.fits: catalog file
        name/nameNNNN.fits: source file (NNNN corresponds to the ID of the source)
        
        Parameters
        ----------
        name : string
               Name of the folder
        overwrite : boolean
                    Overwrite folder if it already exists
        """
        if not os.path.exists(name):
            os.makedirs(name)
        else:
            if overwrite:
                shutil.rmtree(name)
                os.makedirs(name)
                
        for source in self:
            source.write('%s/%s%04d.fits'%(name, name, source.ID))
            
        cat = Catalog.from_sources(self)
        cat.write('%s/%s.fits'%(name, name))
        
    