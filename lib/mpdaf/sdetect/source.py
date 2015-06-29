from astropy.io import fits as pyfits
from astropy.table import Table, Column, vstack

from matplotlib import cm
from matplotlib.patches import Ellipse

import datetime
import logging
import numpy as np
import os.path
import shutil
import warnings

from ..obj import Cube, Image, Spectrum, gauss_image
from ..obj.objs import is_int, is_float
from .catalog import Catalog

emlines = {1215.67: 'Lyalpha1216',
           1550.0: 'CIV1550',
           1909.0: 'CIII]1909',
           2326.0: 'CII2326',
           3726.032: '[OII]3726',
           3728.8149: '[OII]3729',
           3798.6001: 'Htheta3799',
           3834.6599: 'Heta3835',
           3869.0: '[NeIII]3869',
           3888.7: 'Hzeta3888',
           3967.0: '[NeIII]3967',
           4102.0: 'Hdelta4102',
           4340.0: 'Hgamma4340',
           4861.3198: 'Hbeta4861',
           4959.0: '[OIII]4959',
           5007.0: '[OIII]5007',
           6548.0: '[NII]6548',
           6562.7998: 'Halpha6563',
           6583.0: '[NII]6583',
           6716.0: '[SII]6716',
           6731.0: '[SII]6731'}


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
        return -9999.0, -9999.0, 1, wl, flux, ["Lya/[OII]"]
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
                    ["Lya/[OII]"]
                    

class Source(object):
    """This class contains a Source object.

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
              
              Keys give origin of spectra
              ('tot' for total spectrum, TBC).
              
              Values are :class:`mpdaf.obj.Spectrum` object
    images  : :class:`dict`
              Dictionary containing images.
              
              Keys give filter names ('SRC_WHITE' for white image, TBC)
              
              Values are :class:`mpdaf.obj.Image` object
    cubes   : :class:`dict`
                  Dictionary containing small data cubes
                  
                  Keys give a description of the cube
                  
                  Values are :class:`mpdaf.obj.Cube` objects
    tables   : :class:`dict`
                  Dictionary containing tables
                  
                  Keys give a description of each table
                  
                  Values are astropy.Table objects
    """
    
    def __init__(self, header, lines=None, mag=None, z=None,
                 spectra=None, images=None, cubes=None, tables=None):
        """Classic constructor.
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
        # Dictionary TABLES
        if tables is None:
            self.tables = {}
        else:
            self.tables = tables
        # logger
        self.logger = logging.getLogger('mpdaf corelib')

    @classmethod
    def from_data(cls, ID, ra, dec, origin, proba=None, confi=None, extras=None,
                 lines=None, mag=None, z=None,
                 spectra=None, images=None, cubes=None, tables=None):
        """
        Source constructor from a list of data.
        
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
        extras  : dict{key: value} or dict{key: (value, comment)}
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
        tables   : :class:`dict`
                  Dictionary containing tables
                  
                  Keys give a description of each table
                  
                  Values are astropy.Table objects
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
            
        return cls(header, lines, mag, z, spectra, images, cubes, tables)
       
            
    @classmethod
    def from_file(cls, filename):
        """Source constructor from a FITS file.
        
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
        tables= {}
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
            elif extname[:3] == 'TAB':
                tables[extname[4:]] = Table(hdu.data)
        hdulist.close()
        return cls(hdr, lines, mag, z, spectra, images, cubes, tables)
                               
        
    def write(self, filename):
        """Write the source object in a FITS file
        
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
                
        # tables
        for key, tab in self.tables.iteritems():
            tbhdu = pyfits.BinTableHDU(name='TAB_%s'%key, data=np.array(tab))
            hdulist.append(tbhdu)
            
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True, output_verify='fix')
        warnings.simplefilter("default")
        
    def info(self):
        """Print information.
        """
        d = {'class': 'Source', 'method': 'info'}
        for l in repr(self.header).split('\n'):
            if l.split()[0] != 'SIMPLE' and l.split()[0] != 'BITPIX' and \
            l.split()[0] != 'NAXIS' and l.split()[0] != 'EXTEND' and \
            l.split()[0] != 'DATE' and l.split()[0] != 'AUTHOR':
                self.logger.info(l, extra=d)
        print '\n'
        for key, spe in self.spectra.iteritems():
            msg = 'spectra[\'%s\']'%key
            if spe.wave.cunit is None:
                unit = ''
            else:
                unit = spe.wave.cunit
            msg += ',%i elements (%0.2f-%0.2f %s)'%(spe.shape, spe.wave.__getitem__(0), spe.wave.__getitem__(spe.shape - 1), unit)
            data = '.data'
            if spe.data is None:
                data = 'no data'
            noise = '.var'
            if spe.var is None:
                noise = 'no noise'
            msg += ' %s %s'%(data, noise)
            self.logger.info(msg, extra=d)
        for key, ima in self.images.iteritems():
            msg = 'images[\'%s\']'%key
            msg += ' %i X %i' %(ima.shape[0], ima.shape[1])
            data = '.data'
            if ima.data is None:
                data = 'no data'
            noise = '.var'
            if ima.var is None:
                noise = 'no noise'
            msg += ' %s %s'%(data, noise)
            self.logger.info(msg, extra=d)
        for key, cub in self.cubes.iteritems():
            msg = 'cubes[\'%s\']'%key
            msg += ' %i X %i X %i' %(cub.shape[0], cub.shape[1], cub.shape[2])
            data = '.data'
            if cub.data is None:
                data = 'no data'
            noise = '.var'
            if cub.var is None:
                noise = 'no noise'
            msg += ' %s %s'%(data, noise)
            self.logger.info(msg, extra=d)
        for key in self.tables.keys():
            self.logger.info('tables[\'%s\']'%key, extra=d)
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
        """Map values to attributes.
        """
        try:
            return self.header[item]
        except KeyError:
            raise AttributeError(item)
 
    def __setattr__(self, item, value):
        """Map attributes to values.
        """
        if item=='header' or item=='logger' or \
           item=='lines' or item=='mag' or item=='z' or \
           item=='cubes' or item=='images' or item=='spectra' \
           or item=='tables':
            return dict.__setattr__(self, item, value)
        else:
            self.header[item] = value
            
    def add_comment(self, comment, author):
        """Add a user comment to the FITS header of the Source object.
        """
        i = 1
        while 'COMMENT%03d'%i in self.header:
            i += 1
        self.header['COMMENT%03d'%i] = (comment, '%s %s'%(author, str(datetime.date.today())))
    
    def remove_comment(self, ncomment):
        """Remove a comment from the FITS header of the Source object.
        """
        del self.header['COMMENT%03d'%ncomment]
        
    def add_attr(self, key, value, desc=None):
        """Add a new attribute for the current Source object.
        This attribute will be saved as a keyword in the primary FITS header.
        This method could also be used to update a simple Source attribute
        that is saved in the pyfits header.
        
        Parameters
        ----------
        key : string
              Attribute name
        value : integer/float/string
                Attribute value
        desc : string
               Attribute description
        """
        if desc is None:
            self.header[key] = value
        else:
            self.header[key] = (value, desc)
            
    def remove_attr(self, key):
        """Remove an Source attribute from the FITS header of the Source object
        """
        del self.header[key]
        
    def add_z(self, desc, z, errz):
        """Add a redshift value to the z table.
        
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
        """Add a magnitude value to the mag table.
        
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
                
    def add_line(self, cols, values):
        """Add a line to the lines table
        
        Parameters
        ----------
        cols   : list<string>
                 Names of the columns
        values : list<interger/float/string>
                 List of corresponding values
        """
        if self.lines is None:
            types = []
            for val in values:
                if is_int(val):
                    types.append('<i4')
                elif is_float(val):
                    types.append('<f8')
                else:
                    types.append('S20')
            self.lines = Table(rows=[values], names=cols, dtype=types) 
        else:
            # add new columns
            for col in cols:
                if  col not in self.lines.colnames:
                    self.lines[col] = [None]*len(self.lines)
            # add new row
            row = [None]*len(self.lines.colnames)
            for col, val in zip(cols, values):
                row[self.lines.colnames.index(col)] = val
            self.lines.add_row(row)
            
    def add_image(self, image, name, size=None):
        """ Extract an image centered on the source center
        and append it to the images dictionary
        
        Extracted image saved in self.images['name'].
        
        Parameters
        ----------
        image : :class:`mpdaf.obj.Image`
                Input image MPDAF object.
        name  : string
                Name used to distinguish this image
        size  : float or (float, float)
                The total size to extract in arcseconds.
                If None, the size of the white image extension is taken if it exists.
        """
        if size is None:
            try:
                white_ima = self.images['SRC_WHITE']
                size = np.abs(white_ima.get_step() * white_ima.shape)*3600.0
                size[1] /= np.cos(np.deg2rad(self.dec))
            except:
                try:
                    white_ima = self.images['MUSE_WHITE']
                    size = np.abs(white_ima.get_step() * white_ima.shape)*3600.0
                    size[1] /= np.cos(np.deg2rad(self.dec))
                except:
                    raise IOError('Size of the image (in arcsec) is required')
        else:
            if is_int(size) or is_float(size):
                size = (size, size)
        
        size = np.array(size)
        radius = size/2./3600.0
        ra_min = self.ra - radius[1]
        ra_max = self.ra + radius[1]
        dec_min = self.dec - radius[0]
        dec_max = self.dec + radius[0]
        subima = image.truncate(dec_min, dec_max, ra_min, ra_max, mask=False)
        self.images[name] = subima
        
    def add_cube(self, cube, name, size=None, lbda=None):
        """Extract an cube centered on the source center
        and append it to the cubes dictionary
        
        Extracted cube saved in self.cubes['name'].
        
        Parameters
        ----------
        cube : :class:`mpdaf.obj.Cube` or :class:`mpdaf.obj.CubeDisk`
                Input cube MPDAF object.
        name  : string
                Name used to distinguish this image
        size  : float or (float, float)
                The total size to extract in arcseconds.
                If None, the size of the white image extension is taken if it exists.
        lbda  : (float, float) or None
                If not None, tuple giving the wavelength range in Angstrom.
        """
        if size is None:
            try:
                white_ima = self.images['SRC_WHITE']
                size = np.abs(white_ima.get_step() * white_ima.shape)*3600.0
                size[1] /= np.cos(np.deg2rad(self.dec))
            except:
                try:
                    white_ima = self.images['MUSE_WHITE']
                    size = np.abs(white_ima.get_step() * white_ima.shape)*3600.0
                    size[1] /= np.cos(np.deg2rad(self.dec))
                except:
                    raise IOError('Size of the image (in arcsec) is required')
        else:
            if is_int(size) or is_float(size):
                size = (size, size)
        
        size = np.array(size)
        radius = size/2./3600.0
        ra_min = self.ra - radius[1]
        ra_max = self.ra + radius[1]
        dec_min = self.dec - radius[0]
        dec_max = self.dec + radius[0]
        if lbda is None:
            lmin, lmax= cube.wave.get_range()
        else:
            lmin, lmax = lbda
        subcub = cube.truncate([[lmin,dec_min,ra_min], [lmax,dec_max,ra_max]], mask=False)
        self.cubes[name] = subcub

        
    def add_white_image(self, cube, size=10):
        """ Compute the white images from the MUSE data cube
        and appends it to the images dictionary.
        
        White image saved in self.images['MUSE_WHITE'].
        
        Parameters
        ----------
        cube : :class:`mpdaf.obj.Cube`
               MUSE data cube.
        size : float
               The total size to extract in arcseconds.
               By default 10x10arcsec
        """
        print 'add_white_image', size
        subcub = cube.subcube((self.dec, self.ra), size)
        self.images['MUSE_WHITE'] = subcub.sum(axis=0)
        
    def add_narrow_band_images(self, cube, z_desc, eml=None, size=None, width=8, margin=10., fband=3.):
        """Create narrow band images
        
        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)
        
        Narrow-band images are saved in self.images['MUSE_*'].
        
        Parameters
        ----------
        cube   : :class:`mpdaf.obj.Cube`
                 MUSE data cube.
        z_desc : string
                 Redshift description.
                 The redshift value corresponding to this description will be used.
        eml    : dict{float: string}
                 Full catalog of lines
                 Dictionary: key is the wavelength value in Angstrom,
                 value is the name of the line.
                 if None, the following catalog is used:
                 eml = {1216 : 'Lyalpha1216', 1909: 'CIII]1909', 3727: '[OII]3727',
                        4861 : 'Hbeta4861' , 5007: '[OIII]5007', 6563: 'Halpha6563',
                        6724 : '[SII]6724'}
        size  : float
                The total size to extract in arcseconds.    
                If None, the size of the white image extension is taken if it exists.
        width : float
                 Angstrom total width
        margin       : float
                       This off-band is offseted by margin wrt narrow-band limit.
        fband        : float
                       The size of the off-band is fband*narrow-band width.
        """
        if self.z is not None:
            d = {'class': 'Source', 'method': 'add_narrow_band_images'}
            if size is None:
                try:
                    size = self.images['SRC_WHITE'].shape
                    pix = True
                except:
                    try:
                        size = self.images['MUSE_WHITE'].shape
                        pix = True
                    except:
                        raise IOError('Size of the image (in arcsec) is required')
            else:
                if is_int(size) or is_float(size):
                    size = (size, size)
                    pix = False
                
                    
            if eml is None:
                all_lines = np.array([1216, 1909, 3727, 4861, 5007, 6563, 6724])
                all_tags = np.array(['Lyalpha1216', 'CIII]1909', '[OII]3727', 'Hbeta4861', '[OIII]5007', 'Halpha6563', '[SII]6724'])
            else:
                all_lines = np.array(eml.keys())
                all_tags = np.array(eml.values())
                    
            subcub = cube.subcube((self.dec, self.ra), size, pix)
            
            z = self.z['Z'][self.z['Z_DESC']==z_desc]
            
            if z>0:
                minl, maxl = subcub.wave.get_range()/(1+z)
                useful = np.where((all_lines>minl) &  (all_lines<maxl))
                nlines = len(useful[0])
                if nlines>0:
                    lambda_ranges = np.empty((2, nlines))
                    lambda_ranges[0, :] = (1+z)*all_lines[useful]-width/2.0
                    lambda_ranges[1, :] = (1+z)*all_lines[useful]+width/2.0
                    tags = all_tags[useful]
                    for l1, l2, tag in zip(lambda_ranges[0, :], lambda_ranges[1, :], tags):
                        self.logger.info('Doing MUSE_%s'%tag, extra=d)
                        self.images['MUSE_'+tag] = subcub.get_image(wave=(l1, l2), subtract_off=True, margin=margin, fband=fband)
        
    def add_seg_images(self, tags=None, DIR=None, del_sex=True):
        """Run SExtractor on all images listed in tags
        to create segmentation maps. 
        SExtractor will use the default.nnw, default.param, default.sex
        and *.conv files present in the current directory.
        If not present default parameter files are created
        or copied from the directory given in input (DIR).
        
        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)
        
        Parameters
        ----------
        tags : list<string>
               List of tags of selected images
        DIR      : string
                   Directory that contains the configuration files of sextractor
        del_sex  : boolean
                   If False, configuration files of sextractor are not removed.
        """
        d = {'class': 'Source', 'method': 'add_masks'}
        if 'MUSE_WHITE' in self.images:
            if tags is None:
                tags = [tag for tag in self.images.keys() if tag[0:4]!='SEG_' and 'MASK' not in tag]
            
            from ..sdetect.sea import segmentation
            segmentation(self, tags, DIR, del_sex)
        else:
            self.logger.warning('add_seg_images method use the MUSE_WHITE image computed by add_white_image method',
                                extra=d)
            
    def add_masks(self, tags=None):
        """Use the list of segmentation maps to compute the union mask
        and the intersection mask and  the region where no object is detected
        in any segmentation map is saved in the sky mask.
        
        Union is saved as an image of booleans in self.images['MASK_UNION']
        
        Intersection is saved as an image of booleans in self.images['MASK_INTER']
        
        Sky mask is saved as an image of booleans in self.images['MASK_SKY']
        
        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)
        
        Parameters
        ----------
        tags : list<string>
               List of tags of selected segmentation images
        """
        maps = {}
        if tags is None:
            for tag, ima in self.images.iteritems():
                if tag[0:4]=='SEG_':
                    maps[tag[4:]] = ima.data.data
        else:
            for tag in tags:
                if tag[0:4]=='SEG_':
                    maps[tag[4:]] = self.images[tag].data.data
                else:
                    maps[tag] = self.images[tag].data.data
        d = {'class': 'Source', 'method': 'add_masks'}
        if len(maps)==0:
            self.logger.warning('no segmentation images. Use add_seg_images to create them',
                                extra=d)
           
        from ..sdetect.sea import mask_creation
        mask_creation(self, maps)
        
    def add_table(self, tab, name):
        """Append an astropy table to the tables dictionary
        
        Parameters
        ----------
        tab : astropy.table
              Input astropy table object.
        name  : string
                Name used to distingish this table
        """
        self.tables[name] = tab
        
    
        
    def extract_spectra(self, cube,
                        tags_to_try = ['MUSE_WHITE', 'MUSE_LYALPHA1216', 'MUSE_HALPHA6563', 'MUSE_[OII]3727'],
                        skysub=True, psf=None):
        """Extract spectra from the MUSE data cube and from a list of narrow-band images
        (to define spectrum extraction apertures).
        
        If skysub:
        
            The local sky spectrum is saved in self.spectra['MUSE_SKY']
            
            The no-weighting spectrum is saved in self.spectra['MUSE_TOT_SKYSUB']
            
            The weighted spectra are saved in self.spectra['*_SKYSUB'] (for * in tags_to_try)
            
            The potential PSF weighted spectra is saved in self.spectra['MUSE_PSF_SKYSUB']
            
        else:
        
            The no-weighting spectrum is saved in self.spectra['MUSE_TOT']
            
            The weighted spectra are saved in self.spectra['*'] (for * in tags_to_try)
            
            The potential PSF weighted spectra is saved in self.spectra['MUSE_PSF']
        
        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)
        
        Parameters
        ----------
        cube        : :class:`mpdaf.obj.Cube`
                      MUSE data cube.    
        tags_to_try : list<string>
                      List of narrow bands images.
        skysub      : boolean
                      If True, a local sky subtraction is done.
        psf         : np.array
                      The PSF to use for PSF-weighted extraction.
                      This can be a vector of length equal to the wavelength
                      axis to give the FWHM of the Gaussian PSF at each
                      wavelength (in arcsec) or a cube with the PSF to use.
                      psf=None by default (no PSF-weighted extraction).
        """
        d = {'class': 'Source', 'method': 'add_masks'}
        try:
            object_mask = self.images['MASK_UNION'].data.data
        except:
            raise IOError('extract_spectra method use the MASK_UNION computed by add_mask method')
        
        wcs = self.images['MASK_UNION'].wcs
        size = self.images['MASK_UNION'].shape
        
        subcub = cube.subcube((self.dec, self.ra), size, pix=True)
        
        if skysub:
            try:
                sky_mask = self.images['MASK_SKY'].data.data
                if not self.images['MASK_UNION'].wcs.isEqual(wcs):
                    raise IOError('MASK_UNION and MASK_SKY have not the same wcs')
            except:
                raise IOError('extract_spectra method use the MASK_SKY computed by add_mask method')
         
            # Get the sky spectrum to subtract
            sky = subcub.sum(axis=(1,2), weights=sky_mask)
            old_mask = subcub.data.mask.copy()
            subcub.data.mask[np.where(np.tile(sky_mask,(subcub.shape[0],1,1))==0)] = True
            sky = subcub.mean(axis=(1,2))
            self.spectra['MUSE_SKY'] = sky
            subcub.data.mask = old_mask
        
            #substract sky
            subcub = subcub - sky
        
        # extract spectra
        # select narrow bands images
        nb_tags = list(set(tags_to_try) & set(self.images.keys()))
        
        # No weighting
        spec = subcub.sum(axis=(1,2), weights=object_mask)
        if skysub:
            self.spectra['MUSE_TOT_SKYSUB'] = spec
        else:
            self.spectra['MUSE_TOT'] = spec
         
        # Now loop over the narrow-band images we want to use. Apply
        # the object mask and ensure that the weight map within the
        # object mask is >=0.
        # Weighted extractions
        ksel = np.where(object_mask==1)
        for tag in nb_tags:
            if self.images[tag].wcs.isEqual(wcs):
                weight = self.images[tag].data * object_mask
                weight[ksel] = weight[ksel] - np.min(weight[ksel])
                weight = weight.filled(0)
                spec = subcub.sum(axis=(1,2), weights=weight)
                if skysub:
                    self.spectra[tag+'_SKYSUB'] = spec
                else:
                    self.spectra[tag] = spec
             
        # PSF
        if psf is not None:
            if len(psf.shape)==3:
                #PSF cube. The user is responsible for getting the
                #dimensions right
                if psf.shape[0] != subcub.shape[0] or \
                   psf.shape[1] != subcub.shape[1] or \
                   psf.shape[2] != subcub.shape[2]:
                    msg = 'Incorrect dimensions for the PSF cube (%i,%i,%i) (it must be (%i,%i,%i)) '\
                        %(psf.shape[0], psf.shape[1], psf.shape[2],
                          subcub.shape[0], subcub.shape[1], subcub.shape[2])
                    self.logger.warning(msg, extra=d)
                    white_cube = None
                else:
                    white_cube = psf
            elif len(psf.shape)==1 and psf.shape[0]==subcub.shape[0]:
                # a Gaussian expected.
                white_cube = np.zeros_like(subcub.data.data)
                for l in range(subcub.shape[0]):
                    gauss_ima = gauss_image(shape=(subcub.shape[1], subcub.shape[2]),
                                            wcs=subcub.wcs, fwhm=(psf[l], psf[l]), peak=False)
                    white_cube[l,:,:] = gauss_ima.data.data
            else:
                msg = 'Incorrect dimensions for the PSF vector (%i) (it must be (%i)) '\
                        %(psf.shape[0], subcub.shape[0])
                self.logger.warning(msg, extra=d)
                white_cube = None
            if white_cube is not None:
                weight = white_cube * np.tile(object_mask,(subcub.shape[0],1,1))
                spec = subcub.sum(axis=(1,2), weights=weight)
                if skysub:
                    self.spectra['MUSE_PSF_SKYSUB'] = spec
                else:
                    self.spectra['MUSE_PSF'] = spec
                # Insert the PSF weighted flux - here re-normalised? 
                
    def crack_z(self, eml=None, nlines=np.inf):
        """Estimate the best redshift matching the list of emission lines
        
         Algorithm from Johan Richard (johan.richard@univ-lyon1.fr).
         
         This method saves the redshift values in self.z and lists the detected lines in self.lines.
         self.info() could be used to print the results.
     
         Parameters
         ----------
         eml    : dict{float: string}
                  Full catalog of lines to test redshift
                  Dictionary: key is the wavelength value in Angtsrom,
                  value is the name of the line.
                  if None, the following catalog is used:
                  emlines = {1215.67  : 'Lyalpha1216' , 1550.0   : 'CIV1550',
                             1909.0   : 'CIII]1909'   , 2326.0   : 'CII2326', 
                             3726.032 : '[OII]3726'   , 3728.8149: '[OII]3729',
                             3798.6001: 'Htheta3799'  , 3834.6599: 'Heta3835',
                             3869.0   : '[NeIII]3869' , 3888.7   : 'Hzeta3889',
                             3967.0   : '[NeIII]3967' , 4102.0   : 'Hdelta4102',
                             4340.0   : 'Hgamma4340'  , 4861.3198: 'Hbeta4861',
                             4959.0   : '[OIII]4959'  , 5007.0   : '[OIII]5007',
                             6548.0   : '[NII6548]'   , 6562.7998: 'Halpha6563',
                             6583.0   : '[NII]6583'   , 6716.0   : '[SII]6716',
                             6731.0   : '[SII]6731'}
                  
          nlines  : integer
                    estimated the redshift if the list of emission lines is inferior to this value
        """
        #d = {'class': 'Source', 'method': 'crack_z'}
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
                #self.logger.info('crack_z: z=%0.6f err_z=%0.6f'%(z, errz), extra=d)
                #line names
                if 'LINE' not in self.lines.colnames:
                    col = Column(data=None, name='LINE', dtype='S20', length=len(self.lines))
                    self.lines.add_column(col)
                for w, name in zip(wl, lnames):
                    self.lines['LINE'][self.lines['LBDA_OBS']==w] = name
                #self.logger.info('crack_z: lines', extra=d)
                #for l in self.lines.pformat():
                #    self.logger.info(l, extra=d)
                
    def sort_lines(self, nlines_max=25):
        """Sort lines by flux in descending order. 
        
        Parameters
        ----------
        nlines_max : integer
                     Maximum number of stored lines
        """
        if self.lines is not None:
            subtab1 = self.lines[self.lines['LINE']!=""]
            subtab1.sort('FLUX')
            subtab1.reverse()
            n1 = len(subtab1)
            subtab2 = self.lines[self.lines['LINE']==""]
            subtab2.sort('FLUX')
            subtab2.reverse()
            n2 = len(subtab2)
            if (n1+n2)>25:
                n2 = max(nlines_max-n1,0)
            self.lines = vstack([subtab1, subtab2[0:n2]])
                
    def show_ima(self, ax, name, showcenter=None,
                 cuts=None, cmap=cm.gray_r, **kwargs):
        """
        Show image.
        
        Parameters
        ----------
        ax         : matplotlib.axes._subplots.AxesSubplot
                     Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        name       : string
                     Name of image to display.
        showcenter : (float, string)
                     radius in arcsec and color used to plot a circle around the center of the source.
        cuts       : (float, float)
                     Minimum and maximum values to use for the scaling.
        cmap       : matplotlib.cm
                     Color map.
        kwargs     : matplotlib.artist.Artist
                     kwargs can be used to set additional plotting properties.
        """
        if name not in self.images.keys():
            raise ValueError,'Image %s not found'%(name)
        zima = self.images[name]
        if cuts == None:
            vmin = None
            vmax = None
        else:
            vmin, vmax = cuts 
        if 'title' not in kwargs:
            kwargs['title'] = '%s'%(name)
        zima.plot(vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, **kwargs)
        if showcenter is not None:
            rad, col = showcenter
            pix = zima.wcs.sky2pix((self.DEC, self.RA))[0]
            rpix = rad/(3600.0*zima.get_step()[0])
            ell = Ellipse((pix[1],pix[0]), 2*rpix, 2*rpix, 0, fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(1)
            ell.set_edgecolor(col)  
        ax.axis('off')
        return  
    
    def show_spec(self, ax, name, cuts=None, zero=False, sky=None, lines=None, **kwargs): 
        """Display a spectra.
         
        Parameters
        ----------
        ax         : matplotlib.axes._subplots.AxesSubplot
                     Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        name       : string
                     Name of spectra to display.
        cuts       : (float, float)
                     Minimum and maximum values to use for the scaling.
        zero       : float
                     If True, the 0 flux line is plotted in black.
        sky        : :class:`mpdaf.obj.Spectrum`
                     Sky spectra to overplot (default None).
        lines      : string
                     Name of a columns of the lines table containing wavelength values.
                     If not None, overplot red vertical lines at the given wavelengths.
        kwargs     : matplotlib.artist.Artist
                     kwargs can be used to set additional plotting properties.
        """
        spec = self.spectra[name]
        spec.plot(ax=ax, **kwargs)
        if zero:
            ax.axhline(0, color='k')
        if cuts is not None:
            ax.set_ylim(cuts)
        if sky is not None:
            ax2 = ax.twinx()
            if 'lmin' in kwargs:
                sky.plot(ax=ax2, color='k', alpha=0.2, lmin=kwargs['lmin'], lmax=kwargs['lmax'])
            else:
                sky.plot(ax=ax2, color='k', alpha=0.2)
            ax2.axis('off')
        if lines is not None:
            wavelist = self.lines[name]
            for lbda in wavelist:
                ax.axvline(lbda, color='r', **kwargs)
        return 
    
        
class SourceList(list):
    """
        list< :class:`mpdaf.sdetect.Source` >
    """
    
    def write(self, name, path='.', overwrite=True):
        """ Create the directory and saves all sources files and the catalog file in this folder.
        
        path/name.fits: catalog file
        (In FITS table, the maximum number of fields is 999.
        In this case, the catalog is saved as an ascci table).
        
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
        try:
            cat.write(fcat)
            # For FITS tables, the maximum number of fields is 999
        except:
            cat.write(fcat.replace('.fits', '.txt'), format='ascii')
    
