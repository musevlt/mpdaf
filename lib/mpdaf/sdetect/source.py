from astropy.io import fits as pyfits
from astropy.io.fits import Column
from astropy.table import Table

import datetime
import numpy as np
import warnings
import logging

from ..obj import Cube, Image, Spectrum

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
    flag    : integer
              Quality flag
    comment : string
              Users comments
    header  : pyfits.Header
              FITS header instance
               ID = hdr['ID']
        ra = hdr['ra']
        dec = hdr['dec']
        origin = hdr['origin']
        z = hdr.get('z', None)
        errz = hdr.get('errz', None)
        flag = hdr.get('flag', '')
        comment = hdr.get('comment', '')
    """
    
    def __init__(self, header, lines=None, spe={}, ima={}, cub=None):
        """
        """
        if not ('ra' in header and 'dec' in header 
                and 'ID' in header and 'origin' in header):
            raise IOError('ID, ra, dec, origin are mandatory parameters to create a Source object') 
        self.header = header
        self.lines = lines
        self.spe = spe
        self.ima = ima
        self.cub = cub
        self.logger = logging.getLogger('mpdaf corelib')

    @classmethod
    def from_data(cls, ID, ra, dec, origin,
                 lines=None, spe={}, ima={}, cub=None, z=None, errz=None,
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
    
    proba ??? - pmax pour selfi
        """
        header = pyfits.Header()
        header['ID'] = (ID, 'object ID')
        header['ra'] = (np.float32(ra), 'RA in degrees')
        header['dec'] = (np.float32(dec), 'DEC in degrees')
        header['origin'] = (origin, 'software which creates this object')
        if z is not None:
            header['z'] = (np.float32(z), 'redshift')
        if errz is not None:
            header['errz'] = (np.float32(errz), 'redshift error')
        if flag != '':
            header['flag'] = (flag, 'quality flag')
        if comment != '':
            header['comment'] = (comment, 'user comment')
        for key, value in extras.iteritems():
            header[key] = value
            
        return cls(header, lines, spe, ima, cub)
            
            
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
        spe = {}
        ima = {}
        cub = None
        for i in range(1, len(hdulist)):
            hdu = hdulist[i]
            extname = hdu.header['EXTNAME']
            #lines
            if extname == 'LINES':
                lines = Table.read(filename)
            # spe
            elif extname[:3] == 'SPE' and extname[-4:]=='DATA':
                spe_name = extname[4:-5]
                try:
                    ext_var = hdulist.index_of('SPE_'+spe_name+'_STAT')
                    ext = (i, ext_var)
                except:
                    ext = i
                spe[spe_name] = Spectrum(filename, ext=ext)
            #ima
            elif extname[:3] == 'IMA' and extname[-4:]=='DATA':
                ima_name = extname[4:-5]
                try:
                    ext_var = hdulist.index_of('IMA_'+ima_name+'_STAT')
                    ext = (i, ext_var)
                except:
                    ext = i
                ima[ima_name] = Image(filename, ext=ext)
            elif extname == 'CUBE_DATA':
                try:
                    ext_var = hdulist.index_of('CUBE_STAT')
                    ext = (i, ext_var)
                except:
                    ext = i
                cub = Cube(filename, ext=ext, ima=False)
        hdulist.close()
        return cls(hdr, lines, spe, ima, cub)
                               
        
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

        if self.lines is not None:
            lbda = self.lines['LBDA'] 
            err_lbda = self.lines['ERR_LBDA']
            width = self.lines['WIDTH']
            err_width = self.lines['ERR_WIDTH']
            flux = self.lines['FLUX']
            err_flux = self.lines['ERR_FLUX']
            ew = self.lines['EW']
            err_ew = self.lines['ERR_EW']
            name = self.lines['NAME']
            z = self.lines['Z']
        
            cols = [Column(name='LBDA', format='1E', unit='Angstrom',
                       array=lbda),
                    Column(name='ERR_LBDA', format='1E', unit='Angstrom',
                       array=err_lbda),
                    Column(name='WIDTH', format='1E', unit='Angstrom',
                       array=width),
                    Column(name='ERR_WIDTH', format='1E', unit='Angstrom',
                       array=err_width),
                    Column(name='FLUX', format='1E', unit='10**(-20)*erg/s/cm**2',
                       array=flux),
                    Column(name='ERR_FLUX', format='1E', unit='10**(-20)*erg/s/cm**2',
                       array=err_flux),
                    Column(name='EW', format='1E', unit='Angstrom', array=ew),
                    Column(name='ERR_EW', format='1E', unit='Angstrom', array=err_ew),
                    Column(name='NAME', format='A20', unit='', array=name),
                    Column(name='Z', format='1E', unit='', array=z)]

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
        for key, spe in self.spe.iteritems():
            self.logger.info('spe[\'%s\']'%key, extra=d)
            spe.info()
            print '\n'
        for key, ima in self.ima.iteritems():
            self.logger.info('ima[\'%s\']'%key, extra=d)
            ima.info()
            print '\n'
        if self.cub is not None:
            self.logger.info('cub', extra=d)
            self.cub.info()
            print '\n'
        if self.lines is not None:
            self.logger.info('lines', extra=d)
            for l in self.lines.pformat():
                self.logger.info(l, extra=d)
            print '\n'
        
    def get_ra(self):
        """
        """
        return self.header['ra']
    
    def get_dec(self):
        """
        """
        return self.header['dec']
        
    def get_keyword(self, key):
        """
        """
        return self.header[key]
        