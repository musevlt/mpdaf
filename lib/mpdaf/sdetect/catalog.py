import logging
from astropy.io import fits as pyfits
from astropy.io.fits import Column
import datetime
import warnings
import numpy as np

from ..tools.fits import add_mpdaf_method_keywords

class SourceCatalog(object):

    """This class contains a catalog of objects.

    Parameters
    ----------
    catalog : :class:`dict`
              Dictionary.
              Keys are tuple of position (y, x) in degrees.
              Values are numpy array listing maximum lambda value of the detected object.
              self.catalog = {(y, x) = [lbda1, lbda2, ...]}

    Attributes
    ----------
    catalog  : :class:`dict`
               Dictionary.
               Keys are tuple of position (y, x) in degrees.
               Values are numpy array listing maximum lambda value of the detected object.
               self.catalog = {(y, x) = [lbda1, lbda2, ...]}
    wcs      : string
               Type of spatial coordinates
    wave     : string
               Type of spectral coordinates.
    method   : string
               MPDAF method identifier
    params   : list of strings
               Names of parameters
    values   : list
               Values of parameters
    comments : list of strings
               parameters description
    """

    def __init__(self, catalog, wcs='pix', wave='Angstrom', method="", params=[], values=[], comments=[]):
        """Create a Sourcecatlog object from a dictionary.

        Parameters
        ----------
        catalog : :class:`dict`
                  Dictionary.
                  Keys are tuple of position (y, x) in degrees.
                  Values are numpy array listing maximum lambda value of the detected object.
                  self.catalog = {(y, x) = [lbda1, lbda2, ...]}
        """
        self.logger = logging.getLogger('mpdaf corelib')
        self.catalog = catalog
        self.wcs = wcs
        self.wave = wave
        self.method = method
        self.params = params
        self.values = values
        self.comments = comments

    def info(self):
        """Prints information.
        """
        d = {'class': 'SourceCatalog', 'method': 'info'}
        for coord, lbdas in self.catalog.iteritems():
            msg = 'ra=%0.1f dec=%0.1f lbda=%s' % (coord[1], coord[0], str(lbdas))
            self.logger.info(msg, extra=d)

    def write(self, filename):
        """Writes catalog in table fits file
        """
        d = {'class': 'SourceCatalog', 'method': 'write'}
        prihdu = pyfits.PrimaryHDU()
        warnings.simplefilter("ignore")
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        add_mpdaf_method_keywords(prihdu.header, self.method, self.params,
                                  self.values, self.comments)
        xpos = np.array([x[1] for x in self.catalog.keys()], dtype=np.float32)
        ypos = np.array([x[0] for x in self.catalog.keys()], dtype=np.float32)
        lbdas = self.catalog.values()

        cols = [
            Column(name='xpos', format='1E', unit=self.wcs, array=xpos),
            Column(name='ypos', format='1E', unit=self.wcs, array=ypos),
            Column(name='lambda', format='PE', unit=self.wave,
                   array=np.array(lbdas, dtype=np.object))]

        coltab = pyfits.ColDefs(cols)
        tbhdu = pyfits.new_table(coltab)
        hdu = pyfits.HDUList([prihdu, tbhdu])

        hdu.writeto(filename, clobber=True, output_verify='fix')
        warnings.simplefilter("default")
        self.logger.info('SourceCatalog object save in %s' % filename, extra=d)
