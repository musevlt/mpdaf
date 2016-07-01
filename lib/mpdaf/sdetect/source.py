"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c)      2015 Johan Richard <jrichard@univ-lyon1.fr>
Copyright (c) 2015-2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c) 2015-2016 Roland Bacon <roland.bacon@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import absolute_import, division, print_function

import astropy.units as u
import datetime
import glob
import itertools
import logging
import numpy as np
import os.path
import re
import six
import shutil
import warnings

from astropy.io import fits as pyfits
from astropy.table import Table, MaskedColumn, vstack
from functools import partial
from matplotlib import cm
from matplotlib.patches import Ellipse
from numpy import ma
from six.moves import range, zip
from scipy.optimize import leastsq

from ..obj import Cube, Image, Spectrum, gauss_image
from ..obj.objs import is_int, is_float, bounding_box
from ..tools import deprecated
from ..MUSE import FieldsMap, FSF
from ..MUSE.PSF import MOFFAT1
from ..sdetect.sea import segmentation, mask_creation, findCentralDetection
from ..sdetect.sea import union, intersection, compute_spectrum

emlines = {1215.67: 'LYALPHA1216',
           1550.0: 'CIV1550',
           1909.0: 'CIII]1909',
           2326.0: 'CII2326',
           2801.0: 'MgII2801',
           3726.032: '[OII]3726',
           3728.8149: '[OII]3729',
           3798.6001: 'HTHETA3799',
           3834.6599: 'HETA3835',
           3869.0: '[NeIII]3869',
           3888.7: 'HZETA3888',
           3967.0: '[NeIII]3967',
           4102.0: 'HDELTA4102',
           4340.0: 'HGAMMA4340',
           4861.3198: 'HBETA4861',
           4959.0: '[OIII]4959',
           5007.0: '[OIII]5007',
           6548.0: '[NII]6548',
           6562.7998: 'HALPHA6563',
           6583.0: '[NII]6583',
           6716.0: '[SII]6716',
           6731.0: '[SII]6731'}


def vacuum2air(vac):
    """in angstroms."""
    vac = np.array(vac)
    return vac / (1.0 + 2.735182e-4 + 131.4182 / (vac**2) + 2.76249e8 / (vac**4))


def air2vacuum(air):
    """in angstroms."""
    air = np.array(air)
    vactest = air + (air - vacuum2air(air))
    x = np.abs(air - vacuum2air(vactest))
    for i in range(10):
        vactest = vactest + x
        x = np.abs(air - vacuum2air(vactest))
    return vactest


def matchlines(nlines, wl, z, eml):
    """Try to match all the lines given.

    For each line computes the distance in Angstroms to the closest line.
    Add the errors

    Algorithm from Johan Richard (johan.richard@univ-lyon1.fr)

    Parameters
    ----------
    nlines : int
        Number of emission lines
    wl : array of float
        Table of wavelengths
    z : float
        Redshift to test
    eml : dict
        Full catalog of lines to test redshift. key: wavelength, value: name.

    Returns
    -------
    out : (array of float, array of float)
        (list of wavelengths, errors)

    """
    lbdas = np.array(list(eml.keys()))
    a = (wl[:, np.newaxis] / (1 + z) - lbdas[np.newaxis, :]) ** 2.0
    jfound = np.argmin(a, axis=1)
    error = np.diag(a[:, jfound]).sum()
    error = np.sqrt(error / nlines)
    if((nlines >= 2)and(jfound[0] == jfound[1])):
        error = 15.
    return(error, jfound)


def crackz(nlines, wl, flux, eml, zguess=None):
    """Method to estimate the best redshift matching a list of emission lines.

    Algorithm from Johan Richard (johan.richard@univ-lyon1.fr)

    Parameters
    ----------
    nlines : int
        Number of emission lines
    wl : array of float
        Table of observed line wavelengths
    flux : array of float
        Table of line fluxes
    eml : dict
        Full catalog of lines to test redshift
    zguess : float
        Guess redshift to test (only this)

    Returns
    -------
    out : (float, float, int, list of float, list of float, list of str)
        (redshift, redshift error, list of wavelengths, list of fluxes,
        list of lines names)

    """
    errmin = 3.0
    zstep = 0.0002
    if zguess:
        zmin = zguess
        zmax = zguess + zstep
    else:
        zmin = 0.0
        zmax = 7.0
    if(nlines == 0):
        return -9999.0, -9999.0, 0, [], [], []
    lnames = np.array(list(eml.values()))
    if(nlines == 1):
        if zguess:
            (error, jfound) = matchlines(nlines, wl, zguess, eml)
            if(error < errmin):
                return zguess, -9999.0, 1, wl, flux, list(lnames[jfound[0]])
            else:
                return zguess, -9999.0, 1, [], [], []
        else:
            return -9999.0, -9999.0, 1, wl, flux, ["Lya/[OII]"]
    if(nlines > 1):
        found = 0
        lbdas = np.array(list(eml.keys()))
        for z in np.arange(zmin, zmax, zstep):
            (error, jfound) = matchlines(nlines, wl, z, eml)
            if(error < errmin):
                errmin = error
                found = 1
                zfound = z
                jfinal = jfound.copy()
        if((found == 0) and zguess):
            return zguess, -9999.0, 0, [], [], []
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
                return crackz(1, [wl[ksel]], [flux[ksel]], eml)


def _read_ext(cls, hdulist, extname, **kwargs):
    """Read an extension from a FITS HDUList."""
    try:
        obj = cls(hdulist[extname].data, **kwargs)
    except Exception as e:
        raise IOError('%s: Impossible to open extension %s as a %s\n%s' % (
            os.path.basename(hdulist.filename), extname, cls.__name__, e))
    return obj


def _read_mpdaf_obj(cls, hdulist, ext, **kwargs):
    """Read an extension from a FITS HDUList and return an MPDAF object."""
    filename = hdulist.filename()
    try:
        obj = cls(filename=filename, hdulist=hdulist, ext=ext, **kwargs)
    except Exception as e:
        raise IOError('%s: Impossible to open extension %s as a %s\n%s' % (
            os.path.basename(filename), ext, cls.__name__, e))
    return obj


def _read_masked_table(hdulist, extname, **kwargs):
    """Read a masked Table from a FITS HDUList."""
    t = _read_ext(Table, hdulist, extname, masked=True)
    h = hdulist[extname].header
    for i in range(h['TFIELDS']):
        try:
            t.columns[i].unit = h['TUNIT%d' % (i + 1)]
        except:
            pass
    return t

# _read_masked_table = partial(_read_ext, Table, masked=True)
_read_spectrum = partial(_read_mpdaf_obj, Spectrum)
_read_image = partial(_read_mpdaf_obj, Image)
_read_cube = partial(_read_mpdaf_obj, Cube)


class Source(object):

    """This class contains a Source object.

    Attributes
    ----------
    header : `astropy.io.fits.Header`
        FITS header instance
    lines : `astropy.table.Table`
        List of lines
    mag : `astropy.table.Table`
        List of magnitudes
    z : `astropy.table.Table`
        List of redshifts
    spectra : dict
        Spectra dictionary, keys give origin of spectra (``'tot'`` for total
        spectrum, TBC). Values are `~mpdaf.obj.Spectrum` objects.
    images : dict
        Images dictionary, keys give filter names (``'MUSE_WHITE'`` for white
        image, TBC). Values are `~mpdaf.obj.Image` objects.
    cubes : dict
        Cubes dictionary, keys give a description of the cube.
        Values are `~mpdaf.obj.Cube` objects.
    tables : dict
        Tables dictionary, keys give a description of each table.
        Values are `astropy.table.Table` objects.

    """

    def __init__(self, header, lines=None, mag=None, z=None, spectra=None,
                 images=None, cubes=None, tables=None, mask_invalid=True):
        # Check required keywords in the FITS header
        for key in ('RA', 'DEC', 'ID', 'CUBE', 'ORIGIN', 'ORIGIN_V'):
            if key not in header:
                raise IOError('{} keyword is mandatory to create a Source '
                              'object'.format(key))

        self.header = header
        # Table LINES
        self.lines = lines
        # Table MAG
        self.mag = mag
        # Table Z
        self.z = z
        # Dictionary SPECTRA
        self.spectra = spectra or {}
        # Dictionary IMAGES
        self.images = images or {}
        # Dictionary CUBES
        self.cubes = cubes or {}
        # Dictionary TABLES
        self.tables = tables or {}
        # logger
        self._logger = logging.getLogger(__name__)
        # mask invalid
        if mask_invalid:
            self.masked_invalid()

    @classmethod
    def from_data(cls, ID, ra, dec, origin, proba=None, confi=None,
                  extras=None, lines=None, mag=None, z=None, spectra=None,
                  images=None, cubes=None, tables=None, mask_invalid=True):
        """Source constructor from a list of data.

        Parameters
        ----------
        ID : int
            ID of the source
        ra : double
            Right ascension in degrees
        dec : double
            Declination in degrees
        origin : tuple (str, str, str)
            1- Name of the detector software which creates this object
            2- Version of the detector software which creates this object
            3- Name of the FITS data cube from which this object has been
            extracted.
        proba : float
            Detection probability
        confi : int
            Expert confidence index
        extras : dict{key: value} or dict{key: (value, comment)}
            Extra keywords
        lines : `astropy.table.Table`
            List of lines
        mag : astropy.Lines
            List of magnitudes.
        z : `astropy.table.Table`
            List of redshifts
        spectra : dict
            Dictionary containing spectra.
            Keys gives the origin of the spectrum ('tot' for total spectrum,
            TBC).
            Values are `~mpdaf.obj.Spectrum` object
        images : dict
            Dictionary containing small images.
            Keys gives the filter ('MUSE_WHITE' for white image, TBC)
            Values are `~mpdaf.obj.Image` object
        cubes : dict
            Dictionary containing small data cubes
            Keys gives a description of the cube
            Values are `~mpdaf.obj.Cube` objects
        tables : dict
            Dictionary containing tables
            Keys give a description of each table
            Values are `astropy.table.Table` objects
        mask_invalid: bool
            If True (default), iterate on all columns of all tables to mask
            invalid values (Inf, NaN, and -9999).

        """
        header = pyfits.Header()
        header['ID'] = (ID, 'object ID u.unitless %d')
        header['RA'] = (ra, 'RA u.degree %.7f')
        header['DEC'] = (dec, 'DEC u.degree %.7f')
        header['ORIGIN'] = (origin[0], 'detection software')
        header['ORIGIN_V'] = (origin[1], 'version of the detection software')
        header['CUBE'] = (os.path.basename(origin[2]), 'MUSE data cube')
        if proba is not None:
            header['DPROBA'] = (proba, 'Detection probability')
        if confi is not None:
            header['CONFI'] = (confi, 'Confidence index')
        if extras is not None:
            header.update(extras)

        return cls(header, lines, mag, z, spectra, images, cubes, tables,
                   mask_invalid=mask_invalid)

    @classmethod
    def from_file(cls, filename, ext=None, mask_invalid=True):
        """Source constructor from a FITS file.

        Parameters
        ----------
        filename : str
            FITS filename
        ext : str or list of str
              Names of the FITS extensions that will be loaded in the source
              object. Regular expression accepted.
        mask_invalid: bool
            If True (default), iterate on all columns of all tables to mask
            invalid values (Inf, NaN, and -9999).

        """
        hdulist = pyfits.open(filename)
        hdr = hdulist[0].header
        spectra = {}
        images = {}
        cubes = {}
        tables = {}

        if ext is None:
            extnames = [h.name for h in hdulist[1:]]
        elif isinstance(ext, six.string_types):
            extnames = [h.name for h in hdulist[1:] if re.findall(ext, h.name)]
        else:
            extnames = []
            for e in ext:
                extnames += [h.name for h in hdulist[1:]
                             if re.findall(e, h.name)]

        lines = (_read_masked_table(hdulist, 'LINES') if 'LINES' in extnames
                 else None)
        if lines is not None:
            for name in lines.colnames:
                if 'LBDA' in name or 'EQW' in name:
                    lines[name].format = '.2f'
                if 'FLUX' in name or 'FWHM' in name:
                    lines[name].format = '.1f'

        mag = _read_masked_table(hdulist, 'MAG') if 'MAG' in extnames else None
        if mag is not None:
            for name in mag.colnames:
                mag[name].unit = 'unitless'
                if name == 'BAND':
                    mag[name].description = 'Filter name'
                elif name == 'MAG_ERR':
                    mag[name].format = '.3f'
                    mag[name].description = 'Error in AB Magnitude'
                elif name == 'MAG':
                    mag[name].format = '.3f'
                    mag[name].description = 'AB Magnitude'

        z = _read_masked_table(hdulist, 'Z') if 'Z' in extnames else None
        if z is not None:
            for name in z.colnames:
                z[name].unit = 'unitless'
                if name == 'Z_DESC':
                    z[name].description = 'Redshift description'
                elif name == 'Z_MIN':
                    z[name].format = '.4f'
                    z[name].description = 'Lower bound of estimated redshift'
                elif name == 'Z_MAX':
                    z[name].format = '.4f'
                    z[name].description = 'Upper bound of estimated redshift'
                elif name == 'Z_ERR':
                    z[name].format = '.4f'
                    z[name].description = 'Error of estimated redshift'
                elif name == 'Z':
                    z[name].format = '.4f'
                    z[name].description = 'Estimated redshift'

        for i, hdu in enumerate(hdulist[1:]):
            try:
                extname = hdu.name
                if not extname:
                    raise IOError('%s: Extension %d without EXTNAME' % (
                        os.path.basename(filename), i))

                if extname in extnames:
                    start = extname[:3]
                    end = extname[-4:]

                    if end == 'STAT':
                        continue
                    elif end == 'DATA':
                        name = extname[4:-5]
                        stat_ext = '%s_%s_STAT' % (start, name)
                        if stat_ext in hdulist:
                            ext = (extname, stat_ext)
                        else:
                            ext = extname
                        if start == 'SPE':
                            spectra[name] = _read_spectrum(hdulist, ext)
                        elif start == 'IMA':
                            images[name] = _read_image(hdulist, ext)
                        elif start == 'CUB':
                            cubes[name] = _read_cube(hdulist, ext, ima=False)
                    elif start == 'TAB':
                        tables[extname[4:]] = _read_masked_table(hdulist,
                                                                 extname)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(e)
        hdulist.close()
        return cls(hdr, lines, mag, z, spectra, images, cubes, tables,
                   mask_invalid=mask_invalid)

    @classmethod
    def _light_from_file(cls, filename):
        """Source constructor from a FITS file.

        Light: Only data that are stored in catalog are loaded.

        Parameters
        ----------
        filename : str
            FITS filename
        """
        hdulist = pyfits.open(filename)
        hdr = hdulist[0].header

        lines = (_read_masked_table(hdulist, 'LINES') if 'LINES' in hdulist
                 else None)
        mag = _read_masked_table(hdulist, 'MAG') if 'MAG' in hdulist else None
        z = _read_masked_table(hdulist, 'Z') if 'Z' in hdulist else None

        tables = {}
        for i in range(1, len(hdulist)):
            try:
                hdu = hdulist[i]
                if 'EXTNAME' not in hdu.header:
                    raise IOError('%s: Extension %d without EXTNAME' % (
                        os.path.basename(filename), i))

                extname = hdu.header['EXTNAME']
                # tables
                if extname[:3] == 'TAB':
                    tables[extname[4:]] = _read_masked_table(hdulist, extname)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(e)

        hdulist.close()
        return cls(hdr, lines, mag, z, None, None, None, tables)

    def write(self, filename):
        """Write the source object in a FITS file.

        Parameters
        ----------
        filename : str
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
            cols = []
            for colname in self.lines.colnames:
                col = self.lines[colname]
                if col.unit is not None:
                    unit = col.unit.to_string('fits')
                else:
                    unit = None
                try:
                    cols.append(pyfits.Column(
                        name=col.name, format=col.dtype.char,
                        unit=unit, array=np.array(col)))
                except:
                    cols.append(pyfits.Column(
                        name=col.name, format='A20',
                        unit=unit,
                        array=np.array(col)))

            coldefs = pyfits.ColDefs(cols)
            tbhdu = pyfits.BinTableHDU.from_columns(name='LINES',
                                                    columns=coldefs)
            hdulist.append(tbhdu)

        # magnitudes
        if self.mag is not None:
            tbhdu = pyfits.BinTableHDU(name='MAG', data=np.array(self.mag))
            hdulist.append(tbhdu)

        # redshifts
        if self.z is not None:
            tbhdu = pyfits.BinTableHDU(name='Z', data=np.array(self.z))
            hdulist.append(tbhdu)

        # spectra
        for key, spe in six.iteritems(self.spectra):
            ext_name = 'SPE_%s_DATA' % key
            data_hdu = spe.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'SPE_%s_STAT' % key
            stat_hdu = spe.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)

        # images
        for key, ima in six.iteritems(self.images):
            ext_name = 'IMA_%s_DATA' % key
            savemask = 'none' if key.startswith(('MASK_', 'SEG_')) else 'nan'
            data_hdu = ima.get_data_hdu(name=ext_name, savemask=savemask)
            hdulist.append(data_hdu)
            ext_name = 'IMA_%s_STAT' % key
            stat_hdu = ima.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)

        # cubes
        for key, cub in six.iteritems(self.cubes):
            ext_name = 'CUB_%s_DATA' % key
            data_hdu = cub.get_data_hdu(name=ext_name, savemask='nan')
            hdulist.append(data_hdu)
            ext_name = 'CUB_%s_STAT' % key
            stat_hdu = cub.get_stat_hdu(name=ext_name)
            if stat_hdu is not None:
                hdulist.append(stat_hdu)

        # tables
        for key, tab in six.iteritems(self.tables):
            tbhdu = pyfits.BinTableHDU(name='TAB_%s' % key, data=np.array(tab))
            hdulist.append(tbhdu)

        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True, output_verify='fix')
        warnings.simplefilter("default")

    def info(self):
        """Print information."""
        info = self._logger.info
        excluded_cards = {'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE',
                          'AUTHOR'}
        keys = set(self.header.keys())
        coms = set(filter(lambda k: re.match('COM\d\d\d', k), keys))
        hist = set(filter(lambda k: re.match('HIST\d\d\d', k), keys))
        keys = keys - excluded_cards - coms - hist

        for key in itertools.chain(keys, coms, hist):
            info(self.header.cards[key])

        if any([self.spectra, self.images, self.cubes, self.tables]):
            print('')

        for key, spe in six.iteritems(self.spectra):
            data = '' if spe.data is None else '.data'
            noise = '' if spe.var is None else '.var'
            start, end = spe.get_range(unit=u.angstrom)
            info("spectra['%s'], %i elements (%0.2f-%0.2f A) %s %s",
                 key, spe.shape[0], start, end, data, noise)

        for key, ima in six.iteritems(self.images):
            data = '' if ima.data is None else '.data'
            noise = '' if ima.var is None else '.var'
            info("images['%s'], %i X %i %s %s rot=%0.1f deg", key,
                 ima.shape[0], ima.shape[1], data, noise, ima.wcs.get_rot())

        for key, cub in six.iteritems(self.cubes):
            data = '' if cub.data is None else '.data'
            noise = '' if cub.var is None else '.var'
            info("cubes['%s'], %i X %i X %i %s %s rot=%0.1f deg", key,
                 cub.shape[0], cub.shape[1], cub.shape[2], data, noise,
                 cub.wcs.get_rot())

        for key in self.tables:
            info('tables[\'%s\']' % key)

        for name, table in (('lines', self.lines), ('magnitudes', self.mag),
                            ('redshifts', self.z)):
            if table is not None:
                print('')
                info(name)
                for l in table.pformat():
                    info(l)

    def __getattr__(self, item):
        """Map values to attributes."""
        try:
            return self.header[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, item, value):
        """Map attributes to values."""
        if item in ('header', 'lines', 'mag', 'z', 'cubes', 'images',
                    'spectra', 'tables', '_logger'):
            # return dict.__setattr__(self, item, value)
            super(Source, self).__setattr__(item, value)
        else:
            self.header[item] = value

    def add_comment(self, comment, author, date=None):
        """Add a user comment to the FITS header of the Source object.

        Parameters
        ----------
        comment : str
                  Comment
        author : str
                 Initial of the author
        date   : datetime.date
                 Date
                 By default the current local date is used.
        """
        if date is None:
            date = datetime.date.today()
        i = 1
        while 'COM%03d' % i in self.header:
            i += 1
        self.header['COM%03d' % i] = (comment, '%s %s' % (author, str(date)))

    def remove_comment(self, ncomment):
        """Remove a comment from the FITS header of the Source object.

        Parameters
        ----------
        ncomment : int
                   Comment ID
        """
        del self.header['COM%03d' % ncomment]

    def add_history(self, text, author="", date=None):
        """Add a history to the FITS header of the Source object.

        Parameters
        ----------
        text : str
               History text
        author : str
                 Initial of the author.
        date   : datetime.date
                 Date
                 By default the current local date is used.
        """
        if date is None:
            date = datetime.date.today()
        i = 1
        while 'HIST%03d' % i in self.header:
            i += 1
        self.header['HIST%03d' % i] = (text, '%s %s' % (author, str(date)))

    def remove_history(self, nhist):
        """Remove an history from the FITS header of the Source object.

        Parameters
        ----------
        nhist : int
                History ID
        """
        del self.header['HIST%03d' % nhist]

    def add_attr(self, key, value, desc=None, unit=None, fmt=None):
        """Add a new attribute for the current Source object. This attribute
        will be saved as a keyword in the primary FITS header. This method
        could also be used to update a simple Source attribute that is saved in
        the pyfits header.

        Equivalent to self.key = (value, comment)

        Parameters
        ----------
        key : str
            Attribute name
        value : int/float/str
            Attribute value
        desc : str
            Attribute description
        unit : `astropy.units.Unit`
               Attribute units
        fmt : str
              Attribute format ('.2f' for example)
        """
        if desc is None:
            desc = ''
        if unit is not None:
            desc += ' u.%s' % (unit.to_string('fits'))
        if fmt is not None:
            desc += ' %%%s' % fmt
        self.header[key] = (value, desc)

    def remove_attr(self, key):
        """Remove an Source attribute from the FITS header of the Source
        object."""
        del self.header[key]

    def add_z(self, desc, z, errz=0):
        """Add a redshift value to the z table.

        Parameters
        ----------
        desc : str
            Redshift description.
        z : float
            Redshidt value.
        errz : float or (float,float)
            Redshift error (deltaz) or redshift interval (zmin,zmax).
        """
        if np.isscalar(errz):
            if errz == -9999:
                zmin = -9999
                zmax = -9999
            else:
                zmin = z - errz / 2
                zmax = z + errz / 2
        else:
            try:
                zmin, zmax = errz
            except:
                raise ValueError('Wrong type for errz in add_z')
        if self.z is None:
            if z != -9999:
                self.z = Table(names=['Z_DESC', 'Z', 'Z_MIN', 'Z_MAX'],
                               rows=[[desc, z, zmin, zmax]],
                               dtype=('S20', 'f8', 'f8', 'f8'),
                               masked=True)
                self.z['Z'].format = '.4f'
                self.z['Z'].description = 'Estimated redshift'
                self.z['Z'].unit = 'unitless'
                self.z['Z_MIN'].format = '.4f'
                self.z['Z_MIN'].description = 'Lower bound of estimated redshift'
                self.z['Z_MIN'].unit = 'unitless'
                self.z['Z_MAX'].format = '.4f'
                self.z['Z_MAX'].description = 'Upper bound of estimated redshift'
                self.z['Z_MAX'].unit = 'unitless'
                self.z['Z_DESC'].description = 'Type of redshift'
                self.z['Z_DESC'].unit = 'unitless'
        else:
            if desc in self.z['Z_DESC']:
                if z != -9999:
                    self.z['Z'][self.z['Z_DESC'] == desc] = z
                    self.z['Z_MIN'][self.z['Z_DESC'] == desc] = zmin
                    self.z['Z_MAX'][self.z['Z_DESC'] == desc] = zmax
                else:
                    index = np.where((self.z['Z_DESC'] == desc))[0][0]
                    self.z.remove_row(index)
            else:
                if z != -9999:
                    self.z.add_row([desc, z, zmin, zmax])

        if self.z is not None:
            self.z['Z'] = ma.masked_equal(self.z['Z'], -9999)
            self.z['Z_MIN'] = ma.masked_equal(self.z['Z_MIN'], -9999)
            self.z['Z_MAX'] = ma.masked_equal(self.z['Z_MAX'], -9999)

    def add_mag(self, band, m, errm):
        """Add a magnitude value to the mag table.

        Parameters
        ----------
        band : str
            Filter name.
        m : float
            Magnitude value.
        errm : float
            Magnitude error.
        """
        if self.mag is None:
            self.mag = Table(names=['BAND', 'MAG', 'MAG_ERR'],
                             rows=[[band, m, errm]],
                             dtype=('S20', 'f8', 'f8'),
                             masked=True)
            self.mag['MAG'].format = '.3f'
            self.mag['MAG'].description = 'AB Magnitude'
            self.mag['MAG'].unit = 'unitless'
            self.mag['MAG_ERR'].format = '.3f'
            self.mag['MAG_ERR'].description = 'Error in AB Magnitude'
            self.mag['MAG_ERR'].unit = 'unitless'
        else:
            if band in self.mag['BAND']:
                self.mag['MAG'][self.mag['BAND'] == band] = m
                self.mag['MAG_ERR'][self.mag['BAND'] == band] = errm
            else:
                self.mag.add_row([band, m, errm])

    def add_line(self, cols, values, units=None, desc=None, fmt=None,
                 match=None):
        """Add a line to the lines table.

        Parameters
        ----------
        cols : list of str
            Names of the columns
        values : list<int/float/str>
            List of corresponding values
        units : list<astropy.units>
            Unity of each column
        desc : list of str
               Description of each column
        fmt : list of str
               Fromat of each column.
        match : (str, float/int/str, bool)
            Tuple (key, value, False/True) that gives the key to match the
            added line with an existing line.  eg ('LINE','LYALPHA1216', True)
            If the boolean is True, the line will be added even if it is not
            matched.

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
            self.lines = Table(rows=[values], names=cols, dtype=types,
                               masked=True)
            if units is not None:
                for colname, unit in zip(self.lines.colnames, units):
                    self.lines[colname].unit = unit
            if desc is not None:
                for colname, d in zip(self.lines.colnames, desc):
                    self.lines[colname].description = d
            if fmt is not None:
                for colname, f in zip(self.lines.colnames, fmt):
                    self.lines[colname].format = f
        else:
            # add new columns
            if units is None:
                units = [None] * len(cols)
            if desc is None:
                desc = [None] * len(cols)
            if fmt is None:
                fmt = [None] * len(cols)
            for col, val, unit, d, f in zip(cols, values, units, desc, fmt):
                if col not in self.lines.colnames:
                    nlines = len(self.lines)
                    if is_int(val):
                        typ = '<i4'
                    elif is_float(val):
                        typ = '<f8'
                    else:
                        typ = 'S20'
                    col = MaskedColumn(ma.masked_array(np.empty(nlines),
                                                       mask=np.ones(nlines)),
                                       name=col, dtype=typ, unit=unit,
                                       description=d, format=f)
                    self.lines.add_column(col)

            if match is not None:
                if len(match) == 2:
                    matchkey, matchval = match
                    add_if_not_matched = False
                else:
                    matchkey, matchval, add_if_not_matched = match

            if match is not None and matchkey in self.lines.colnames:
                l = np.argwhere(self.lines[matchkey] == six.b(matchval))
                if len(l) > 0:
                    for col, val, unit in zip(cols, values, units):
                        if unit is None or unit == self.lines[col].unit:
                            self.lines[col][l] = val
                        else:
                            self.lines[col][l] = (val * unit).to(self.lines[col].unit).value
                    return
                else:
                    if not add_if_not_matched:
                        return

            # add new row
            ncol = len(self.lines.colnames)
            row = [None] * ncol
            mask = np.ones(ncol)
            for col, val, unit in zip(cols, values, units):
                i = self.lines.colnames.index(col)
                if unit is None or unit == self.lines[col].unit:
                    row[i] = val
                else:
                    row[i] = (val * unit).to(self.lines[col].unit).value
                mask[i] = 0
            self.lines.add_row(row, mask=mask)

    def add_image(self, image, name, size=None, minsize=2.0,
                  unit_size=u.arcsec, rotate=False):
        """Extract an small image centered on the source center from the input
        image and append it to the images dictionary.

        Extracted image saved in self.images['name'].

        Parameters
        ----------
        image : `~mpdaf.obj.Image`
            Input image MPDAF object.
        name : str
            Name used to distinguish this image
        size : float
            The size to extract. It corresponds to the size along the delta
            axis and the image is square. If None, the size of the white image
            extension is taken if it exists.
        unit_size : `astropy.units.Unit`
            Size and minsize unit.
            Arcseconds by default (use None for size in pixels)
        minsize : float
            The minimum size of the output image.
        rotate : bool
            if True, the image is rotated to the same PA as the white-light
            image.

        """
        if size is None:
            try:
                white_ima = self.images['MUSE_WHITE']
            except:
                raise IOError('Size of the image is required')
            if white_ima.wcs.sameStep(image.wcs):
                size = white_ima.shape[0]
                if unit_size is not None:
                    minsize /= image.wcs.get_step(unit=unit_size)[0]
                    unit_size = None
            else:
                size = white_ima.wcs.get_step(unit=u.arcsec)[0] * white_ima.shape[0]
                if unit_size is None:
                    minsize *= image.wcs.get_step(unit=u.arcsec)[0]
                else:
                    if unit_size != u.arcsec:
                        minsize = (minsize * unit_size).to(u.arcsec).value
                unit_size = u.arcsec
        if rotate:
            try:
                white_ima = self.images['MUSE_WHITE']
            except:
                raise IOError('MUSE_WHITE image is required to get the PA')
            pa_white = white_ima.get_rot()
            pa = image.get_rot()
            if np.abs(pa_white - pa) > 1.e-3:
                subima = image.subimage((self.dec, self.ra), size * 1.5, minsize=minsize,
                                        unit_center=u.deg, unit_size=unit_size)
                uniq = np.unique(subima.data.data)
                if ((uniq == 0) | (uniq == 1)).all():
                    subima = subima.rotate(pa_white - pa, order=0)
                else:
                    subima = subima.rotate(pa_white - pa)
                subima = subima.subimage((self.dec, self.ra), size, minsize=minsize,
                                         unit_center=u.deg, unit_size=unit_size)
            else:
                subima = image.subimage((self.dec, self.ra), size, minsize=minsize,
                                        unit_center=u.deg, unit_size=unit_size)
        else:
            subima = image.subimage((self.dec, self.ra), size, minsize=minsize,
                                    unit_center=u.deg, unit_size=unit_size)
        if subima is None:
            self._logger.warning('Image %s not added. Source outside or at the'
                                 ' edges', name)
            return
        self.images[name] = subima

    def add_cube(self, cube, name, size=None, lbda=None,
                 unit_size=u.arcsec, unit_wave=u.angstrom):
        """Extract a cube centered on the source center and append it to the
        cubes dictionary.

        Extracted cube saved in self.cubes['name'].

        Parameters
        ----------
        cube : `~mpdaf.obj.Cube`
            Input cube MPDAF object.
        name : str
            Name used to distinguish this cube
        size : float
            The size to extract. It corresponds to the size along the delta
            axis and the image is square. If None, the size of the white image
            extension is taken if it exists.
        lbda : (float, float) or None
            If not None, tuple giving the wavelength range.
        unit_size : `astropy.units.Unit`
            unit of the size value (arcseconds by default)
            If None, size is in pixels
        unit_wave : `astropy.units.Unit`
            Wavelengths unit (angstrom by default)
            If None, inputs are in pixels

        """
        if size is None:
            try:
                white_ima = self.images['MUSE_WHITE']
            except:
                raise IOError('Size of the image is required')
            if white_ima.wcs.sameStep(cube.wcs):
                size = white_ima.shape[0]
                unit_size = None
            else:
                size = white_ima.wcs.get_step(unit=u.arcsec)[0] * white_ima.shape[0]
                unit_size = u.arcsec

        subcub = cube.subcube(center=(self.dec, self.ra), size=size, lbda=lbda,
                              unit_center=u.deg, unit_size=unit_size,
                              unit_wave=unit_wave)
        self.cubes[name] = subcub

    def add_white_image(self, cube, size=5, unit_size=u.arcsec):
        """Compute the white images from the MUSE data cube and appends it to
        the images dictionary.

        White image saved in self.images['MUSE_WHITE'].

        Parameters
        ----------
        cube : `~mpdaf.obj.Cube`
            MUSE data cube.
        size : float
            The total size to extract in arcseconds.
            It corresponds to the size along the delta axis and the image is
            square.  By default 5x5arcsec
        unit_size : `astropy.units.Unit`
            unit of the size value (arcseconds by default)
            If None, size is in pixels
        """
        subcub = cube.subcube(center=(self.dec, self.ra), size=size,
                              unit_center=u.deg, unit_size=unit_size)
        self.images['MUSE_WHITE'] = subcub.mean(axis=0)

    def add_FSF(self, cube):
        """Compute the mean FSF using the FSF keywords presents if the FITS
        header of the mosaic cube.

        Parameters
        ----------
        cube : `~mpdaf.obj.Cube`
                Input cube MPDAF object.
        """
        try:
            FSF_mode = cube.primary_header['FSFMODE']
        except:
            raise IOError('Cannot compute FSF int the FSF keywords are not'
                          'present in the primary header of the cube')
        if FSF_mode != 'MOFFAT1':
            raise IOError('This method is coded only for FSFMODE=MOFFAT1')

        if cube.primary_header['NFIELDS'] == 1:  # just one FSF
            nf = 0
            beta = cube.primary_header['FSF%02dBET' % nf]
            a = cube.primary_header['FSF%02dFWA' % nf]
            b = cube.primary_header['FSF%02dFWB' % nf]
        else:
            # load field map
            fmap = FieldsMap(cube.filename, extname='FIELDMAP')
            # load info from the white image
            try:
                white = self.images['MUSE_WHITE']
            except:
                raise IOError('Cannot compute FSF if the MUSE_WHITE image'
                              'doesn t exist.')
            size = white.shape[0]
            center = np.asarray((self.dec, self.ra))
            center = cube.wcs.sky2pix(center, unit=u.deg)[0]
            size = int(size + 0.5)
            radius = size / 2.
            [sy, sx], _, _ = bounding_box(
                form="rectangle", center=center, radii=radius,
                shape=cube.shape[1:])

            # compute corresponding sub field map
            subfmap = fmap[sy, sx]
            # weights
            w = np.array(subfmap.compute_weights())
            w *= white._data[np.newaxis, :, :]
            w = np.ma.sum(np.ma.masked_invalid(w), axis=(1, 2))
            w /= np.ma.sum(np.ma.masked_invalid(w))
            w = w.data
            # FSF
            ksel = np.where(w != 0)
            if len(ksel[0]) == 1:  # only one field
                nf = ksel[0][0] + 1
                beta = cube.primary_header['FSF%02dBET' % nf]
                a = cube.primary_header['FSF%02dFWA' % nf]
                b = cube.primary_header['FSF%02dFWB' % nf]
            else:  # several fields
                nf = 99
                # FSF model
                Nfsf = 13
                step_arcsec = cube.wcs.get_step(unit=u.arcsec)[0]
                FSF_model = FSF(FSF_mode)
                # compute FSF for minimum and maximum wavelength
                lbda1, lbda2 = cube.wave.get_range()
                FSF1 = np.zeros((Nfsf, Nfsf))
                FSF2 = np.zeros((Nfsf, Nfsf))
                for i in ksel[0]:
                    _i = i + 1
                    beta = cube.primary_header['FSF%02dBET' % _i]
                    a = cube.primary_header['FSF%02dFWA' % _i]
                    b = cube.primary_header['FSF%02dFWB' % _i]
                    kernel1 = FSF_model.get_FSF(lbda1, step_arcsec, Nfsf,
                                                beta=beta, a=a, b=b)[0]
                    kernel2 = FSF_model.get_FSF(lbda2, step_arcsec, Nfsf,
                                                beta=beta, a=a, b=b)[0]
                    FSF1 += w[i] * kernel1
                    FSF2 += w[i] * kernel2
                # fit beta, fwhm1 and fwhm2 on PSF1 and PSF2
                moffatfit = lambda v: np.ravel(\
                    MOFFAT1(lbda1, step_arcsec, Nfsf, v[0], v[1], v[2])[0] - FSF1 +\
                    MOFFAT1(lbda2, step_arcsec, Nfsf, v[0], v[1], v[2])[0] - FSF2)
                v0 = [beta, a, b]
                v = leastsq(moffatfit, v0)[0]
                beta = v[0]
                a = v[1]
                b = v[2]

        self.header['FSFMODE'] = FSF_mode
        self.header['FSF%02dBET' % nf] = np.around(beta, decimals=2)
        self.header['FSF%02dFWA' % nf] = np.around(a, decimals=3)
        self.header['FSF%02dFWB' % nf] = np.float('%.3e' % b)

    def add_narrow_band_images(self, cube, z_desc, eml=None, size=None,
                               unit_size=u.arcsec, width=8, is_sum=False,
                               subtract_off=True, margin=10., fband=3.):
        """Create narrow band images from a redshift value and a catalog of
        lines.

        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)

        Narrow-band images are saved in ``self.images['NB_']``.

        Parameters
        ----------
        cube : `~mpdaf.obj.Cube`
            MUSE data cube.
        z_desc : str
            Redshift description. The redshift value corresponding to
            this description will be used.
        eml : dict{float: str}
            Full catalog of lines
            Dictionary: key is the wavelength value in Angstrom,
            value is the name of the line.
            if None, the following catalog is used::

                eml = {1216 : 'LYALPHA', 1908: 'SUMCIII1907',
                        3727: 'SUMOII3726', 4863: 'HBETA' ,
                        5007: 'OIII5007', 6564: 'HALPHA'}

        size : float
            The total size to extract. It corresponds to the size along the
            delta axis and the image is square. If None, the size of the white
            image extension is taken if it exists.
        unit_size : `astropy.units.Unit`
            unit of the size value (arcseconds by default)
            If None, size is in pixels
        width : float
            Narrow-band width(in angstrom).
        is_sum : bool
            if True the image is computed as the sum over the wavelength axis,
            otherwise this is the average.
        subtract_off : bool
            If True, subtracting off nearby data.
            The method computes the subtracted flux by using the algorithm
            from Jarle Brinchmann (jarle@strw.leidenuniv.nl)::

                # if is_sum is False
                sub_flux = mean(flux[lbda1-margin-fband*(lbda2-lbda1)/2: lbda1-margin] +
                                flux[lbda2+margin: lbda2+margin+fband*(lbda2-lbda1)/2])

                # or if is_sum is True:
                sub_flux = sum(flux[lbda1-margin-fband*(lbda2-lbda1)/2: lbda1-margin] +
                                flux[lbda2+margin: lbda2+margin+fband*(lbda2-lbda1)/2]) /fband

        margin : float
            This off-band is offseted by margin wrt narrow-band limit (in
            angstrom).
        fband : float
            The size of the off-band is ``fband x narrow-band width`` (in
            angstrom).

        """
        if self.z is None:
            self._logger.warning('Cannot generate narrow band image if the '
                                 'redshift is None.')
            return

        if size is None:
            try:
                white = self.images['MUSE_WHITE']
            except:
                raise IOError('Size of the image is required')

            if white.wcs.sameStep(cube.wcs):
                size = white.shape[0]
                unit_size = None
            else:
                size = white.wcs.get_step(unit=u.arcsec)[0] * white.shape[0]
                unit_size = u.arcsec

        subcub = cube.subcube(center=(self.dec, self.ra), size=size,
                              unit_center=u.deg, unit_size=unit_size)

        z = self.z['Z'][self.z['Z_DESC'] == six.b(z_desc)]

        if z > 0:
            if eml is None:
                all_lines = np.array([1216, 1908, 3727, 4863, 5007, 6564])
                all_tags = np.array(['LYALPHA', 'SUMCIII1907', 'SUMOII3726',
                                     'HBETA', 'OIII5007', 'HALPHA'])
            else:
                all_lines = np.array(list(eml.keys()))
                all_tags = np.array(list(eml.values()))

            minl, maxl = subcub.wave.get_range(unit=u.angstrom) / (1 + z)
            useful = np.where((all_lines > minl) & (all_lines < maxl))
            nlines = len(useful[0])
            if nlines > 0:
                lambda_ranges = np.empty((2, nlines))
                lambda_ranges[0, :] = (1 + z) * all_lines[useful] - width / 2.0
                lambda_ranges[1, :] = (1 + z) * all_lines[useful] + width / 2.0
                tags = all_tags[useful]
                for l1, l2, tag in zip(lambda_ranges[0, :],
                                       lambda_ranges[1, :], tags):
                    # self._logger.debug('Generate narrow band image for NB_%s'
                    #                   ' with z=%s', tag, z[0])
                    self.images['NB_' + tag] = subcub.get_image(
                        wave=(l1, l2), is_sum=is_sum,
                        subtract_off=subtract_off, margin=margin,
                        fband=fband, unit_wave=u.angstrom)

    def add_narrow_band_image_lbdaobs(self, cube, tag, lbda, size=None,
                                      unit_size=u.arcsec, width=8,
                                      is_sum=False, subtract_off=True,
                                      margin=10., fband=3.):
        """Create narrow band image around an observed wavelength value.

        Parameters
        ----------
        cube : `~mpdaf.obj.Cube`
            MUSE data cube.
        tag : str
            key used to identify the new narrow band image in the images
            dictionary.
        lbda : float
            Observed wavelength value in angstrom.
        size : float
            The total size to extract in arcseconds. It corresponds to the size
            along the delta axis and the image is square. If None, the size of
            the white image extension is taken if it exists.
        unit_size : `astropy.units.Unit`
            unit of the size value (arcseconds by default)
            If None, size is in pixels
        width : float
            Angstrom total width
        is_sum : bool
            if True the image is computed as the sum over the wavelength axis,
            otherwise this is the average.
        subtract_off : bool
            If True, subtracting off nearby data.
        margin : float
            This off-band is offseted by margin wrt narrow-band limit (in
            angstrom).
        fband : float
            The size of the off-band is fband*narrow-band width (in angstrom).

        """
        if size is None:
            try:
                white_ima = self.images['MUSE_WHITE']
            except:
                raise IOError('Size of the image (in arcsec) is required')
            if white_ima.wcs.sameStep(cube.wcs):
                size = white_ima.shape[0]
                unit_size = None
            else:
                size = white_ima.wcs.get_step(unit=u.arcsec)[0] * white_ima.shape[0]
                unit_size = u.arcsec

        l1 = lbda - width / 2.0
        l2 = lbda + width / 2.0

        lmin, lmax = cube.wave.get_range(unit=u.angstrom)
        if l1 < lmin:
            l1 = lmin
        if l2 > lmax:
            l2 = lmax

        subcub = cube.subcube(center=(self.dec, self.ra), size=size,
                              unit_center=u.deg, unit_size=unit_size)
        self.images[tag] = subcub.get_image(wave=(l1, l2), is_sum=is_sum,
                                            subtract_off=subtract_off,
                                            margin=margin, fband=fband,
                                            unit_wave=u.angstrom)

    def add_seg_images(self, tags=None, DIR=None, del_sex=True):
        """Run SExtractor on all images to create segmentation maps.

        SExtractor will use the ``default.nnw``, ``default.param``,
        ``default.sex`` and ``*.conv`` files present in the current directory.
        If not present default parameter files are created or copied from the
        directory given in input (DIR).

        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)

        Parameters
        ----------
        tags : list of str
            List of tags of selected images
        DIR : str
            Directory that contains the configuration files of sextractor
        del_sex : bool
            If False, configuration files of sextractor are not removed.

        """
        if 'MUSE_WHITE' in self.images:
            if tags is None:
                tags = [tag for tag in self.images
                        if tag[0:4] != 'SEG_' and 'MASK' not in tag]

            segmentation(self, tags, DIR, del_sex)
        else:
            self._logger.warning('add_seg_images method use the MUSE_WHITE '
                                 'image computed by add_white_image method')

    @deprecated('`add_masks` method is deprecated, use `find_sky_mask` or '
                '`find_union_mask` or `find_intersection_mask`')
    def add_masks(self, tags=None):
        """Use the list of segmentation maps to compute the union mask and the
        intersection mask and the region where no object is detected in any
        segmentation map is saved in the sky mask.

        Masks are saved as boolean images:
        - Union is saved in ``self.images['MASK_UNION']``.
        - Intersection is saved in ``self.images['MASK_INTER']``.
        - Sky mask is saved in ``self.images['MASK_SKY']``.

        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl).

        Parameters
        ----------
        tags : list of str
            List of tags of selected segmentation images

        """
        maps = {}
        if tags is None:
            for tag, ima in six.iteritems(self.images):
                if tag[0:4] == 'SEG_':
                    maps[tag[4:]] = ima.data.data
        else:
            for tag in tags:
                if tag[0:4] == 'SEG_':
                    maps[tag[4:]] = self.images[tag].data.data
                else:
                    maps[tag] = self.images[tag].data.data
        if len(maps) == 0:
            self._logger.warning('no segmentation images. Use add_seg_images '
                                 'to create them')

        mask_creation(self, maps)

    def find_sky_mask(self, seg_tags, sky_mask='MASK_SKY'):
        """Loop over all segmentation images and use the region where no object is
        detected in any segmentation map as our sky image.

        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)

        Parameters
        ----------
        seg_tags : list of str
            List of tags of selected segmentation images.
        sky_mask : str
            Name of the sky mask image.
        """
        shape = self.images[seg_tags[0]].shape
        wcs = self.images[seg_tags[0]].wcs
        mask = np.ones(shape, dtype=np.bool)
        for key in seg_tags:
            im = self.images[key]
            if im.shape[0] == shape[0] and im.shape[1] == shape[1]:
                mask &= (~np.asarray(im.data, dtype=bool))
            else:
                raise IOError('segmentation maps have not the same dimensions')
        self.images[sky_mask] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                      data=mask)

    def find_union_mask(self, seg_tags, union_mask='MASK_UNION'):
        """Use the list of segmentation maps to compute the union mask.

        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl):

        1- Select on each segmentation map the object at the centre of the map.
        (the algo supposes that each objects have different labels)
        2- compute the union of these selected objects

        Parameters
        ----------
        tags : list of str
            List of tags of selected segmentation images
        union_mask : str
            Name of the union mask image.
        """
        wcs = self.images['MUSE_WHITE'].wcs
        yc, xc = wcs.sky2pix((self.DEC, self.RA), unit=u.deg)[0]
        maps = {}
        for tag in seg_tags:
            if tag[0:4] == 'SEG_':
                maps[tag[4:]] = self.images[tag].data.data
            else:
                maps[tag] = self.images[tag].data.data

        r = findCentralDetection(maps, yc, xc, tolerance=3)
        self.images[union_mask] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                        data=union(list(r['seg'].values())))

    def find_intersection_mask(self, seg_tags, inter_mask='MASK_INTER'):
        """Use the list of segmentation maps to compute the instersection mask.

        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl):

        1- Select on each segmentation map the object at the centre of the map.
        (the algo supposes that each objects have different labels)
        2- compute the intersection of these selected objects

        Parameters
        ----------
        tags : list of str
            List of tags of selected segmentation images
        inter_mask : str
            Name of the intersection mask image.

        """
        wcs = self.images['MUSE_WHITE'].wcs
        yc, xc = wcs.sky2pix((self.DEC, self.RA), unit=u.deg)[0]
        maps = {}
        for tag in seg_tags:
            if tag[0:4] == 'SEG_':
                maps[tag[4:]] = self.images[tag].data.data
            else:
                maps[tag] = self.images[tag].data.data

        r = findCentralDetection(maps, yc, xc, tolerance=3)
        self.images[inter_mask] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                        data=intersection(list(r['seg'].values())))

    def add_table(self, tab, name):
        """Append an astropy table to the tables dictionary.

        Parameters
        ----------
        tab : astropy.table.Table
            Input astropy table object.
        name : str
            Name used to distinguish this table

        """
        self.tables[name] = tab

    def extract_spectra(self, cube, obj_mask='MASK_UNION', sky_mask='MASK_SKY',
                        tags_to_try=('MUSE_WHITE', 'NB_LYALPHA',
                                     'NB_HALPHA', 'NB_SUMOII3726'),
                        skysub=True, psf=None, lbda=None,
                        unit_wave=u.angstrom):
        """Extract spectra from the MUSE data cube and from a list of
        narrow-band images (to define spectrum extraction apertures).

        First, this method computes a subcube that has the same size along the
        spatial axis as the image that contains the mask of the objet.

        Then, the no-weighting spectrum is computed as the sum of the subcube
        weighted by the mask of the object.  It is saved in
        ``self.spectra['MUSE_TOT']``.

        The weighted spectra are computed as the sum of the subcube weighted by
        the corresponding narrow bands image.  They are saved in
        ``self.spectra[nb_ima]`` (for nb_ima in tags_to_try).

        If psf:
            The potential PSF weighted spectrum is computed as the sum of the
            subcube weighted by mutliplication of the mask of the objetct and
            the PSF. It is saved in self.spectra['MUSE_PSF']

        If skysub:
            The local sky spectrum is computed as the average of the subcube
            weighted by the sky mask image.
            It is saved in ``self.spectra['MUSE_SKY']``

            The other spectra are computed on the sky-subtracted subcube and
            they are saved in ``self.spectra['*_SKYSUB']``.

        Algorithm from Jarle Brinchmann (jarle@strw.leidenuniv.nl)

        The weighted sum conserves the flux by :

        - Taking into account bad pixels in the addition.
        - Normalizing with the median value of weighting sum/no-weighting sum

        Parameters
        ----------
        cube : `~mpdaf.obj.Cube`
            MUSE data cube.
        obj_mask : str
            Name of the image that contains the mask of the object.
        sky_mask : str
            Name of the sky mask image.
        tags_to_try : list of str
            List of narrow bands images.
        skysub : bool
            If True, a local sky subtraction is done.
        psf : numpy.ndarray
            The PSF to use for PSF-weighted extraction.
            This can be a vector of length equal to the wavelength
            axis to give the FWHM of the Gaussian PSF at each
            wavelength (in arcsec) or a cube with the PSF to use.
            psf=None by default (no PSF-weighted extraction).
        lbda : (float, float) or none
            if not none, tuple giving the wavelength range.
        unit_wave : `astropy.units.Unit`
            Wavelengths unit (angstrom by default)
            If None, inputs are in pixels

        """
        if obj_mask not in self.images:
            raise ValueError('key %s not present in the images dictionary'
                             % obj_mask)

        if skysub and sky_mask not in self.images:
            raise ValueError('key %s not present in the images dictionary'
                             % sky_mask)

        ima = self.images[obj_mask]

        if ima.wcs.sameStep(cube.wcs):
            size = ima.shape[0]
            unit_size = None
        else:
            size = ima.wcs.get_step(unit=u.arcsec)[0] * ima.shape[0]
            unit_size = u.arcsec

        subcub = cube.subcube(center=(self.dec, self.ra), size=size,
                              unit_center=u.deg, unit_size=unit_size,
                              lbda=lbda, unit_wave=unit_wave)
        if ima.wcs.isEqual(subcub.wcs):
            object_mask = ima.data.data
        else:
            object_mask = ima.resample(
                newdim=(subcub.shape[1], subcub.shape[2]),
                newstart=subcub.wcs.get_start(unit=u.deg),
                newstep=subcub.wcs.get_step(unit=u.arcsec),
                order=0, unit_start=u.deg, unit_step=u.arcsec).data.data

        if skysub:
            if self.images[sky_mask].wcs.isEqual(subcub.wcs):
                skymask = self.images[sky_mask].data.data
            else:
                skymask = self.images[sky_mask].resample(
                    newdim=(subcub.shape[1], subcub.shape[2]),
                    newstart=subcub.wcs.get_start(unit=u.deg),
                    newstep=subcub.wcs.get_step(unit=u.arcsec),
                    order=0, unit_start=u.deg, unit_step=u.arcsec).data.data

            # Get the sky spectrum to subtract
            # FIXME: next line seems useless
            sky = compute_spectrum(subcub, weights=skymask)
            old_mask = subcub.data.mask.copy()
            subcub.data.mask[np.where(
                np.tile(skymask, (subcub.shape[0], 1, 1)) == 0)] = True
            sky = subcub.mean(axis=(1, 2))
            self.spectra['MUSE_SKY'] = sky
            subcub.data.mask = old_mask

            # substract sky
            subcub = subcub - sky

        # extract spectra
        # select narrow bands images
        nb_tags = list(set(tags_to_try) & set(self.images))

        # No weighting
        spec = compute_spectrum(subcub, weights=object_mask)
        if skysub:
            self.spectra['MUSE_TOT_SKYSUB'] = spec
        else:
            self.spectra['MUSE_TOT'] = spec

        # Now loop over the narrow-band images we want to use. Apply
        # the object mask and ensure that the weight map within the
        # object mask is >=0.
        # Weighted extractions
        ksel = (object_mask != 0)
        for tag in nb_tags:
            if self.images[tag].wcs.isEqual(subcub.wcs):
                weight = self.images[tag].data * object_mask
                weight[ksel] -= np.min(weight[ksel])
                weight = weight.filled(0)
                spec = compute_spectrum(subcub, weights=weight)
                if skysub:
                    self.spectra[tag + '_SKYSUB'] = spec
                else:
                    self.spectra[tag] = spec

        # PSF
        if psf is not None:
            if len(psf.shape) == 3:
                # PSF cube. The user is responsible for getting the
                # dimensions right
                if not np.array_equal(psf.shape, subcub.shape):
                    self._logger.warning(
                        'Incorrect dimensions for the PSF cube (%s) (it must '
                        'be (%s)) ', psf.shape, subcub.shape)
                    white_cube = None
                else:
                    white_cube = psf
            elif len(psf.shape) == 1 and psf.shape[0] == subcub.shape[0]:
                # a Gaussian expected.
                white_cube = np.zeros_like(subcub.data.data)
                for l in range(subcub.shape[0]):
                    gauss_ima = gauss_image(
                        shape=(subcub.shape[1], subcub.shape[2]),
                        wcs=subcub.wcs, fwhm=(psf[l], psf[l]), peak=False,
                        unit_fwhm=u.arcsec)
                    white_cube[l, :, :] = gauss_ima.data.data
            else:
                self._logger.warning('Incorrect dimensions for the PSF vector '
                                     '(%i) (it must be (%i)) ', psf.shape[0],
                                     subcub.shape[0])
                white_cube = None
            if white_cube is not None:
                weight = white_cube * np.tile(object_mask,
                                              (subcub.shape[0], 1, 1))
                spec = compute_spectrum(subcub, weights=weight)
                if skysub:
                    self.spectra['MUSE_PSF_SKYSUB'] = spec
                else:
                    self.spectra['MUSE_PSF'] = spec
                # Insert the PSF weighted flux - here re-normalised?

    def crack_z(self, eml=None, nlines=np.inf, cols=('LBDA_OBS', 'FLUX'),
                z_desc='EMI', zguess=None):
        """Estimate the best redshift matching the list of emission lines.

        Algorithm from Johan Richard (johan.richard@univ-lyon1.fr).

        This method saves the redshift values in ``self.z`` and lists the
        detected lines in ``self.lines``.  ``self.info()`` could be used to
        print the results.

        Parameters
        ----------
        eml : dict{float: str}
            Full catalog of lines to test redshift
            Dictionary: key is the wavelength value in Angtsrom,
            value is the name of the line.
            if None, the following catalog is used::

                emlines = {
                    1215.67: 'LYALPHA1216'  , 1550.0: 'CIV1550'       ,
                    1908.0: 'CIII]1909'     , 2326.0: 'CII2326'       ,
                    3726.032: '[OII]3726'   , 3728.8149: '[OII]3729'  ,
                    3798.6001: 'HTHETA3799' , 3834.6599: 'HETA3835'   ,
                    3869.0: '[NEIII]3869'   , 3888.7: 'HZETA3889'     ,
                    3967.0: '[NEIII]3967'   , 4102.0: 'HDELTA4102'    ,
                    4340.0: 'HGAMMA4340'    , 4861.3198: 'HBETA4861'  ,
                    4959.0: '[OIII]4959'    , 5007.0: '[OIII]5007'    ,
                    6548.0: '[NII6548]'     , 6562.7998: 'HALPHA6563' ,
                    6583.0: '[NII]6583'     , 6716.0: '[SII]6716'     ,
                    6731.0: '[SII]6731'
                }

        nlines : int
            estimated the redshift if the number of emission lines is
            inferior to this value
        cols : (str, str)
            tuple (wavelength column name, flux column name)
            Two columns of self.lines that will be used to define the emission
            lines.
        z_desc : str
            Estimated redshift will be saved in self.z table under these name.
        zguess : float
            Guess redshift. Test if this redshift is a match and fills the
            detected lines

        """
        nline_max = nlines
        if eml is None:
            eml = emlines
        col_lbda, col_flux = cols
        if self.lines is None:
            raise IOError('invalid self.lines table')
        if col_lbda not in self.lines.colnames:
            raise IOError('invalid colum name %s' % col_lbda)
        if col_flux not in self.lines.colnames:
            raise IOError('invalid colum name %s' % col_flux)

        try:
            # vacuum wavelengths
            wl = air2vacuum(np.array(self.lines[col_lbda]))
            flux = np.array(self.lines[col_flux])
            nlines = len(wl)
        except:
            self._logger.info('Impossible to estimate the redshift, no '
                              'emission lines')
            return

        z, errz, nlines, wl, flux, lnames = crackz(nlines, wl, flux, eml,
                                                   zguess)
        # observed wavelengths
        wl = vacuum2air(wl)

        if nlines > 0:
            if nlines < nline_max:
                # redshift
                self.add_z(z_desc, z, errz)
                self._logger.info('crack_z: z=%0.6f err_z=%0.6f' % (z, errz))
                # line names
                if 'LINE' not in self.lines.colnames:
                    nlines = len(self.lines)
                    col = MaskedColumn(ma.masked_array(np.array([''] * nlines),
                                                       mask=np.ones(nlines)),
                                       name='LINE', dtype='S20',
                                       unit='unitless',
                                       description='line name')
                    self.lines.add_column(col)
                for w, name in zip(wl, lnames):
                    self.lines['LINE'][self.lines[col_lbda] == w] = name
                self._logger.info('crack_z: lines')
                for l in self.lines.pformat():
                    self._logger.info(l)
            else:
                self._logger.info('Impossible to estimate the redshift, the '
                                  'number of emission lines is inferior to %d',
                                  nline_max)
        else:
            self._logger.info('Impossible to estimate the redshift, no '
                              'emission lines')

    def sort_lines(self, nlines_max=25):
        """Sort lines by flux in descending order.

        Parameters
        ----------
        nlines_max : int
            Maximum number of stored lines

        """
        if self.lines is not None:
            if isinstance(self.lines['LINE'], MaskedColumn):
                self.lines['LINE'] = self.lines['LINE'].filled('')

            subtab1 = self.lines[self.lines['LINE'] != b'']
            subtab1.sort('FLUX')
            subtab1.reverse()
            n1 = len(subtab1)
            subtab2 = self.lines[self.lines['LINE'] == b'']
            subtab2.sort('FLUX')
            subtab2.reverse()
            n2 = len(subtab2)
            if (n1 + n2) > nlines_max:
                n2 = max(nlines_max - n1, 0)
            self.lines = vstack([subtab1, subtab2[0:n2]])

    def show_ima(self, ax, name, showcenter=None,
                 cuts=None, cmap=cm.gray_r, **kwargs):
        """Show image.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        name : str
            Name of image to display.
        showcenter : (float, str)
            radius in arcsec and color used to plot a circle around the center
            of the source.
        cuts : (float, float)
            Minimum and maximum values to use for the scaling.
        cmap : matplotlib.cm
            Color map.
        kwargs : matplotlib.artist.Artist
            kwargs can be used to set additional plotting properties.

        """
        if name not in self.images:
            raise ValueError('Image %s not found' % name)
        zima = self.images[name]
        if cuts is None:
            vmin = None
            vmax = None
        else:
            vmin, vmax = cuts
        if 'title' not in kwargs:
            kwargs['title'] = '%s' % (name)
        zima.plot(vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, **kwargs)
        if showcenter is not None:
            rad, col = showcenter
            pix = zima.wcs.sky2pix((self.DEC, self.RA))[0]
            rpix = rad / zima.wcs.get_step(unit=u.arcsec)[0]
            ell = Ellipse((pix[1], pix[0]), 2 * rpix, 2 * rpix, 0, fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(1)
            ell.set_edgecolor(col)
        ax.axis('off')
        return

    def show_spec(self, ax, name, cuts=None, zero=False, sky=None, lines=None,
                  **kwargs):
        """Display a spectra.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        name : str
            Name of spectra to display.
        cuts : (float, float)
            Minimum and maximum values to use for the scaling.
        zero : float
            If True, the 0 flux line is plotted in black.
        sky : `~mpdaf.obj.Spectrum`
            Sky spectra to overplot (default None).
        lines : str
            Name of a columns of the lines table containing wavelength values.
            If not None, overplot red vertical lines at the given wavelengths.
        kwargs : matplotlib.artist.Artist
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
                sky.plot(ax=ax2, color='k', alpha=0.2, lmin=kwargs['lmin'],
                         lmax=kwargs['lmax'])
            else:
                sky.plot(ax=ax2, color='k', alpha=0.2)
            ax2.axis('off')
        if lines is not None:
            wavelist = self.lines[lines]
            for lbda in wavelist:
                ax.axvline(lbda, color='r')
        return

    def masked_invalid(self, tables=None):
        """Mask where invalid values occur (NaNs or infs or -9999 or '')."""
        if tables is not None:
            tables = [tables] if isinstance(tables, Table) else tables
        else:
            tables = ([self.lines, self.mag, self.z] +
                      list(self.tables.values()))

        for tab in tables:
            if tab is not None:
                for name, col in tab.columns.items():
                    try:
                        tab[name] = ma.masked_invalid(col)
                        tab[name] = ma.masked_equal(col, -9999)
                    except:
                        pass


class SourceList(list):

    """Handles a list of `mpdaf.sdetect.Source` objects."""

    source_class = Source

    def write(self, name, path='.', overwrite=True, fmt='default'):
        """Create the directory and saves all sources files and the catalog
        file in this folder.

        ``path/name.fits``: catalog file
        (In FITS table, the maximum number of fields is 999.
        In this case, the catalog is saved as an ascci table).

        ``path/name/nameNNNN.fits``: source file (NNNN corresponds to the ID of
        the source)

        Parameters
        ----------
        name : str
            Name of the catalog
        path : str
            path where the catalog will be saved.
        overwrite : bool
            Overwrite the catalog if it already exists
        fmt : str, 'working' or 'default'
            Format of the catalog. The format differs for the LINES table.
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
            source.write('%s/%s-%04d.fits' % (path2, name, source.ID))

        fcat = '%s/%s.fits' % (path, name)
        if overwrite and os.path.isfile(fcat):
            os.remove(fcat)

        from .catalog import Catalog
        cat = Catalog.from_sources(self, fmt)
        try:
            cat.write(fcat)
        except:
            cat.write(fcat.replace('.fits', '.txt'), format='ascii')

    @classmethod
    def from_path(cls, path):
        """Read a SourceList object from the path of a directory containing
        source files.

        Parameters
        ----------
        path : str
            Directory containing Source files
        """
        if not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))

        slist = cls()
        for f in glob.glob(path + '/*.fits'):
            slist.append(self.source_class.from_file(f))

        return slist
