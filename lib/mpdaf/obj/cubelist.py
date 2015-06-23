"""cube.py manages Cube objects."""

import ctypes
import logging
import numpy as np
import os

from astropy.table import Table
from ctypes import c_char_p

from .cube import CubeDisk, Cube
from .objs import is_float, is_int
from ..tools.fits import add_mpdaf_method_keywords, copy_keywords


class CubeList(object):

    """This class manages a list of cube FITS filenames.

    Parameters
    ----------
    files : list of strings
            List of cubes fits filenames

    Attributes
    ----------
    files  : list of strings
             List of cubes fits filenames
    nfiles : integer
             Number of files.
    shape  : array of 3 integers)
             Lengths of data in Z and Y and X
             (python notation (nz,ny,nx)).
    fscale : float
             Flux scaling factor (1 by default).
    wcs    : :class:`mpdaf.obj.WCS`
             World coordinates.
    wave   : :class:`mpdaf.obj.WaveCoord`
             Wavelength coordinates
    unit   : string
             Possible data unit type. None by default.
    """

    def __init__(self, files):
        """Creates a CubeList object.

        Parameters
        ----------
        files : list of strings
                List of cubes fits filenames
        """
        self.logger = logging.getLogger('mpdaf corelib')
        self.files = files
        self.nfiles = len(files)
        self.cubes = [CubeDisk(f) for f in self.files]
        self.shape = None
        self.fscale = None
        self.wcs = None
        self.wave = None
        self.unit = None
        self.check_compatibility()

    def __getitem__(self, item):
        """Apply a slice on all the cubes.

        See :meth:`mpdaf.obj.CubeDisk.__getitem__` for details.
        """
        if not (isinstance(item, tuple) and len(item) == 3):
            raise ValueError('Operation forbidden')

        return np.array([cube[item] for cube in self.cubes])

    def info(self):
        """Prints information."""
        for cube in self.cubes:
            cube.info()

    def check_dim(self):
        """Checks if all cubes have same dimensions."""
        d = {'class': 'CubeList', 'method': 'check_dim'}
        shapes = np.array([cube.shape for cube in self.cubes])

        if not np.all(shapes == shapes[0]):
            msg = 'all cubes have not same dimensions'
            self.logger.warning(msg, extra=d)
            for i in range(self.nfiles):
                msg = '%i X %i X %i cube (%s)' % (shapes[i, 0], shapes[i, 1],
                                                  shapes[i, 2], self.files[i])
                self.logger.warning(msg, extra=d)
            return False
        else:
            self.shape = shapes[0, :]
            return True

    def check_wcs(self):
        """Checks if all cubes have same world coordinates."""
        d = {'class': 'CubeList', 'method': 'check_wcs'}
        self.wcs = self.cubes[0].wcs
        self.wave = self.cubes[0].wave

        for f, cube in zip(self.files, self.cubes):
            if not cube.wcs.isEqual(self.wcs) or \
                    not cube.wave.isEqual(self.wave):
                if not cube.wcs.isEqual(self.wcs):
                    msg = 'all cubes have not same spatial coordinates'
                    self.logger.warning(msg, extra=d)
                    self.logger.info(self.files[0], extra=d)
                    self.wcs.info()
                    self.logger.info(f, extra=d)
                    cube.wcs.info()
                if not cube.wave.isEqual(self.wave):
                    msg = 'all cubes have not same spectral coordinates'
                    self.logger.warning(msg, extra=d)
                    self.logger.info(self.files[0], extra=d)
                    self.wave.info()
                    self.logger.info(f, extra=d)
                    cube.wave.info()
                return False
        return True

    def check_fscale(self):
        """Checks if all cubes have same unit and same scale factor."""
        d = {'class': 'CubeList', 'method': 'check_fscale'}
        fscale = np.array([cube.fscale for cube in self.cubes])
        unit = np.array([cube.unit for cube in self.cubes])

        self.fscale = fscale[0]
        self.unit = unit[0]
        if len(np.unique(fscale)) == 1 and len(np.unique(unit)) == 1:
            return True
        else:
            if len(np.unique(fscale)) > 1:
                msg = ('All cubes have not same scale factor. Scale from the '
                       'first cube will be used.')
                self.logger.warning(msg, extra=d)
                for i in range(self.nfiles):
                    msg = 'FSCALE=%s (%s)' % (fscale[i], self.files[i])
                    self.logger.info(msg, extra=d)
            if len(np.unique(unit)) > 1:
                msg = ('All cubes have not same unit. Unit from the first cube'
                       ' will be used.')
                self.logger.warning(msg, extra=d)
                for i in range(self.nfiles):
                    msg = 'BUNIT=%s (%s)' % (unit[i], self.files[i])
                    self.logger.info(msg, extra=d)
            return False

    def check_compatibility(self):
        """Checks if all cubes are compatible."""
        return self.check_dim() and self.check_wcs() and self.check_fscale()

    def save_combined_cube(self, data, var=None, method='', keywords=None):
        c = Cube(data=data.reshape(self.shape), wcs=self.wcs,
                 wave=self.wave, unit=self.unit, var=var)
        hdr = c.primary_header
        nfiles = len(self.files)
        copy_keywords(self.cubes[0].primary_header, hdr,
                      ('ORIGIN', 'TELESCOP', 'INSTRUME', 'EQUINOX',
                       'RADECSYS', 'EXPTIME', 'OBJECT'))
        try:
            hdr['EXPTIME'] = hdr['EXPTIME'] * nfiles
        except:
            pass
        try:
            c.data_header['OBJECT'] = self.cubes[0].data_header['OBJECT']
        except:
            pass

        if keywords is not None:
            params, values, comments = zip(*keywords)
        else:
            params, values, comments = [], [], []

        add_mpdaf_method_keywords(hdr, method, params, values, comments)

        files = ','.join(os.path.basename(f) for f in self.files)
        hdr['NFILES'] = (nfiles, 'number of files merged in this cube')
        hdr['FILES'] = (files, 'list of files merged in this cube')
        return c

    def median(self):
        """combines cubes in a single data cube using median.

        Parameters
        ----------
        output      : string
                      DATACUBE_<output>.fits will contain the merged cube

                      EXPMAP_<output>.fits will contain an exposure map
                      data cube which counts the number of exposures used
                      for the combination of each pixel.
        output_path : string
                      Output path where resulted cubes are stored.

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`, :class:`mpdaf.obj.Cube`, Table
              cube, expmap, statpix

              cube will contain the merged cube

              expmap will contain an exposure map
              data cube which counts the number of exposures used
              for the combination of each pixel.

              statpix is a table that will give the number of Nan
              pixels pixels per exposures
              (columns are FILENAME and NPIX_NAN)
        """
        # load the library, using numpy mechanisms
        path = os.path.dirname(__file__)[:-4]
        libCmethods = np.ctypeslib.load_library("libCmethods", path)
        # define argument types
        charptr = ctypes.POINTER(ctypes.c_char)
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')
        # setup argument types
        libCmethods.mpdaf_merging_median.argtypes = [
            charptr, array_1d_double, array_1d_int, array_1d_int]
        # run C method
        npixels = self.shape[0]*self.shape[1]*self.shape[2]
        data = np.empty(npixels, dtype=np.float64)
        expmap = np.empty(npixels, dtype=np.int32)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        libCmethods.mpdaf_merging_median(c_char_p('\n'.join(self.files)), data,
                                         expmap, valid_pix)

        # no valid pixels
        no_valid_pix = [npixels - npix for npix in valid_pix]
        stat_pix = Table([self.files, no_valid_pix],
                         names=['FILENAME', 'NPIX_NAN'])

        expmap = self.save_combined_cube(expmap, method='obj.cubelist.median')
        cube = self.save_combined_cube(data, method='obj.cubelist.median')
        return cube, expmap, stat_pix

    def combine(self, nmax=2, nclip=5.0, nstop=2, var='propagate', mad=False):
        """combines cubes in a single data cube using sigma clipped mean.

        Parameters
        ----------
        nmax        : integer
                      maximum number of clipping iterations
        nclip       : float or (float,float)
                      Number of sigma at which to clip.
                      Single clipping parameter or lower / upper clipping
                      parameters.
        nstop       : integer
                      If the number of not rejected pixels is less
                      than this number, the clipping iterations stop.
        var         : string
                      'propagate', 'stat_mean', 'stat_one'

                      'propagate': the variance is the sum of the variances
                      of the N individual exposures divided by N**2.

                      'stat_mean': the variance of each combined pixel
                      is computed as the variance derived from the comparison
                      of the N individual exposures divided N-1.

                      'stat_one': the variance of each combined pixel is
                      computed as the variance derived from the comparison
                      of the N individual exposures.
        mad         : boolean
                      

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`, :class:`mpdaf.obj.Cube`, astropy.table
              cube, expmap, statpix

              cube will contain the merged cube

              expmap will contain an exposure map
              data cube which counts the number of exposures used
              for the combination of each pixel.

              statpix is a table that will give the number of Nan
              pixels and rejected pixels per exposures
              (columns are FILENAME, NPIX_NAN and NPIX_REJECTED)
        """
        if is_int(nclip) or is_float(nclip):
            nclip_low = nclip
            nclip_up = nclip
        else:
            nclip_low = nclip[0]
            nclip_up = nclip[1]

        # load the library, using numpy mechanisms
        path = os.path.dirname(__file__)[:-4]
        libCmethods = np.ctypeslib.load_library("libCmethods", path)
        # define argument types
        charptr = ctypes.POINTER(ctypes.c_char)
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                              flags='CONTIGUOUS')

        # returned arrays
        npixels = self.shape[0]*self.shape[1]*self.shape[2]
        data = np.empty(npixels, dtype=np.float64)
        vardata = np.empty(npixels, dtype=np.float64)
        expmap = np.empty(npixels, dtype=np.int32)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        select_pix = np.zeros(self.nfiles, dtype=np.int32)

        if var == 'propagate':
            var_mean = 0
        elif var == 'stat_mean':
            var_mean = 1
        else:
            var_mean = 2
            
        # setup argument types
        libCmethods.mpdaf_merging_sigma_clipping.argtypes = [
                charptr, array_1d_double, array_1d_double, array_1d_int,
                array_1d_int, array_1d_int, ctypes.c_int, ctypes.c_double,
                ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        # run C method
        libCmethods.mpdaf_merging_sigma_clipping(
                c_char_p('\n'.join(self.files)), data, vardata, expmap,
                select_pix, valid_pix, nmax, np.float64(nclip_low),
                np.float64(nclip_up), nstop, np.int32(var_mean), np.int32(mad))

        # no valid pixels
        no_valid_pix = [npixels - npix for npix in valid_pix]
        rejected_pix = [valid - select for valid, select in zip(valid_pix,
                                                                select_pix)]
        statpix = Table([self.files, no_valid_pix, rejected_pix],
                        names=['FILENAME', 'NPIX_NAN', 'NPIX_REJECTED'])

        keywords = [('nmax', nmax, 'max number of clipping iterations'),
                    ('nclip_low', nclip_low, 'lower clipping parameter'),
                    ('nclip_up', nclip_up, 'upper clipping parameter'),
                    ('nstop', nstop, 'clipping minimum number'),
                    ('var', var, 'type of variance')]
        expmap = self.save_combined_cube(expmap, method='obj.cubelist.merging',
                                         keywords=keywords)
        cube = self.save_combined_cube(data, var=vardata.reshape(self.shape),
                                       method='obj.cubelist.merging',
                                       keywords=keywords)
        return cube, expmap, statpix
