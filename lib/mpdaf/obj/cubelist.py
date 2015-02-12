"""cube.py manages Cube objects."""

import ctypes
import logging
import numpy as np
import os

from astropy.io import fits
from ctypes import c_char_p

from .cube import CubeDisk, Cube
from .objs import is_float, is_int
from ..tools.fits import add_mpdaf_method_keywords


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
                    self.wcs = None
                if not cube.wave.isEqual(self.wave):
                    msg = 'all cubes have not same spectral coordinates'
                    self.logger.warning(msg, extra=d)
                    self.logger.info(self.files[0], extra=d)
                    self.wave.info()
                    self.logger.info(f, extra=d)
                    cube.wave.info()
                    self.wave = None
                return False
        return True

    def check_fscale(self):
        """Checks if all cubes have same unit and same scale factor."""
        d = {'class': 'CubeList', 'method': 'check_fscale'}
        fscale = np.array([cube.fscale for cube in self.cubes])
        unit = np.array([cube.unit for cube in self.cubes])

        if len(np.unique(fscale)) == 1 and len(np.unique(unit)):
            self.fscale = fscale[0]
            self.unit = unit[0]
            return True
        else:
            if len(np.unique(fscale)) > 1:
                msg = 'all cubes have not same scale factor'
                self.logger.info(msg, extra=d)
                for i in range(self.nfiles):
                    msg = 'FSCALE=%g (%s)' % (fscale[i], self.files[i])
                    self.logger.info(msg, extra=d)
            if len(np.unique(unit)) > 1:
                msg = 'all cubes have not same unit'
                self.logger.info(msg, extra=d)
                for i in range(self.nfiles):
                    msg = 'BUNIT=%g (%s)' % (unit[i], self.files[i])
                    self.logger.info(msg, extra=d)
            return False

    def check_compatibility(self):
        """Checks if all cubes are compatible."""
        return self.check_dim() and self.check_wcs() and self.check_fscale()

    def median(self, output, output_path='.'):
        """merges cubes in a single data cube using median.

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
        out : :class:`mpdaf.obj.Cube`
        """
        import mpdaf

        cubepath = output_path + '/DATACUBE_' + output + '.fits'
        try:
            os.remove(cubepath)
            os.remove(output_path + '/EXPMAP_' + output + '.fits')
        except OSError:
            pass

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods", mpdaf.__path__[0])
        # define argument types
        charptr = ctypes.POINTER(ctypes.c_char)
        # setup argument types
        libCmethods.mpdaf_merging_median.argtypes = [charptr, charptr]
        # run C method
        libCmethods.mpdaf_merging_median(c_char_p('\n'.join(self.files)),
                                         c_char_p(output),
                                         c_char_p(output_path))

        # update header
        hdu = fits.open(cubepath, mode='update')
        add_mpdaf_method_keywords(hdu[0].header, "obj.cubelist.median",
                                  [], [], [])
        files = ','.join(os.path.basename(f) for f in self.files)
        hdu[0].header['NFILES'] = (len(self.files),
                                   'number of files merged in this cube')
        hdu[0].header['FILES'] = (files, 'list of files merged in this cube')
        hdu.flush()
        hdu.close()

    def merging(self, output, output_path='.', nmax=2, nclip=5.0, nstop=2, var_mean=True):
        """merges cubes in a single data cube using sigma clipped mean.

        Parameters
        ----------
        output      : string
                      DATACUBE_<output>.fits will contain the merged cube

                      EXPMAP_<output>.fits will contain an exposure map
                      data cube which counts the number of exposures used
                      for the combination of each pixel.

                      NOVALID_<output>.txt will give the number of invalid
                      pixels per exposures.
        output_path : string
                      Output path where resulted cubes are stored.
        nmax        : integer
                      maximum number of clipping iterations
        nclip       : float or (float,float)
                      Number of sigma at which to clip.
                      Single clipping parameter or lower / upper clipping parameters
        nstop       : integer
                      If the number of not rejected pixels is less
                      than this number, the clipping iterations stop.
        var_mean    : boolean
                      True: the variance of each combined pixel is computed
                      as the variance derived from the comparison of the
                      N individual exposures divided N-1.

                      False: the variance of each combined pixel is computed
                      as the variance derived from the comparison of the
                      N individual exposures.

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`
        """
        import mpdaf

        cubepath = output_path + '/DATACUBE_' + output + '.fits'

        if is_int(nclip) or is_float(nclip):
            nclip_low = nclip
            nclip_up = nclip
        else:
            nclip_low = nclip[0]
            nclip_up = nclip[1]

        try:
            os.remove(cubepath)
            os.remove(output_path + '/EXPMAP_' + output + '.fits')
            os.remove(output_path + '/NOVALID_' + output + '.txt')
        except OSError:
            pass

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods", mpdaf.__path__[0])
        # define argument types
        charptr = ctypes.POINTER(ctypes.c_char)
        # setup argument types
        libCmethods.mpdaf_merging_sigma_clipping.argtypes = [
            charptr, charptr, charptr, ctypes.c_int, ctypes.c_double,
            ctypes.c_double, ctypes.c_int, ctypes.c_int
        ]
        # run C method
        libCmethods.mpdaf_merging_sigma_clipping(
            c_char_p('\n'.join(self.files)), c_char_p(output),
            c_char_p(output_path), nmax, np.float64(nclip_low),
            np.float64(nclip_up), nstop, np.int32(var_mean)
        )

        # update header
        hdu = fits.open(cubepath, mode='update')
        add_mpdaf_method_keywords(
            hdu[0].header, "obj.cubelist.merging",
            ['nmax', 'nclip_low', 'nclip_up', 'nstop', 'var_mean'],
            [nmax, nclip_low, nclip_up, nstop, var_mean],
            ['max number of clipping iterations',
             'lower clipping parameter',
             'upper clipping parameter',
             'clipping minimum number',
             'variance divided or not by N-1']
        )
        files = ','.join(os.path.basename(f) for f in self.files)
        hdu[0].header['NFILES'] = (len(self.files),
                                   'number of files merged in this cube')
        hdu[0].header['FILES'] = (files, 'list of files merged in this cube')
        hdu.flush()
        hdu.close()
