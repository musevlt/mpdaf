"""cube.py manages Cube objects."""

import numpy as np
from .cube import CubeDisk
from .objs import is_float, is_int


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
    """

    def __init__(self, files):
        """Creates a CubeList object.

        Parameters
        ----------
        files : list of strings
                List of cubes fits filenames
        """
        self.files = files
        self.nfiles = len(files)
        self.cubes = [CubeDisk(f) for f in self.files]
        self.shape = None
        self.fscale = None
        self.wcs = None
        self.wave = None
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
        shapes = np.array([cube.shape for cube in self.cubes])

        if not np.all(shapes == shapes[0]):
            print 'all cubes have not same dimensions'
            for i in range(self.nfiles):
                print '%i X %i X %i cube (%s)' % (shapes[i, 0], shapes[i, 1],
                                                  shapes[i, 2], self.files[i])
            print ''
            return False
        else:
            self.shape = shapes[0, :]
            return True

    def check_wcs(self):
        """Checks if all cubes have same world coordinates."""
        self.wcs = self.cubes[0].wcs
        self.wave = self.cubes[0].wave

        for f, cube in zip(self.files, self.cubes):
            if not cube.wcs.isEqual(self.wcs) or \
                    not cube.wave.isEqual(self.wave):
                if not cube.wcs.isEqual(self.wcs):
                    print 'all cubes have not same spatial coordinates'
                    print self.files[0]
                    self.wcs.info()
                    print ''
                    print f
                    cube.wcs.info()
                    print ''
                    self.wcs = None
                if not cube.wave.isEqual(self.wave):
                    print 'all cubes have not same spectral coordinates'
                    print self.files[0]
                    self.wave.info()
                    print ''
                    print f
                    cube.wave.info()
                    print ''
                    self.wave = None
                return False
        return True

    def check_fscale(self):
        """Checks if all cubes have same scale factor."""
        fscale = np.array([cube.fscale for cube in self.cubes])

        if len(np.unique(fscale)) == 1:
            self.fscale = fscale[0]
            return True
        else:
            print 'all cubes have not same scale factor'
            for i in range(self.nfiles):
                print 'fscale=%g (%s)' % (fscale[i], self.files[i])
            print ''
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
        """
        import mpdaf
        import ctypes

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods", mpdaf.__path__[0])
        # define argument types
        charptr = ctypes.POINTER(ctypes.c_char)
        # setup argument types
        libCmethods.mpdaf_merging_median.argtypes = [charptr, charptr]
        # run C method
        libCmethods.mpdaf_merging_median(ctypes.c_char_p('\n'.join(self.files)), ctypes.c_char_p(output), ctypes.c_char_p(output_path))

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
        """
        import mpdaf
        import ctypes
        if is_int(nclip) or is_float(nclip):
            nclip_low = nclip
            nclip_up = nclip
        else:
            nclip_low = nclip[0]
            nclip_up = nclip[1]

        # load the library, using numpy mechanisms
        libCmethods = np.ctypeslib.load_library("libCmethods", mpdaf.__path__[0])
        # define argument types
        charptr = ctypes.POINTER(ctypes.c_char)
        # setup argument types
        libCmethods.mpdaf_merging_sigma_clipping.argtypes = [charptr, charptr, charptr, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
        # run C method
        libCmethods.mpdaf_merging_sigma_clipping(ctypes.c_char_p('\n'.join(self.files)), ctypes.c_char_p(output), ctypes.c_char_p(output_path), nmax, np.float64(nclip_low), np.float64(nclip_up), nstop, np.int32(var_mean))
