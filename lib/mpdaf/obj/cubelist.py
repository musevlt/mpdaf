"""cube.py manages Cube objects."""

import ctypes
import logging
import numpy as np
import os

from astropy.table import Table
from astropy.utils.console import ProgressBar
from ctypes import c_char_p
from numpy import ma, allclose, array_equal

from .cube import CubeDisk, Cube
from .objs import is_float, is_int
from ..tools.fits import add_mpdaf_method_keywords, copy_keywords

__all__ = ['CubeList', 'CubeMosaic']


class CubeList(object):

    """Manages a list of cubes and handles the combination.

    To run the combination, all the cubes must have the same dimensions and be
    on the same WCS grid.

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
        Lengths of data in Z and Y and X (python notation (nz,ny,nx)).
    fscale : float
        Flux scaling factor (1 by default).
    wcs    : :class:`mpdaf.obj.WCS`
        World coordinates.
    wave   : :class:`mpdaf.obj.WaveCoord`
        Wavelength coordinates
    unit   : string
        Possible data unit type. None by default.

    """

    checkers = ('check_dim', 'check_wcs', 'check_fscale')

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
        self._set_defaults()
        self.check_compatibility()

    def _set_defaults(self):
        self.shape = self.cubes[0].shape
        self.wcs = self.cubes[0].wcs
        self.wave = self.cubes[0].wave
        self.fscale = self.cubes[0].fscale
        self.unit = self.cubes[0].unit

    def __getitem__(self, item):
        """Apply a slice on all the cubes.

        See :meth:`mpdaf.obj.CubeDisk.__getitem__` for details.
        """
        if not (isinstance(item, tuple) and len(item) == 3):
            raise ValueError('Operation forbidden')

        return np.array([cube[item] for cube in self.cubes])

    def info(self, verbose=False):
        """Prints information."""
        d = {'class': 'CubeList', 'method': 'info'}
        rows = [(os.path.basename(c.filename),
                 'x'.join(str(s) for s in c.shape),
                 str(c.wcs.wcs.wcs.crpix), str(c.wcs.wcs.wcs.crval))
                for c in self.cubes]
        t = Table(rows=rows, names=('filename', 'shape', 'crpix', 'crval'))
        for line in t.pformat():
            self.logger.info(line, extra=d)

        if verbose:
            self.logger.info('Detailed information per file:', extra=d)
            for cube in self.cubes:
                cube.info()

    def check_dim(self):
        """Checks if all cubes have same dimensions."""
        shapes = np.array([cube.shape for cube in self.cubes])

        if not np.all(shapes == self.shape):
            d = {'class': 'CubeList', 'method': 'check_dim'}
            self.logger.warning('all cubes have not same dimensions', extra=d)
            for i in range(self.nfiles):
                self.logger.warning('%i X %i X %i cube (%s)', shapes[i, 0],
                                    shapes[i, 1], shapes[i, 2], self.files[i],
                                    extra=d)
            return False
        else:
            return True

    def check_wcs(self):
        """Checks if all cubes have same world coordinates."""
        d = {'class': 'CubeList', 'method': 'check_wcs'}
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
        for checker in self.checkers:
            getattr(self, checker)()

    def save_combined_cube(self, data, var=None, method='', keywords=None,
                           expnb=None, object_name=None, save_unit=True):
        d = {'class': 'CubeList', 'method': 'merging'}
        self.logger.info('Creating combined cube object', extra=d)

        if data.ndim != 3:
            data = data.reshape(self.shape)
        if var is not None and var.ndim != 3:
            var = var.reshape(self.shape)

        c = Cube(shape=self.shape, wcs=self.wcs, wave=self.wave,
                 unit=self.unit if save_unit else None)
        c.data = ma.asarray(data)
        c.var = var

        hdr = c.primary_header
        copy_keywords(
            self.cubes[0].primary_header, hdr,
            ('ORIGIN', 'TELESCOP', 'INSTRUME', 'RA', 'DEC', 'EQUINOX',
             'RADECSYS', 'EXPTIME', 'MJD-OBS', 'DATE-OBS', 'PI-COI',
             'OBSERVER', 'OBJECT', 'ESO INS DROT POSANG', 'ESO INS MODE',
             'ESO DET READ CURID', 'ESO INS TEMP11 VAL', 'ESO OBS ID',
             'ESO OBS NAME', 'ESO OBS START', 'ESO TEL AIRM END',
             'ESO TEL AIRM START', 'ESO TEL AMBI FWHM END',
             'ESO TEL AMBI FWHM START'))

        if expnb is not None and 'EXPTIME' in hdr:
            hdr['EXPTIME'] = hdr['EXPTIME'] * expnb

        if object_name is not None:
            c.primary_header['OBJECT'] = object_name
            c.data_header['OBJECT'] = object_name
        elif 'OBJECT' in self.cubes[0].data_header:
            c.data_header['OBJECT'] = self.cubes[0].data_header['OBJECT']

        if keywords is not None:
            params, values, comments = zip(*keywords)
        else:
            params, values, comments = [], [], []

        add_mpdaf_method_keywords(hdr, method, params, values, comments)
        hdr['NFILES'] = (len(self.files), 'number of files merged in the cube')

        # Put the list of merged files in comments instead of using a CONTINUE
        # keyword, because MuseWise has a size limit for keyword values ...
        hdr['comment'] = 'List of cubes merged in this cube:'
        for f in self.files:
            hdr['comment'] = '- ' + os.path.basename(f)
        return c

    def median(self):
        """Combines cubes in a single data cube using median.

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`, :class:`mpdaf.obj.Cube`, Table
              cube, expmap, statpix

              - ``cube`` will contain the merged cube
              - ``expmap`` will contain an exposure map data cube which counts
                the number of exposures used for the combination of each pixel.
              - ``statpix`` is a table that will give the number of Nan pixels
                pixels per exposures (columns are FILENAME and NPIX_NAN)

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
        npixels = np.prod(self.shape)
        data = np.empty(npixels, dtype=np.float64)
        expmap = np.empty(npixels, dtype=np.int32)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        libCmethods.mpdaf_merging_median(c_char_p('\n'.join(self.files)), data,
                                         expmap, valid_pix)

        # no valid pixels
        no_valid_pix = npixels - valid_pix
        stat_pix = Table([self.files, no_valid_pix],
                         names=['FILENAME', 'NPIX_NAN'])

        kwargs = dict(expnb=expmap.max(), method='obj.cubelist.median')
        expmap = self.save_combined_cube(expmap, save_unit=False, **kwargs)
        cube = self.save_combined_cube(data, **kwargs)
        return cube, expmap, stat_pix

    def combine(self, nmax=2, nclip=5.0, nstop=2, var='propagate', mad=False):
        """combines cubes in a single data cube using sigma clipped mean.

        Parameters
        ----------
        nmax  : integer
                maximum number of clipping iterations
        nclip : float or (float,float)
                Number of sigma at which to clip.
                Single clipping parameter or lower / upper clipping parameters.
        nstop : integer
                If the number of not rejected pixels is less
                than this number, the clipping iterations stop.
        var   : string
                ``propagate``, ``stat_mean``, ``stat_one``

                - ``propagate``: the variance is the sum of the variances
                  of the N individual exposures divided by N**2.
                - ``stat_mean``: the variance of each combined pixel
                  is computed as the variance derived from the comparison
                  of the N individual exposures divided N-1.
                - ``stat_one``: the variance of each combined pixel is
                  computed as the variance derived from the comparison
                  of the N individual exposures.

        mad   : boolean
                use MAD (median absolute deviation) statistics for
                sigma-clipping

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`, :class:`mpdaf.obj.Cube`, astropy.table
              cube, expmap, statpix

              - ``cube`` will contain the merged cube
              - ``expmap`` will contain an exposure map data cube which counts
                the number of exposures used for the combination of each pixel.
              - ``statpix`` is a table that will give the number of Nan pixels
                and rejected pixels per exposures (columns are FILENAME,
                NPIX_NAN and NPIX_REJECTED)

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
        rej = (valid_pix - select_pix) / valid_pix.astype(float) * 100.0
        rej = " ".join("{:.2f}%".format(p) for p in rej)
        d = {'class': 'CubeList', 'method': 'merging'}
        self.logger.info("%% of rejected pixels per files: %s", rej, extra=d)
        no_valid_pix = npixels - valid_pix
        rejected_pix = valid_pix - select_pix
        statpix = Table([self.files, no_valid_pix, rejected_pix],
                        names=['FILENAME', 'NPIX_NAN', 'NPIX_REJECTED'])

        keywords = [('nmax', nmax, 'max number of clipping iterations'),
                    ('nclip_low', nclip_low, 'lower clipping parameter'),
                    ('nclip_up', nclip_up, 'upper clipping parameter'),
                    ('nstop', nstop, 'clipping minimum number'),
                    ('var', var, 'type of variance')]
        kwargs = dict(expnb=expmap.max(), keywords=keywords,
                      method='obj.cubelist.merging')
        expmap = self.save_combined_cube(expmap, save_unit=False, **kwargs)
        cube = self.save_combined_cube(data, var=vardata, **kwargs)
        return cube, expmap, statpix

    def pymedian(self):
        d = {'class': 'CubeList', 'method': 'median'}
        try:
            import fitsio
        except ImportError:
            self.logger.error('fitsio is required !')
            raise

        data = [fitsio.FITS(f)[1] for f in self.files]
        # shape = data[0].get_dims()
        cube = np.empty(self.shape, dtype=np.float64)
        expmap = np.empty(self.shape, dtype=np.int32)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        nl = self.shape[0]

        self.logger.info('Looping on the %d planes of the cube', nl, extra=d)
        for l in ProgressBar(xrange(nl)):
            arr = np.array([c[l, :, :][0] for c in data])
            cube[l, :, :] = np.nanmedian(arr, axis=0)
            expmap[l, :, :] = (~np.isnan(arr)).astype(int).sum(axis=0)
            valid_pix += (~np.isnan(arr)).astype(int).sum(axis=1).sum(axis=1)

        # no valid pixels
        npixels = np.prod(self.shape)
        no_valid_pix = npixels - valid_pix
        stat_pix = Table([self.files, no_valid_pix],
                         names=['FILENAME', 'NPIX_NAN'])

        kwargs = dict(expnb=expmap.max(), method='obj.cubelist.pymedian')
        expmap = self.save_combined_cube(expmap, save_unit=False, **kwargs)
        cube = self.save_combined_cube(cube, **kwargs)
        return cube, expmap, stat_pix

    def pycombine(self, nmax=2, nclip=5.0, var='propagate', nstop=2, nl=None,
                  object_name=None):
        d = {'class': 'CubeList', 'method': 'merging'}
        try:
            import fitsio
        except ImportError:
            self.logger.error('fitsio is required !', extra=d)
            raise

        try:
            from ..merging import sigma_clip
        except:
            self.logger.error('The `merging` module must have been compiled to'
                              'use this method', extra=d)
            raise

        if is_int(nclip) or is_float(nclip):
            nclip_low, nclip_up = nclip, nclip
        else:
            nclip_low, nclip_up = nclip

        if var == 'propagate':
            var_mean = 0
        elif var == 'stat_mean':
            var_mean = 1
        else:
            var_mean = 2

        info = self.logger.info
        info("Merging cube using sigma clipped mean", extra=d)
        info("nmax = %d", nmax, extra=d)
        info("nclip_low = %f", nclip_low, extra=d)
        info("nclip_high = %f", nclip_up, extra=d)

        if nl is not None:
            self.shape[0] = nl

        data = [fitsio.FITS(f)['DATA'] for f in self.files]
        cube = np.empty(self.shape, dtype=np.float64)
        expmap = np.zeros(self.shape, dtype=np.int32)
        rejmap = np.zeros(self.shape, dtype=np.int32)
        vardata = np.empty(self.shape, dtype=np.float64)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        select_pix = np.zeros(self.nfiles, dtype=np.int32)
        nl = self.shape[0]
        fshape = (self.shape[1], self.shape[2], self.nfiles)
        arr = np.empty(fshape, dtype=float)
        starr = np.empty(fshape, dtype=float)

        if var_mean == 0:
            stat = [fitsio.FITS(f)['STAT'] for f in self.files]

        info('Looping on the %d planes of the cube', nl, extra=d)
        for l in xrange(nl):
            if l % 100 == 0:
                info('%d/%d', l, nl, extra=d)
            for i, f in enumerate(data):
                arr[:, :, i] = f[l, :, :][0]
            if var_mean == 0:
                for i, f in enumerate(stat):
                    starr[:, :, i] = f[l, :, :][0]

            sigma_clip(arr, starr, cube, vardata, expmap, rejmap, valid_pix,
                       select_pix, l, nmax, nclip_low, nclip_up, nstop,
                       var_mean)

        arr = None
        data = None

        # no valid pixels
        npixels = np.prod(self.shape)
        no_valid_pix = npixels - valid_pix
        rejected_pix = valid_pix - select_pix
        rej = rejected_pix / valid_pix.astype(float) * 100.0
        rej = " ".join("{:.2f}".format(p) for p in rej)
        info("%% of rejected pixels per files: %s", rej, extra=d)

        stat_pix = Table([self.files, no_valid_pix, rejected_pix],
                         names=['FILENAME', 'NPIX_NAN', 'NPIX_REJECTED'])

        keywords = [('nmax', nmax, 'max number of clipping iterations'),
                    ('nclip_low', nclip, 'lower clipping parameter'),
                    ('nclip_up', nclip, 'upper clipping parameter'),
                    ('var', var, 'type of variance')]
        kwargs = dict(expnb=expmap.max(), object_name=object_name,
                      keywords=keywords, method='obj.cubelist.pymerging')
        cube = self.save_combined_cube(cube, var=vardata, **kwargs)
        expmap = self.save_combined_cube(expmap, save_unit=False, **kwargs)
        rejmap = self.save_combined_cube(rejmap, save_unit=False, **kwargs)
        return cube, expmap, stat_pix, rejmap


class CubeMosaic(CubeList):

    """Manages a list of cubes and handles the combination to make a mosaic.

    To run the combination, all the cubes must be on the same WCS grid. The
    values from the ``CRPIX`` keywords will be used as offsets to put a cube
    inside the combined cube.

    This class inherits from :class:`mpdaf.obj.Cube`.

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
        Lengths of data in Z and Y and X (python notation (nz,ny,nx)).
    fscale : float
        Flux scaling factor (1 by default).
    wcs    : :class:`mpdaf.obj.WCS`
        World coordinates.
    wave   : :class:`mpdaf.obj.WaveCoord`
        Wavelength coordinates
    unit   : string
        Possible data unit type. None by default.

    """

    checkers = ('check_dim', 'check_wcs', 'check_fscale')

    def __init__(self, files, output_wcs):
        """Creates a CubeMosaic object.

        Parameters
        ----------
        files : list of strings
            List of cubes fits filenames.
        output_wcs : str
            Path to a cube FITS file, this cube is used to define the output
            cube: shape, WCS, fscale and unit are needed, it must have the XCS
            grid as the input cubes.

        """
        self.out = CubeDisk(output_wcs)
        super(CubeMosaic, self).__init__(files)

    def __getitem__(self, item):
        raise ValueError('Operation forbidden')

    def info(self, verbose=False):
        super(CubeMosaic, self).info(verbose=verbose)
        d = {'class': 'CubeMosaic', 'method': 'info'}
        self.logger.info('Output WCS:', extra=d)
        self.logger.info('- shape: %s', 'x'.join(str(s) for s in self.shape),
                         extra=d)
        self.logger.info('- crpix: %s', self.wcs.wcs.wcs.crpix, extra=d)
        self.logger.info('- crval: %s', self.wcs.wcs.wcs.crval, extra=d)

    def _set_defaults(self):
        self.shape = self.out.shape
        self.wcs = self.out.wcs
        self.wave = self.out.wave
        self.fscale = self.out.fscale
        self.unit = self.out.unit

    def check_wcs(self):
        """Checks if all cubes use the same projection."""
        d = {'class': 'CubeMosaic', 'method': 'check_wcs'}
        wcs = self.wcs
        cdelt1 = wcs.get_step()
        rot = wcs.get_rot()

        for f, cube in zip(self.files, self.cubes):
            cw = cube.wcs
            valid = [allclose(wcs.wcs.wcs.crval, cw.wcs.wcs.crval),
                     # allclose(wcs.wcs.wcs.cd, cw.wcs.wcs.cd),
                     array_equal(wcs.wcs.wcs.ctype, cw.wcs.wcs.ctype),
                     allclose(cdelt1, cw.get_step()),
                     allclose(rot, cw.get_rot())]
            if not all(valid):
                msg = 'all cubes have not same spatial coordinates'
                self.logger.warning(msg, extra=d)
                self.logger.info(valid, extra=d)
                self.logger.info(self.files[0], extra=d)
                self.wcs.info()
                self.logger.info(f, extra=d)
                cube.wcs.info()
                return False

        for f, cube in zip(self.files, self.cubes):
            if not cube.wave.isEqual(self.wave):
                msg = 'all cubes have not same spectral coordinates'
                self.logger.warning(msg, extra=d)
                self.logger.info(self.files[0], extra=d)
                self.wave.info()
                self.logger.info(f, extra=d)
                cube.wave.info()
                return False
        return True

    def check_dim(self):
        """Checks if all cubes have same dimensions."""
        shapes = np.array([cube.shape for cube in self.cubes])
        assert len(np.unique(shapes[:, 0])) == 1, (
            'Cubes must have the same spectral range.')

    def pycombine(self, nmax=2, nclip=5.0, var='propagate', nstop=2, nl=None,
                  object_name=None):
        d = {'class': 'CubeMosaic', 'method': 'merging'}
        try:
            import fitsio
        except ImportError:
            self.logger.error('fitsio is required !', extra=d)
            raise

        try:
            from ..merging import sigma_clip
        except:
            self.logger.error('The `merging` module must have been compiled to'
                              'use this method', extra=d)
            raise

        if is_int(nclip) or is_float(nclip):
            nclip_low, nclip_up = nclip, nclip
        else:
            nclip_low, nclip_up = nclip

        info = self.logger.info
        info("Merging cube using sigma clipped mean", extra=d)
        info("nmax = %d", nmax, extra=d)
        info("nclip_low = %f", nclip_low, extra=d)
        info("nclip_high = %f", nclip_up, extra=d)

        if nl is not None:
            self.shape[0] = nl

        data = [fitsio.FITS(f)[1] for f in self.files]
        offsets = np.array([-cube.wcs.wcs.wcs.crpix[::-1]
                            for cube in self.cubes], dtype=int) + 1
        shapes = np.array([cube.shape[1:] for cube in self.cubes])

        cube = np.empty(self.shape, dtype=np.float64)
        vardata = np.empty(self.shape, dtype=np.float64)
        expmap = np.empty(self.shape, dtype=np.int32)
        rejmap = np.empty(self.shape, dtype=np.int32)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        select_pix = np.zeros(self.nfiles, dtype=np.int32)
        nl = self.shape[0]
        fshape = (self.shape[1], self.shape[2], self.nfiles)
        arr = np.empty(fshape, dtype=float)

        info('Looping on the %d planes of the cube', nl, extra=d)
        for l in xrange(nl):
            if l % 100 == 0:
                info('%d/%d', l, nl)
            arr.fill(np.nan)
            for i, f in enumerate(data):
                x, y = offsets[i]
                arr[x:x+shapes[i][0], y:y+shapes[i][1], i] = f[l, :, :][0]

            sigma_clip(arr, cube, vardata, expmap, rejmap, valid_pix,
                       select_pix, l, nmax, nclip_low, nclip_up, nstop)

        arr = None
        data = None

        # no valid pixels
        npixels = np.prod(self.shape)
        no_valid_pix = npixels - valid_pix
        rejected_pix = valid_pix - select_pix
        rej = rejected_pix / valid_pix.astype(float) * 100.0
        rej = " ".join("{:.2f}".format(p) for p in rej)
        info("%% of rejected pixels per files: %s", rej, extra=d)
        stat_pix = Table([self.files, no_valid_pix, rejected_pix],
                         names=['FILENAME', 'NPIX_NAN', 'NPIX_REJECTED'])

        keywords = [('nmax', nmax, 'max number of clipping iterations'),
                    ('nclip_low', nclip, 'lower clipping parameter'),
                    ('nclip_up', nclip, 'upper clipping parameter'),
                    ('var', var, 'type of variance')]
        kwargs = dict(expnb=np.nanmedian(expmap), object_name=object_name,
                      keywords=keywords, method='obj.cubemosaic.pymerging')
        cube = self.save_combined_cube(cube, var=vardata, **kwargs)
        expmap = self.save_combined_cube(expmap, save_unit=False, **kwargs)
        rejmap = self.save_combined_cube(rejmap, save_unit=False, **kwargs)
        return cube, expmap, stat_pix, rejmap
