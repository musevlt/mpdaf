"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2014-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2015 Johan Richard <jrichard@univ-lyon1.fr>

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

from __future__ import absolute_import, division

import logging
import numpy as np
import os

from astropy import units as u
from astropy.table import Table
from astropy.utils.console import ProgressBar
from ctypes import c_char_p
from numpy import allclose, array_equal
from six.moves import range, zip

from .cube import Cube
from ..tools.fits import add_mpdaf_method_keywords, copy_keywords

__all__ = ('CubeList', 'CubeMosaic')

# List of keywords that will be copied to the combined cube
KEYWORDS_TO_COPY = (
    'ORIGIN', 'TELESCOP', 'INSTRUME', 'RA', 'DEC', 'EQUINOX', 'RADECSYS',
    'EXPTIME', 'MJD-OBS', 'DATE-OBS', 'PI-COI', 'OBSERVER', 'OBJECT',
    'ESO INS DROT POSANG', 'ESO INS MODE', 'ESO DET READ CURID',
    'ESO INS TEMP11 VAL', 'ESO OBS ID', 'ESO OBS NAME', 'ESO OBS START',
    'ESO TEL AIRM END', 'ESO TEL AIRM START', 'ESO TEL AMBI FWHM END',
    'ESO TEL AMBI FWHM START'
)


class CubeList(object):

    """Manages a list of cubes and handles the combination.

    To run the combination, all the cubes must have the same dimensions and be
    on the same WCS grid.

    Parameters
    ----------
    files : list of str
        List of cubes fits filenames

    Attributes
    ----------
    files : list of str
        List of cubes fits filenames
    nfiles : integer
        Number of files.
    scales : list of doubles
        List of scales
    shape : array of 3 integers)
        Lengths of data in Z and Y and X (python notation (nz,ny,nx)).
    wcs : `mpdaf.obj.WCS`
        World coordinates.
    wave : `mpdaf.obj.WaveCoord`
        Wavelength coordinates
    unit : str
        Possible data unit type. None by default.
    """

    checkers = ('check_dim', 'check_wcs')

    def __init__(self, files, scalelist=None):
        """Create a CubeList object.

        Parameters
        ----------
        files : list of str
            List of cubes fits filenames
        scalelist: list of float
            (optional) list of scales to be applied to each cube
        """
        self._logger = logging.getLogger(__name__)
        self.files = files
        self.nfiles = len(files)
        if scalelist:
            self.scales = np.array(scalelist)
        else:
            self.scales = np.ones(self.nfiles)
        self.cubes = [Cube(filename=f) for f in self.files]
        self._set_defaults()
        self.check_compatibility()

    def _set_defaults(self):
        self.shape = self.cubes[0].shape
        self.wcs = self.cubes[0].wcs
        self.wave = self.cubes[0].wave
        self.unit = self.cubes[0].unit

    def __getitem__(self, item):
        """Apply a slice on all the cubes.

        See `mpdaf.obj.Cube.__getitem__` for details.
        """
        if not (isinstance(item, tuple) and len(item) == 3):
            raise ValueError('Operation forbidden')

        return np.array([cube[item] for cube in self.cubes])

    def info(self, verbose=False):
        """Print information."""
        rows = [(os.path.basename(c.filename),
                 'x'.join(str(s) for s in c.shape),
                 str(c.wcs.wcs.wcs.crpix), str(c.wcs.wcs.wcs.crval))
                for c in self.cubes]
        t = Table(rows=rows, names=('filename', 'shape', 'crpix', 'crval'))
        for line in t.pformat():
            self._logger.info(line)

        if verbose:
            self._logger.info('Detailed information per file:')
            for cube in self.cubes:
                cube.info()

    def check_dim(self):
        """Checks if all cubes have same dimensions."""
        shapes = np.array([cube.shape for cube in self.cubes])

        if not np.all(shapes == self.shape):
            self._logger.warning('all cubes have not same dimensions')
            for i in range(self.nfiles):
                self._logger.warning('%i X %i X %i cube (%s)', shapes[i, 0],
                                     shapes[i, 1], shapes[i, 2], self.files[i])
            return False
        else:
            return True

    def check_wcs(self):
        """Checks if all cubes have same world coordinates."""
        for f, cube in zip(self.files, self.cubes):
            if not cube.wcs.isEqual(self.wcs) or \
                    not cube.wave.isEqual(self.wave):
                if not cube.wcs.isEqual(self.wcs):
                    msg = 'all cubes have not same spatial coordinates'
                    self._logger.warning(msg)
                    self._logger.info(self.files[0])
                    self.wcs.info()
                    self._logger.info(f)
                    cube.wcs.info()
                if not cube.wave.isEqual(self.wave):
                    msg = 'all cubes have not same spectral coordinates'
                    self._logger.warning(msg)
                    self._logger.info(self.files[0])
                    self.wave.info()
                    self._logger.info(f)
                    cube.wave.info()
                return False
        return True

    def check_compatibility(self):
        """Checks if all cubes are compatible."""
        for checker in self.checkers:
            getattr(self, checker)()

    def save_combined_cube(self, data, var=None, method='', keywords=None,
                           expnb=None, unit=None, header=None):
        self._logger.info('Creating combined cube object')

        if data.ndim != 3:
            data = data.reshape(self.shape)
        if var is not None and var.ndim != 3:
            var = var.reshape(self.shape)

        c = Cube(wcs=self.wcs, wave=self.wave, data=data, var=var, copy=False,
                 dtype=data.dtype, unit=unit or self.unit, mask=np.ma.nomask)

        hdr = c.primary_header
        copy_keywords(self.cubes[0].primary_header, hdr, KEYWORDS_TO_COPY)

        if expnb is not None and 'EXPTIME' in hdr:
            hdr['EXPTIME'] = hdr['EXPTIME'] * expnb

        if header is not None:
            c.primary_header.update(header)

        # For MuseWise ?
        if 'OBJECT' in c.primary_header:
            c.data_header['OBJECT'] = c.primary_header['OBJECT']
        elif 'OBJECT' in self.cubes[0].data_header:
            c.data_header['OBJECT'] = self.cubes[0].data_header['OBJECT']

        if keywords is not None:
            params, values, comments = list(zip(*keywords))
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

    def median(self, header=None):
        """Combines cubes in a single data cube using median.

        Returns
        -------
        out : `~mpdaf.obj.Cube`, `mpdaf.obj.Cube`, Table
            cube, expmap, statpix

            - ``cube`` will contain the merged cube
            - ``expmap`` will contain an exposure map data cube which counts
              the number of exposures used for the combination of each pixel.
            - ``statpix`` is a table that will give the number of Nan pixels
              pixels per exposures (columns are FILENAME and NPIX_NAN)

        """
        from ..tools.ctools import ctools

        # run C method
        npixels = np.prod(self.shape)
        data = np.empty(npixels, dtype=np.float64)
        expmap = np.empty(npixels, dtype=np.int32)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        ctools.mpdaf_merging_median(c_char_p('\n'.join(self.files)), data,
                                    expmap, valid_pix)

        # no valid pixels
        no_valid_pix = npixels - valid_pix
        stat_pix = Table([self.files, no_valid_pix],
                         names=['FILENAME', 'NPIX_NAN'])

        kwargs = dict(expnb=np.nanmedian(expmap), method='obj.cubelist.median',
                      header=header)
        expmap = self.save_combined_cube(expmap, unit=u.dimensionless_unscaled,
                                         **kwargs)
        cube = self.save_combined_cube(data, **kwargs)
        return cube, expmap, stat_pix

    def combine(self, nmax=2, nclip=5.0, nstop=2, var='propagate', mad=False,
                header=None):
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
        var   : str
            ``propagate``, ``stat_mean``, ``stat_one``

            - ``propagate``: the variance is the sum of the variances
              of the N individual exposures divided by N**2.
            - ``stat_mean``: the variance of each combined pixel
              is computed as the variance derived from the comparison
              of the N individual exposures divided N-1.
            - ``stat_one``: the variance of each combined pixel is
              computed as the variance derived from the comparison
              of the N individual exposures.

        mad : boolean
            Use MAD (median absolute deviation) statistics for sigma-clipping

        Returns
        -------
        out : `~mpdaf.obj.Cube`, `mpdaf.obj.Cube`, astropy.table
            cube, expmap, statpix

            - ``cube`` will contain the merged cube
            - ``expmap`` will contain an exposure map data cube which counts
              the number of exposures used for the combination of each pixel.
            - ``statpix`` is a table that will give the number of Nan pixels
              and rejected pixels per exposures (columns are FILENAME,
              NPIX_NAN and NPIX_REJECTED)

        """
        from ..tools.ctools import ctools

        if np.isscalar(nclip):
            nclip_low = nclip
            nclip_up = nclip
        else:
            nclip_low = nclip[0]
            nclip_up = nclip[1]

        # returned arrays
        npixels = self.shape[0] * self.shape[1] * self.shape[2]
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

        # run C method
        ctools.mpdaf_merging_sigma_clipping(
            c_char_p('\n'.join(self.files)), data, vardata, expmap,
            self.scales, select_pix, valid_pix, nmax, np.float64(nclip_low),
            np.float64(nclip_up), nstop, np.int32(var_mean), np.int32(mad))

        # no valid pixels
        rej = (valid_pix - select_pix) / valid_pix.astype(float) * 100.0
        rej = " ".join("{:.2f}%".format(p) for p in rej)
        self._logger.info("%% of rejected pixels per files: %s", rej)
        no_valid_pix = npixels - valid_pix
        rejected_pix = valid_pix - select_pix
        statpix = Table([self.files, no_valid_pix, rejected_pix],
                        names=['FILENAME', 'NPIX_NAN', 'NPIX_REJECTED'])

        keywords = [('nmax', nmax, 'max number of clipping iterations'),
                    ('nclip_low', nclip_low, 'lower clipping parameter'),
                    ('nclip_up', nclip_up, 'upper clipping parameter'),
                    ('nstop', nstop, 'clipping minimum number'),
                    ('var', var, 'type of variance')]
        kwargs = dict(expnb=np.nanmedian(expmap), keywords=keywords,
                      header=header, method='obj.cubelist.merging')
        expmap = self.save_combined_cube(expmap, unit=u.dimensionless_unscaled,
                                         **kwargs)
        cube = self.save_combined_cube(data, var=vardata, **kwargs)
        return cube, expmap, statpix

    def pymedian(self, header=None):
        try:
            import fitsio
        except ImportError:
            self._logger.error('fitsio is required !')
            raise

        data = [fitsio.FITS(f)[1] for f in self.files]
        # shape = data[0].get_dims()
        cube = np.empty(self.shape, dtype=np.float64)
        expmap = np.empty(self.shape, dtype=np.int32)
        valid_pix = np.zeros(self.nfiles, dtype=np.int32)
        nl = self.shape[0]

        self._logger.info('Looping on the %d planes of the cube', nl)
        for l in ProgressBar(range(nl)):
            arr = np.array([c[l, :, :][0] for c in data])
            cube[l, :, :] = np.nanmedian(arr, axis=0)
            expmap[l, :, :] = (~np.isnan(arr)).astype(int).sum(axis=0)
            valid_pix += (~np.isnan(arr)).astype(int).sum(axis=1).sum(axis=1)

        # no valid pixels
        npixels = np.prod(self.shape)
        no_valid_pix = npixels - valid_pix
        stat_pix = Table([self.files, no_valid_pix],
                         names=['FILENAME', 'NPIX_NAN'])

        kwargs = dict(expnb=np.nanmedian(expmap), header=header,
                      method='obj.cubelist.pymedian')
        expmap = self.save_combined_cube(expmap, unit=u.dimensionless_unscaled,
                                         **kwargs)
        cube = self.save_combined_cube(cube, **kwargs)
        return cube, expmap, stat_pix

    def pycombine(self, nmax=2, nclip=5.0, var='propagate', nstop=2, nl=None,
                  header=None):
        try:
            import fitsio
        except ImportError:
            self._logger.error('fitsio is required !')
            raise

        try:
            from ..merging import sigma_clip
        except:
            self._logger.error('The `merging` module must have been compiled '
                               'to use this method')
            raise

        if np.isscalar(nclip):
            nclip_low, nclip_up = nclip, nclip
        else:
            nclip_low, nclip_up = nclip

        if var == 'propagate':
            var_mean = 0
        elif var == 'stat_mean':
            var_mean = 1
        else:
            var_mean = 2

        info = self._logger.info

        info("Merging cube using sigma clipped mean")
        info("nmax = %d", nmax)
        info("nclip_low = %f", nclip_low)
        info("nclip_high = %f", nclip_up)

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

        info('Looping on the %d planes of the cube', nl)

        for l in range(nl):
            if l % 100 == 0:
                info('%d/%d', l, nl)
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
        info("%% of rejected pixels per files: %s", rej)

        stat_pix = Table([self.files, no_valid_pix, rejected_pix],
                         names=['FILENAME', 'NPIX_NAN', 'NPIX_REJECTED'])

        keywords = [('nmax', nmax, 'max number of clipping iterations'),
                    ('nclip_low', nclip, 'lower clipping parameter'),
                    ('nclip_up', nclip, 'upper clipping parameter'),
                    ('var', var, 'type of variance')]
        kwargs = dict(expnb=np.nanmedian(expmap), header=header,
                      keywords=keywords, method='obj.cubelist.pymerging')
        cube = self.save_combined_cube(cube, var=vardata, **kwargs)
        expmap = self.save_combined_cube(expmap, unit=u.dimensionless_unscaled,
                                         **kwargs)
        rejmap = self.save_combined_cube(rejmap, unit=u.dimensionless_unscaled,
                                         **kwargs)
        return cube, expmap, stat_pix, rejmap


class CubeMosaic(CubeList):

    """Manages a list of cubes and handles the combination to make a mosaic.

    To run the combination, all the cubes must be on the same WCS grid. The
    values from the ``CRPIX`` keywords will be used as offsets to put a cube
    inside the combined cube.

    This class inherits from `mpdaf.obj.CubeList`.

    Parameters
    ----------
    files : list of str
        List of cubes fits filenames

    Attributes
    ----------
    files : list of str
        List of cubes fits filenames
    nfiles : integer
        Number of files.
    shape : array of 3 integers)
        Lengths of data in Z and Y and X (python notation (nz,ny,nx)).
    wcs : `mpdaf.obj.WCS`
        World coordinates.
    wave : `mpdaf.obj.WaveCoord`
        Wavelength coordinates
    unit : str
        Possible data unit type. None by default.

    """

    checkers = ('check_dim', 'check_wcs')

    def __init__(self, files, output_wcs):
        """Create a CubeMosaic object.

        Parameters
        ----------
        files : list of str
            List of cubes fits filenames.
        output_wcs : str
            Path to a cube FITS file, this cube is used to define the output
            cube: shape, WCS and unit are needed, it must have the same WCS
            grid as the input cubes.
        """
        self.out = Cube(output_wcs)
        super(CubeMosaic, self).__init__(files)

    def __getitem__(self, item):
        raise ValueError('Operation forbidden')

    def info(self, verbose=False):
        super(CubeMosaic, self).info(verbose=verbose)
        self._logger.info('Output WCS:')
        self._logger.info('- shape: %s', 'x'.join(str(s) for s in self.shape))
        self._logger.info('- crpix: %s', self.wcs.wcs.wcs.crpix)
        self._logger.info('- crval: %s', self.wcs.wcs.wcs.crval)

    def _set_defaults(self):
        self.shape = self.out.shape
        self.wcs = self.out.wcs
        self.wave = self.out.wave
        self.unit = self.out.unit

    def check_wcs(self):
        """Checks if all cubes use the same projection."""
        wcs = self.wcs
        cdelt1 = wcs.get_step()
        cunit = wcs.unit
        rot = wcs.get_rot()
        logger = self._logger

        for f, cube in zip(self.files, self.cubes):
            cw = cube.wcs
            valid = [allclose(wcs.wcs.wcs.crval, cw.wcs.wcs.crval),
                     # allclose(wcs.wcs.wcs.cd, cw.wcs.wcs.cd),
                     array_equal(wcs.wcs.wcs.ctype, cw.wcs.wcs.ctype),
                     allclose(cdelt1, cw.get_step(unit=cunit)),
                     allclose(rot, cw.get_rot())]
            if not all(valid):
                logger.warning('all cubes have not same spatial coordinates')
                logger.info(valid)
                logger.info(self.files[0])
                self.wcs.info()
                logger.info(f)
                cube.wcs.info()
                return False

        for f, cube in zip(self.files, self.cubes):
            if not cube.wave.isEqual(self.wave):
                logger.warning('all cubes have not same spectral coordinates')
                logger.info(self.files[0])
                self.wave.info()
                logger.info(f)
                cube.wave.info()
                return False
        return True

    def check_dim(self):
        """Checks if all cubes have same dimensions."""
        shapes = np.array([cube.shape for cube in self.cubes])
        assert len(np.unique(shapes[:, 0])) == 1, (
            'Cubes must have the same spectral range.')

    def pycombine(self, nmax=2, nclip=5.0, var='propagate', nstop=2, nl=None,
                  header=None):
        try:
            import fitsio
        except ImportError:
            self._logger.error('fitsio is required !')
            raise

        try:
            from ..merging import sigma_clip
        except:
            self._logger.error('The `merging` module must have been compiled '
                               'to use this method')
            raise

        if np.isscalar(nclip):
            nclip_low, nclip_up = nclip, nclip
        else:
            nclip_low, nclip_up = nclip

        info = self._logger.info
        info("Merging cube using sigma clipped mean")
        info("nmax = %d", nmax)
        info("nclip_low = %f", nclip_low)
        info("nclip_high = %f", nclip_up)

        if var == 'propagate':
            var_mean = 0
        elif var == 'stat_mean':
            var_mean = 1
        else:
            var_mean = 2

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
        starr = np.empty(fshape, dtype=float)

        if var_mean == 0:
            stat = [fitsio.FITS(f)['STAT'] for f in self.files]

        info('Looping on the %d planes of the cube', nl)
        for l in range(nl):
            if l % 100 == 0:
                info('%d/%d', l, nl)
            arr.fill(np.nan)
            for i, f in enumerate(data):
                x, y = offsets[i]
                arr[x:x + shapes[i][0], y:y + shapes[i][1], i] = f[l, :, :][0]
            if var_mean == 0:
                starr.fill(np.nan)
                for i, f in enumerate(stat):
                    x, y = offsets[i]
                    nx, ny = shapes[i]
                    starr[x:x + nx, y:y + ny, i] = f[l, :, :][0]

            sigma_clip(arr, starr, cube, vardata, expmap, rejmap, valid_pix,
                       select_pix, l, nmax, nclip_low, nclip_up, nstop,
                       var_mean)

        arr = None
        starr = None
        stat = None
        data = None

        # no valid pixels
        npixels = np.prod(self.shape)
        no_valid_pix = npixels - valid_pix
        rejected_pix = valid_pix - select_pix
        rej = rejected_pix / valid_pix.astype(float) * 100.0
        rej = " ".join("{:.2f}".format(p) for p in rej)
        info("%% of rejected pixels per files: %s", rej)
        stat_pix = Table([self.files, no_valid_pix, rejected_pix],
                         names=['FILENAME', 'NPIX_NAN', 'NPIX_REJECTED'])

        keywords = [('nmax', nmax, 'max number of clipping iterations'),
                    ('nclip_low', nclip, 'lower clipping parameter'),
                    ('nclip_up', nclip, 'upper clipping parameter'),
                    ('var', var, 'type of variance')]
        kwargs = dict(expnb=np.nanmedian(expmap), header=header,
                      keywords=keywords, method='obj.cubemosaic.pymerging')
        cube = self.save_combined_cube(cube, var=vardata, **kwargs)
        expmap = self.save_combined_cube(expmap, unit=u.dimensionless_unscaled,
                                         **kwargs)
        rejmap = self.save_combined_cube(rejmap, unit=u.dimensionless_unscaled,
                                         **kwargs)
        return cube, expmap, stat_pix, rejmap
