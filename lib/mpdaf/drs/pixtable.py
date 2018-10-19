"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2012-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2012-2014 Aurelien Jarno <aurelien.jarno@univ-lyon1.fr>
Copyright (c)      2013 Johan Richard <jrichard@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

import astropy.units as u
import datetime
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings

from astropy.io import fits
from astropy.io.fits import Column, ImageHDU
from astropy.table import Table
from os.path import basename

from ..obj import Image, Spectrum, WaveCoord, WCS
from ..tools import add_mpdaf_method_keywords, copy_header, deprecated

try:
    import numexpr
except ImportError:
    numexpr = False

__all__ = ('PixTable', 'PixTableMask', 'PixTableAutoCalib',
           'plot_autocal_factors')

NIFUS = 24
NSLICES = 48
SKY_SEGMENTS = [0, 5000, 5265, 5466, 5658, 5850, 6120, 6440, 6678, 6931, 7211,
                7450, 7668, 7900, 8120, 8330, 8565, 8731, 9012, 9275, 10000]

KEYWORD = 'HIERARCH ESO DRS MUSE PIXTABLE'
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


def _get_file_basename(f):
    """Return a string with the basename of f if f is not None"""
    return '' if f is None else basename(f)


class PixTableMask(object):

    """PixTableMask class.

    This class manages input/output for MUSE pixel mask files

    Parameters
    ----------
    filename : str or None
        Name of the FITS table containing the masked column. If a PixTableMask
        object is loaded from a FITS file, the others parameters are not read
        but loaded from the FITS file.
    maskfile : str or None
        Name of the FITS image masking some objects.
    maskcol : array of bool or None
        pixtable's column corresponding to the mask
    pixtable : str or None
        Name of the corresponding pixel table.

    Attributes
    ----------
    filename : str
        Name of the FITS table containing the masked column.
    maskfile : str
        Name of the FITS image masking some objects.
    maskcol : array of bool
        pixtable's column corresponding to the mask
    pixtable : str
        Name of the corresponding pixel table.

    """

    def __init__(self, filename=None, maskfile=None, maskcol=None,
                 pixtable=None):
        if filename is None:
            self.maskfile = maskfile
            self.maskcol = maskcol
            self.pixtable = pixtable
        else:
            hdulist = fits.open(filename)
            self.maskfile = hdulist[0].header['mask']
            self.pixtable = hdulist[0].header['pixtable']
            self.maskcol = np.bool_(hdulist['maskcol'].data[:, 0])

    def write(self, filename):
        """Save the object in a FITS file.

        Parameters
        ----------
        filename : str
            The FITS filename.
        """
        prihdu = fits.PrimaryHDU()
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        add_mpdaf_method_keywords(prihdu.header,
                                  'mpdaf.drs.pixtable.mask_column',
                                  [], [], [])
        prihdu.header['pixtable'] = (basename(self.pixtable), 'pixtable')
        prihdu.header['mask'] = (basename(self.maskfile),
                                 'file to mask out all bright obj')
        hdulist = [prihdu]
        nrows = self.maskcol.shape[0]
        hdulist.append(ImageHDU(
            name='maskcol', data=np.int32(self.maskcol.reshape((nrows, 1)))))
        hdu = fits.HDUList(hdulist)
        hdu[1].header['BUNIT'] = 'boolean'
        hdu.writeto(filename, overwrite=True, output_verify='fix')


class PixTableAutoCalib(object):

    """PixTableAutoCalib class.

    This class manages input/output for file containing auto calibration
    results of MUSE pixel table files

    Parameters
    ----------
    filename : str
        The FITS file name. If PixTableAutoCalib object is loaded from a FITS
        file, the others parameters are not read but loaded from the FITS file.
    method : str or None
        Name of the auto calibration method.
    maskfile : str or None
        Name of the FITS image masking some objects.
    pixtable : str or None
        Name of the corresponding pixel table.
    ifu : array of int or None
        channel numbers.
    sli : array of int or None
        slice numbers.
    quad : array of int or None
        Detector quadrant numbers.
    npts : array of int or None
        number of remaining pixels.
    corr : array of float or None
        correction value.

    Attributes
    ----------
    filename : str
        The FITS file name.
    method : str
        Name of the auto calibration method.
    maskfile : str
        Name of the FITS image masking some objects.
    pixtable : str
        Name of the corresponding pixel table.
    ifu : array of int
        channel numbers.
    sli : array of int
        slice numbers.
    quad : array of int or None
        Detector quadrant numbers.
    npts : array of int
        number of remaining pixels.
    corr : array of float
        correction value.
    """

    def __init__(self, filename=None, method=None, maskfile=None,
                 pixtable=None, ifu=None, sli=None, quad=None, npts=None,
                 corr=None):
        if filename is None:
            self.method = method
            self.maskfile = maskfile
            self.pixtable = pixtable
            self.ifu = ifu
            self.sli = sli
            self.quad = quad
            self.npts = npts
            self.corr = corr
        else:
            hdulist = fits.open(filename)
            self.method = hdulist[0].header['method']
            self.maskfile = hdulist[0].header['mask']
            self.pixtable = hdulist[0].header['pixtable']
            self.ifu = hdulist['ifu'].data[:, 0]
            self.sli = hdulist['sli'].data[:, 0]
            try:
                self.quad = hdulist['quad'].data[:, 0]
            except KeyError:
                self.quad = None
            self.npts = hdulist['npts'].data[:, 0]
            self.corr = hdulist['corr'].data[:, 0]

    def as_table(self):
        return Table([self.ifu, self.sli, self.quad, self.corr, self.npts],
                     names=('ifu', 'slice', 'quad', 'correction', 'npts'),
                     copy=False)

    def write(self, filename):
        """Save the object in a FITS file."""
        prihdu = fits.PrimaryHDU()
        warnings.simplefilter("ignore")
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        add_mpdaf_method_keywords(prihdu.header,
                                  'mpdaf.drs.PixTableAutoCalib.write',
                                  [], [], [])
        prihdu.header['method'] = (self.method, 'auto calib method')
        prihdu.header['pixtable'] = (basename(self.pixtable), 'pixtable')
        prihdu.header['mask'] = (basename(self.maskfile),
                                 'file to mask out all bright obj')

        shape = (self.corr.shape[0], 1)
        hdulist = [
            prihdu,
            ImageHDU(name='ifu', data=np.int32(self.ifu.reshape(shape))),
            ImageHDU(name='sli', data=np.int32(self.sli.reshape(shape))),
            ImageHDU(name='quad', data=np.int32(self.quad.reshape(shape))),
            ImageHDU(name='npts', data=np.int32(self.npts.reshape(shape))),
            ImageHDU(name='corr', data=np.float64(self.corr.reshape(shape)))]
        hdu = fits.HDUList(hdulist)
        hdu.writeto(filename, overwrite=True, output_verify='fix')
        warnings.simplefilter("default")


def write(filename, xpos, ypos, lbda, data, dq, stat, origin, weight=None,
          primary_header=None, save_as_ima=True, wcs=u.pix, wave=u.angstrom,
          unit_data=u.count):
    """Save the object in a FITS file.

    Parameters
    ----------
    filename : str
        The FITS filename.
    save_as_ima : bool
        If True, pixtable is saved as multi-extension FITS
    """
    fits.conf.extension_name_case_sensitive = True
    warnings.simplefilter("ignore")

    if primary_header is not None:
        header = copy_header(primary_header)
    else:
        header = fits.Header()
    header['date'] = (str(datetime.datetime.now()), 'creation date')
    header['author'] = ('MPDAF', 'origin of the file')
    prihdu = fits.PrimaryHDU(header=header)

    if save_as_ima:
        nrows = xpos.shape[0]
        hdulist = [
            prihdu,
            ImageHDU(name='xpos', data=np.float32(xpos.reshape((nrows, 1)))),
            ImageHDU(name='ypos', data=np.float32(ypos.reshape((nrows, 1)))),
            ImageHDU(name='lambda', data=np.float32(lbda.reshape((nrows, 1)))),
            ImageHDU(name='data', data=np.float32(data.reshape((nrows, 1)))),
            ImageHDU(name='dq', data=np.int32(dq.reshape((nrows, 1)))),
            ImageHDU(name='stat', data=np.float32(stat.reshape((nrows, 1)))),
            ImageHDU(name='origin', data=np.int32(origin.reshape((nrows, 1)))),
        ]
        if weight is not None:
            hdulist.append(
                ImageHDU(name='weight',
                         data=np.float32(weight.reshape((nrows, 1)))))
        hdu = fits.HDUList(hdulist)
        hdu[1].header['BUNIT'] = wcs.to_string('fits')
        hdu[2].header['BUNIT'] = wcs.to_string('fits')
        hdu[3].header['BUNIT'] = wave.to_string('fits')
        hdu[4].header['BUNIT'] = unit_data.to_string('fits')
        hdu[6].header['BUNIT'] = (unit_data**2).to_string('fits')

    else:
        cols = [
            Column(name='xpos', format='1E', unit=wcs.to_string('fits'),
                   array=np.float32(xpos)),
            Column(name='ypos', format='1E', unit=wcs.to_string('fits'),
                   array=np.float32(ypos)),
            Column(name='lambda', format='1E', unit=wave.to_string('fits'),
                   array=lbda),
            Column(name='data', format='1E', unit=unit_data.to_string('fits'),
                   array=np.float32(data)),
            Column(name='dq', format='1J', array=np.int32(dq)),
            Column(name='stat', format='1E',
                   unit=(unit_data**2).to_string('fits'),
                   array=np.float32(stat)),
            Column(name='origin', format='1J', array=np.int32(origin)),
        ]

        if weight is not None:
            cols.append(Column(name='weight', format='1E',
                               array=np.float32(weight)))
        coltab = fits.ColDefs(cols)
        tbhdu = fits.TableHDU(fits.FITS_rec.from_columns(coltab))
        hdu = fits.HDUList([prihdu, tbhdu])

    hdu.writeto(filename, overwrite=True, output_verify='fix')
    warnings.simplefilter("default")


def plot_autocal_factors(filename, savefig=None, plot_rejected=False,
                         sharex=True, sharey=True, figsize=4, cmap='Spectral',
                         plot_npts=False):
    """Plot the corrections computed by `PixTable.selfcalibrate`.

    This also works for the AUTOCAL_FACTORS table from the DRS.

    Parameters
    ----------
    filename : str, `astropy.table.Table`
        The corrections table, either MPDAF's PixTableAutoCalib, or
        the DRS AUTOCAL_FACTORS.
    savefig : str
        File to which the plot is saved.
    plot_rejected : bool
        Also plot the rejected corrections. This only works with the DRS
        AUTOCAL_FACTORS table.
    sharex, sharey : bool
        Controls sharing of properties among x/y axes.
    figsize : float
        Size of an individual plot.
    cmap : str
        Colormap.

    """
    t = Table.read(filename) if isinstance(filename, str) else filename
    fig, axes = plt.subplots(6, 4, figsize=(4 * figsize, 6 * figsize),
                             sharex=sharex, sharey=sharey)
    base = plt.cm.get_cmap(cmap)
    palette = base(np.linspace(0, 1, 48)).tolist()
    cm = base.from_list('Custom cmap', palette, len(palette))
    key = 'npts' if plot_npts else 'corr'

    # Handle the old format from MPDAF (columns have been renamed in the DRS
    # version)
    if 'slice' in t.colnames:
        t.rename_column('slice', 'sli')
    if 'correction' in t.colnames:
        t.rename_column('correction', 'corr')

    for ifu in range(1, 25):
        ax = axes.flat[ifu - 1]
        tt = t[t['ifu'] == ifu]
        for sl in range(1, 49):
            ts = tt[tt['sli'] == sl]
            ax.plot(ts['quad'], ts[key], color=palette[sl - 1])
            if 'corr_orig' in ts.colnames and plot_rejected and \
                    not plot_npts and not np.isnan(ts['corr_orig']).all():
                ax.scatter(ts['quad'], ts['corr_orig'], s=20,
                           color=palette[sl - 1])
        ax.set_title('IFU %s' % ifu)
        ax.grid(True)

    title = 'Number of points' if plot_npts else 'Correction factor'
    fig.suptitle('{} (y) for each wavelength segment (x), each slice (color) '
                 'and each IFU'.format(title), fontsize=16)

    bounds = np.linspace(0, 48, 49)
    norm = mpl.colors.BoundaryNorm(bounds, cm.N)
    ax2 = fig.add_axes([0.1, 0.96, 0.8, 0.01])
    mpl.colorbar.ColorbarBase(
        ax2, cmap=cm, norm=norm, spacing='proportional', ticks=bounds + .5,
        boundaries=bounds, format='%1i', orientation='horizontal')

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if savefig:
        fig.savefig(savefig)


class PixTable(object):

    """PixTable class.

    This class manages input/output for MUSE pixel table files. The FITS file
    is opened with memory mapping. Just the primary header and table dimensions
    are loaded. The methods ``get_xpos``, ``get_ypos``, ``get_lambda``,
    ``get_data``, ``get_dq``, ``get_stat`` and ``get_origin`` must be used to
    get columns data.

    Parameters
    ----------
    filename : str
        The FITS file name. None by default.

    Attributes
    ----------
    filename : str
        The FITS file name. None if any.
    primary_header : `astropy.io.fits.Header`
        The primary header.
    nrows : int
        Number of rows.
    nifu : int
        Number of merged IFUs that went into this pixel table.
    wcs : `astropy.units.Unit`
        Type of spatial coordinates of this pixel table (u.pix, u.deg or u.rad)
    wave : `astropy.units.Unit`
        Type of spectral coordinates of this pixel table
    ima : bool
        If True, pixtable is saved as multi-extension FITS image
        instead of FITS binary table.

    """

    def __init__(self, filename, xpos=None, ypos=None, lbda=None, data=None,
                 dq=None, stat=None, origin=None, weight=None,
                 primary_header=None, save_as_ima=True, wcs=u.pix,
                 wave=u.angstrom, unit_data=u.count):
        self._logger = logging.getLogger(__name__)
        self.filename = filename
        self.wcs = wcs
        self.wave = wave
        self.ima = save_as_ima

        self.xpos = None
        self.ypos = None
        self.lbda = None
        self.data = None
        self.stat = None
        self.dq = None
        self.origin = None
        self.weight = None
        self.nrows = 0
        self.nifu = 0
        self.unit_data = unit_data
        self.xc = 0.0
        self.yc = 0.0

        if filename is not None:
            self.hdulist = fits.open(self.filename, memmap=1)
            self.primary_header = self.hdulist[0].header
            self.nrows = self.hdulist[1].header["NAXIS2"]
            self.ima = self.hdulist[1].header['XTENSION'] == 'IMAGE'

            if self.ima:
                self.wcs = u.Unit(self.hdulist['xpos'].header['BUNIT'])
                self.wave = u.Unit(self.hdulist['lambda'].header['BUNIT'])
                self.unit_data = u.Unit(self.hdulist['data'].header['BUNIT'])
            else:
                self.wcs = u.Unit(self.hdulist[1].header['TUNIT1'])
                self.wave = u.Unit(self.hdulist[1].header['TUNIT3'])
                self.unit_data = u.Unit(self.hdulist[1].header['TUNIT4'])
        else:
            self.hdulist = None
            if (xpos is None or ypos is None or lbda is None or
                    data is None or dq is None or stat is None or
                    origin is None or primary_header is None):
                self.primary_header = fits.Header()
            else:
                self.primary_header = primary_header
                self.xpos = np.asarray(xpos)
                self.ypos = np.asarray(ypos)
                self.lbda = np.asarray(lbda)
                self.data = np.asarray(data)
                self.stat = np.asarray(stat)
                self.dq = np.asarray(dq)
                self.origin = np.asarray(origin)
                self.nrows = xpos.shape[0]

                for attr in (self.ypos, self.lbda, self.data, self.stat,
                             self.dq, self.origin):
                    if attr.shape[0] != self.nrows:
                        raise IOError('input data with different dimensions')

                if weight is None or weight.shape[0] == self.nrows:
                    self.weight = weight
                else:
                    raise IOError('input data with different dimensions')

        if self.nrows != 0:
            # Merged IFUs that went into this pixel tables
            try:
                self.nifu = self.get_keyword("MERGED")
            except KeyError:
                self.nifu = 1

            projection = self.projection
            if projection == 'projected':  # spheric coordinates
                keyx, keyy = 'RA', 'DEC'
            elif projection == 'positioned':
                keyx, keyy = 'CRVAL1', 'CRVAL2'
            else:
                raise Exception('Unknown projection: %s' % projection)

            try:
                # center in degrees
                cunit = u.Unit(self.primary_header["CUNIT1"])
                self.xc = (self.primary_header[keyx] * cunit).to(u.deg).value
                self.yc = (self.primary_header[keyy] * cunit).to(u.deg).value
            except:
                try:
                    # center in pixels
                    self.xc = self.primary_header[keyx]
                    self.yc = self.primary_header[keyy]
                except:
                    pass

    def __repr__(self):
        msg = "<{}({} rows, {} ifus, {})>".format(
            self.__class__.__name__, self.nrows, self.nifu, self.projection)
        if self.skysub:
            msg = msg[:-2] + ", sky-subtracted)>"
        if self.fluxcal:
            msg = msg[:-2] + ", flux-calibrated)>"
        return msg

    @property
    def fluxcal(self):
        """If True, this pixel table was flux-calibrated."""
        try:
            return self.get_keyword("FLUXCAL")
        except KeyError:
            return False

    @property
    def skysub(self):
        """If True, this pixel table was sky-subtracted."""
        try:
            return self.get_keyword("SKYSUB")
        except KeyError:
            return False

    @property
    def projection(self):
        """Return the projection type.

        - 'positioned' for positioned pixtables
        - 'projected' for reduced pixtables

        """
        wcs = self.get_keyword("WCS")
        return wcs.split(' ')[0]

    def copy(self):
        """Copy PixTable object in a new one and returns it."""
        result = PixTable(self.filename)
        result.wcs = self.wcs
        result.wave = self.wave
        result.unit_data = self.unit_data
        result.ima = self.ima

        if self.xpos is not None:
            result.xpos = self.xpos.__copy__()
        if self.ypos is not None:
            result.ypos = self.ypos.__copy__()
        if self.lbda is not None:
            result.lbda = self.lbda.__copy__()
        if self.data is not None:
            result.data = self.data.__copy__()
        if self.stat is not None:
            result.stat = self.stat.__copy__()
        if self.dq is not None:
            result.dq = self.dq.__copy__()
        if self.origin is not None:
            result.origin = self.origin.__copy__()
        if self.weight is not None:
            result.weight = self.weight.__copy__()

        result.nrows = self.nrows
        result.nifu = self.nifu

        result.primary_header = self.primary_header.copy()

        result.xc = self.xc
        result.yc = self.yc

        return result

    def info(self):
        """Print information."""
        hdr = self.primary_header
        self._logger.info("%i merged IFUs went into this pixel table",
                          self.nifu)
        if self.skysub:
            self._logger.info("This pixel table was sky-subtracted")
        if self.fluxcal:
            self._logger.info("This pixel table was flux-calibrated")
        self._logger.info('%s (%s)', hdr["%s WCS" % KEYWORD],
                          hdr.comments["%s WCS" % KEYWORD])
        try:
            self._logger.info(self.hdulist.info())
        except:
            self._logger.info('No\tName\tType\tDim')
            self._logger.info('0\tPRIMARY\tcard\t()')
            # print "1\t\tTABLE\t(%iR,%iC)" % (self.nrows,self.ncols)

    def write(self, filename, save_as_ima=True):
        """Save the object in a FITS file.

        Parameters
        ----------
        filename : str
            The FITS filename.
        save_as_ima : bool
            If True, pixtable is saved as multi-extension FITS image
            instead of FITS binary table.
        """
        write(filename, self.get_xpos(), self.get_ypos(),
              self.get_lambda(), self.get_data(), self.get_dq(),
              self.get_stat(), self.get_origin(), self.get_weight(),
              self.primary_header, save_as_ima, self.wcs, self.wave,
              self.unit_data)

        self.filename = filename
        self.ima = save_as_ima

    def get_column(self, name, ksel=None):
        """Load a column and return it.

        Parameters
        ----------
        name : str or attribute
            Name of the column.
        ksel : output of np.where
            Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        attr_name = 'lbda' if name == 'lambda' else name
        attr = getattr(self, attr_name)
        if attr is not None:
            if ksel is None:
                return attr
            else:
                return attr[ksel]
        else:
            if self.hdulist is None:
                return None
            else:
                if ksel is None:
                    if self.ima:
                        column = self.hdulist[name].data[:, 0]
                    else:
                        column = self.hdulist[1].data.field(name)
                else:
                    if isinstance(ksel, tuple):
                        ksel = ksel[0]
                    if self.ima:
                        column = self.hdulist[name].data[ksel, 0]
                    else:
                        column = self.hdulist[1].data.field(name)[ksel]

                if np.issubdtype(column.dtype, np.floating):
                    # Ensure that float values are converted to double
                    column = column.astype(float)
                return column

    def set_column(self, name, data, ksel=None):
        """Set a column (or a part of it).

        Parameters
        ----------
        name : str or attribute
            Name of the column.
        data : numpy.array
            data values
        ksel : output of np.where
            Elements depending on a condition.
        """
        attr_name = 'lbda' if name == 'lambda' else name
        data = np.asarray(data)
        if ksel is None:
            assert data.shape[0] == self.nrows, 'Wrong dimension number'
            setattr(self, attr_name, data)
        else:
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, getattr(self, 'get_' + name)())
            attr = getattr(self, attr_name)
            attr[ksel] = data

    def get_xpos(self, ksel=None, unit=None):
        """Load the xpos column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('xpos', ksel=ksel)
        else:
            return (self.get_column('xpos', ksel=ksel) *
                    self.wcs).to(unit).value

    def set_xpos(self, xpos, ksel=None, unit=None):
        """Set xpos column (or a part of it).

        Parameters
        ----------
        xpos : numpy.array
            xpos values
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            unit of the xpos column in input.
        """
        if unit is not None:
            xpos = (xpos * unit).to(self.wcs).value
        self.set_column('xpos', xpos, ksel=ksel)
        self.set_keyword("LIMITS X LOW", float(self.xpos.min()))
        self.set_keyword("LIMITS X HIGH", float(self.xpos.max()))

    def get_ypos(self, ksel=None, unit=None):
        """Load the ypos column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('ypos', ksel=ksel)
        else:
            return (self.get_column('ypos', ksel=ksel) *
                    self.wcs).to(unit).value

    def set_ypos(self, ypos, ksel=None, unit=None):
        """Set ypos column (or a part of it).

        Parameters
        ----------
        ypos : numpy.array
            ypos values
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            unit of the ypos column in input.
        """
        if unit is not None:
            ypos = (ypos * unit).to(self.wcs).value
        self.set_column('ypos', ypos, ksel=ksel)
        self.set_keyword("LIMITS Y LOW", float(self.ypos.min()))
        self.set_keyword("LIMITS Y HIGH", float(self.ypos.max()))

    def get_lambda(self, ksel=None, unit=None):
        """Load the lambda column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('lambda', ksel=ksel)
        else:
            return (self.get_column('lambda', ksel=ksel) *
                    self.wave).to(unit).value

    def set_lambda(self, lbda, ksel=None, unit=None):
        """Set lambda column (or a part of it).

        Parameters
        ----------
        lbda : numpy.array
            lbda values
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            unit of the lambda column in input.
        """
        if unit is not None:
            lbda = (lbda * unit).to(self.wave).value
        self.set_column('lambda', lbda, ksel=ksel)
        self.set_keyword("LIMITS LAMBDA LOW", float(self.lbda.min()))
        self.set_keyword("LIMITS LAMBDA HIGH", float(self.lbda.max()))

    def get_data(self, ksel=None, unit=None):
        """Load the data column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('data', ksel=ksel)
        else:
            return (self.get_column('data', ksel=ksel) *
                    self.unit_data).to(unit).value

    def set_data(self, data, ksel=None, unit=None):
        """Set data column (or a part of it).

        Parameters
        ----------
        data : numpy.array
            data values
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            unit of the data column in input.
        """
        if unit is not None:
            data = (data * unit).to(self.unit_data).value
        self.set_column('data', data, ksel=ksel)

    def get_stat(self, ksel=None, unit=None):
        """Load the stat column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            Unit of the returned data.

        Returns
        -------
        out : numpy.array
        """
        if unit is None:
            return self.get_column('stat', ksel=ksel)
        else:
            return (self.get_column('stat', ksel=ksel) *
                    (self.unit_data**2)).to(unit).value

    def set_stat(self, stat, ksel=None, unit=None):
        """Set stat column (or a part of it).

        Parameters
        ----------
        stat : numpy.array
            stat values
        ksel : output of np.where
            Elements depending on a condition.
        unit : `astropy.units.Unit`
            unit of the stat column in input.
        """
        if unit is not None:
            stat = (stat * unit).to(self.unit_data**2).value
        self.set_column('stat', stat, ksel=ksel)

    def get_dq(self, ksel=None):
        """Load the dq column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        return self.get_column('dq', ksel=ksel)

    def set_dq(self, dq, ksel=None):
        """Set dq column (or a part of it).

        Parameters
        ----------
        dq : numpy.array
            dq values
        ksel : output of np.where
            Elements depending on a condition.
        """
        self.set_column('dq', dq, ksel=ksel)

    def get_origin(self, ksel=None):
        """Load the origin column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        return self.get_column('origin', ksel=ksel)

    def set_origin(self, origin, ksel=None):
        """Set origin column (or a part of it).

        Parameters
        ----------
        origin : numpy.array
            origin values
        ksel : output of np.where
            Elements depending on a condition.
        """
        self.set_column('origin', origin, ksel=ksel)
        ifu = self.origin2ifu(self.origin)
        sli = self.origin2slice(self.origin)
        self.set_keyword("LIMITS IFU LOW", int(ifu.min()))
        self.set_keyword("LIMITS IFU HIGH", int(ifu.max()))
        self.set_keyword("LIMITS SLICE LOW", int(sli.min()))
        self.set_keyword("LIMITS SLICE HIGH", int(sli.max()))

        # merged pixtable
        if self.nifu > 1:
            self.set_keyword("MERGED", len(np.unique(ifu)))

    def get_weight(self, ksel=None):
        """Load the weight column and return it.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition.

        Returns
        -------
        out : numpy.array
        """
        try:
            wght = self.get_keyword("WEIGHTED")
        except KeyError:
            wght = False
        return self.get_column('weight', ksel=ksel) if wght else None

    def set_weight(self, weight, ksel=None):
        """Set weight column (or a part of it).

        Parameters
        ----------
        weight : numpy.array
            weight values
        ksel : output of np.where
            Elements depending on a condition.
        """
        self.set_column('weight', weight, ksel=ksel)

    def get_exp(self):
        """Load the exposure numbers and return it as a column.

        Returns
        -------
        out : numpy.memmap
        """
        try:
            nexp = self.get_keyword("COMBINED")
            exp = np.empty(shape=self.nrows)
            for i in range(1, nexp + 1):
                first = self.get_keyword("EXP%i FIRST" % i)
                last = self.get_keyword("EXP%i LAST" % i)
                exp[first:last + 1] = i
        except:
            exp = None
        return exp

    def select_lambda(self, lbda, unit=u.angstrom):
        """Return a mask corresponding to the given wavelength range.

        Parameters
        ----------
        lbda : (float, float)
            (min, max) wavelength range in angstrom.
        unit : `astropy.units.Unit`
            Unit of the wavelengths in input.

        Returns
        -------
        out : array of bool
            mask
        """
        arr = self.get_lambda()
        mask = np.zeros(self.nrows, dtype=bool)
        if numexpr:
            for l1, l2 in lbda:
                l1 = (l1 * unit).to(self.wave).value
                l2 = (l2 * unit).to(self.wave).value
                mask |= numexpr.evaluate('(arr >= l1) & (arr < l2)')
        else:
            for l1, l2 in lbda:
                l1 = (l1 * unit).to(self.wave).value
                l2 = (l2 * unit).to(self.wave).value
                mask |= (arr >= l1) & (arr < l2)
        return mask

    def select_stacks(self, stacks, origin=None):
        """Return a mask corresponding to given stacks.

        Parameters
        ----------
        stacks : list of int
            Stacks numbers (1,2,3 or 4)
        Returns
        -------
        out : array of bool
            mask
        """
        from ..MUSE import Slicer
        assert min(stacks) > 0
        assert max(stacks) < 5
        sl = sorted([Slicer.sky2ccd(i) for st in stacks
                     for i in range(1 + 12 * (st - 1), 12 * st - 1)])
        self._logger.debug('Extract stack %s -> slices %s', stacks, sl)
        return self.select_slices(sl, origin=origin)

    def select_slices(self, slices, origin=None):
        """Return a mask corresponding to given slices.

        Parameters
        ----------
        slices : list of int
            Slice number on the CCD.

        Returns
        -------
        out : array of bool
            mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_sli = self.origin2slice(col_origin)
        if numexpr:
            mask = np.zeros(self.nrows, dtype=bool)
            for s in slices:
                mask |= numexpr.evaluate('col_sli == s')
            return mask
        else:
            return np.in1d(col_sli, slices)

    def select_ifus(self, ifus, origin=None):
        """Return a mask corresponding to given ifus.

        Parameters
        ----------
        ifu : int or list
            IFU number.

        Returns
        -------
        out : array of bool
            mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_ifu = self.origin2ifu(col_origin)
        if numexpr:
            mask = np.zeros(self.nrows, dtype=bool)
            for ifu in ifus:
                mask |= numexpr.evaluate('col_ifu == ifu')
            return mask
        else:
            return np.in1d(col_ifu, ifus)

    def select_exp(self, exp, col_exp):
        """Return a mask corresponding to given exposure numbers.

        Parameters
        ----------
        exp : list of int
            List of exposure numbers

        Returns
        -------
        out : array of bool
            mask
        """
        mask = np.zeros(self.nrows, dtype=bool)
        if numexpr:
            for iexp in exp:
                mask |= numexpr.evaluate('col_exp == iexp')
        else:
            for iexp in exp:
                mask |= (col_exp == iexp)
        return mask

    def select_xpix(self, xpix, origin=None):
        """Return a mask corresponding to given detector pixels.

        Parameters
        ----------
        xpix : list
            [(min, max)] pixel range along the X axis

        Returns
        -------
        out : array of bool
            mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_xpix = self.origin2xpix(col_origin)
        if hasattr(xpix, '__iter__'):
            mask = np.zeros(self.nrows, dtype=bool)
            if numexpr:
                for x1, x2 in xpix:
                    mask |= numexpr.evaluate('(col_xpix >= x1) & '
                                             '(col_xpix < x2)')
            else:
                for x1, x2 in xpix:
                    mask |= (col_xpix >= x1) & (col_xpix < x2)
        else:
            x1, x2 = xpix
            if numexpr:
                mask = numexpr.evaluate('(col_xpix >= x1) & (col_xpix < x2)')
            else:
                mask = (col_xpix >= x1) & (col_xpix < x2)
        return mask

    def select_ypix(self, ypix, origin=None):
        """Return a mask corresponding to given detector pixels.

        Parameters
        ----------
        ypix : list
            [(min, max)] pixel range along the Y axis

        Returns
        -------
        out : array of bool
            mask
        """
        col_origin = origin if origin is not None else self.get_origin()
        col_ypix = self.origin2ypix(col_origin)
        if hasattr(ypix, '__iter__'):
            mask = np.zeros(self.nrows, dtype=bool)
            if numexpr:
                for y1, y2 in ypix:
                    mask |= numexpr.evaluate('(col_ypix >= y1) & '
                                             '(col_ypix < y2)')
            else:
                for y1, y2 in ypix:
                    mask |= (col_ypix >= y1) & (col_ypix < y2)
        else:
            y1, y2 = ypix
            if numexpr:
                mask = numexpr.evaluate('(col_ypix >= y1) & (col_ypix < y2)')
            else:
                mask = (col_ypix >= y1) & (col_ypix < y2)
        return mask

    def select_sky(self, sky):
        """Return a mask corresponding to the given aperture on the sky
        (center, size and shape)

        Parameters
        ----------
        sky : (float, float, float, char)
            (y, x, size, shape) extract an aperture on the sky, defined by
            a center (y, x) in degrees/pixel, a shape ('C' for circular, 'S'
            for square) and size (radius or half side length) in arcsec/pixels.

        Returns
        -------
        out : array of bool
            mask
        """
        xpos, ypos = self.get_pos_sky()  # in degree or pixel here
        mask = np.zeros(self.nrows, dtype=bool)
        if numexpr:
            pi = np.pi  # NOQA
            for y0, x0, size, shape in sky:
                if shape == 'C':
                    if self.wcs == u.deg or self.wcs == u.rad:
                        mask |= numexpr.evaluate(
                            '(((xpos - x0) * 3600 * cos(y0 * pi / 180.)) ** 2 '
                            '+ ((ypos - y0) * 3600) ** 2) < size ** 2')
                    else:
                        mask |= numexpr.evaluate(
                            '((xpos - x0) ** 2 + (ypos - y0) ** 2) < size ** 2')
                elif shape == 'S':
                    if self.wcs == u.deg or self.wcs == u.rad:
                        mask |= numexpr.evaluate(
                            '(abs((xpos - x0) * 3600 * cos(y0 * pi / 180.)) < size) '
                            '& (abs((ypos - y0) * 3600) < size)')
                    else:
                        mask |= numexpr.evaluate(
                            '(abs(xpos - x0) < size) & (abs(ypos - y0) < size)')
                else:
                    raise ValueError('Unknown shape parameter')
        else:
            for y0, x0, size, shape in sky:
                if shape == 'C':
                    if self.wcs == u.deg or self.wcs == u.rad:
                        mask |= (((xpos - x0) * 3600
                                  * np.cos(y0 * DEG2RAD)) ** 2
                                 + ((ypos - y0) * 3600) ** 2) \
                            < size ** 2
                    else:
                        mask |= ((xpos - x0) ** 2
                                 + (ypos - y0) ** 2) < size ** 2
                elif shape == 'S':
                    if self.wcs == u.deg or self.wcs == u.rad:
                        mask |= (np.abs((xpos - x0) * 3600
                                        * np.cos(y0 * DEG2RAD)) < size) \
                            & (np.abs((ypos - y0) * 3600) < size)
                    else:
                        mask |= (np.abs(xpos - x0) < size) \
                            & (np.abs(ypos - y0) < size)
                else:
                    raise ValueError('Unknown shape parameter')
        return mask

    def extract_from_mask(self, mask):
        """Return a new pixtable extracted with the given mask.

        Parameters
        ----------
        mask : numpy.ndarray
            Mask (array of bool).

        Returns
        -------
        out : PixTable
        """
        if np.count_nonzero(mask) == 0:
            return None

        hdr = self.primary_header.copy()

        # xpos
        xpos = self.get_xpos(ksel=mask)
        hdr["%s LIMITS X LOW" % KEYWORD] = float(xpos.min())
        hdr["%s LIMITS X HIGH" % KEYWORD] = float(xpos.max())

        # ypos
        ypos = self.get_ypos(ksel=mask)
        hdr["%s LIMITS Y LOW" % KEYWORD] = float(ypos.min())
        hdr["%s LIMITS Y HIGH" % KEYWORD] = float(ypos.max())

        # lambda
        lbda = self.get_lambda(ksel=mask)
        hdr["%s LIMITS LAMBDA LOW" % KEYWORD] = float(lbda.min())
        hdr["%s LIMITS LAMBDA HIGH" % KEYWORD] = float(lbda.max())

        # origin
        origin = self.get_origin(ksel=mask)
        ifu = self.origin2ifu(origin)
        sl = self.origin2slice(origin)
        hdr["%s LIMITS IFU LOW" % KEYWORD] = int(ifu.min())
        hdr["%s LIMITS IFU HIGH" % KEYWORD] = int(ifu.max())
        hdr["%s LIMITS SLICE LOW" % KEYWORD] = int(sl.min())
        hdr["%s LIMITS SLICE HIGH" % KEYWORD] = int(sl.max())

        # merged pixtable
        if self.nifu > 1:
            hdr["%s MERGED" % KEYWORD] = len(np.unique(ifu))

        # combined exposures
        selfexp = self.get_exp()
        if selfexp is not None:
            newexp = selfexp[mask]
            numbers_exp = np.unique(newexp)
            hdr["%s COMBINED" % KEYWORD] = len(numbers_exp)
            for iexp, i in zip(numbers_exp, range(1, len(numbers_exp) + 1)):
                k = np.where(newexp == iexp)
                hdr["%s EXP%i FIRST" % (KEYWORD, i)] = k[0][0]
                hdr["%s EXP%i LAST" % (KEYWORD, i)] = k[0][-1]
            for i in range(len(numbers_exp) + 1,
                           self.get_keyword("COMBINED") + 1):
                del hdr["%s EXP%i FIRST" % (KEYWORD, i)]
                del hdr["%s EXP%i LAST" % (KEYWORD, i)]

        # return sub pixtable
        data = self.get_data(ksel=mask)
        stat = self.get_stat(ksel=mask)
        dq = self.get_dq(ksel=mask)
        weight = self.get_weight(ksel=mask)
        return PixTable(None, xpos, ypos, lbda, data, dq, stat, origin,
                        weight, hdr, self.ima, self.wcs, self.wave,
                        unit_data=self.unit_data)

    def extract(self, filename=None, sky=None, lbda=None, ifu=None, sl=None,
                xpix=None, ypix=None, exp=None, stack=None, method='and'):
        """Extracts a subset of a pixtable using the following criteria:

        - aperture on the sky (center, size and shape)
        - wavelength range
        - IFU numbers
        - slice numbers
        - detector pixels
        - exposure numbers
        - stack numbers

        The arguments can be either single value or a list of values to select
        multiple regions.

        Parameters
        ----------
        filename : str
            The FITS filename used to save the resulted object.
        sky : (float, float, float, char)
            (y, x, size, shape) extract an aperture on the sky, defined by
            a center (y, x) in degrees/pixel, a shape ('C' for circular, 'S'
            for square) and size (radius or half side length) in arcsec/pixels.
        lbda : (float, float)
            (min, max) wavelength range in angstrom.
        ifu : int or list
            IFU number.
        sl : int or list
            Slice number on the CCD.
        xpix : (int, int) or list
            (min, max) pixel range along the X axis
        ypix : (int, int) or list
            (min, max) pixel range along the Y axis
        exp : list of int
            List of exposure numbers
        stack : list of int
            List of stack numbers
        method : 'and' or 'or'
                 Logical operation used to merge the criteria

        Returns
        -------
        out : PixTable
        """
        if self.nrows == 0:
            return None

        if isinstance(sky, tuple):
            sky = [sky]
        if isinstance(lbda, tuple):
            lbda = [lbda]
        if np.isscalar(ifu):
            ifu = [ifu]
        if np.isscalar(sl):
            sl = [sl]
        if np.isscalar(stack):
            stack = [stack]

        if method == 'and':
            kmask = np.ones(self.nrows, dtype=bool)
            lfunc = np.logical_and
        elif method == 'or':
            kmask = np.zeros(self.nrows, dtype=bool)
            lfunc = np.logical_or

        # Do the selection on the sky
        if sky is not None:
            lfunc(kmask, self.select_sky(sky), out=kmask)

        # Do the selection on wavelengths
        if lbda is not None:
            lfunc(kmask, self.select_lambda(lbda, unit=u.angstrom), out=kmask)

        # Do the selection on the origin column
        if (ifu is not None) or (sl is not None) or (stack is not None) or \
                (xpix is not None) or (ypix is not None):
            origin = self.get_origin()
            if sl is not None:
                lfunc(kmask, self.select_slices(sl, origin=origin), out=kmask)
            if stack is not None:
                lfunc(kmask, self.select_stacks(stack, origin=origin),
                      out=kmask)
            if ifu is not None:
                lfunc(kmask, self.select_ifus(ifu, origin=origin), out=kmask)
            if xpix is not None:
                lfunc(kmask, self.select_xpix(xpix, origin=origin), out=kmask)
            if ypix is not None:
                lfunc(kmask, self.select_ypix(ypix, origin=origin), out=kmask)
            origin = None

        # Do the selection on the exposure numbers
        if exp is not None:
            col_exp = self.get_exp()
            if col_exp is not None:
                lfunc(kmask, self.select_exp(exp, col_exp), out=kmask)

        # Compute the new pixtable
        pix = self.extract_from_mask(kmask)
        if pix is not None and filename is not None:
            pix.filename = filename
            pix.write(filename)
        return pix

    def origin2ifu(self, origin):
        """Converts the origin value and returns the ifu number.

        Parameters
        ----------
        origin : int
            Origin value.

        Returns
        -------
        out : int
        """
        return (origin >> 6) & 0x1f

    def origin2slice(self, origin):
        """Converts the origin value and returns the slice number.

        Parameters
        ----------
        origin : int
            Origin value.

        Returns
        -------
        out : int
        """
        return origin & 0x3f

    def origin2ypix(self, origin):
        """Converts the origin value and returns the y coordinates.

        Parameters
        ----------
        origin : int
            Origin value.

        Returns
        -------
        out : float
        """
        return ((origin >> 11) & 0x1fff) - 1

    def origin2xoffset(self, origin, ifu=None, sli=None):
        """Converts the origin value and returns the x coordinates offset.

        Parameters
        ----------
        origin : int
            Origin value.

        Returns
        -------
        out : float
        """
        col_ifu = ifu if ifu is not None else self.origin2ifu(origin)
        col_slice = sli if sli is not None else self.origin2slice(origin)
        key = "EXP0 IFU%02d SLICE%02d XOFFSET"

        if isinstance(origin, np.ndarray):
            ifus = np.unique(col_ifu)
            slices = np.unique(col_slice)
            offsets = np.zeros((ifus.max() + 1, slices.max() + 1),
                               dtype=np.int32)
            for ifu in ifus:
                for sl in slices:
                    offsets[ifu, sl] = self.get_keyword(key % (ifu, sl))

            xoffset = offsets[col_ifu, col_slice]
        else:
            xoffset = self.get_keyword(key % (col_ifu, col_slice))
        return xoffset

    def origin2xpix(self, origin, ifu=None, sli=None):
        """Converts the origin value and returns the x coordinates.

        Parameters
        ----------
        origin : int
            Origin value.

        Returns
        -------
        out : float
        """
        return (self.origin2xoffset(origin, ifu=ifu, sli=sli) +
                ((origin >> NIFUS) & 0x7f) - 1)

    def origin2coords(self, origin):
        """Converts the origin value and returns (ifu, slice, ypix, xpix).

        Parameters
        ----------
        origin : int
            Origin value.

        Returns
        -------
        out : (int, int, float, float)
        """
        ifu, sli = self.origin2ifu(origin), self.origin2slice(origin)
        return (ifu, sli, self.origin2ypix(origin),
                self.origin2xpix(origin, ifu=ifu, sli=sli))

    def _get_pos_sky(self, xpos, ypos):
        if self.projection == 'projected':  # spheric coordinates
            phi = xpos
            theta = ypos + np.pi / 2
            dp = self.yc * DEG2RAD
            ra = np.arctan2(np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.cos(dp) +
                            np.cos(theta) * np.sin(dp) * np.cos(phi)) * RAD2DEG
            xpos_sky = self.xc + ra
            ypos_sky = np.arcsin(np.sin(theta) * np.sin(dp) -
                                 np.cos(theta) * np.cos(dp) * np.cos(phi)) * RAD2DEG
        else:
            if self.wcs == u.deg:
                # dp = self.yc * DEG2RAD
                xpos_sky = self.xc + xpos
                ypos_sky = self.yc + ypos
            elif self.wcs == u.rad:
                # dp = self.yc * DEG2RAD
                xpos_sky = self.xc + xpos * RAD2DEG
                ypos_sky = self.yc + ypos * RAD2DEG
            else:
                xpos_sky = self.xc + xpos
                ypos_sky = self.yc + ypos
        return xpos_sky, ypos_sky

    def _get_pos_sky_numexpr(self, xpos, ypos):
        pi = np.pi  # NOQA
        xc = self.xc  # NOQA
        yc = self.yc  # NOQA
        if self.projection == 'projected':  # spheric coordinates
            phi = xpos  # NOQA
            theta = numexpr.evaluate("ypos + pi/2")
            dp = numexpr.evaluate("yc * pi / 180")
            ra = numexpr.evaluate("arctan2(cos(theta) * sin(phi), sin(theta) * cos(dp) + cos(theta) * sin(dp) * cos(phi)) * 180 / pi")
            xpos_sky = numexpr.evaluate("xc + ra")
            ypos_sky = numexpr.evaluate("arcsin(sin(theta) * sin(dp) - cos(theta) * cos(dp) * cos(phi)) * 180 / pi")
        else:
            if self.wcs == u.deg:
                # dp = numexpr.evaluate("yc * pi / 180")
                xpos_sky = numexpr.evaluate("xc + xpos")
                ypos_sky = numexpr.evaluate("yc + ypos")
            elif self.wcs == u.rad:
                # dp = numexpr.evaluate("yc * pi / 180")
                xpos_sky = numexpr.evaluate("xc + xpos * 180 / pi")
                ypos_sky = numexpr.evaluate("yc + ypos * 180 / pi")
            else:
                xpos_sky = numexpr.evaluate("xc + xpos")
                ypos_sky = numexpr.evaluate("yc + ypos")
        return xpos_sky, ypos_sky

    def get_pos_sky(self, xpos=None, ypos=None):
        """Return the absolute position on the sky in degrees/pixel.

        Parameters
        ----------
        xpos : numpy.array
            xpos values
        ypos : numpy.array
            ypos values

        Returns
        -------
        xpos_sky, ypos_sky : numpy.array, numpy.array
        """
        if xpos is None:
            xpos = self.get_xpos()
        if ypos is None:
            ypos = self.get_ypos()
        if numexpr:
            return self._get_pos_sky_numexpr(xpos, ypos)
        else:
            return self._get_pos_sky(xpos, ypos)

    def get_keyword(self, key):
        """Return the keyword value corresponding to key, adding the keyword
        prefix (``'HIERARCH ESO DRS MUSE PIXTABLE'``).

        Parameters
        ----------
        key : str
            Keyword.

        Returns
        -------
        out : keyword value

        """
        return self.primary_header['{} {}'.format(KEYWORD, key)]

    def set_keyword(self, key, val):
        """Set the keyword value corresponding to key, adding the keyword
        prefix (``'HIERARCH ESO DRS MUSE PIXTABLE'``).

        Parameters
        ----------
        key : str
            Keyword.
        val : str or int or float
            Value.

        """
        self.primary_header['{} {}'.format(KEYWORD, key)] = val

    def reconstruct_det_image(self, xstart=None, ystart=None,
                              xstop=None, ystop=None):
        """Reconstructs the image on the detector from the pixtable.

        The pixtable must concerns only one IFU, otherwise an exception is
        raised.

        Returns
        -------
        out : `~mpdaf.obj.Image`
        """
        if self.nrows == 0:
            return None

        if self.nifu != 1:
            raise ValueError('Pixtable contains multiple IFU')

        col_data = self.get_data()
        col_origin = self.get_origin()

        ifu = np.empty(self.nrows, dtype='uint16')
        sl = np.empty(self.nrows, dtype='uint16')
        xpix = np.empty(self.nrows, dtype='uint16')
        ypix = np.empty(self.nrows, dtype='uint16')

        ifu, sl, ypix, xpix = self.origin2coords(col_origin)
        if len(np.unique(ifu)) != 1:
            raise ValueError('Pixtable contains multiple IFU')

        if xstart is None:
            xstart = xpix.min()
        if xstop is None:
            xstop = xpix.max()
        if ystart is None:
            ystart = ypix.min()
        if ystop is None:
            ystop = ypix.max()
        # xstart, xstop = xpix.min(), xpix.max()
        # ystart, ystop = ypix.min(), ypix.max()
        image = np.zeros((ystop - ystart + 1,
                          xstop - xstart + 1), dtype='float') * np.NaN
        image[ypix - ystart, xpix - xstart] = col_data

        wcs = WCS(crval=(ystart, xstart))
        return Image(data=image, wcs=wcs, unit=self.unit_data, copy=False)

    def reconstruct_det_waveimage(self):
        """Reconstructs an image of wavelength values on the detector from the
        pixtable. The pixtable must concerns only one IFU, otherwise an
        exception is raised.

        Returns
        -------
        out : `~mpdaf.obj.Image`
        """
        if self.nrows == 0:
            return None

        if self.nifu != 1:
            raise ValueError('Pixtable contains multiple IFU')

        col_origin = self.get_origin()
        col_lambdas = self.get_lambda()

        ifu = np.empty(self.nrows, dtype='uint16')
        sl = np.empty(self.nrows, dtype='uint16')
        xpix = np.empty(self.nrows, dtype='uint16')
        ypix = np.empty(self.nrows, dtype='uint16')

        ifu, sl, ypix, xpix = self.origin2coords(col_origin)
        if len(np.unique(ifu)) != 1:
            raise ValueError('Pixtable contains multiple IFU')

        xstart, xstop = xpix.min(), xpix.max()
        ystart, ystop = ypix.min(), ypix.max()
        image = np.zeros((ystop - ystart + 1, xstop - xstart + 1),
                         dtype='float')
        image[ypix - ystart, xpix - xstart] = col_lambdas

        wcs = WCS(crval=(ystart, xstart))

        return Image(data=image, wcs=wcs, unit=self.wave, copy=False)

    def mask_column(self, maskfile=None):
        """Compute the mask column corresponding to a mask file.

        Parameters
        ----------
        maskfile : str
            Path to a FITS image file with WCS information, used to mask
            out bright continuum objects present in the FoV. Values must
            be 0 for the background and >0 for objects.

        Returns
        -------
        out : `mpdaf.drs.PixTableMask`

        """
        if maskfile is None:
            return np.zeros(self.nrows, dtype=bool)

        pos = np.array(self.get_pos_sky()[::-1]).T
        ima_mask = Image(maskfile, dtype=bool)
        sky = ima_mask.wcs.sky2pix(pos, nearest=True, unit=u.deg).T
        mask = ima_mask.data.data[sky[0], sky[1]]

        return PixTableMask(maskfile=maskfile, maskcol=mask,
                            pixtable=self.filename)

    @deprecated('This method was used for the old auto-calibration algorithm '
                'and is now useless.')
    def sky_ref(self, pixmask=None, dlbda=1.0, nmax=2, nclip=5.0, nstop=2,
                lmin=None, lmax=None):
        """Compute the reference sky spectrum using sigma clipped median.

        Algorithm from Kurt Soto (kurt.soto@phys.ethz.ch)

        Parameters
        ----------
        pixmask : `mpdaf.drs.PixTableMask`
            Column corresponding to a mask file (previously computed by
            ``mask_column``).
        dlbda : double
            Wavelength step in angstrom
        nmax : int
            Maximum number of clipping iterations
        nclip : float or (float,float)
            Number of sigma at which to clip. Single clipping parameter or
            lower/upper clipping parameters
        nstop : int
            If the number of not rejected pixels is less than this number,
            the clipping iterations stop.

        Returns
        -------
        out : `~mpdaf.obj.Spectrum`
        """

        from ..tools.ctools import ctools

        # mask
        if pixmask is None:
            maskfile = ''
            mask = np.zeros(self.nrows, dtype=bool)
        else:
            maskfile = _get_file_basename(pixmask.maskfile)
            mask = pixmask.maskcol

        # sigma clipped parameters
        if np.isscalar(nclip):
            nclip_low, nclip_up = nclip, nclip
        else:
            nclip_low, nclip_up = nclip

        # wavelength step
        lbda = self.get_lambda(unit=u.angstrom)
        lmin = lmin or np.min(lbda) - dlbda / 2.0
        lmax = lmax or np.max(lbda) + dlbda / 2.0
        n = int((lmax - lmin) / dlbda) + 1

        data = self.get_data()
        data = data.astype(np.float64)
        lbda = lbda.astype(np.float64)
        mask = mask.astype(np.int32)
        result = np.empty(n, dtype=np.float64)
        # run C method
        ctools.mpdaf_sky_ref(data, lbda, mask, data.shape[0],
                             np.float64(lmin),
                             np.float64(dlbda), n, nmax,
                             np.float64(nclip_low),
                             np.float64(nclip_up), nstop, result)
        wave = WaveCoord(crpix=1.0, cdelt=dlbda, crval=np.min(lbda),
                         cunit=u.angstrom, shape=n)

        spe = Spectrum(data=result, wave=wave, unit=self.unit_data, copy=False)
        add_mpdaf_method_keywords(spe.primary_header,
                                  "drs.pixtable.sky_ref",
                                  ['pixtable', 'mask', 'dlbda', 'nmax',
                                   'nclip_low', 'nclip_up', 'nstop'],
                                  [_get_file_basename(self.filename), maskfile,
                                   dlbda, nmax, nclip_low, nclip_up, nstop],
                                  ['pixtable',
                                   'file to mask out all bright objects',
                                   'wavelength step',
                                   'max number of clipping iterations',
                                   'lower clipping parameter',
                                   'upper clipping parameter',
                                   'clipping minimum number'])
        return spe

    def subtract_slice_median(self, skyref, pixmask):
        raise AttributeError('This method was replaced with .selfcalibrate')

    def divide_slice_median(self, skyref, pixmask):
        raise AttributeError('This method was replaced with .selfcalibrate')

    @deprecated('The self-calibration method is available in the DRS since '
                'version 2.4, with the ``autocalib="deepfield"`` parameter. '
                'This should be prefered as it is more efficient, and includes'
                ' some bugfixes and new features.')
    def selfcalibrate(self, pixmask=None, corr_clip=15.0, logfile=None):
        """Correct the background level of the slices.

        This requires a Pixtable that is *not sky-subtracted*. It uses the mean
        sky level as a reference, and compute a multiplicative correction to
        apply to each slice to bring its background level to the reference one.

        A mask of sources can be provided. A `~mpdaf.drs.PixTableMask` must be
        computed from a 2D mask file with `~mpdaf.drs.PixTable.mask_column`.
        This mask will be merged with the 'DQ' column from the pixtable.

        The method works on wavelength bins, defined in
        `mpdaf.drs.pixtable.SKY_SEGMENTS`. These bins have been chosen so that
        their edges do not fall on a sky line, with a 200A to 300A width. They
        can also be modified (by setting a new list to this global variable).

        Then, the algorithm can be summarized as follow::

            - foreach lambda bin:
                - foreach ifu:
                    - foreach slice:
                        - foreach pixel:
                            - compute the mean flux
                        - compute the median flux of the slice
                    - compute the median flux of the ifu
                    - foreach slice:
                        - if too few points, use the mean ifu flux
                - compute the total mean flux

                - foreach ifu:
                    - foreach slice:
                        - compute the correction: total_flux / slice_flux
                    - compute the mean and stddev of the corrections
                    - for slices where |correction - mean| > corr_clip*std_dev:
                        - use the mean ifu correction: total_flux / ifu_flux

            - foreach ifu:
                - foreach slice:
                    - foreach lambda bin:
                        - reject spikes in the correction curves, using
                          a comparison with the correction from the next and
                          previous lambda bin.

        Parameters
        ----------
        pixmask : `mpdaf.drs.PixTableMask`
            Column corresponding to a mask file (previously computed by
            `~mpdaf.drs.PixTable.mask_column`).
        corr_clip : float
            Clipping threshold for slice corrections in one IFU.
        logfile : str
            Path to a file to which the log will be written.

        Returns
        -------
        out : `mpdaf.drs.PixTableAutoCalib`

        """
        from ..tools.ctools import ctools

        if pixmask is None:
            maskfile = ''
            maskcol = np.zeros(self.nrows, dtype=bool)
        else:
            maskfile = _get_file_basename(pixmask.maskfile)
            maskcol = pixmask.maskcol

        origin = self.get_origin()
        ifu = np.asarray(self.origin2ifu(origin), dtype=np.int32)
        sli = np.asarray(self.origin2slice(origin), dtype=np.int32)
        xpix = np.asarray((origin >> NIFUS) & 0x7f, dtype=np.int32)
        origin = None

        data = np.asarray(self.get_data(), dtype=np.float64)
        stat = np.asarray(self.get_stat(), dtype=np.float64)
        lbda = np.asarray(self.get_lambda(), dtype=np.float64)
        mask = np.asarray(maskcol | self.get_dq().astype(bool), dtype=np.int32)

        skyseg = np.array(SKY_SEGMENTS, dtype=np.int32)
        nquad = len(skyseg) - 1
        ncorr = NIFUS * NSLICES * nquad
        result = np.empty_like(data, dtype=np.float64)
        result_stat = np.empty_like(data, dtype=np.float64)
        corr = np.empty(ncorr, dtype=np.float64)
        npts = np.empty(ncorr, dtype=np.int32)

        if logfile is None:
            logfile = ''
        logfile = logfile.encode('utf8')

        ctools.mpdaf_slice_median(
            result, result_stat, corr, npts, ifu, sli, data, stat, lbda,
            data.shape[0], mask, xpix, nquad, skyseg, corr_clip, logfile)

        # set pixtable data
        self.set_data(result)
        self.set_stat(result_stat)

        # store parameters of the method in FITS keywords
        add_mpdaf_method_keywords(
            self.primary_header, "drs.pixtable.selfcalibrate",
            ['mask', 'corr_clip'], [maskfile, corr_clip],
            ['file to mask out all bright objects',
             'Clipping threshold for slice corrections in one IFU'])

        return PixTableAutoCalib(
            method='drs.pixtable.selfcalibrate', maskfile=maskfile,
            pixtable=_get_file_basename(self.filename),
            ifu=np.resize(np.repeat(np.arange(1, NIFUS + 1), NSLICES), ncorr),
            sli=np.resize(np.arange(1, NSLICES + 1), ncorr),
            quad=np.repeat(np.arange(1, nquad + 1), NSLICES * NIFUS),
            npts=npts, corr=corr)
