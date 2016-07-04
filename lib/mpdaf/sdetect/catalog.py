"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c) 2015-2016 Roland Bacon <roland.bacon@univ-lyon1.fr>

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

import glob
import logging
import numpy as np
import os
import six
import sys

from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, hstack, vstack
from astropy import units as u
from matplotlib.patches import Ellipse
from six.moves import range, zip

INVALID = {
    type(1): -9999, np.int_: -9999,
    type(1.0): np.nan, np.float_: np.nan,
    type('1'): '', np.str_: '',
    type(False): -9999, np.bool_: -9999
}


class Catalog(Table):

    """This class inherits from `astropy.table.Table`.
    Its goal is to manage a list of objects.
    """

    def __init__(self, *args, **kwargs):
        super(Catalog, self).__init__(*args, **kwargs)
        self._logger = logging.getLogger(__name__)
        for name in ('ra', 'dec', 'z'):
            if name in self.colnames:
                self.rename_column(name, name.upper())
        self.masked_invalid()

    @classmethod
    def from_sources(cls, sources, fmt='default'):
        """Construct a catalog from a list of source objects.

        The new catalog will contain all data stored in the primary headers
        and in the tables extensions of the sources:

        * a column per header fits
        * two columns per magnitude band:
          [BAND] [BAND]_ERR
        * three columns per redshift
          Z_[Z_DESC], Z_[Z_DESC]_MIN and Z_[Z_DESC]_MAX
        * several columns per line.

        The lines columns depend of the format.
        By default the columns names are created around unique LINE name
        [LINE]_[LINES columns names].
        But it is possible to use a working format.
        [LINES columns names]_xxx
        where xxx is the number of lines present in each source.

        Parameters
        ----------
        sources : list< `mpdaf.sdetect.Source` >
            List of `mpdaf.sdetect.Source` objects
        fmt : str 'working'|'default'
            Format of the catalog. The format differs for the LINES table.

        """
        logger = logging.getLogger(__name__)
        # union of all headers keywords without mandatory FITS keywords
        h = sources[0].header

        d = dict(list(zip(list(h.keys()), [(type(c[1]), c[2]) for c in h.cards])))
        for source in sources[1:]:
            h = source.header
            d.update(dict(list(zip(list(h.keys()), [(type(c[1]), c[2]) for c in h.cards]))))

        excluded_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE',
                          'AUTHOR']
        i = 1
        while 'COM%03d' % i in list(d.keys()):
            excluded_cards.append('COM%03d' % i)
            i += 1
        i = 1
        while 'HIST%03d' % i in list(d.keys()):
            excluded_cards.append('HIST%03d' % i)
            i += 1

        d = {key: value for key, value in d.items() if key not in excluded_cards}
        names_hdr = list(d.keys())
        tuple_hdr = list(d.values())
        # sort mandatory keywords
        index = names_hdr.index('ID')
        names_hdr.insert(0, names_hdr.pop(index))
        tuple_hdr.insert(0, tuple_hdr.pop(index))
        index = names_hdr.index('RA')
        names_hdr.insert(1, names_hdr.pop(index))
        tuple_hdr.insert(1, tuple_hdr.pop(index))
        index = names_hdr.index('DEC')
        names_hdr.insert(2, names_hdr.pop(index))
        tuple_hdr.insert(2, tuple_hdr.pop(index))
        index = names_hdr.index('ORIGIN')
        names_hdr.insert(3, names_hdr.pop(index))
        tuple_hdr.insert(3, tuple_hdr.pop(index))
        index = names_hdr.index('ORIGIN_V')
        names_hdr.insert(4, names_hdr.pop(index))
        tuple_hdr.insert(4, tuple_hdr.pop(index))
        index = names_hdr.index('CUBE')
        names_hdr.insert(5, names_hdr.pop(index))
        tuple_hdr.insert(5, tuple_hdr.pop(index))

        dtype_hdr = [c[0] for c in tuple_hdr]
        desc_hdr = [c[1][:c[1].find('u.')] if c[1].find('u.') != -1 else c[1][:c[1].find('%')] if c[1].find('%') != -1 else c[1] for c in tuple_hdr]
        unit_hdr = [c[1][c[1].find('u.'):].split()[0][2:] if c[1].find('u.') != -1 else None for c in tuple_hdr]
        format_hdr = [c[1][c[1].find('%'):].split()[0] if c[1].find('%') != -1 else None for c in tuple_hdr]

        # magnitudes
        lmag = [len(source.mag) for source in sources if source.mag is not None]
        if len(lmag) != 0:
            names_mag = list(set(np.concatenate([source.mag['BAND'].data.data for source in sources
                                                 if source.mag is not None])))
            names_mag += ['%s_ERR' % mag for mag in names_mag]
            names_mag.sort()
        else:
            names_mag = []

        # redshifts
        lz = [len(source.z) for source in sources if source.z is not None]
        if len(lz) != 0:
            names_z = list(set(np.concatenate([source.z['Z_DESC'].data.data for source in sources
                                               if source.z is not None])))
            names_z = ['Z_%s' % z for z in names_z]
            colnames = list(set(np.concatenate([source.z.colnames for source in sources if source.z is not None])))
            if 'Z_ERR' in colnames:
                names_err = ['%s_ERR' % z for z in names_z]
            else:
                names_err = []
            if 'Z_MIN' in colnames:
                names_min = ['%s_MIN' % z for z in names_z]
                names_max = ['%s_MAX' % z for z in names_z]
            else:
                names_min = []
                names_max = []
            names_z += names_err
            names_z += names_min
            names_z += names_max
            names_z.sort()
        else:
            names_z = []

        # lines
        llines = [len(source.lines) for source in sources if source.lines is not None]
        if len(llines) == 0:
            names_lines = []
            dtype_lines = []
            units_lines = []
        else:
            if fmt == 'default':
                names_lines = []
                d = {}
                unit = {}
                for source in sources:
                    if source.lines is not None and 'LINE' in source.lines.colnames:
                        colnames = source.lines.colnames
                        colnames.remove('LINE')

                        for col in colnames:
                            d[col] = source.lines.dtype[col]
                            unit[col] = source.lines[col].unit

                        for line, mask in zip(source.lines['LINE'].data.data, source.lines['LINE'].data.mask):
                            if not mask and line != 'None':
                                try:
                                    float(line)
                                    logger.warning('source %d: line labeled \"%s\" not loaded' % (source.ID, line))
                                except:
                                    names_lines += ['%s_%s' % (line.replace('_', ''), col) for col in colnames]

                names_lines = list(set(np.concatenate([names_lines])))
                names_lines.sort()
                dtype_lines = [d['_'.join(name.split('_')[1:])] for name in names_lines]
                units_lines = [unit['_'.join(name.split('_')[1:])] for name in names_lines]
            elif fmt == 'working':
                lmax = max(llines)
                d = {}
                unit = {}
                for source in sources:
                    if source.lines is not None:
                        for col in source.lines.colnames:
                            d[col] = source.lines.dtype[col]
                            unit[col] = source.lines[col].unit
                if lmax == 1:
                    names_lines = sorted(d)
                    dtype_lines = [d[key] for key in sorted(d)]
                    units_lines = [unit[key] for key in sorted(d)]
                else:
                    names_lines = []
                    inames_lines = sorted(d)
                    for i in range(1, lmax + 1):
                        names_lines += [col + '%03d' % i for col in inames_lines]
                    dtype_lines = [d[key] for key in sorted(d)] * lmax
                    units_lines = [unit[key] for key in sorted(d)] * lmax
            else:
                raise IOError('Catalog creation: invalid format. It must be default or working.')

        data_rows = []
        for source in sources:
            # header
            h = source.header
            keys = list(h.keys())
            row = []
            for key, typ in zip(names_hdr, dtype_hdr):
                if typ == type('1'):
                    row += ['%s' % h[key] if key in keys else INVALID[typ]]
                else:
                    row += [h[key] if key in keys else INVALID[typ]]

            # magnitudes
            if len(lmag) != 0:
                if source.mag is None:
                    row += [np.nan for key in names_mag]
                else:
                    keys = source.mag['BAND']
                    for key in names_mag:
                        if key in keys:
                            row += [source.mag['MAG'][source.mag['BAND'] == key].data.data[0]]
                        elif key[-4:] == '_ERR' and key[:-4] in keys:
                            row += [source.mag['MAG_ERR'][source.mag['BAND'] == key[:-4]].data.data[0]]
                        else:
                            row += [np.nan]
            # redshifts
            if len(lz) != 0:
                if source.z is None:
                    row += [np.nan for key in names_z]
                else:
                    keys = source.z['Z_DESC']
                    for key in names_z:
                        key = key[2:]
                        if key in keys:
                            row += [source.z['Z'][source.z['Z_DESC'] == key].data.data[0]]
                        elif key[-4:] == '_MAX' and key[:-4] in keys:
                            row += [source.z['Z_MAX'][source.z['Z_DESC'] == key[:-4]].data.data[0]]
                        elif key[-4:] == '_MIN' and key[:-4] in keys:
                            row += [source.z['Z_MIN'][source.z['Z_DESC'] == key[:-4]].data.data[0]]
                        elif key[-4:] == '_ERR' and key[:-4] in keys:
                            row += [source.z['Z_ERR'][source.z['Z_DESC'] == key[:-4]].data.data[0]]
                        else:
                            row += [np.nan]
            # lines
            if len(llines) != 0:
                if source.lines is None:
                    for typ in dtype_lines:
                        row += [INVALID[typ.type]]
                else:
                    if fmt == 'default' and 'LINE' in source.lines.colnames:
                        copy = source.lines['LINE'].data.data
                        for i in range(len(source.lines)):
                            source.lines['LINE'][i] = source.lines['LINE'][i].replace('_', '')
                        for name, typ in zip(names_lines, dtype_lines):
                            colname = '_'.join(name.split('_')[1:])
                            line = name.split('_')[0]
                            if 'LINE' in source.lines.colnames and \
                               colname in source.lines.colnames and \
                               line in source.lines['LINE'].data.data:
                                row += [source.lines[colname][source.lines['LINE'] == line].data.data[0]]
                            else:
                                row += [INVALID[typ.type]]
                        for i in range(len(source.lines)):
                            source.lines['LINE'][i] = copy[i]
                    elif fmt == 'working':
                        keys = source.lines.colnames
                        if lmax == 1:
                            row += [source.lines[key][0] if key in keys else INVALID[typ.type] for key, typ in zip(names_lines, dtype_lines)]
                        else:
                            try:
                                subtab1 = source.lines[source.lines['LINE'] != ""]
                                subtab2 = source.lines[source.lines['LINE'] == ""]
                                lines = vstack([subtab1, subtab2])
                            except:
                                lines = source.lines
                            n = len(lines)
                            for key, typ in zip(names_lines, dtype_lines):
                                if key[:-3] in keys and int(key[-3:]) <= n:
                                    row += [lines[key[:-3]][int(key[-3:]) - 1]]
                                else:
                                    row += [INVALID[typ.type]]
                    else:
                        pass

            # final row
            data_rows.append(row)

        dtype = dtype_hdr

        # magnitudes
        if len(lmag) != 0:
            dtype += ['f8' for i in range(len(names_mag))]
        # redshifts
        if len(lz) != 0:
            dtype += ['f8' for i in range(len(names_z))]
        # lines
        if len(llines) != 0:
            dtype += dtype_lines

        # create Table
        names = names_hdr + names_mag + names_z + names_lines

        t = cls(rows=data_rows, names=names, masked=True, dtype=dtype)

        # format
        for name, desc, unit, fmt in zip(names_hdr, desc_hdr, unit_hdr, format_hdr):
            t[name].description = desc
            t[name].unit = unit
            t[name].format = fmt
        for name in names_z:
            t[name].format = '.4f'
            t[name].unit = 'unitless'
            if name[-3:] == 'MIN':
                t[name].description = 'Lower bound of estimated redshift'
            elif name[-3:] == 'MAX':
                t[name].description = 'Upper bound of estimated redshift'
            elif name[-3:] == 'ERR':
                t[name].description = 'Error of estimated redshift'
            else:
                t[name].description = 'Estimated redshift'
        for name in names_mag:
            t[name].format = '.3f'
            t[name].unit = 'unitless'
            if name[-3:] == 'ERR':
                t[name].description = 'Error in AB Magnitude'
            else:
                t[name].description = 'AB Magnitude'
        if len(llines) != 0:
            for name, unit in zip(names_lines, units_lines):
                t[name].unit = unit
                if 'LBDA' in name or 'EQW' in name:
                    t[name].format = '.2f'
                if 'FLUX' in name or 'FWHM' in name:
                    t[name].format = '.1f'
        return t

    @classmethod
    def from_path(cls, path, fmt='default', pattern='*.fits'):
        """Create a Catalog object from the path of a directory containing
        source files.

        Parameters
        ----------
        path : str
            Directory containing Source files
        fmt : str 'working'|'default'
            Format of the catalog. The format differs for the LINES table.
        pattern : str
            Pattern used to select the files, default to ``*.fits``.

        """
        logger = logging.getLogger(__name__)

        if not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))

        from .source import Source

        slist = []
        filenames = []
        files = glob.glob(os.path.join(path, pattern))
        n = len(files)
        logger.info('Building catalog from path %s', path)

        for f in files:
            try:
                slist.append(Source._light_from_file(f))
                filenames.append(os.path.basename(f))
            except KeyboardInterrupt:
                return
            except Exception:
                logger.warning('source %s not loaded', f, exc_info=True)
            sys.stdout.write("\r\x1b[K %i%%" % (100 * len(filenames) / n))
            sys.stdout.flush()

        sys.stdout.write("\r\x1b[K ")
        sys.stdout.flush()

        t = cls.from_sources(slist, fmt)
        t['FILENAME'] = filenames

        return t

    def masked_invalid(self):
        """Mask where invalid values occur (NaNs or infs or -9999 or '')."""
        for col in self.colnames:
            try:
                self[col] = np.ma.masked_invalid(self[col])
                self[col] = np.ma.masked_equal(self[col], -9999)
            except:
                pass

    def match(self, cat2, radius=1, colc1=('RA', 'DEC'), colc2=('RA', 'DEC'),
              full_output=True):
        """Match elements of the current catalog with an other (in RA, DEC).

        Parameters
        ----------
        cat2 : `astropy.table.Table`
            Catalog to match.
        radius : float
            Matching size in arcsec (default 1).
        colc1: tuple
            ('RA','DEC') name of ra,dec columns of input table
        colc2: tuple
            ('RA','DEC') name of ra,dec columns of cat2

        Returns
        -------

        if full_output is True

        out : astropy.Table, astropy.Table, astropy.Table
             match, nomatch1, nomatch2
        else

        out : astropy.Table
              match

        match: table of matched elements in RA,DEC
        nomatch1: sub-table of non matched elements of the current catalog
        nomatch2: sub-table of non matched elements of the catalog cat2

        """
        coord1 = SkyCoord(self[colc1[0]], self[colc1[1]],
                          unit=(u.degree, u.degree))
        coord2 = SkyCoord(cat2[colc2[0]], cat2[colc2[1]],
                          unit=(u.degree, u.degree))
        id2, d2d, d3d = coord1.match_to_catalog_sky(coord2)
        id1 = np.arange(len(self))
        kmatch = d2d < radius * u.arcsec
        id2match = id2[kmatch]
        d2match = d2d[kmatch]
        id1match = id1[kmatch]
        # search non unique index
        m = np.zeros_like(id2match, dtype=bool)
        m[np.unique(id2match, return_index=True)[1]] = True
        duplicate = id2match[~m]
        if len(duplicate) > 0:
            self._logger.debug('Found %d duplicate in matching catalogs',
                               len(duplicate))
            to_remove = []
            for k in duplicate:
                mask = id2match == k
                idlist = np.arange(len(id2match))[mask]
                to_remove.append(idlist[d2match[mask].argsort()[1:]].tolist())
            id2match = np.delete(id2match, to_remove)
            id1match = np.delete(id1match, to_remove)
            d2match = np.delete(d2match, to_remove)
        match1 = self[id1match]
        match2 = cat2[id2match]
        match = hstack([match1, match2], join_type='exact')
        match.add_column(Column(data=d2match.to(u.arcsec), name='Distance',
                                dtype=float))
        if full_output:
            id1notmatch = np.in1d(range(len(self)), id1match,
                                  assume_unique=True, invert=True)
            id2notmatch = np.in1d(range(len(cat2)), id2match,
                                  assume_unique=True, invert=True)
            nomatch2 = cat2[id2notmatch]
            nomatch1 = self[id1notmatch]
            self._logger.debug('Cat1 Nelt %d Matched %d Not Matched %d'
                               % (len(self), len(match1), len(nomatch1)))
            self._logger.debug('Cat2 Nelt %d Matched %d Not Matched %d'
                               % (len(cat2), len(match2), len(nomatch2)))
            return match, nomatch1, nomatch2
        else:
            self._logger.debug('Cat1 Nelt %d Cat2 Nelt %d Matched %d'
                               % (len(self), len(cat2), len(match1)))
            return match

    def select(self, wcs, ra='RA', dec='DEC', margin=0):
        """Select all sources from catalog which are inside the given WCS
        and return a new catalog.

        Parameters
        ----------
        wcs    : `~mpdaf.obj.WCS`
                 Image WCS
        ra     : str
                 Name of the column that contains RA values in degrees.
        dec    : str
                 Name of the column that contains DEC values in degrees.
        margin : int
                 Margin from the edges.

        Returns
        -------
        out : `mpdaf.sdetect.Catalog`

        """
        arr = np.vstack([self[dec].data, self[ra].data]).T
        cen = wcs.sky2pix(arr, unit=u.deg).T
        sel = ((cen[0] > margin) & (cen[0] < wcs.naxis1 - margin) &
               (cen[1] > margin) & (cen[1] < wcs.naxis2 - margin))
        return self[sel]

    def edgedist(self, wcs, ra='RA', dec='DEC'):
        """Return the smallest distance of all catalog sources center to the
        edge of the WCS of the given image.

        Parameters
        ----------
        wcs : `~mpdaf.obj.WCS`
              Image WCS
        ra  : str
              Name of the column that contains RA values in degrees
        dec : str
              Name of the column that contains DEC values in degrees

        Returns
        -------
        out : `numpy.array` in arcsec units

        """
        dim = np.array([wcs.naxis1, wcs.naxis2])
        pix = wcs.sky2pix(np.array([self[dec], self[ra]]).T, unit=u.deg)
        dist = np.hstack([pix, dim - pix]).min(axis=1)
        step = wcs.get_step()[0] * u.deg
        dist *= step.to(u.arcsec)
        return dist

    def plot_symb(self, ax, wcs, ra='RA', dec='DEC',
                  symb=0.4, col='k', alpha=1.0, **kwargs):
        """This function plots the sources location from the catalog.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        wcs : `mpdaf.obj.WCS`
            Image WCS
        ra : str
            Name of the column that contains RA values (in degrees)
        dec : str
            Name of the column that contains DEC values (in degrees)
        symb : list or str or float
            - List of 3 columns names containing FWHM1,
                FWHM2 and ANGLE values to define the ellipse of each source.
            - Column name containing value that will be used
                to define the circle size of each source.
            - float in the case of circle with constant size in arcsec
        col : str
            Symbol color.
        alpha : float
            Symbol transparency
        kwargs : matplotlib.artist.Artist
            kwargs can be used to set additional plotting properties.

        """
        if isinstance(symb, (list, tuple)) and len(symb) == 3:
            stype = 'ellipse'
            fwhm1, fwhm2, angle = symb
        elif isinstance(symb, six.string_types):
            stype = 'circle'
            fwhm = symb
        elif np.isscalar(symb):
            stype = 'fix'
            size = symb
        else:
            raise IOError('wrong symbol')

        if ra not in self.colnames:
            raise IOError('column %s not found in catalog' % ra)
        if dec not in self.colnames:
            raise IOError('column %s not found in catalog' % dec)

        step = wcs.get_step(unit=u.arcsec)
        for src in self:
            cen = wcs.sky2pix([src[dec], src[ra]], unit=u.deg)[0]
            if stype == 'ellipse':
                f1 = src[fwhm1] / step[0]  # /cos(dec) ?
                f2 = src[fwhm2] / step[1]
                pa = src[angle] * 180 / np.pi
            elif stype == 'circle':
                f1 = src[fwhm] / step[0]
                f2 = f1
                pa = 0
            elif stype == 'fix':
                f1 = size / step[0]
                f2 = f1
                pa = 0
            ell = Ellipse((cen[1], cen[0]), 2 * f1, 2 * f2, pa, fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ell.set_edgecolor(col)

    def plot_id(self, ax, wcs, iden='ID', ra='RA', dec='DEC',
                symb=0.2, alpha=0.5, col='k', **kwargs):
        """This function displays the id of the catalog.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        wcs : `mpdaf.obj.WCS`
            Image WCS
        iden : str
            Name of the column that contains ID values
        ra : str
            Name of the column that contains RA values
        dec : str
            Name of the column that contains DEC values
        symb : float
            Size of the circle in arcsec
        col : str
            Symbol color.
        alpha : float
            Symbol transparency
        kwargs : matplotlib.artist.Artist
            kwargs can be used to set additional plotting properties.

        """
        if ra not in self.colnames:
            raise IOError('column %s not found in catalog' % ra)
        if dec not in self.colnames:
            raise IOError('column %s not found in catalog' % dec)
        if iden not in self.colnames:
            raise IOError('column %s not found in catalog' % iden)

        cat = self.select(wcs, ra, dec)
        size = 2 * symb / wcs.get_step(unit=u.arcsec)[0]
        for src in cat:
            cen = wcs.sky2pix([src[dec], src[ra]], unit=u.deg)[0]
            ax.text(cen[1], cen[0] + size, src[iden], ha='center', color=col,
                    **kwargs)
            ell = Ellipse((cen[1], cen[0]), size, size, 0, fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ell.set_edgecolor(col)
