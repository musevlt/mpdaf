"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c) 2015-2018 Roland Bacon <roland.bacon@univ-lyon1.fr>
Copyright (c)      2018 David Carton <cartondj@gmail.com>

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

import glob
import logging
import numpy as np
import os
import sys

from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, MaskedColumn, hstack, vstack, join
from astropy import units as u
from matplotlib.patches import Circle, Rectangle, Ellipse, RegularPolygon


from ..tools import deprecated, LowercaseOrderedDict

INVALID = {
    type(1): -9999, np.int_: -9999, np.int32: -9999,
    type(1.0): np.nan, np.float_: np.nan,
    type('1'): '', np.str_: '',
    type(False): -9999, np.bool_: -9999
}

# List of required keywords and their type
MANDATORY_KEYS = ['ID', 'RA', 'DEC', 'FROM', 'FROM_V', 'CUBE', 'CUBE_V']
MANDATORY_TYPES = [int, np.float64, np.float64, str, str, str, str]
# List of exluded keywords
EXCLUDED_CARDS = {'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'AUTHOR'}


class Catalog(Table):

    """This class inherits from `astropy.table.Table`.
    Its goal is to manage a list of objects.
    """

    # These are default column names to be used if not provided by metadata.
    # They are not stored in the meta directly, and therefore not written when
    # the catalog is writen to disk.
    _idname_default = 'ID'
    _raname_default = 'RA'
    _decname_default = 'DEC'

    def __init__(self, *args, **kwargs):
        """Initialize a Catalog instance

        Parameters
        ----------
        idname : str, optional
            Table column name containing object IDs
        raname : str, optional
            Table column name containing object RA coords
        decname : str, optional
            Table column name containing object DEC coords
        Remaining args and kwargs are passed to `astropy.table.Table.__init__`.

        """
        #pop kwargs for PY2 compatibility
        idname = kwargs.pop('idname', None)
        raname = kwargs.pop('raname', None)
        decname = kwargs.pop('decname', None)

        super(Catalog, self).__init__(*args, **kwargs)
        self._logger = logging.getLogger(__name__)
        if self.masked:
            self.masked_invalid()

        #replace Table.meta OrderedDict with a case insenstive version
        self.meta = LowercaseOrderedDict(self.meta)

        #set column names in metadata
        if idname is not None:
            self.meta['idname'] = idname
        if raname is not None:
            self.meta['raname'] = raname
        if decname is not None:
            self.meta['decname'] = decname

    @staticmethod
    def _merge_meta(catalogs, join_keys=None, suffix=None):
        """Returns a metadata object combined from a set of catalogs.

        Parameters
        ----------
        catalogs : list< `mpdaf.sdetect.Catalog` >
            List of `mpdaf.sdetect.Catalog` objects to be combined
        suffix : list< str >
            Column suffixes that are appended column names when tables are
            joined. Must be the same length as the number of catalogs.
            Defaults to ['_1', '_2', '_3', etc.]
        """
        if join_keys is None:
            join_keys = []

        n_cat = len(catalogs)

        if suffix is None:
            suffix = tuple(['_{}'.format(i+1) for i in range(n_cat)])

        cat0 = catalogs[0]
        # create output with same type as meta of first catalog
        out = type(cat0.meta)()

        n_cat = len(catalogs)
        for i_cat, cat in enumerate(catalogs):
            s = suffix[i_cat]
            for key, value in cat.meta.items():
                out[key+s] = value

        # special treatment for keys that identify columns
        col_keys = ['idname', 'raname', 'decname']
        for i_cat, cat in enumerate(catalogs):
            for key in col_keys:
                # check key exists
                try:
                    col_name = cat.meta[key]
                except KeyError:
                    continue

                # check actually found in catalog
                if col_name not in cat.columns:
                    continue

                # is col_name supposed to be joined
                if col_name not in join_keys:

                    # check if name is dupilcated in other catalogs, and thus a
                    # suffix is needed
                    n = sum([1 for c in catalogs if col_name in c.columns])
                    if n > 1: #not unique
                        key += suffix[i_cat]
                        col_name += suffix[i_cat]
                out[key] = col_name

        #set default column keys to the first table
        #e.g. raname = raname_1
        for key in col_keys:
            try:
                out[key] = out[key+suffix[0]]
            except KeyError:
                pass

        return out

    @classmethod
    def from_sources(cls, sources, fmt='default'):
        """Construct a catalog from a list of source objects.

        The new catalog will contain all data stored in the primary headers
        and in the tables extensions of the sources:

        * a column per header fits
          ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE',
          'AUTHOR', 'COM*' and 'HIST*' are excluded)
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

        ###############################################
        # List the columns (name/type) of the catalog #
        ###############################################

        # union of all headers keywords without mandatory FITS keywords
        d = {}
        for source in sources:
            d.update({k: (type(v), com) for k, v, com in source.header.cards})

        keys = set(d.keys()) - EXCLUDED_CARDS

        if 'CUBE_V' not in keys:
            logger.warning('CUBE_V keyword in missing. It will be soon '
                           'mandatory and its absecne will return an error')
            d['CUBE_V'] = (str, 'datacube version')

        names_hdr = MANDATORY_KEYS + list(keys - set(MANDATORY_KEYS))
        tuple_hdr = [d[k] for k in names_hdr]
        dtype_hdr = (MANDATORY_TYPES +
                     [c[0] for c in tuple_hdr[len(MANDATORY_TYPES):]])

        desc_hdr = [c[:c.find('u.')] if c.find('u.') != -1
                    else c[:c.find('%')] if c.find('%') != -1
                    else c for _, c in tuple_hdr]
        unit_hdr = [c[c.find('u.'):].split()[0][2:]
                    if c.find('u.') != -1 else None for _, c in tuple_hdr]
        format_hdr = [c[c.find('%'):].split()[0]
                      if c.find('%') != -1 else None for _, c in tuple_hdr]

        has_mag = any(source.mag for source in sources)
        has_z = any(source.z for source in sources)

        names_mag = []
        names_z = []
        names_lines = []
        dtype_lines = []
        units_lines = []

        # magnitudes
        if has_mag:
            names_mag = list(set(np.concatenate(
                [s.mag['BAND'].data for s in sources if s.mag is not None])))
            names_mag += ['%s_ERR' % mag for mag in names_mag]
            names_mag.sort()

        # redshifts
        if has_z:
            names_z = list(set(np.concatenate([
                s.z['Z_DESC'].data for s in sources if s.z is not None])))
            names_z = ['Z_%s' % z for z in names_z]
            colnames = list(set(np.concatenate([
                s.z.colnames for s in sources if s.z is not None])))
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

        # lines
        llines = [len(source.lines) for source in sources
                  if source.lines is not None]
        if len(llines) > 0:
            if fmt == 'default':
                d = {}
                unit = {}
                for source in sources:
                    if source.lines is not None and \
                            'LINE' in source.lines.colnames:
                        colnames = source.lines.colnames
                        colnames.remove('LINE')

                        for col in colnames:
                            d[col] = source.lines.dtype[col]
                            unit[col] = source.lines[col].unit

                        for line in source.lines['LINE'].data:
                            if line is not None:
                                try:
                                    float(line)
                                    logger.warning(
                                        'source %d: line labeled \"%s\" not '
                                        'loaded', source.ID, line)
                                except Exception:
                                    names_lines += [
                                        '%s_%s' % (line.replace('_', ''), col)
                                        for col in colnames
                                    ]

                names_lines = list(set(np.concatenate([names_lines])))
                names_lines.sort()
                dtype_lines = [d['_'.join(name.split('_')[1:])]
                               for name in names_lines]
                units_lines = [unit['_'.join(name.split('_')[1:])]
                               for name in names_lines]
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
                        names_lines += [col + '%03d' % i
                                        for col in inames_lines]
                    dtype_lines = [d[key] for key in sorted(d)] * lmax
                    units_lines = [unit[key] for key in sorted(d)] * lmax
            else:
                raise IOError('Catalog creation: invalid format. It must be '
                              'default or working.')

        ###############################################
        # Set the data row by row                     #
        ###############################################

        data_rows = []
        for source in sources:
            # header
            h = source.header
            keys = list(h.keys())
            row = []
            for key, typ in zip(names_hdr, dtype_hdr):
                if typ == type('1'):
                    row += [('%s' % h[key]).replace('\n', ' ')
                            if key in keys else INVALID[typ]]
                else:
                    k = [h[key] if key in keys else INVALID[typ]]
                    if type(k[0]) == type('1'):
                        raise ValueError('column %s: could not convert string to %s' % (key, typ))
                    row += k

            # magnitudes
            if has_mag:
                if source.mag is None:
                    row += [np.nan for key in names_mag]
                else:
                    keys = source.mag['BAND']
                    for key in names_mag:
                        if key in keys:
                            row += [source.mag['MAG'][source.mag['BAND'] == key].data[0]]
                        elif key[-4:] == '_ERR' and key[:-4] in keys:
                            row += [source.mag['MAG_ERR'][source.mag['BAND'] == key[:-4]].data[0]]
                        else:
                            row += [np.nan]

            # redshifts
            if has_z:
                if source.z is None:
                    row += [np.nan for key in names_z]
                else:
                    keys = source.z['Z_DESC']
                    for key in names_z:
                        key = key[2:]
                        if key in keys:
                            row += [source.z['Z'][source.z['Z_DESC'] == key].data[0]]
                        elif key[-4:] == '_MAX' and key[:-4] in keys:
                            row += [source.z['Z_MAX'][source.z['Z_DESC'] == key[:-4]].data[0]]
                        elif key[-4:] == '_MIN' and key[:-4] in keys:
                            row += [source.z['Z_MIN'][source.z['Z_DESC'] == key[:-4]].data[0]]
                        elif key[-4:] == '_ERR' and key[:-4] in keys:
                            row += [source.z['Z_ERR'][source.z['Z_DESC'] == key[:-4]].data[0]]
                        else:
                            row += [np.nan]

            # lines
            if len(llines) != 0:
                if source.lines is None:
                    for typ in dtype_lines:
                        row += [INVALID[typ.type]]
                else:
                    if fmt == 'default':
                        if 'LINE' not in source.lines.colnames:
                            logger.warning(
                                'source %d:LINE column not present in LINE '
                                'table. LINE information will be not loaded '
                                'with the default format.', source.ID)
                            for typ in dtype_lines:
                                row += [INVALID[typ.type]]
                        else:
                            copy = source.lines['LINE'].data.copy()
                            for i in range(len(source.lines)):
                                source.lines['LINE'][i] = source.lines['LINE'][i].replace('_', '')
                            for name, typ in zip(names_lines, dtype_lines):
                                colname = '_'.join(name.split('_')[1:])
                                line = name.split('_')[0]
                                if 'LINE' in source.lines.colnames and \
                                   colname in source.lines.colnames and \
                                   line in source.lines['LINE'].data:
                                    row += [source.lines[colname][source.lines['LINE'] == line].data[0]]
                                else:
                                    row += [INVALID[typ.type]]
                            source.lines['LINE'] = copy
                    elif fmt == 'working':
                        keys = source.lines.colnames
                        if lmax == 1:
                            row += [source.lines[key][0] if key in keys
                                    else INVALID[typ.type]
                                    for key, typ in zip(names_lines, dtype_lines)]
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
        if has_mag:
            dtype += ['f8' for i in range(len(names_mag))]
        # redshifts
        if has_z:
            dtype += ['f8' for i in range(len(names_z))]
        # lines
        if len(llines) != 0:
            dtype += dtype_lines

        # create Table
        names = names_hdr + names_mag + names_z + names_lines

        # raise a warning if the type is not the same between each source
        for i in range(len(names_hdr)):
            check = set([type(r[i]) for r in data_rows])
            if len(check) > 1:
                logger.warning('column %s is defined with different types(%s) '
                               'that will be converted to %s',
                               names[i], check, dtype[i])

        t = cls(rows=data_rows, names=names, masked=True, dtype=dtype)

        # format
        for name, desc, unit, fmt in zip(names_hdr, desc_hdr, unit_hdr,
                                         format_hdr):
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
        """Construct a catalog from a list of source objects
        which are contains in the directory given as input.

        The new catalog will contain all data stored in the primary headers
        and in the tables extensions of the sources:

        * a column per header fits
          ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE',
          'AUTHOR', 'COM*' and 'HIST*' are excluded)
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
        files.sort()
        n = len(files)
        logger.info('Building catalog from path %s', path)

        for f in files:
            try:
                slist.append(Source.from_file(f))
                filenames.append(os.path.basename(f))
            except KeyboardInterrupt:
                return
            except Exception as inst:
                logger.warning('source %s not loaded (%s)', f, inst)
            sys.stdout.write("\r\x1b[K %i%%" % (100 * len(filenames) / n))
            sys.stdout.flush()

        sys.stdout.write("\r\x1b[K ")
        sys.stdout.flush()

        t = cls.from_sources(slist, fmt)
        t['FILENAME'] = filenames

        return t

    def write(self, *args, **kwargs):
        """Write this Table object out in the specified format.

        This function provides the Table interface to the astropy unified I/O
        layer.  This allows easily writing a file in many supported data
        formats using syntax such as::

          >>> from astropy.table import Table
          >>> dat = Table([[1, 2], [3, 4]], names=('a', 'b'))
          >>> dat.write('table.dat', format='ascii')  # doctest: +SKIP

        The arguments and keywords (other than ``format``) provided to this
        function are passed through to the underlying data reader (e.g.
        `~astropy.io.ascii.write`).

        """
        # Try to detect if the file is saved as FITS. In this case, Astropy
        # 3.0 serialize .format and .description in yaml comments, which causes
        # the table to not be readable by Catalog.read (inside Astropy it is
        # then read as QTable which is not parent class from Catalog).
        # So for now we just remove .format and .description in this case.
        # https://github.com/astropy/astropy/issues/7181
        if (kwargs.get('format') == 'fits' or (
                isinstance(args[0], str) and args[0].endswith('.fits'))):
            t = self.copy()
            for col in t.itercols():
                col.format = None
                col.description = None
            super(Catalog, t).write(*args, **kwargs)
        else:
            super(Catalog, self).write(*args, **kwargs)

    def masked_invalid(self):
        """Mask where invalid values occur (NaNs or infs or -9999 or '')."""
        for col in self.colnames:
            try:
                self[col][:] = np.ma.masked_invalid(self[col])
                self[col][:] = np.ma.masked_equal(self[col], -9999)
            except Exception:
                pass

    def hstack(self, cat2, **kwargs):
        """Peforms an `astropy.table.hstack` with another catalog, but also
        handles the metadata correctly.

        Parameters
        ----------
        cat2 : `astropy.table.Table`
            Catalog to stack with.
        Remaining args and kwargs are passed to `astropy.table.hstack`, excecpt
        metadata_conflicts.

        Returns
        -------
        stacked : `Catalog` object
            New catalog containing the stacked data

        """
        #convert cat2 to Catalog object
        if not isinstance(cat2, Catalog):
            cat2 = Catalog(cat2, copy=False)

        #suppress metadata conflict warnings
        kwargs['metadata_conflicts'] = 'silent'
        stacked = hstack([self, cat2], **kwargs)
        stacked.meta = self._merge_meta([self, cat2])

        return stacked

    def join(self, cat2, **kwargs):
        """Peforms an `astropy.table.join` with another catalog, but also
        handles the metadata correctly.

        Parameters
        ----------
        cat2 : `astropy.table.Table`
            Right catalog to join with.
        Remaining args and kwargs are passed to `astropy.table.join`, excecpt
        metadata_conflicts.

        Returns
        -------
        joined : `~astropy.table.Table` object
            New table containing the result of the join operation.

        """
        #convert cat2 to Catalog object
        if not isinstance(cat2, Catalog):
            cat2 = Catalog(cat2, copy=False)

        #suppress metadata conflict warnings
        keys = kwargs.get('keys', None)
        kwargs['metadata_conflicts'] = 'silent'
        joined = join(self, cat2, **kwargs)

        joined.meta = self._merge_meta([self, cat2], join_keys=keys)

        return joined

    def match(self, cat2, radius=1, colc1=(None, None), colc2=(None, None),
              full_output=True, **kwargs):
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
        full_output: bool
            output flag
        **kwargs
            Other arguments are passed to
            `astropy.coordinates.match_coordinates_sky`.

        Returns
        -------
        out : astropy.Table, astropy.Table, astropy.Table

            If ``full_output`` is True, return a tuple ``(match, nomatch1,
            nomatch2)`` where:

            - match: table of matched elements in RA,DEC.
            - nomatch1: sub-table of non matched elements of the current
              catalog.
            - nomatch2: sub-table of non matched elements of the catalog cat2.

            If ``full_output`` is False, only ``match`` is returned.

        """

        #convert cat2 to Catalog object
        if not isinstance(cat2, Catalog):
            cat2_class = cat2.__class__
            cat2 = Catalog(cat2, copy=False)
        else:
            cat2_class = None

        col1_ra = colc1[0] or self.meta.get('raname', self._raname_default)
        col1_dec = colc1[1] or self.meta.get('decname', self._decname_default)

        col2_ra = colc2[0] or cat2.meta.get('raname', cat2._raname_default)
        col2_dec = colc2[1] or cat2.meta.get('decname', cat2._decname_default)


        coord1 = self.to_skycoord(ra=col1_ra, dec=col1_dec)
        coord2 = cat2.to_skycoord(ra=col2_ra, dec=col2_dec)
        id2, d2d, d3d = coord1.match_to_catalog_sky(coord2, **kwargs)
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
                to_remove += idlist[d2match[mask].argsort()[1:]].tolist()
            id2match = np.delete(id2match, to_remove)
            id1match = np.delete(id1match, to_remove)
            d2match = np.delete(d2match, to_remove)
        match1 = self[id1match]
        match2 = cat2[id2match]
        match = match1.hstack(match2, join_type='exact')
        match.add_column(Column(data=d2match.to(u.arcsec), name='Distance',
                                dtype=float))

        if full_output:
            id1notmatch = np.in1d(range(len(self)), id1match,
                                  assume_unique=True, invert=True)
            id2notmatch = np.in1d(range(len(cat2)), id2match,
                                  assume_unique=True, invert=True)
            nomatch2 = cat2[id2notmatch]
            nomatch1 = self[id1notmatch]
            self._logger.debug('Cat1 Nelt %d Matched %d Not Matched %d',
                               len(self), len(match1), len(nomatch1))
            self._logger.debug('Cat2 Nelt %d Matched %d Not Matched %d',
                               len(cat2), len(match2), len(nomatch2))

            #convert nomatch2 back to original cat2 type
            if cat2_class:
                nomatch2 = cat2_class(nomatch2, copy=False)

            return match, nomatch1, nomatch2
        else:
            self._logger.debug('Cat1 Nelt %d Cat2 Nelt %d Matched %d',
                               len(self), len(cat2), len(match1))
            return match

    def nearest(self, coord, colcoord=(None, None), ksel=1, maxdist=None,
                **kwargs):
        """Return the nearest sources with respect to the given coordinate.

        Parameters
        ----------
        coord: tuple
           ra,dec in decimal degree, or HH:MM:SS,DD:MM:SS
        colcoord: tuple of str
           column names of coordinate: default ('RA','DEC')
        ksel: int
           Number of sources to return, default 1 (if None return all sources
           sorted by distance)
        maxdist: float
           Maximum distance to source in arcsec, default None
        **kwargs
            Other arguments are passed to
            `astropy.coordinates.match_coordinates_sky`.

        Returns
        -------
        `astropy.table.Table`
            The corresponding catalog of matched sources with the additional
            Distance column (arcsec).

        """
        if not isinstance(coord, SkyCoord):
            ra, dec = coord
            if isinstance(ra, str) and ':' in ra:
                unit = (u.hourangle, u.deg)
            else:
                unit = (u.deg, u.deg)
            coord = SkyCoord(ra, dec, unit=unit, frame='fk5')

        if coord.shape == ():
            coord = coord.reshape(1)

        col_ra = colcoord[0] or self.meta.get('raname', self._raname_default)
        col_dec = colcoord[1] or self.meta.get('decname', self._decname_default)
        src_coords = self.to_skycoord(ra=col_ra, dec=col_dec)
        idx, d2d, d3d = src_coords.match_to_catalog_sky(coord, **kwargs)
        dist = d2d.arcsec
        ksort = dist.argsort()

        cat = self[ksort]
        dist = dist[ksort]
        if maxdist is not None:
            kmax = dist <= maxdist
            cat = cat[kmax]
            dist = dist[kmax]

        cat['Distance'] = dist
        cat['Distance'].format = '.2f'

        if ksel is not None:
            cat = cat[:ksel]

        return cat

    def match3Dline(self, cat2, linecolc1, linecolc2, spatial_radius=1,
                    spectral_window=5, suffix=('_1', '_2'), full_output=True,
                    colc1=(None, None), colc2=(None, None), **kwargs):
        """3D Match elements of the current catalog with an other using
        spatial (RA, DEC) and list of spectral lines location.

        Parameters
        ----------
        cat2 : `astropy.table.Table`
            Catalog to match.
        linecolc1: list of float
            List of column names containing the wavelengths of the input catalog
        linecolc2: list of float
            List of column names containing the wavelengths of the cat2
        spatial_radius : float
            Matching radius size in arcsec (default 1).
        spectral_window : float (default 5)
            Matching wavelength window in spectral unit (default 5).
        colc1: tuple
            ('RA','DEC') name of ra,dec columns of input catalog
        colc2: tuple
            ('RA','DEC') name of ra,dec columns of cat2
        full_output: bool
            output flag
        other arguments are passed to astropy match_to_catalog_sky


        Returns
        -------
        out : astropy.Table, astropy.Table, astropy.Table

            If ``full_output`` is True, return a tuple ``(match3d, match2d,
            nomatch1, nomatch2)`` where:

            - match3d, match2d: table of matched elements in RA,DEC.
            - nomatch1: sub-table of non matched elements of the current
              catalog.
            - nomatch2: sub-table of non matched elements of the catalog cat2.

            If ``full_output`` is False, only ``match`` is returned.

        """

        #convert cat2 to Catalog object
        if not isinstance(cat2, Catalog):
            cat2_class = cat2.__class__
            cat2 = Catalog(cat2, copy=False)
        else:
            cat2_class = None

        col1_ra = colc1[0] or self.meta.get('raname', self._raname_default)
        col1_dec = colc1[1] or self.meta.get('decname', self._decname_default)

        col2_ra = colc2[0] or cat2.meta.get('raname', cat2._raname_default)
        col2_dec = colc2[1] or cat2.meta.get('decname', cat2._decname_default)

        # rename all catalogs columns with _1 or _2
        self._logger.debug('Rename Catalog columns with %s or %s suffix',
                           suffix[0], suffix[1])
        tcat1 = self.copy()
        tcat2 = cat2.copy()
        for name in tcat1.colnames:
            tcat1.rename_column(name, name + suffix[0])
        for name in tcat2.colnames:
            tcat2.rename_column(name, name + suffix[1])
        colc1 = (col1_ra + suffix[0], col1_dec + suffix[0])
        linecolc1 = [col + suffix[0] for col in linecolc1]
        colc2 = (col2_ra + suffix[1], col2_dec + suffix[1])
        linecolc2 = [col + suffix[1] for col in linecolc2]

        self._logger.debug('Performing spatial match')
        if full_output:
            match, unmatch1, unmatch2 = tcat1.match(
                tcat2, radius=spatial_radius, colc1=colc1, colc2=colc2,
                full_output=full_output, **kwargs)
        else:
            match = tcat1.match(
                tcat2, radius=spatial_radius, colc1=colc1, colc2=colc2,
                full_output=full_output, **kwargs)            
        match.meta = self._merge_meta([tcat1, tcat2], suffix=suffix)

        tcat1._logger.debug('Performing line match')
        # create matched line colonnes
        match.add_column(MaskedColumn(length=len(match), name='NLMATCH',
                                      dtype='int'), index=1)
        # reorder columns
        match.add_column(match['Distance'], index=2, name='DIST')
        match['DIST'].format = '.2f'
        match.remove_column('Distance')
        for col in linecolc1:
            l = tcat1.colnames.index(col)
            match.add_columns([MaskedColumn(length=len(match), dtype='bool'),
                               MaskedColumn(length=len(match), dtype='S30'),
                               MaskedColumn(length=len(match), dtype='float')],
                              names=['M_' + col, 'L_' + col, 'E_' + col],
                              indexes=[l, l, l])
            match['E_' + col].format = '.2f'
            match['M_' + col] = False
        # perform match for lines
        for r in match:
            # Match lines
            nmatch = 0
            for c1 in linecolc1:
                l1 = r[c1]
                if np.ma.is_masked(l1):
                    continue
                for c2 in linecolc2:
                    l2 = r[c2]
                    if np.ma.is_masked(l2):
                        continue
                    err = abs(l2 - l1)
                    if err < spectral_window:
                        nmatch += 1
                        r['M_' + c1] = True
                        r['L_' + c1] = c2
                        r['E_' + c1] = err
            r['NLMATCH'] = nmatch

        if full_output:
            match3d = match[match['NLMATCH'] > 0]
            match2d = match[match['NLMATCH'] == 0]
            self._logger.info('Matched 3D: %d Matched 2D: %d Cat1 unmatched: '
                              '%d Cat2 unmatched: %d', len(match3d),
                              len(match2d), len(unmatch1), len(unmatch2))

            #convert unmatch2 back to original cat2 type
            if cat2_class:
                unmatch2 = cat2_class(unmatch2, copy=False)

            return (match3d, match2d, unmatch1, unmatch2)
        else:
            self._logger.info('Matched 3D: %d',
                              len(match[match['NLMATCH'] > 0]))
            return match

    def select(self, wcs, ra=None, dec=None, margin=0, mask=None):
        """Select all sources from catalog which are inside the given WCS
        and return a new catalog.

        Parameters
        ----------
        wcs : `~mpdaf.obj.WCS`
            Image WCS
        ra : str
            Name of the column that contains RA values in degrees.
        dec : str
            Name of the column that contains DEC values in degrees.
        margin : int
            Margin from the edges (pixels).
        mask : array-like
            Mask used to filter sources (1 to mask).

        Returns
        -------
        `mpdaf.sdetect.Catalog`
            The catalog with selected rows.

        """

        ra = ra or self.meta.get('raname', self._raname_default)
        dec = dec or self.meta.get('decname', self._decname_default)

        arr = np.vstack([self[dec].data, self[ra].data]).T
        cen = wcs.sky2pix(arr, unit=u.deg).T
        sel = ((cen[0] > margin) & (cen[0] < wcs.naxis2 - margin) &
               (cen[1] > margin) & (cen[1] < wcs.naxis1 - margin))
        if mask is not None:
            # select sources that are not masked
            sel[sel] = ~mask[(cen[0, sel] + 0.5).astype(int),
                             (cen[1, sel] + 0.5).astype(int)]
        return self[sel]

    def edgedist(self, wcs, ra=None, dec=None):
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
        `numpy.ndarray`
            The distance in arcsec units.

        """
        ra = ra or self.meta.get('raname', self._raname_default)
        dec = dec or self.meta.get('decname', self._decname_default)

        dim = np.array([wcs.naxis2, wcs.naxis1])
        pix = wcs.sky2pix(np.array([self[dec], self[ra]]).T, unit=u.deg)
        dist = np.hstack([pix, dim - pix]).min(axis=1)
        return dist * wcs.get_step(unit=u.arcsec)[0]

    def to_skycoord(self, ra=None, dec=None, frame='fk5', unit='deg'):
        """Return an `astropy.coordinates.SkyCoord` object."""
        ra = ra or self.meta.get('raname', self._raname_default)
        dec = dec or self.meta.get('decname', self._decname_default)

        from astropy.coordinates import SkyCoord
        return SkyCoord(ra=self[ra], dec=self[dec],
                        unit=(unit, unit), frame=frame)

    def to_ds9_regions(self, outfile, ra=None, dec=None, radius=1,
                       frame='fk5', unit_pos='deg', unit_radius='arcsec'):
        """Return an `astropy.coordinates.SkyCoord` object."""
        try:
            from regions import CircleSkyRegion, write_ds9
        except ImportError:
            self._logger.error("the 'regions' package is needed for this")
            raise
        ra = ra or self.meta.get('raname', self._raname_default)
        dec = dec or self.meta.get('decname', self._decname_default)
        center = self.to_skycoord(ra=ra, dec=dec, frame=frame, unit=unit_pos)
        radius = radius * u.Unit(unit_radius)
        regions = [CircleSkyRegion(center=c, radius=radius) for c in center]
        write_ds9(regions, filename=outfile, coordsys=frame)

    def plot_symb(self, ax, wcs, label=False, esize=0.8, lsize=None, etype='o',
                  ltype=None, ra=None, dec=None, id=None, ecol='k', lcol=None,
                  alpha=1.0, fill=False, fontsize=8, expand=1.7, ledgecol=None,
                  lfacecol=None, npolygon=3, **kwargs):
        """This function plots the sources location from the catalog.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        wcs : `mpdaf.obj.WCS`
            Image WCS.
        label: bool
            If True catalog ID are displayed.
        esize : float
            symbol size in arcsec (used only if lsize is not set).
        lsize : str
            Column name containing the size in arcsec.
        etype : str
            Type of symbol: o (circle, size=diameter), s (square, size=length),
            p (polygon, size=diameter) used only if ltype is not set.
        ltype : str
            Name of column that contain the symbol to use.
        ra : str
            Name of the column that contains RA values (in degrees).
        dec : str
            Name of the column that contains DEC values (in degrees).
        id : str
            Name of the column that contains ID.
        lcol: str
            Name of the column that contains Color.
        ecol : str
            Symbol color (only used if lcol is not set).
        alpha : float
            Symbol transparency.
        fill: bool
            If True filled symbol are used.
        expand: float
            Expand factor to write label.
        ledgecol: str
            Name of the column that contains the edge color.
        lfacecol: str
            Name of the column that contains the fqce color.
        **kwargs
            kwargs can be used to set additional plotting properties.

        """
        ra = ra or self.meta.get('raname', self._raname_default)
        dec = dec or self.meta.get('decname', self._decname_default)
        id = id or self.meta.get('idname', self._idname_default)

        if (ltype is None) and (etype not in ['o', 's', 'p']):
            raise IOError('Unknown symbol %s' % etype)
        if (ltype is not None) and (ltype not in self.colnames):
            raise IOError('column %s not found in catalog' % ltype)
        if (lsize is not None) and (lsize not in self.colnames):
            raise IOError('column %s not found in catalog' % lsize)
        if (lcol is not None) and (lcol not in self.colnames):
            raise IOError('column %s not found in catalog' % lcol)
        if (ledgecol is not None) and (ledgecol not in self.colnames):
            raise IOError('column %s not found in catalog' % ledgecol)
        if (lfacecol is not None) and (lfacecol not in self.colnames):
            raise IOError('column %s not found in catalog' % lfacecol)
        if ra not in self.colnames:
            raise IOError('column %s not found in catalog' % ra)
        if dec not in self.colnames:
            raise IOError('column %s not found in catalog' % dec)
        if label and (id not in self.colnames):
            raise IOError('column %s not found in catalog' % id)

        texts = []

        step = wcs.get_step(unit=u.arcsec)

        arr = np.vstack([self[dec].data, self[ra].data]).T
        arr = wcs.sky2pix(arr, unit=u.deg)

        for src, cen in zip(self, arr):
            yy, xx = cen
            if (xx < 0) or (yy < 0) or (xx > wcs.naxis1) or (yy > wcs.naxis2):
                continue
            vsize = esize if lsize is None else src[lsize]
            pixsize = vsize / step[0]
            vtype = etype if ltype is None else src[ltype]
            vcol = None
            vedgecol = 'none'
            vfacecol = 'none'
            vfill = True
            if (lcol is None) and (ledgecol is None) and (lfacecol is None):
                vcol = ecol
                vfill = fill
            if lcol is not None:
                vcol = src[lcol]
            if ledgecol is not None:
                vcol = None
                vfill = False
                vedgecol = src[ledgecol]
            if lfacecol is not None:
                vfill = True
                vcol = None
                vfacecol = src[lfacecol]
            if vtype == 'o':
                if vcol is not None:
                    s = Circle((xx, yy), 0.5 * pixsize, fill=fill,
                               ec=vcol.rstrip(), alpha=alpha, **kwargs)
                else:
                    s = Circle((xx, yy), 0.5 * pixsize, fill=vfill,
                               edgecolor=vedgecol.rstrip(),
                               facecolor=vfacecol.rstrip(),
                               alpha=alpha, **kwargs)
            elif vtype == 's':
                if vcol is not None:
                    s = Rectangle((xx - pixsize / 2, yy - pixsize / 2),
                                  pixsize, pixsize, fill=fill,
                                  ec=vcol.rstrip(), alpha=alpha, **kwargs)
                else:
                    s = Rectangle((xx - pixsize / 2, yy - pixsize / 2),
                                  pixsize, pixsize, fill=vfill,
                                  edgecolor=vedgecol.rstrip(),
                                  facecolor=vfacecol.rstrip(),
                                  alpha=alpha, **kwargs)
            elif vtype == 'p':
                if vcol is not None:
                    s = RegularPolygon((xx, yy), npolygon, 0.5 * pixsize,
                                       fill=fill, ec=vcol.rstrip(),
                                       alpha=alpha, **kwargs)
                else:
                    s = RegularPolygon((xx, yy), npolygon, 0.5 * pixsize,
                                       fill=vfill, edgecolor=vedgecol.rstrip(),
                                       facecolor=vfacecol.rstrip(),
                                       alpha=alpha, **kwargs)
            ax.add_artist(s)
            if label and (not np.ma.is_masked(src[id])):
                texts.append((ax.text(xx, yy, src[id], ha='center',
                                      fontsize=fontsize), cen[1], cen[0]))
            s.set_clip_box(ax.bbox)

        if label and len(texts) > 0:
            text, x, y = zip(*texts)
            try:
                from adjustText import adjust_text
            except ImportError:
                self._logger.error("the 'adjustText' package is needed to "
                                   "avoid labels overlap")
            else:
                adjust_text(text, x=x, y=y, ax=ax, only_move={text: 'xy'},
                            expand_points=(expand, expand))

    @deprecated('plot_id is deprecated, use plot_symb with label=True instead')
    def plot_id(self, ax, wcs, iden=None, ra=None, dec=None, symb=0.2,
                alpha=0.5, col='k', ellipse_kwargs=None, **kwargs):
        """This function displays the id of the catalog.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
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
        ellipse_kwargs : dict
            Additional properties for `matplotlib.patches.Ellipse`.
        **kwargs
            Additional properties for ``ax.text``.

        """
        iden = iden or self.meta.get('idname', self._idname_default)
        ra = ra or self.meta.get('raname', self._raname_default)
        dec = dec or self.meta.get('decname', self._decname_default)

        if ra not in self.colnames:
            raise IOError('column %s not found in catalog' % ra)
        if dec not in self.colnames:
            raise IOError('column %s not found in catalog' % dec)
        if iden not in self.colnames:
            raise IOError('column %s not found in catalog' % iden)

        ellipse_kwargs = ellipse_kwargs or {}
        cat = self.select(wcs, ra, dec)
        size = 2 * symb / wcs.get_step(unit=u.arcsec)[0]
        for src in cat:
            cen = wcs.sky2pix([src[dec], src[ra]], unit=u.deg)[0]
            ax.text(cen[1], cen[0] + size, src[iden], ha='center', color=col,
                    **kwargs)
            ell = Ellipse((cen[1], cen[0]), size, size, 0, fill=False,
                          alpha=alpha, edgecolor=col, clip_box=ax.bbox,
                          **ellipse_kwargs)
            ax.add_artist(ell)
