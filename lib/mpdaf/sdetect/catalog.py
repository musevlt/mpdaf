"""
Copyright (c) 2010-2017 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2017 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c) 2015-2017 Roland Bacon <roland.bacon@univ-lyon1.fr>

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
from astropy.table import Table, Column, MaskedColumn, hstack, vstack
from astropy import units as u
from matplotlib.patches import Ellipse
from six.moves import range, zip

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

    def __init__(self, *args, **kwargs):
        super(Catalog, self).__init__(*args, **kwargs)
        self._logger = logging.getLogger(__name__)
        if self.masked:
            self.masked_invalid()

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

        # magnitudes
        if has_mag:
            names_mag = list(set(np.concatenate(
                [s.mag['BAND'].data for s in sources if s.mag is not None])))
            names_mag += ['%s_ERR' % mag for mag in names_mag]
            names_mag.sort()
        else:
            names_mag = []

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
        else:
            names_z = []

        # lines
        llines = [len(source.lines) for source in sources
                  if source.lines is not None]
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
                                except:
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

    def masked_invalid(self):
        """Mask where invalid values occur (NaNs or infs or -9999 or '')."""
        for col in self.colnames:
            try:
                self[col][:] = np.ma.masked_invalid(self[col])
                self[col][:] = np.ma.masked_equal(self[col], -9999)
            except:
                pass

    def match(self, cat2, radius=1, colc1=('RA', 'DEC'), colc2=('RA', 'DEC'),
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
        other arguments are passed to astropy match_to_catalog_sky

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
        #for name in match1.colnames:
        #    match1.remove_indices(name)
        match2 = cat2[id2match]
        #for name in match2.colnames:
        #    match2.remove_indices(name)
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
        
    def nearest(self, coord, colcoord=('RA', 'DEC'), ksel=1, maxdist=None, **kwargs):
        """ return the nearest sources with respect to the given coordinate

        Parameters
        ----------
        coord: tuple
           ra,dec in decimal degree, or HH:MM:SS,DD:MM:SS
        colcoord: tuple of str
           column names of coordinate: default ('RA','DEC')
        ksel: int
           Number of sources to return, default 1 (if None return all sources sorted by distance)
        maxdist: float
           Maximum distance to source in arcsec, default None

        Returns
        -------
        cat: `astropy.table.Table`
          the corresponding catalog of matched sources with the additional Distance column (arcsec)
        """
        colra,coldec = colcoord
        cra, cdec = _get_coord(coord[0], coord[1])
        xcoord = SkyCoord([cra], [cdec], unit=(u.deg, u.deg))
        src_coords = SkyCoord(self[colra], self[coldec], unit=(u.deg, u.deg))
        idx, d2d, d3d = src_coords.match_to_catalog_sky(xcoord, **kwargs)
        dist = d2d.arcsec
        ksort = dist.argsort()
        if ksel is not None:
            ksort = ksort[:ksel]
        cat = self[ksort]
        dist = dist[ksort]
        if maxdist is not None:
            kmax = dist <= maxdist
            cat = cat[kmax]
            dist = dist[kmax]
        cat['Distance'] = dist
        cat['Distance'].format = '.2f'
        return cat    

    def match3Dline(self, cat2, linecolc1, linecolc2, spatial_radius=1, spectral_window=5, 
                    colc1=('RA', 'DEC'), colc2=('RA', 'DEC'), 
                    suffix=('_1','_2'),
                    full_output=True, **kwargs):
        """3D Match elements of the current catalog with an other using spatial (RA, DEC) and list of spectral lines location.

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

        if full_output is True

        out : astropy.Table, astropy.Table, astropy.Table
             match3d, match2d, nomatch1, nomatch2
        else

        out : astropy.Table
              match

        match: table of matched elements in RA,DEC
        nomatch1: sub-table of non matched elements of the current catalog
        nomatch2: sub-table of non matched elements of the catalog cat2

        """
        # rename all catalogs columns with _1 or _2 
        self._logger.debug('Rename Catalog columns with %s or %s suffix',suffix[0],suffix[1])
        for name in self.colnames:
            self.rename_column(name, name+suffix[0])
        for name in cat2.colnames:
            cat2.rename_column(name, name+suffix[1])
        colc1 = (colc1[0]+suffix[0], colc1[1]+suffix[0])
        linecolc1 = [col+suffix[0] for col in linecolc1]
        colc2 = (colc2[0]+suffix[1], colc2[1]+suffix[1])
        linecolc2 = [col+suffix[1] for col in linecolc2]
        
        self._logger.debug('Performing spatial match')
        match,unmatch1,unmatch2 = self.match(cat2, radius=spatial_radius, colc1=colc1, 
                                             colc2=colc2, full_output=full_output, **kwargs)
        self._logger.debug('Performing line match')
        # create matched line colonnes
        match.add_column(MaskedColumn(length=len(match), name='NLMATCH', dtype='int'), index=1)
        # reorder columns
        match.add_column(match['Distance'], index=2, name='DIST')
        match['DIST'].format = '.2f'
        match.remove_column('Distance')
        for col in linecolc1:
            l = self.colnames.index(col)
            match.add_columns([MaskedColumn(length=len(match),dtype='bool'),
                               MaskedColumn(length=len(match), dtype='S30'),
                               MaskedColumn(length=len(match), dtype='float')],
                              names=['M_'+col,'L_'+col,'E_'+col], indexes=[l,l,l])       
            match['E_'+col].format = '.2f'
            match['M_'+col] = False
        # perform match for lines
        for r in match:
            # Match lines
            nmatch = 0
            for c1 in linecolc1:
                l1 = r[c1]
                if np.ma.is_masked(l1): continue
                for c2 in linecolc2:
                    l2 = r[c2]
                    if np.ma.is_masked(l2): continue
                    err = abs(l2-l1)
                    if err < spectral_window:
                        nmatch += 1
                        r['M_'+c1] = True
                        r['L_'+c1] = c2
                        r['E_'+c1] = err
            r['NLMATCH'] = nmatch 
            
        if full_output:
            match3d = match[match['NLMATCH']>0]
            match2d = match[match['NLMATCH']==0]
            self._logger.info('Matched 3D: %d Matched 2D: %d Cat1 unmatched: %d Cat2 unmatched: %d',
                            len(match3d),len(match2d),len(unmatch1),len(unmatch2))              
            return (match3d,match2d,unmatch1,unmatch2) 
        else:
            self._logger.info('Matched 3D: %d', len(match[match['NLMATCH']>0]))
            return match
        
    def select(self, wcs, ra='RA', dec='DEC', margin=0):
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

        Returns
        -------
        out : `mpdaf.sdetect.Catalog`

        """
        arr = np.vstack([self[dec].data, self[ra].data]).T
        cen = wcs.sky2pix(arr, unit=u.deg).T
        sel = ((cen[0] > margin) & (cen[0] < wcs.naxis2 - margin) &
               (cen[1] > margin) & (cen[1] < wcs.naxis1 - margin))
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
        dim = np.array([wcs.naxis2, wcs.naxis1])
        pix = wcs.sky2pix(np.array([self[dec], self[ra]]).T, unit=u.deg)
        dist = np.hstack([pix, dim - pix]).min(axis=1)
        return dist * wcs.get_step(unit=u.arcsec)[0]

    def to_skycoord(self, ra='RA', dec='DEC', frame='fk5', unit='deg'):
        """Return an `astropy.coordinates.SkyCoord` object."""
        from astropy.coordinates import SkyCoord
        return SkyCoord(ra=self[ra], dec=self[dec],
                        unit=(unit, unit), frame=frame)

    def to_ds9_regions(self, outfile, ra='RA', dec='DEC', radius=1,
                       frame='fk5', unit_pos='deg', unit_radius='arcsec'):
        """Return an `astropy.coordinates.SkyCoord` object."""
        try:
            from regions import CircleSkyRegion, write_ds9
        except ImportError:
            self._logger.error("the 'regions' package is needed for this")
            raise
        center = self.to_skycoord(ra=ra, dec=dec, frame=frame, unit=unit_pos)
        radius = radius * u.Unit(unit_radius)
        regions = [CircleSkyRegion(center=c, radius=radius) for c in center]
        write_ds9(regions, filename=outfile, coordsys=frame)

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
        kwargs : dict
            Additional properties for `matplotlib.patches.Ellipse`.

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
            ell = Ellipse((cen[1], cen[0]), 2 * f1, 2 * f2, pa, fill=False,
                          alpha=alpha, edgecolor=col, clip_box=ax.bbox,
                          **kwargs)
            ax.add_artist(ell)

    def plot_id(self, ax, wcs, iden='ID', ra='RA', dec='DEC', symb=0.2,
                alpha=0.5, col='k', ellipse_kwargs=None, **kwargs):
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
        ellipse_kwargs : dict
            Additional properties for `matplotlib.patches.Ellipse`.
        kwargs : dict
            Additional properties for ``ax.text``.

        """
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

def _get_coord(ra, dec):
    """ translate coordinate from HH:MM:SS to decimal deg"""
    if isinstance(ra, str) and ':' in ra:
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        logger.debug('Translating RA,DEC in decimal degre: {}, {}'.format(c.ra.value, c.dec.value))
        return (c.ra.value, c.dec.value)
    else:
        return (ra, dec)