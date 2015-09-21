
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table, hstack, vstack
from astropy import units as u

from matplotlib.patches import Ellipse

import glob
import logging
import numpy as np
import os.path
import sys


class Catalog(Table):

    """This class contains a catalog of objects.

    Inherits from :class:`astropy.table.Table`.

    """

    def __init__(self, data=None, masked=None, names=None,
                 dtype=None, meta=None, copy=True, rows=None):
        Table.__init__(self, data, masked, names, dtype, meta, copy, rows)
        self.logger = logging.getLogger('mpdaf corelib')
        if self.colnames.count('ra') != 0:
            self.rename_column('ra', 'RA')
        if self.colnames.count('dec') != 0:
            self.rename_column('dec', 'DEC')
        if self.colnames.count('z') != 0:
            self.rename_column('z', 'Z')

    @classmethod
    def from_sources(cls, sources, fmt='default'):
        """Construct a catalog from a list of source objects.

        Parameters
        ----------
        sources : list< :class:`mpdaf.sdetect.Source` >
        """
        invalid = {type(1): -9999, np.int_: -9999, 
                   type(1.0): np.nan, np.float_: np.nan,
                   type('1'): '', np.str_: '',
                   type(False): -9999, np.bool_: -9999}
        #invalid = {type(1): np.ma.masked_array([-9999], mask=[1], fill_value=-9999), type(1.0): np.ma.masked_array([np.nan], mask=[1], fill_value=np.nan), type('1'): '', type(False): -1}
        # union of all headers keywords without mandatory FITS keywords

        h = sources[0].header
        d = dict(zip(h.keys(), [type(v) for v in h.values()]))
        for source in sources[1:]:
            h = source.header
            d.update(dict(zip(h.keys(), [type(v) for v in h.values()])))
        d = {key: value for key, value in d.items() if key not in [
            'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'AUTHOR']}
        names_hdr = d.keys()
        dtype_hdr = d.values()
        # sort mandatory keywords
        index = names_hdr.index('ID')
        names_hdr.insert(0, names_hdr.pop(index))
        dtype_hdr.insert(0, dtype_hdr.pop(index))
        index = names_hdr.index('RA')
        names_hdr.insert(1, names_hdr.pop(index))
        dtype_hdr.insert(1, dtype_hdr.pop(index))
        index = names_hdr.index('DEC')
        names_hdr.insert(2, names_hdr.pop(index))
        dtype_hdr.insert(2, dtype_hdr.pop(index))
        index = names_hdr.index('ORIGIN')
        names_hdr.insert(3, names_hdr.pop(index))
        dtype_hdr.insert(3, dtype_hdr.pop(index))
        index = names_hdr.index('ORIGIN_V')
        names_hdr.insert(4, names_hdr.pop(index))
        dtype_hdr.insert(4, dtype_hdr.pop(index))
        index = names_hdr.index('CUBE')
        names_hdr.insert(5, names_hdr.pop(index))
        dtype_hdr.insert(5, dtype_hdr.pop(index))
        
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
            if 'Z_ERR' in source.z.colnames:
                names_err = ['%s_ERR' % z for z in names_z]
            else:
                names_err = []
            if 'Z_MIN' in source.z.colnames:
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
        else:
            if fmt == 'default':
                names_lines = []
                d = {}
                for source in sources:
                    if source.lines is not None and 'LINE' in source.lines.colnames:
                        colnames = source.lines.colnames
                        colnames.remove('LINE')
                        
                        for col in colnames:
                            d[col] = source.lines.dtype[col]
                        
                        for line, mask in zip(source.lines['LINE'].data.data, source.lines['LINE'].data.mask):
                            if not mask:
                                names_lines += ['%s_%s'%(line,col) for col in colnames]
                                
                names_lines = list(set(np.concatenate([names_lines])))
                names_lines.sort()
                dtype_lines = [d['_'.join(name.split('_')[1:])] for name in names_lines]
            elif fmt == 'working':
                lmax = max(llines)
                d = {}
                for source in sources:
                    if source.lines is not None:
                        for col in source.lines.colnames:
                            d[col] = source.lines.dtype[col]
                if lmax == 1:
                    names_lines = sorted(d)
                    dtype_lines = [d[key] for key in sorted(d)]
                else:
                    names_lines = []
                    inames_lines = sorted(d)
                    for i in range(1, lmax + 1):
                        names_lines += [col + '%03d' % i for col in inames_lines]
                    dtype_lines = [d[key] for key in sorted(d)] * lmax
            else:
                raise IOError('Catalog creation: invalid format. It must be dafault or working.')


        data_rows = []
        for source in sources:
            # header
            h = source.header
            keys = h.keys()
            row = []
            for key,typ in zip(names_hdr, dtype_hdr):
                if typ==type('1'):
                    row += ['%s'%h[key] if key in keys else invalid[typ]]
                else:
                    row += [h[key] if key in keys else invalid[typ]]

            # magnitudes
            if len(lmag) != 0:
                if source.mag is None:
                    row += ['' for key in names_mag]
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
                    row += ['' for key in names_z]
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
                        row += [invalid[typ.type]]
                    #row += [None for key in names_lines]
                else:
                    if fmt=='default':
                        for name, typ in zip(names_lines, dtype_lines):
                            colname = '_'.join(name.split('_')[1:])
                            line = name.split('_')[0]
                            if 'LINE' in source.lines.colnames and \
                               colname in source.lines.colnames and \
                               line in source.lines['LINE'].data.data:
                                row += [source.lines[colname][source.lines['LINE'] == line].data.data[0]]
                            else:
                                row += [invalid[typ.type]]
                    elif fmt=='working':
                        keys = source.lines.colnames
                        if lmax == 1:
                            row += [source.lines[key][0] if key in keys else invalid[typ.type] for key,typ in zip(names_lines, dtype_lines)]
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
                                    row += [invalid[typ.type]]
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
        t['ID'].format = '%d'
        t['RA'].format = '%.7f'
        t['DEC'].format = '%.7f'
        for names in names_z:
            t[names].format = '%.6f'
        for names in names_lines:
            if names[:4] == 'LBDA':
                t[names].format = '%0.2f'
            if names[:4] == 'FLUX':
                t[names].format = '%0.4f'
                
        # mask nan
        for col in t.colnames:
            try:
                t[col] = np.ma.masked_invalid(t[col])
#                 t[col] = np.ma.masked_equal(t[col], None)
#                 t[col] = np.ma.masked_equal(t[col], 'None')
                t[col] = np.ma.masked_equal(t[col], -9999)
                t[col] = np.ma.masked_equal(t[col], np.nan)
                t[col] = np.ma.masked_equal(t[col], '')
            except:
                pass
            
        return t
    
    @classmethod
    def from_path(cls, path, fmt='default'):
        """Create a Catalog object from the path of a directory containing source files

        Parameters
        ----------
        path : string
               Directory containing Source files
        """
        logger = logging.getLogger('mpdaf corelib')
        d = {'class': 'Catalog', 'method': 'from_path'}
        
        if not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))
        
        from .source import Source

        slist = []
        filenames = []
        files = glob.glob(path+'/*.fits')
        n = len(files)
        
        logger.info('Building catalog from path %s'%path ,extra=d)
        
        for f in files:
            slist.append(Source._light_from_file(f))
            filenames.append(os.path.basename(f))
            sys.stdout.write("\r\x1b[K %i%%"%(100*len(filenames)/n))
            sys.stdout.flush()
            
        #output = ""
        sys.stdout.write("\r\x1b[K ")
        sys.stdout.flush()

        t = cls.from_sources(slist, fmt)
        t['FILENAME'] = filenames
        
        return t

    def match(self, cat2, radius=1):
        """Match elements of the current catalog with an other (in RA, DEC).

        Parameters
        ----------
        cat2   : astropy.Table
                 Catalog to match
        radius : float
                 Matching size in arcsec

        Returns
        -------
        match, nomatch, nomatch2 : astropy.Table, astropy.Table, astropy.Table

                                   1- match table of matched elements in RA,DEC

                                   2- sub-table of non matched elements of the current catalog

                                   3- sub-table of non matched elements of the catalog cat2
        """
        d = {'class': 'Catalog', 'method': 'match'}

        coord1 = SkyCoord(ra=self['RA'] * u.degree, dec=self['DEC'] * u.degree)
        coord2 = SkyCoord(ra=cat2['RA'] * u.degree, dec=cat2['DEC'] * u.degree)
        id1, id2, d2d, d3d = search_around_sky(coord1, coord2, radius * u.arcsec)
        id1_notin_2 = np.in1d(range(len(self)), id1, invert=True)
        id2_notin_1 = np.in1d(range(len(cat2)), id2, invert=True)
        self.logger.info('Cat1 Nelt %d Match %d Not Matched %d'\
                         % (len(self), len(id1), len(self[id1_notin_2])),\
                         extra=d)
        self.logger.info('Cat2 Nelt %d Match %d Not Matched %d'\
                         % (len(cat2), len(id2), len(cat2[id2_notin_1])), \
                         extra=d)
        match = hstack([self[id1], cat2[id2]])
        nomatch = self[id1_notin_2]
        nomatch2 = cat2[id2_notin_1]
        return match, nomatch, nomatch2

    def select(self, wcs, ra='RA', dec='DEC'):
        """Select all sources from catalog which are inside the WCS of image
        and return a new catalog.

        Parameters
        ----------
        wcs : :class:`mpdaf.obj.WCS`
              Image WCS
        ra  : string
              Name of the column that contains RA values
        dec : string
              Name of the column that contains DEC values

        Returns
        -------
        out : :class:`mpdaf.sdetect.Catalog`
        """
        ksel = []
        for k, src in enumerate(self):
            cen = wcs.sky2pix([src[dec], src[ra]])[0]
            if cen[0] >= 0 and cen[0] <= wcs.naxis1 and cen[1] >= 0 \
                    and cen[1] <= wcs.naxis2:
                ksel.append(k)
        return self[ksel]

    def plot_symb(self, ax, wcs, ra='RA', dec='DEC', symb=0.4, col='k',
                  alpha=1.0, **kwargs):
        """This function plots the sources location from the catalog

        Parameters
        ----------
        ax  : matplotlib.axes._subplots.AxesSubplot
              Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        wcs : :class:`mpdaf.obj.WCS`
              Image WCS
        ra  : string
              Name of the column that contains RA values
        dec : string
              Name of the column that contains DEC values
        symb : list or string or float

               - List of 3 columns names containing FWHM1,
                 FWHM2 and ANGLE values to define the ellipse of each source.
               - Column name containing value that will be used
                 to define the circle size of each source.
               - float in the case of circle with constant size in arcsec

        col : string
              Symbol color.
        alpha : float
                Symbol transparency
        kwargs : matplotlib.artist.Artist
                 kwargs can be used to set additional plotting properties.

        """
        if type(symb) in [list, tuple] and len(symb) == 3:
            stype = 'ellipse'
            fwhm1, fwhm2, angle = symb
        elif type(symb) is str:
            stype = 'circle'
            fwhm = symb
        elif type(symb) in [int, float]:
            stype = 'fix'
            size = symb
        else:
            raise IOError('wrong symbol')

        if ra not in self.colnames:
            raise IOError('column %s not found in catalog' % ra)
        if dec not in self.colnames:
            raise IOError('column %s not found in catalog' % dec)

        for src in self:
            cen = wcs.sky2pix([src[dec], src[ra]])[0]
            if stype == 'ellipse':
                f1 = src[fwhm1] / (3600.0 * wcs.get_step()[0])  # /cos(dec) ?
                f2 = src[fwhm2] / (3600.0 * wcs.get_step()[1])
                pa = src[angle] * 180 / np.pi
            elif stype == 'circle':
                f1 = src[fwhm] / (3600.0 * wcs.get_step()[0])
                f2 = f1
                pa = 0
            elif stype == 'fix':
                f1 = size / (3600.0 * wcs.get_step()[0])
                f2 = f1
                pa = 0
            ell = Ellipse((cen[1], cen[0]), 2 * f1, 2 * f2, pa, fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ell.set_edgecolor(col)

    def plot_id(self, ax, wcs, iden='ID', ra='RA', dec='DEC', symb=0.2,
                alpha=0.5, col='k', **kwargs):
        """ This function displays the id of the catalog

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
             Matplotlib axis instance (eg ax = fig.add_subplot(2,3,1)).
        wcs : :class:`mpdaf.obj.WCS`
              Image WCS
        iden : string
               Name of the column that contains ID values
        ra  : string
              Name of the column that contains RA values
        dec : string
              Name of the column that contains DEC values
        symb : float
               Size of the circle in arcsec
        col : string
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

        cat = self.select(wcs)
        size = 2 * symb / (3600.0 * wcs.get_step()[0])
        for src in cat:
            cen = wcs.sky2pix([src[dec], src[ra]])[0]
            ax.text(cen[1], cen[0] + size, src[iden], ha='center', color=col,
                    **kwargs)
            ell = Ellipse((cen[1], cen[0]), size, size, 0, fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ell.set_edgecolor(col)
