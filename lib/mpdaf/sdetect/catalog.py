
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

import logging
import numpy as np

from astropy.table import Table, hstack

class Catalog(Table):
    """This class contains a catalog of objects.
       This is a subclass of Table class from astropy.table
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
    def from_sources(cls, sources):
        """ constructs a catalog from a list of source objects.
        
        Parameters
        ----------
        sources : list< :class:`mpdaf.sdetect.Source` >
        """ 
        # union of all headers keywords without mandatory FITS keywords
        
        h = sources[0].header
        d = dict(zip(h.keys(), [type(v) for v in h.values()]))
        for source in sources[1:]:
            h = source.header
            d.update(dict(zip(h.keys(), [type(v) for v in h.values()])))
        d = {key: value for key, value in d.items() if key not in ['SIMPLE', 'BITPIX', 'NAXIS','EXTEND','DATE','AUTHOR']}
        names_hdr = d.keys()
        dtype_hdr = d.values()
        #sort mandatory keywords
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
         
        #lines
        llines = [len(source.lines) for source in sources if source.lines is not None]
        if len(llines) != 0:
            lmax = max(llines)
            d = {}
            for source in sources:
                if source.lines is not None:
                    for col in source.lines.colnames:
                        d[col] = source.lines.dtype[col]
            if lmax ==1:
                names_lines = sorted(d)
                dtype_lines = [d[key] for key in sorted(d)]
            else:
                names_lines = []
                inames_lines = sorted(d)
                for i in range(1,lmax+1):
                    names_lines += [col+'%03d'%i for col in inames_lines]
                dtype_lines = [d[key] for key in sorted(d)]*lmax
        else:
            names_lines = []
            dtype_lines = []
            
        #magnitudes
        lmag = [len(source.mag) for source in sources if source.mag is not None]
        if len(lmag) != 0:
            names_mag = list(set(np.concatenate([source.mag['BAND'] for source in sources
                             if source.mag is not None])))
            names_mag += ['%s_ERR'%mag for mag in names_mag]
            names_mag.sort()
        else:
            names_mag = []
        
        #redshifts
        lz = [len(source.z) for source in sources if source.z is not None]
        if len(lz) != 0:
            names_z = list(set(np.concatenate([source.z['Z_DESC'] for source in sources
                             if source.z is not None])))
            names_z = ['Z_%s'%z for z in names_z]
            names_z += ['%s_ERR'%z for z in names_z]
            names_z.sort()
        else:
            names_z = []
            
        
        data_rows = []
        for source in sources:
            # header
            h = source.header
            keys = h.keys()
            row = [h[key] if key in keys else None for key in names_hdr]
            # lines
            if len(llines) != 0:
                if source.lines is None:
                    row += [None for key in names_lines]
                else:
                    keys = source.lines.colnames
                    if lmax ==1:
                        row += [source.lines[key][0] if key in keys else None for key in names_lines]
                    else:
                        n = len(source.lines)
                        row += [source.lines[key[:-3]][int(key[-3:])-1] if key[:-3] in keys and int(key[-3:])<=n else None for key in names_lines]
            #magnitudes
            if len(lmag) != 0:
                if source.mag is None:
                    row += [None for key in names_mag]
                else:
                    keys = source.mag['BAND']
                    for key in names_mag:
                        if key in keys:
                            row += [float(source.mag['MAG'][source.mag['BAND']==key])]
                        elif key[-4:]=='_ERR' and key[:-4] in keys:
                            row += [float(source.mag['MAG_ERR'][source.mag['BAND']==key[:-4]])]
                        else:
                            row += [None]
            #redshifts
            if len(lz) !=0:
                if source.z is None:
                    row += [None for key in names_z]
                else:
                    keys = source.z['Z_DESC']
                    for key in names_z:
                        key = key[2:]
                        if key in keys:
                            row += [float(source.z['Z'][source.z['Z_DESC']==key])]
                        elif key[-4:]=='_ERR' and key[:-4] in keys:
                            row += [float(source.z['Z_ERR'][source.z['Z_DESC']==key[:-4]])]
                        else:
                            row += [None]
                    
            # final row
            data_rows.append(row)
            
        dtype = dtype_hdr
        #lines
        if len(llines) != 0:
#             dtype += ['<f8' for i in range(len(names_lines))]
            dtype += dtype_lines
        #magnitudes
        if len(lmag) != 0:
            dtype += ['<f8' for i in range(len(names_mag))]
        #redshifts
        if len(lz) !=0:
            dtype += ['<f8' for i in range(len(names_z))]
            
        #create Table
        names = names_hdr + names_lines + names_mag + names_z
        t = cls(rows=data_rows, names=names, masked=True, dtype=dtype)
        
        #format
        t['ID'].format = '%d'
        t['RA'].format = '%.7f'
        t['DEC'].format = '%.7f'
        for names in names_z:
            t[names].format = '%.6f'
        for names in names_lines:
            if names[:4]=='LBDA':
                t[names].format = '%0.2f'
            if names[:4]=='FLUX':
                t[names].format = '%0.4f'
        return t   
    
    def match(self, cat2, radius=1):
        """ Matchs elements of the current catalog with an other (in RA, DEC).
        
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
        
        coord1 = SkyCoord(ra=self['RA']*u.degree, dec=self['DEC']*u.degree)
        coord2 = SkyCoord(ra=cat2['RA']*u.degree, dec=cat2['DEC']*u.degree)
        id1, id2, d2d, d3d = search_around_sky(coord1, coord2, radius*u.arcsec)
        id1_notin_2 = np.in1d(range(len(self)), id1, invert=True)
        id2_notin_1 = np.in1d(range(len(cat2)), id2, invert=True)
        self.logger.info('Cat1 Nelt %d Match %d Not Matched %d'\
                         %(len(self),len(id1),len(self[id1_notin_2])),\
                          extra=d)
        self.logger.info('Cat2 Nelt %d Match %d Not Matched %d'\
                         %(len(cat2),len(id2),len(cat2[id2_notin_1])), \
                         extra=d)
        match = hstack([self[id1], cat2[id2]])
        nomatch = self[id1_notin_2]
        nomatch2 = cat2[id2_notin_1]
        return match, nomatch, nomatch2
    