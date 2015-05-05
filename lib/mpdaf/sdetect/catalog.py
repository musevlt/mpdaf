
from astropy.io import fits as pyfits
from astropy.io.fits import Column
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

import datetime
import logging
import warnings
import numpy as np
from collections import OrderedDict

from ..tools.fits import add_mpdaf_method_keywords

from astropy.table import Table, hstack

class Catalog(Table):
    """This class contains a catalog of objects.
    table : astropy.table
            Astropy table
            
    astropy.table object
    """
    
    def __init__(self, data=None, masked=None, names=None,
                 dtype=None, meta=None, copy=True, rows=None):
        Table.__init__(self, data, masked, names, dtype, meta, copy, rows)
        self.logger = logging.getLogger('mpdaf corelib')

        
    @classmethod
    def from_sources(cls, sources):
        """ constructs a catalog from a list of source objects.
        
        Parameters
        ----------
        sources : list< :class:`mpdaf.sdetect.Source` >
        """ 
        # union of all headers keywords without mandatory FITS keywords
        names_hdr = list(set(np.concatenate([source.header.keys() 
                                         for source in sources]))
                     - set(['SIMPLE', 'BITPIX', 'NAXIS',
                            'EXTEND','DATE','AUTHOR']))
        #sort mandatory keywords
        names_hdr.insert(0, names_hdr.pop(names_hdr.index('ID')))
        names_hdr.insert(1, names_hdr.pop(names_hdr.index('RA')))
        names_hdr.insert(2, names_hdr.pop(names_hdr.index('DEC')))
        names_hdr.insert(3, names_hdr.pop(names_hdr.index('ORIGIN')))
        names_hdr.insert(4, names_hdr.pop(names_hdr.index('ORIGIN_V')))
        names_hdr.insert(5, names_hdr.pop(names_hdr.index('CUBE')))
         
        #lines
        llines = [len(source.lines) for source in sources if source.lines is not None]
        if len(llines) != 0:
            lmax = max(llines)
            if lmax ==1:
                names_lines = list(OrderedDict.fromkeys(np.concatenate(
                            [source.lines.colnames for source in sources
                             if source.lines is not None])))
            else:
                names_lines = []
                inames_lines = list(OrderedDict.fromkeys(np.concatenate(
                            [source.lines.colnames for source in sources
                             if source.lines is not None])))
                for i in range(1,lmax+1):
                    names_lines += [col+'%03d'%i for col in inames_lines]
        else:
            names_lines = []
            
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
            names_z = ['z_%s'%z for z in names_z]
            names_z += ['%s_ERR'%z for z in names_z]
            names_z.sort()
        else:
            names_z = []
        
        data_rows = []
        for source in sources:
            # header
            h = source.header
            keys = h.keys()
            row = [h[key] if key in keys else '' for key in names_hdr]
            # lines
            if len(llines) != 0:
                if source.lines is None:
                    row += ['' for key in names_lines]
                else:
                    keys = source.lines.colnames
                    if lmax ==1:
                        row += [source.lines[key][0] if key in keys else '' for key in names_lines]
                    else:
                        n = len(source.lines)
                        row += [source.lines[key[:-3]][int(key[-3:])-1] if key[:-3] in keys and int(key[-3:])<=n else '' for key in names_lines]
            #magnitudes
            if len(lmag) != 0:
                if source.mag is None:
                    row += ['' for key in names_mag]
                else:
                    keys = source.mag['BAND']
                    for key in names_mag:
                        if key in keys:
                            row += [float(source.mag['MAG'][source.mag['BAND']==key])]
                        elif key[-4:]=='_ERR' and key[:-4] in keys:
                            row += [float(source.mag['MAG_ERR'][source.mag['BAND']==key[:-4]])]
                        else:
                            row += [""]
            #redshifts
            if len(lz) !=0:
                if source.z is None:
                    row += ['' for key in names_z]
                else:
                    keys = source.z['Z_DESC']
                    for key in names_z:
                        key = key[2:]
                        if key in keys:
                            row += [float(source.z['Z'][source.z['Z_DESC']==key])]
                        elif key[-4:]=='_ERR' and key[:-4] in keys:
                            row += [float(source.z['Z_ERR'][source.z['Z_DESC']==key[:-4]])]
                        else:
                            row += [""]
                    
            # final row
            data_rows.append(row)
        
        #create Table
        names = names_hdr + names_lines + names_mag + names_z
        return cls(rows=data_rows, names=names)    
    
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
    