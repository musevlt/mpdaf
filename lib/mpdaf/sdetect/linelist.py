"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2019 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2017-2019 Roland Bacon <roland.bacon@univ-lyon1.fr>
Copyright (c) 2018-2019 Yannick Roehlly <yannick.roehlly@univ-lyon1.fr>

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

import numpy as np
from astropy.table import Table

from ..obj import vactoair
from ..obj import airtovac  # noqa - for backward compatibility


__all__ = ['get_emlines', 'z_from_linepos']

# list of useful emission lines
# name (id), vacuum wave A (c), line type (em/is) (tp), 
# main line(1/0) (s), doublet (average/0) (d)
# line family (0=abs, 1=Balmer, 2=Forbidden, 3=Resonant) (f)
# vdisp (0/1) (v), display name (n)
emlines = np.array([
    ('LYALPHA',  1215.67, 'em', 1,      0, 3, 0, "Lyα"),   
    ('NV1238',  1238.82, 'em', 0, 1240.8, 2, 0, None), # NV
    ('NV1243',  1242.80, 'em', 0, 1240.8, 2, 0, "NV"),
    ('SiII1260', 1260.42, 'is', 0,      0, 0, 0, "Siɪɪ"),
    ('OI1302',   1302.17, 'is', 0,      0, 0, 0, "Oɪ"),
    ('SIII1304', 1304.37, 'is', 0,      0, 0, 0, "Siɪɪ"),
    ('CII1334',  1334.53, 'is', 0,      0, 0, 0, "Cɪɪ"),
    ('SIIV1394', 1393.76, 'is', 0,      0, 0, 0, None), # SiIV
    ('SIIV1403', 1402.77, 'is', 0,      0, 0, 0, "Siɪᴠ"),
    ('CIV1548',  1548.20, 'em', 1, 1549.5, 3, 0, None), #CIV
    ('CIV1551',  1550.77, 'em', 1, 1549.5, 3, 0, "Cɪᴠ"),
    ('FEII1608', 1608.45, 'is', 0,      0, 0, 0, None), #FeII
    ('FEII1611', 1611.20, 'is', 0,      0, 0, 0, "Feɪɪ"),
    ('HEII1640', 1640.42, 'em', 0,      0, 2, 0, "Heɪɪ"),
    ('OIII1660', 1660.81, 'em', 0,      0, 2, 0, None), #OIII
    ('OIII1666', 1666.15, 'em', 0,      0, 2, 0, "Oɪɪɪ]"),
    ('ALII1671', 1670.79, 'is', 0,      0, 0, 0, "Alɪɪ"),
    ('AL1854',   1854.10, 'is', 0,      0, 0, 0, None), #AlIII
    ('AL1862',   1862.17, 'is', 0,      0, 0, 0, "Alɪɪɪ"),
    ('CIII1907', 1906.68, 'em', 1, 1907.7, 2, 0, None), #CIII]
    ('CIII1909', 1908.73, 'em', 1, 1907.7, 2, 0, "Cɪɪɪ]"),
    ('CII2324',  2324.21, 'em', 0, 2326.0, 2, 0, None), #CII 
    ('CII2326',  2326.11, 'em', 0, 2326.0, 2, 0, "Cɪɪ]"), 
    ('CII2328',  2327.64, 'em', 0, 2326.0, 2, 0, None), # CII
    ('CII2329',  2328.84, 'em', 0, 2326.0, 2, 0, None), # CII
    ('FEII2344', 2344.21, 'is', 0,      0, 0, 0, None), #FeII
    ('FEII2374', 2374.46, 'is', 0,      0, 0, 0, None), #FeII
    ('FEII2383', 2382.76, 'is', 0,      0, 0, 0, "Feɪɪ"),
    ('NEIV2422', 2421.83, 'em', 0, 2423.0, 2, 0, None),
    ('NEIV2424', 2424.42, 'em', 0, 2423.0, 2, 0, "Neɪᴠ"),
    ('FEII2587', 2586.65, 'is', 0,      0, 0, 0, None), #FeII
    ('FEII2600', 2600.17, 'is', 0,      0, 0, 0, "Feɪɪ"),
    ('MGII2796', 2796.35, 'em', 0, 2800.0, 3, 0, None), #MgII
    ('MGII2803', 2803.53, 'em', 0, 2800.0, 3, 0, "Mgɪɪ"),
    ('MGI2853',  2852.97, 'is', 0,      0, 0, 0, "Mgɪ"),
    ('NEV3427',  3426.85, 'em', 0,      0, 2, 0, "Neᴠ"),
    ('OII3727',  3727.09, 'em', 1, 3727.5, 2, 0, None), #OII
    ('OII3729',  3729.88, 'em', 1, 3727.5, 2, 0, "[Oɪɪ]"),
    ('H11',      3771.70, 'em', 0,      0, 1, 0, "H11"),
    ('H10',      3798.98, 'em', 0,      0, 1, 0, "H10"),
    ('H9',       3836.47, 'em', 0,      0, 1, 0, "H9"),
    ('NEIII3870',3870.16, 'em', 1,      0, 2, 0, "[Neɪɪɪ]"),
    ('CAK',      3933.66, 'is', 0,      0, 0, 0, None), #CaK
    ('CAH',      3968.45, 'is', 0,      0, 0, 0, "CaHK"),
    ('HEI3890',  3889.73, 'em', 0,      0, 2, 0, None), #HeI
    ('H8',       3890.15, 'em', 0,      0, 1, 0, "H8"),
    ('NEIII3967',3968.91, 'em', 0,      0, 2, 0, None), #NeIII
    ('HEPSILON', 3971.20, 'em', 0,      0, 1, 0, "Hε"),
    ('HDELTA',   4102.89, 'em', 1,      0, 1, 0, "Hδ"),
    ('CAG',      4304.57, 'is', 0,      0, 0, 0, "Gband"),
    ('HGAMMA',   4341.68, 'em', 1,      0, 1, 0, "Hγ"),
    ('OIII4364', 4364.44, 'em', 0,      0, 2, 0, None), #OIII
    ('HBETA',    4862.68, 'em', 1,      0, 1, 0, "Hβ"),
    ('OIII4960', 4960.30, 'em', 1,      0, 2, 0, None), #OIII
    ('OIII5008', 5008.24, 'em', 1,      0, 2, 0, "[Oɪɪɪ]"),
    ('MGB',      5175.44, 'is', 0,      0, 0, 0, "Mgb"),
    ('HEI5877',  5877.25, 'em', 0,      0, 2, 0, None), #HeI
    ('NAD',      5891.94, 'is', 0,      0, 0, 0, "NaD"),
    ('OI6302',   6302.05, 'em', 0,      0, 2, 0, "[Oɪ]"),
    ('NII6550',  6549.85, 'em', 0,      0, 2, 0, None), #NII
    ('HALPHA',   6564.61, 'em', 1,      0, 1, 0, "Hα"),
    ('NII6585',  6585.28, 'em', 1,      0, 2, 0, None), #NII
    ('SII6718',  6718.29, 'em', 1,      0, 2, 0, None),
    ('SII6733',  6732.67, 'em', 1,      0, 2, 0, "[Sɪɪ]"),
    ('ARIII7138',7137.80, 'em', 0,      0, 2, 0, "[Arɪɪɪ]"),
], dtype=[('id', 'U20'), ('c', '<f4'), ('tp', 'U2'),
          ('s', '<i4'), ('d', '<f4'),
          ('f', '<i4'), ('v', '<i4'), ('n', 'U10')])


def get_emlines(iden=None, z=0, vac=True, lbrange=None, margin=25, sel=None,
                ltype=None, doublet=False, restframe=False, table=False,
                family=None, exlbrange=None):
    """Return list of emission lines

    Parameters
    ----------
    iden : str or list of str
        identifiers, eg 'LYALPHA', ['OII3727','OII3729'] default None
    z : float
        redshift (0)
    vac : bool
        if False return wavelength in air
    lbrange : array-like
        wavelength range ex [4750,9350] default None
    margin : float
        margin in A to select a line (25)
    sel : int
        select line which has sel value (1=major,0=minor)
    ltype : str
        select line with the given type ('em', 'is')
    doublet : bool
        if true return only doublet
    restframe : bool
        if true the wavelength are not reshifted but the
        selection with lbrange take into account the redshift
    table : bool
        if True return an astropy table
    family : int
        select line with the given family (0=abs, 1=Balmer, 2=Forbidden,
        3=Resonant).
    exlbrange : array-like
        wavelength range to exclude in observed frame (ex for AO spectra)
    """
    em = emlines.copy()
    if iden is not None:
        if isinstance(iden, str):
            em = em[em['id'] == iden]
            if len(em) == 0:
                return None
        elif isinstance(iden, (list, tuple, np.ndarray)):
            em = em[np.isin(em['id'], iden)]

    kd = np.where(em['d'] > 0)
    if not restframe:
        em['d'][kd] *= 1 + z
        for key in ['c']:
            em[key] = (1 + z) * em[key]
    if not vac:
        for key in ['c']:
            em[key] = vactoair(em[key])
        em['d'][kd] = vactoair(em['d'][kd])
    if lbrange is not None:
        if restframe:
            lbda = em['c'] * (1 + z)
            em = em[lbda - margin >= lbrange[0]]
            lbda = em['c'] * (1 + z)
            em = em[lbda + margin <= lbrange[1]]
        else:
            em = em[em['c'] - margin >= lbrange[0]]
            em = em[em['c'] + margin <= lbrange[1]]
    if exlbrange is not None:
        if restframe:
            em = em[(em['c']*(1+z) < exlbrange[0]) | (em['c']*(1+z) > exlbrange[1])]
        else:
            em = em[(em['c'] < exlbrange[0]) | (em['c'] > exlbrange[1])]
        
    if sel is not None:
        em = em[em['s'] == sel]
    if ltype is not None:
        em = em[em['tp'] == ltype]
    if family is not None:
        em = em[em['f'] == family]
    if doublet:
        em = em[em['d'] > 0]
        
    print(em)

    if not table:
        return em
    else:
        return Table(
            data=[em['id'], em['c'], em['tp'], em['s'], em['d'],
                  em['f'], em['v'],
                  np.asarray([str(name).encode('utf8') for name in em['n']], dtype='|S7')],
            names=['LINE', 'LBDA_OBS', 'TYPE', 'MAIN',
                   'DOUBLET', 'FAMILY', 'VDISP', 'DNAME'],
            masked=True
        )


def z_from_linepos(iden, wavelength, vac=True):
    """Returns the redshift at which a line has the given position.

    Parameters
    ----------
    iden : str
        Identifier of the line.
    wavelength : float
        Position of the line in Angstrom.
    vac : bool, optional
        If True, the position is given in vacuum, else it is given in air.

    Returns
    -------
    redshift: float

    Raises
    ------
    ValueError is `iden` does not refer to a known line.

    """
    if not vac:
        wavelength = airtovac(wavelength)

    em = get_emlines(iden)

    if em is None:
        raise ValueError("The line is unknown.")

    restframe_wavelenth = em['c'][0]
    redshift = wavelength / restframe_wavelenth - 1

    return redshift
