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
# name (id), vacuum wave A (c), lower limit (lo), upper limit (up), type (tp),
# main line(1/0) (s), doublet (average/0) (d)
# line family (0=abs, 1=Balmer, 2=Forbidden, 3=Resonant) (f)
# display name (n)
emlines = np.array([
    ('LYALPHA',  1215.67, 1204.00, 1226.00, 'em', 1,      0, 3, "Lyα"),
    ('SiII1260', 1260.42, None,    None,    'is', 0,      0, 0, "Siɪɪ"),
    ('NeV1238',  1238.82, None,    None,    'em', 0, 1240.8, 0, None), # NeV
    ('NeV1243',  1242.80, None,    None,    'em', 0, 1240.8, 0, "Nev"),
    ('OI1302',   1302.17, None,    None,    'is', 0,      0, 0, "Oɪ"),
    ('SIII1304', 1304.37, None,    None,    'is', 0,      0, 0, "Siɪɪ"),
    ('CII1334',  1334.53, None,    None,    'is', 0,      0, 0, "Cɪɪ"),
    ('SIIV1394', 1393.76, 1393.76, 1393.76, 'is', 0,      0, 0, None), # SiIV
    ('SIIV1403', 1402.77, 1402.77, 1402.77, 'is', 0,      0, 0, "Siɪᴠ"),
    ('CIV1548',  1548.20, None,    None,    'em', 1, 1549.5, 2, None), #CIV
    ('CIV1551',  1550.77, None,    None,    'em', 1, 1549.5, 2, "Cɪᴠ"),
    ('FEII1608', 1608.45, None,    None,    'is', 0,      0, 0, None), #FeII
    ('FEII1611', 1611.20, None,    None,    'is', 0,      0, 0, "Feɪɪ"),
    ('HEII1640', 1640.42, 1630.0, 1651.0,   'em', 0,      0, 2, "Heɪɪ"),
    ('OIII1666', 1666.15, None,    None,    'em', 0,      0, 2, "Oɪɪɪ]"),
    ('ALII1671', 1670.79, None,    None,    'is', 0,      0, 0, "Alɪɪ"),
    ('AL1854',   1854.10, None,    None,    'is', 0,      0, 0, None), #AlIII
    ('AL1862',   1862.17, None,    None,    'is', 0,      0, 0, "Alɪɪɪ"),
    ('CIII1907', 1906.68, 1896.0, 1920.0,   'em', 1, 1907.7, 2, None), #CIII]
    ('CIII1909', 1908.73, 1898.0, 1920.0,   'em', 1, 1907.7, 2, "Cɪɪɪ]"),
    ('CII2326',  2326.00, None,    None,    'em', 0,      0, 2, "Cɪɪ]"),
    ('FEII2344', 2344.21, 2330.0, 2354.0,   'is', 0,      0, 0, None), #FeII
    ('FEII2374', 2374.46, 2364.0, 2384.0,   'is', 0,      0, 0, None), #FeII
    ('FEII2383', 2382.76, 2372.0, 2392.0,   'is', 0,      0, 0, "Feɪɪ"),
    ('NEIV2422', 2421.83, 2411.0, 2431.0,   'em', 0, 2423.0, 2, None),
    ('NEIV2424', 2424.42, 2414.0, 2434.0,   'em', 0, 2423.0, 2, "Neɪᴠ"),
    ('FEII2587', 2586.65, 2576.0, 2596.0,   'is', 0,      0, 0, None), #FeII
    ('FEII2600', 2600.17, 2590.0, 2610.0,   'is', 0,      0, 0, "Feɪɪ"),
    ('MGII2796', 2796.35, 2786.0, 2806.0,   'em', 0, 2800.0, 3, None), #MgII
    ('MGII2803', 2803.53, 2793.0, 2813.0,   'em', 0, 2800.0, 3, "Mgɪɪ"),
    ('MGI2853',  2852.97, None,    None,    'is', 0,      0, 0, "Mgɪ"),
    ('NEV3427',  3426.85, 3416.0, 3436.0,   'em', 0,      0, 2, "Neᴠ"),
    ('OII3727',  3727.09, 3717.0, 3737.0,   'em', 1, 3727.5, 2, None), #OII
    ('OII3729',  3729.88, 3719.0, 3739.0,   'em', 1, 3727.5, 2, "[Oɪɪ]"),
    ('H11',      3771.70, 3760.0, 3780.0,   'em', 0,      0, 1, "H11"),
    ('H10',      3798.98, 3787.0, 3809.0,   'em', 0,      0, 1, "H10"),
    ('H9',       3836.47, 3825.0, 3845.0,   'em', 0,      0, 1, "H9"),
    ('NEIII3870',3870.16, 3859.0, 3879.0,   'em', 1,      0, 2, "[Neɪɪɪ]"),
    ('CAK',      3933.66, 3919.0, 3949.0,   'is', 0,      0, 0, None), #CaK
    ('CAH',      3968.45, 3953.0, 3983.0,   'is', 0,      0, 0, "CaHK"),
    ('HEI3890',  3889.73, 3879.0, 3899.0,   'em', 0,      0, 2, None), #HeI
    ('H8',       3890.15, 3879.0, 3899.0,   'em', 0,      0, 1, "H8"),
    ('NEIII3967',3968.91, 3957.0, 3977.0,   'em', 0,      0, 2, None), #NeIII
    ('HEPSILON', 3971.20, 3960.0, 3980.0,   'em', 0,      0, 1, "Hε"),
    ('HDELTA',   4102.89, 4092.0, 4111.0,   'em', 1,      0, 1, "Hδ"),
    ('CAG',      4304.57, 4305.6, 4305.6,   'is', 0,      0, 0, "Gband"),
    ('HGAMMA',   4341.68, 4330.0, 4350.0,   'em', 1,      0, 1, "Hγ"),
    ('OIII4364', 4364.44, 4350.0, 4378.0,   'em', 0,      0, 2, None), #OIII
    ('HBETA',    4862.68, 4851.0, 4871.0,   'em', 1,      0, 1, "Hβ"),
    ('OIII4960', 4960.30, 4949.0, 4969.0,   'em', 1,      0, 2, None), #OIII
    ('OIII5008', 5008.24, 4997.0, 5017.0,   'em', 1,      0, 2, "[Oɪɪɪ]"),
    ('MGB',      5175.44,   None,   None,   'is', 0,      0, 0, "Mgb"),
    ('HEI5877',  5877.25, 5866.0, 5886.0,   'em', 0,      0, 2, None), #HeI
    ('NAD',      5891.94, 5881.0, 5906.0,   'is', 0,      0, 0, "NaD"),
    ('OI6302',   6302.05, 6290.0, 6310.0,   'em', 0,      0, 2, "[Oɪ]"),
    ('NII6550',  6549.85, 6533.0, 6553.0,   'em', 0,      0, 2, None), #NII
    ('HALPHA',   6564.61, 6553.0, 6573.0,   'em', 1,      0, 1, "Hα"),
    ('NII6585',  6585.28, 6573.0, 6593.0,   'em', 1,      0, 2, None), #NII
    ('SII6718',  6718.29, 6704.0, 6724.0,   'em', 1,      0, 2, None),
    ('SII6733',  6732.67, 6724.0, 6744.0,   'em', 1,      0, 2, "[Sɪɪ]"),
    ('ARIII7138',7137.80, 7130.0, 7147.0,   'em', 0,      0, 2, "[Arɪɪɪ]"),
], dtype=[('id', 'U20'), ('c', '<f4'), ('lo', '<f4'),
          ('up', '<f4'), ('tp', 'U2'), ('s', '<i4'), ('d', '<f4'),
          ('f', '<i4'), ('n', 'U10')])


def get_emlines(iden=None, z=0, vac=True, lbrange=None, margin=25, sel=None,
                ltype=None, doublet=False, restframe=False, table=False,
                family=None):
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

    """
    em = emlines.copy()
    if iden is not None:
        if isinstance(iden, str):
            em = em[em['id'] == iden]
            if len(em) == 0:
                return None
        elif isinstance(iden, (list, tuple, np.ndarray)):
            em = em[np.in1d(em['id'], iden)]

    kd = np.where(em['d'] > 0)
    if not restframe:
        em['d'][kd] *= 1 + z
        for key in ['c', 'lo', 'up']:
            em[key] = (1 + z) * em[key]
    if not vac:
        for key in ['c', 'lo', 'up']:
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
    if sel is not None:
        em = em[em['s'] == sel]
    if ltype is not None:
        em = em[em['tp'] == ltype]
    if family is not None:
        em = em[em['f'] == family]
    if doublet:
        em = em[em['d'] > 0]

    if not table:
        return em
    else:
        return Table(
            data=[em['id'], em['c'], em['lo'], em['up'], em['tp'], em['d'],
                  em['f'], em['n']],
            names=['LINE', 'LBDA_OBS', 'LBDA_LOW', 'LBDA_UP', 'TYPE',
                   'DOUBLET', 'FAMILY', 'DNAME']
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
