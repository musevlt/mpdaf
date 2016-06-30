"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

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
from __future__ import absolute_import, print_function

import numpy as np
from astropy.table import Table

__all__ = ['get_emlines']

# list of useful emission lines
# name (id), vacuum wave A (c), lower limit (lo), upper limit (up), type (tp),
# main line(1/0) (s), doublet (averge/0) (d)
emlines = np.array([
    ('LYALPHA', 1215.67, 1204.0, 1226.0, 'em', 1, 0),
    ('CIV1546', 1545.86, 1536.0, 1556.0, 'em', 0, 0),
    ('HEII1640', 1640.42, 1630.0, 1651.0, 'em', 0, 0),
    ('OIII1660', 1660.8101, 1650.0, 1670.0, 'em', 0, 0),
    ('CIII1907', 1906.6801, 1896.0, 1920.0, 'em', 1, 1907.7),
    ('CIII1909', 1908.73, 1898.0, 1920.0, 'em', 1, 1907.7),
    ('NEIV2422', 2421.8301, 2411.0, 2431.0, 'em', 0, 2423.0),
    ('NEIV2424', 2424.4199, 2414.0, 2434.0, 'em', 0, 2423.0),
    ('NEV3426', 3426.8501, 3416.0, 3436.0, 'em', 0, 0),
    ('OII3726', 3727.0901, 3717.0, 3737.0, 'em', 1, 3727.5),
    ('OII3729', 3729.8799, 3719.0, 3739.0, 'em', 1, 3727.5),
    ('H11', 3771.7, 3760.0, 3780.0, 'em', 0, 0),
    ('H10', 3798.98, 3787.0, 3809.0, 'em', 0, 0),
    ('H9', 3836.47, 3825.0, 3845.0, 'em', 0, 0),
    ('NEIII3869', 3870.1599, 3859.0, 3879.0, 'em', 1, 0),
    ('HEI3889', 3889.73, 3879.0, 3899.0, 'em', 0, 0),
    ('H8', 3890.1499, 3879.0, 3899.0, 'em', 0, 0),
    ('NEIII3967', 3968.9099, 3957.0, 3977.0, 'em', 0, 0),
    ('HEPSILON', 3971.2, 3960.0, 3980.0, 'em', 0, 0),
    ('HDELTA', 4102.8901, 4092.0, 4111.0, 'em', 1, 0),
    ('HGAMMA', 4341.6802, 4330.0, 4350.0, 'em', 1, 0),
    ('OIII4363', 4364.4399, 4350.0, 4378.0, 'em', 0, 0),
    ('HBETA', 4862.6802, 4851.0, 4871.0, 'em', 1, 0),
    ('OIII4959', 4960.2998, 4949.0, 4969.0, 'em', 1, 0),
    ('OIII5007', 5008.2402, 4997.0, 5017.0, 'em', 1, 0),
    ('HEI5876', 5877.25, 5866.0, 5886.0, 'em', 0, 0),
    ('OI6300', 6302.0498, 6290.0, 6310.0, 'em', 0, 0),
    ('NII6548', 6549.8501, 6533.0, 6553.0, 'em', 0, 0),
    ('HALPHA', 6564.6099, 6553.0, 6573.0, 'em', 1, 0),
    ('NII6584', 6585.2798, 6573.0, 6593.0, 'em', 1, 0),
    ('SII6717', 6718.29, 6704.0, 6724.0, 'em', 1, 0),
    ('SII6731', 6732.6699, 6724.0, 6744.0, 'em', 1, 0),
    ('ARIII7135', 7137.7998, 7130.0, 7147.0, 'em', 0, 0),
    ('FEII2344', 2344.21, 2330.0, 2354.0, 'is', 0, 0),
    ('FEII2374', 2374.46, 2364.0, 2384.0, 'is', 0, 0),
    ('FEII2382', 2382.76, 2372.0, 2392.0, 'is', 0, 0),
    ('FEII2586', 2586.6499, 2576.0, 2596.0, 'is', 0, 0),
    ('FEII2600', 2600.1699, 2590.0, 2610.0, 'is', 0, 0),
    ('MGII2796', 2796.3501, 2786.0, 2806.0, 'em', 0, 0),
    ('MGII2803', 2803.53, 2793.0, 2813.0, 'em', 0, 0),
    ('CAK', 3933.6599, 3919.0, 3949.0, 'is', 0, 0),
    ('CAH', 3968.45, 3953.0, 3983.0, 'is', 0, 0),
    ('NAD', 5891.9399, 5881.0, 5906.0, 'is', 0, 0)
], dtype=[('id', 'S20'), ('c', '<f4'), ('lo', '<f4'), ('up', '<f4'),
          ('tp', 'S2'), ('s', '<i4'), ('d', '<f4')])


def get_emlines(iden=None, z=0, vac=True, lbrange=None, margin=25, sel=None,
                ltype=None, doublet=False, restframe=False, table=False):
    """Return list of emission lines

    Parameters
    ----------
    iden: str or list of str
        identifiers, eg 'LYALPHA', ['OII3727','OII3729'] default None
    z: float
        redshift (0)
    vac: bool
        if False return wavelength in air
    lbrange: array-like
        wavelength range ex [4750,9350] default None
    margin: float
        margin in A to select a line (20)
    sel:
        select line which has sel value (1,0)
    ltype:
        select line with the given type ('em','is')
    doublet: bool
        if true return only doublet
    restframe: bool
        if true the wavelength are not reshifted but the
        selection with lbrange take into account the redshift
    table: bool
        if True return an astropy table

    """
    em = emlines.copy()
    if iden is not None:
        if type(iden) is str:
            em = em[em['id'] == iden]
            if len(em) == 0:
                return None
        elif type(iden) is list or type(iden) is np.ndarray:
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
            em = em[lbda + margin <= lbrange[1]]
        else:
            em = em[em['c'] - margin >= lbrange[0]]
            em = em[em['c'] + margin <= lbrange[1]]
    if sel is not None:
        em = em[em['s'] == sel]
    if ltype is not None:
        em = em[em['tp'] == ltype]
    if doublet:
        em = em[em['d'] > 0]

    if not table:
        return em
    else:
        tab = Table(data=[em['id'], em['c'], em['lo'], em['up'], em['tp'], em['d']],
                    names=['LINE', 'LBDA_OBS', 'LBDA_LOW', 'LBDA_UP', 'TYPE', 'DOUBLET'])
        return tab


def vactoair(vacwl):
    """
    Calculate the approximate wavelength in air for vacuum wavelengths

    Parameters
    ----------
    vacwl : ndarray
       Vacuum wavelengths.

    This uses an approximate formula from the IDL astronomy library
    vactoair.pro.
    """

    wave2 = vacwl * vacwl
    n = 1.0 + 2.735182e-4 + 131.4182 / wave2 + 2.76249e8 / (wave2 * wave2)

    # Do not extrapolate to very short wavelengths.
    if not isinstance(vacwl, np.ndarray):
        if vacwl < 2000:
            n = 1.0
    else:
        ignore = np.where(vacwl < 2000)
        n[ignore] = 1.0

    return vacwl / n


def airtovac(airwl):
    """
    Convert air wavelengths to vacuum wavelengths

    Parameters
    ----------
    vacwl : ndarray
       Vacuum wavelengths.

    This uses the IAU standard as implemented in the IDL astronomy
    library airtovac.pro
    """

    sigma2 = (1e4 / airwl)**2.        # Convert to wavenumber squared
    n = 1.0 + (6.4328e-5 + 2.94981e-2 / (146. - sigma2) +
               2.5540e-4 / (41. - sigma2))

    if not isinstance(airwl, np.ndarray):
        if airwl < 2000:
            n = 1.0
    else:
        ignore = np.where(airwl < 2000)
        n[ignore] = 1.0

    return airwl * n


if __name__ == '__main__':
    em = get_emlines(doublet=True, z=3.0, vac=False)
    #em = get_emlines(z=0, vac=False, lbrange=(4750,9350), margin=20, sel=0, ltype='is')
    print(em)
    print('done')
