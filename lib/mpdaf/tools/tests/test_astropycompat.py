# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from astropy.table import Table
from mpdaf.tools import zscale, table_to_hdu, write_hdulist_to


def test_zscale():
    np.random.seed(42)
    data = np.random.randn(100, 100) * 5 + 10
    vmin, vmax = zscale(data)
    np.testing.assert_allclose(vmin, -9.6, atol=0.1)
    np.testing.assert_allclose(vmax, 25.4, atol=0.1)

    data = list(range(1000)) + [np.nan]
    vmin, vmax = zscale(data)
    np.testing.assert_allclose(vmin, 0, atol=0.1)
    np.testing.assert_allclose(vmax, 999, atol=0.1)

    data = list(range(100))
    vmin, vmax = zscale(data)
    np.testing.assert_allclose(vmin, 0, atol=0.1)
    np.testing.assert_allclose(vmax, 99, atol=0.1)


def test_table_to_hdu(tmpdir):
    table = Table([[1, 2, 3], ['a', 'b', 'c'], [2.3, 4.5, 6.7]],
                  names=['a', 'b', 'c'], dtype=['i', 'U1', 'f'])
    hdu = table_to_hdu(table)
    assert isinstance(hdu, fits.BinTableHDU)
    filename = str(tmpdir.join('test_table_to_hdu.fits'))
    write_hdulist_to(hdu, filename, overwrite=True)
