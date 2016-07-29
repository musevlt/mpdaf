# -*- coding: utf-8 -*-

import numpy as np
from mpdaf.tools.astropycompat import zscale


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
