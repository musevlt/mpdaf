"""Copyright 2010-2016 CNRS/CRAL

This file is part of MPDAF.

MPDAF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version

MPDAF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MPDAF.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import
import numpy as np


def mag_RJohnson():
    lbda = 6940.
    lmin = 5200.
    lmax = 9600.

    tck = (np.array([5200., 5200., 5200., 5200.,
                     5600., 5800., 6000., 6200.,
                     6400., 6600., 6800., 7000.,
                     7200., 7400., 7600., 7800.,
                     8000., 8200., 8400., 8600.,
                     8800., 9000., 9200., 9600.,
                     9600., 9600., 9600.]),
           np.array([8.60663864e-19, -5.78293411e-02, 2.15658682e-01,
                     4.99845310e-01, 7.13913421e-01, 7.84501007e-01,
                     8.88082550e-01, 9.43168793e-01, 9.79242280e-01,
                     1.01986209e+00, 9.41309368e-01, 8.54900440e-01,
                     7.39088871e-01, 5.68744075e-01, 4.05934831e-01,
                     3.27516603e-01, 1.43998756e-01, 1.16488372e-01,
                     5.00477576e-02, 4.33205980e-02, 7.78626802e-03,
                     1.11068660e-02, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
           3)

    return (lbda, lmin, lmax, tck)


def mag_F606W():
    lbda = 6060.
    lmin = 4630.02
    lmax = 7473.17

    tck = (np.array([4630.02, 4630.02, 4630.02, 4630.02, 4826.83, 4928.36,
                     5032.03, 5137.87, 5245.94, 5356.28, 5468.94, 5583.98,
                     5701.42, 5821.35, 5943.79, 6068.81, 6196.47, 6326.8,
                     6459.87, 6595.75, 6734.48, 6876.14, 7020.77, 7168.44,
                     7473.17, 7473.17, 7473.17, 7473.17]),
           np.array([-1.29035924e-18, -1.77410175e-01, 5.09004115e-01,
                     4.55477447e-01, 5.25807339e-01, 7.02509926e-01,
                     5.53219843e-01, 7.74280131e-01, 7.79317284e-01,
                     9.17635174e-01, 8.36973020e-01, 8.34482266e-01,
                     1.02299100e+00, 9.13253500e-01, 1.04773227e+00,
                     8.92808809e-01, 1.01173936e+00, 9.09378601e-01,
                     9.59431407e-01, 6.61189121e-01, 4.25683387e-01,
                     4.87088936e-01, -3.27742078e-01, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00]),
           3)
    return (lbda, lmin, lmax, tck)