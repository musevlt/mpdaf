"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2012-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
