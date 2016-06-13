"""Copyright 2010-2016 CNRS/CRAL

This file is part of MPDAF.

MPDAF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version

MPDAF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MPDAF.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import, division

import numpy as np
from scipy import special


class LSF(object):

    """This class manages MUSE LSF models.

    :param type: type of LSF
        if type is 'qsim_v1', This is a simple model where the LSF
        is supposed to be constant over the filed of view.
        It uses a simple parametric model of variation with wavelength.

        The model is a convolution of a step function with a gaussian.
        The resulting function is then sample by the pixel size.
        The slit width is assumed to be constant (2.09 pixels).
        The gaussian sigma parameter is a polynomial approximation
        of order 3 with wavelength.
    :type type: 'qsim_v1'

    Attributes
    ----------

    type (string) : LSF type.
    """

    def __init__(self, type="qsim_v1"):
        """Manages LSF model.

        :param type: type of LSF
            if type is 'qsim_v1', This is a simple model
            where the LSF is supposed
            to be constant over the filed of view.
            It uses a simple parametric model of variation with wavelength.

            The model is a convolution of a step function with a gaussian.
            The resulting function is then sample by the pixel size.
            The slit width is assumed to be constant (2.09 pixels).
            The gaussian sigma parameter is a polynomial approximation
            of order 3 with wavelength.
        :type type: 'qsim_v1'
        """
        self.type = type

    def get_LSF(self, lbda, step, size, **kargs):
        """returns an array containing the LSF.

        :param lbda: wavelength value in A
        :type lbda: float
        :param step: size of the pixel in A
        :type step: float
        :param size: number of pixels
        :type size: odd integer
        :param kargs: kargs can be used to set LSF parameters.
        :rtype: np.array
        """
        if self.type == "qsim_v1":
            T = lambda x: np.exp((-x ** 2) / 2.0) + np.sqrt(2.0 * np.pi) \
                * x * special.erf(x / np.sqrt(2.0)) / 2.0
            c = np.array([-0.09876662, 0.44410609, -0.03166038, 0.46285363])
            sigma = lambda x: c[3] + c[2] * x + c[1] * x ** 2 + c[0] * x ** 3

            x = (lbda - 6975.0) / 4650.0
            h_2 = 2.09 / 2.0
            sig = sigma(x)
            dy_2 = step / 1.25 / 2.0

            k = size // 2
            y = np.arange(-k, k + 1)

            y1 = (y - h_2) / sig
            y2 = (y + h_2) / sig

            lsf = T(y2 + dy_2) - T(y2 - dy_2) - T(y1 + dy_2) + T(y1 - dy_2)

        lsf /= lsf.sum()
        return lsf

    def size(self, lbda, step, epsilon, **kargs):
        """returns the LSF size in pixels.

        :param lbda: wavelength value in A
        :type lbda: float
        :param step: size of the pixel in A
        :type step: float
        :param epsilon: this factor is used to determine the size of LSF
            (min(LSF)<max(LSF)*epsilon)
        :type epsilon: float
        :rtype: integer
        """
        x0 = lbda
        # x = np.array([x0])
        diff = -1
        k = 0
        while diff < 0:
            k = k + 1
            g = self.get_LSF(x0, step, 2 * k + 1, **kargs)
            diff = epsilon * g[1] - g[0]
        size = k * 2 + 1
        return size
