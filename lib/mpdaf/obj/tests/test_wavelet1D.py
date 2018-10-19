# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2017 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
from ..wavelet1D import wavelet_transform, wavelet_backTransform, cleanSignal


def test_wavelet1D():
    def gauss(x, amplitude, mu, sigma):
        return amplitude * np.exp(-(x - mu)**2 / (2.0 * sigma**2))

    stdDev = 5.0
    levels = 3
    sigmaCutoff = 5.0
    epsilon = 0.05

    np.random.seed(42)
    x = np.arange(-20, 20)
    signal = gauss(x, 50.0, 0.0, 5.0)
    noise = np.random.normal(0, stdDev, np.size(signal))
    stdDevList = [stdDev for _ in range(-20, 20)]
    signal_final = signal + noise
    wavelet_signal = wavelet_transform(signal_final, levels)
    reconstructed = wavelet_backTransform(wavelet_signal)
    denoised = cleanSignal(signal_final, stdDevList, levels,
                           sigmaCutoff=sigmaCutoff, epsilon=epsilon)
    assert np.abs(np.std(signal_final - denoised) - stdDev) < 1
    assert np.abs(np.std(signal - reconstructed) - stdDev) < 1
