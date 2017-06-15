# -*- coding: utf-8 -*-

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
