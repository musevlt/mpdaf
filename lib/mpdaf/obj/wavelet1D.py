#####################################################################
# Wavelet filtering in 1 dimension
# v1.0, copyright Markus Rexroth, EPFL, 2016
# License: BSD 3-clause license
# Based with permission on a script by Remy Joseph, EPFL
#####################################################################
# We use the "a trous" method with B3 spline scaling function
# and the resulting h coefficients
# For details, see e.g. the paper Rexroth et al. 2017
# or the appendix A in the book "Astronomical
# image and data analysis" by Starck and Murtagh
# Note that we use a slightly modified algorithm: We calculate the
# wavelets by using 2 convolutions, not one.
# We account for this in the wavelet - image space back transformation
# The chosen B3 scaling function is well suited for isotropic signals
# (e.g. Gaussian emission lines). For anisotropic signals, it might
# be better to pick another scaling function.

import numpy as np
from scipy.ndimage import convolve1d

# See book by Starck
H_COEFFICIENTS_LIST = np.array([1 / 16.0, 1 / 4.0, 3 / 8.0, 1 / 4.0, 1 / 16.0])


def test_levels(signal, levels):
    """Test if the chosen levels are too many for our signal (sampling
    condition).
    """
    signal = np.asarray(signal)
    h_length = len(H_COEFFICIENTS_LIST)
    signalSize = signal.size
    max_level = np.log2((signalSize - 1.0) / (h_length - 1.0))
    if levels > max_level:
        # If the (level+1)-th h array is larger than the signal array, the
        # final smoothing function for level number of wavelets is wider than
        # the signal range itself, which has to be avoided. Thus h_length
        # + (2**level -1)*(h_length-1) > signalSize must be avoided (the second
        # term = number of 0s in h array)
        levels = int(np.floor(max_level))
        raise IOError("Attention: The chosen number of levels exceeds the "
                      "number allowed (sampling condition). Thus it was "
                      "automatically set to the maximum number allowed = {}"
                      .format(levels))
    return levels


def get_h_coefficients(levels):
    """Build the list of h coefficient arrays.

    The spacing between the values is 2**i_level -1 (see Starck book).

    """
    h_length = len(H_COEFFICIENTS_LIST)
    h_coefficients = []
    for i in range(levels):
        # Array long enough to store the spacing and the values
        max_length = ((2**i) - 1) * (h_length - 1) + h_length
        h_array = np.zeros(max_length)
        counter = 0
        for index in range(len(h_array)):
            # Now we enter the values and ensure that the spacing between them
            # is correct
            if index % (2**i) == 0:
                h_array[index] = H_COEFFICIENTS_LIST[counter]
                counter += 1
        h_coefficients.append(h_array)

    return h_coefficients


def wavelet_transform(signal, levels):
    """Transform a signal into wavelet space.

    levels: the number of wavelet levels we use.

    """
    signal = np.array(signal, dtype=float)
    signal[np.isnan(signal)] = 0.0

    # Test if the chosen levels are too many for our signal
    levels = test_levels(signal, levels)

    h_coefficients = get_h_coefficients(levels)

    # We need "levels" (e.g. 10) wavelet coefficient arrays
    wavelet_coefficients = []
    for i in range(levels):
        # Depending on the level, we have different h arrays for convolution.
        convolved = convolve1d(signal, h_coefficients[i], mode='wrap')
        convolved_2 = convolve1d(convolved, h_coefficients[i], mode='wrap')
        # Calculate the wavelet coefficients and add them to list
        wavelet_coefficients.append(signal - convolved_2)
        # Overwrite signal with convolved signal for the next iteration
        signal = convolved

    # Get the coefficients for the scaling function
    scaling_coefficients = convolved
    # Merge all coefficients into 1 list
    wavelet_coefficients.append(scaling_coefficients)
    return np.array(wavelet_coefficients)


def wavelet_backTransform(coefficients):
    """Transform from wavelet to real space."""
    # We have array number = levels + 1 (due to the smoothing coefficients)
    coefficients = np.array(coefficients, dtype=float)
    levels = coefficients.shape[0] - 1

    h_coefficients = get_h_coefficients(levels)

    # We convolve each wavelet and the smoothing function with the
    # corresponding h array to re-transform into image space.
    # We have to go in reverse order.

    # Initialize the coefficients for the convolution
    signal = coefficients[levels]
    for i in range(levels):
        # levels-1 because python begins counting at 0
        signal_convolved = convolve1d(signal, h_coefficients[levels - 1 - i],
                                      mode='wrap')
        # We add the corresponding coefficients for the next convolution
        signal = signal_convolved + coefficients[levels - 1 - i]

    return signal


def cleanSignal(signal, noise, levels, sigmaCutoff=5.0, epsilon=0.05):
    """Filter an input signal by using the 1 standard deviation noise estimate
    and wavelets epsilon is the iteration-stop parameter for extracting signal
    from the residual signal.
    """
    signal = np.array(signal, dtype=float)
    noise = np.array(noise, dtype=float)

    # If we have missing values, set the signal to 0 and the noise to 100000
    # standard deviations to downweigh the signal
    nans = np.isnan(signal) & np.isnan(noise)
    if nans.any():
        signal[nans] = 0.0
        noise[nans] = np.inf

    sigmaCutoff = float(sigmaCutoff)
    epsilon = float(epsilon)
    signalSize = signal.size

    # Test if the chosen levels are too many for our signal
    levels = test_levels(signal, levels)

    # We have a dirac function, 0 everywhere except 1 in the central pixel
    # (pixel value = integral over signal in area of pixel = 1 for a dirac
    # function)
    diracSignal = np.zeros(signalSize)
    diracSignal[signalSize // 2] = 1.0

    # We calculate the wavelet coefficients. We do not need the coefficients
    # for the "smoothing function", thus we delete the last array (.shape[0]
    # gives us the number of arrays)
    dirac_coefficients = wavelet_transform(diracSignal, levels)[:-1]

    # We calculate the wavelet coefficients for the noise (= 1 sigma**2):
    # convolve each row to obtain the variance
    variance_coefficients = convolve1d(dirac_coefficients**2.0, noise**2.0,
                                       mode='wrap')

    # Get standard deviation array of arrays
    noise_coef = variance_coefficients**(1 / 2.0)

    # Create the multiresolution support
    signal_coef = np.abs(wavelet_transform(signal, levels))

    # The support is 0 for every non-significant wavelet and 1 for every
    # significant wavelet/smoothing function coefficient.  If the coefficient
    # is less than (x+1)*sigma detection, we consider it as noise. We increase
    # the threshold for i = 0 = high frequencies, as typically noise has high
    # frequencies and signal has lower frequencies.
    M_support = np.vstack([
        signal_coef[0] >= ((sigmaCutoff + 1.0) * noise_coef[0]),
        signal_coef[1:-1] >= (sigmaCutoff * noise_coef[1:]),
        # Last row: smoothing function coefficients are always all significant
        np.ones(signalSize, dtype=bool)
    ])

    # Do the cleaning

    cleaned_signal = np.zeros(signalSize)
    # Initialize the standard deviation of the residual signal
    residual_signal_sigma_old = 0.0
    # Initialize the residual signal
    residual_signal = signal
    residual_signal_sigma = np.std(residual_signal)

    # We can still extract signal from the residual. We do this here after the
    # first iteration until the epsilon condition is false and add it to the
    # already extracted signal

    while np.abs((residual_signal_sigma_old - residual_signal_sigma) /
                 residual_signal_sigma) > epsilon:
        # We continue to extract until the standard deviation doesn't change
        # too much
        residual_coefficients = wavelet_transform(residual_signal, levels)

        # We clean the non-significant wavelets
        residual_coefficients[~M_support] = 0.0

        cleaned_signal += wavelet_backTransform(residual_coefficients)
        residual_signal = signal - cleaned_signal
        residual_signal_sigma_old = residual_signal_sigma
        residual_signal_sigma = np.std(residual_signal)

    return cleaned_signal


def test(stdDev=5.0, random='yes', levels=3, sigmaCutoff=5.0, epsilon=0.05):
    """Test function for the functions defined above."""
    def gauss(x, amplitude, mu, sigma):
        return amplitude * np.exp(-(x - mu)**2 / (2.0 * sigma**2))

    x = np.arange(-20, 20)
    signal = gauss(x, 50.0, 0.0, 5.0)
    if random == 'yes':
        noise = np.random.normal(0, stdDev, np.size(signal))
    elif random == 'no':
        noise = np.array([1.57439344, 3.25840322, -2.5442794, 0.22431039,
                          2.66510799, -4.31975032, -2.38717285, -4.00639705,
                          -3.73459199, 7.390269, -6.76915999, 0.49809766,
                          9.03401679, -6.20273076, -1.08940482, 1.40441121,
                          10.27251409, -0.77355778, -4.31925802, -2.38433115,
                          5.43225019, 3.12188464, -2.53535334, -8.8736281,
                          -2.88545833, 2.2022186, 2.6992551, 4.13129327,
                          -1.13410893, -2.58064627, 1.01763015, 4.97684862,
                          -0.58867364, -1.41797943, -3.36680655, -7.25776942,
                          1.70606578, 2.10790981, 0.12066381, -0.16763699])
    else:
        raise ValueError("Error: random keyword must be yes or no!")

    stddevlist = stdDev * np.ones_like(x)
    signal_final = signal + noise
    wavelet_signal = wavelet_transform(signal_final, levels)
    reconstructed = wavelet_backTransform(wavelet_signal)
    denoised = cleanSignal(signal_final, stddevlist, levels,
                           sigmaCutoff=sigmaCutoff, epsilon=epsilon)

    import matplotlib.pyplot as plt
    plt.plot(x, signal_final, color='blue', label='signal+noise')
    plt.plot(x, reconstructed, 'r--', label='reconstructed')
    plt.plot(x, signal, 'b--', label='signal')
    plt.plot(x, denoised, color='black', ls='dashed', label='denoised')
    plt.plot(x, signal - denoised, 'k--', label='residual')
    plt.legend()
