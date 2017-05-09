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




import numpy
import scipy.ndimage.filters
import matplotlib.pyplot as plt








#### Function definitions

# Test if the chosen levels are too many for our signal (sampling condition)
def test_levels(signal, levels):
    signal = numpy.array(signal) # Tranform for speed to numpy array
    h_coefficients_list = numpy.array([1/16.0, 1/4.0, 3/8.0, 1/4.0, 1/16.0]) # See book by Starck
    h_length = len(h_coefficients_list)
    signalSize = signal.size
    if levels > numpy.log2((signalSize-1.0)/(h_length-1.0)): # If the (level+1)-th h array is larger than the signal array, the final smoothing function for level number of wavelets 
							     # is wider than the signal range itself, which has to be avoided. Thus h_length + (2**level -1)*(h_length-1) > signalSize 
							     # must be avoided (the second term = number of 0s in h array)
        levels = int(numpy.floor(numpy.log2((signalSize-1.0)/(h_length-1.0))))
        print("\nAttention: The chosen number of levels exceeds the number allowed (sampling condition)."
              + " Thus it was automatically set to the maximum number allowed = " + str(levels) + "\n")
    return levels




# Transform a signal into wavelet space
def wavelet_transform(signal, levels): # levels is the number of wavelet levels we use
    signal = numpy.array(signal) # Tranform for speed to numpy array
    for i,element in enumerate(signal):
        if numpy.isnan(element) == True:
            signal[i] = 0.0			# If we have missing values, set the signal to 0 
    signalSize = signal.size

    # Test if the chosen levels are too many for our signal (sampling condition)
    levels = test_levels(signal, levels)

    wavelet_coefficients = []
    h_coefficients_list = numpy.array([1/16.0, 1/4.0, 3/8.0, 1/4.0, 1/16.0]) # See book by Starck
    h_length = len(h_coefficients_list)
    h_coefficients = [] # We build the list of h coefficient arrays. The spacing between the values is 2**i_level -1 (see Starck book)
    for i in range(0,levels):
        h_array = [0 for index in range(((2**i)-1)*(h_length-1) + h_length)] # Array long enough to store the spacing and the values
        counter = 0
        for index in range(len(h_array)):
            if index%(2**i) == 0: # Now we enter the values and ensure that the spacing between them is correct
                h_array[index] = h_coefficients_list[counter]
                counter += 1
        h_coefficients.append(numpy.array(h_array)) # Append and use numpy for speed
    for i in range(0,levels): # We need "levels" (e.g. 10) wavelet coefficient arrays
        signal_convolved = scipy.ndimage.filters.convolve1d(signal*1.0,h_coefficients[i]*1.0, mode='wrap') # Depending on the level, we have different h arrays for convolution. We need *1.0 to convert to float
        signal_convolved_2 = scipy.ndimage.filters.convolve1d(signal_convolved*1.0,h_coefficients[i]*1.0, mode='wrap')
        wavelet_coefficients.append(signal-signal_convolved_2) # Calculate the wavelet coefficients and add them to list
        signal = signal_convolved # Overwrite signal with convolved signal for the next iteration
    scaling_coefficients = signal_convolved # Get the coefficients for the scaling function
    waveletTransform_coefficients = wavelet_coefficients # Merge all coefficients into 1 list
    waveletTransform_coefficients.append(scaling_coefficients)
    return numpy.array(waveletTransform_coefficients)

    


# Transform from wavelet to real space
def wavelet_backTransform(waveletTransform_coefficients):
    levels = numpy.array(waveletTransform_coefficients).shape[0]-1 # We have array number = levels + 1 (due to the smoothing coefficients)
    h_coefficients_list = numpy.array([1/16.0, 1/4.0, 3/8.0, 1/4.0, 1/16.0]) # See book by Starck
    h_length = len(h_coefficients_list)
    h_coefficients = [] # We build the list of h coefficient arrays. The spacing between the values is 2**i_level -1 (see Starck book)
    for i in range(0,levels):
        h_array = [0 for index in range(((2**i)-1)*(h_length-1) + h_length)] # Array long enough to store the spacing and the values
        counter = 0
        for index in range(len(h_array)):
            if index%(2**i) == 0: # Now we enter the values and ensure that the spacing between them is correct
                h_array[index] = h_coefficients_list[counter]
                counter += 1
        h_coefficients.append(numpy.array(h_array)) # Append and use numpy for speed
    # We convolve each wavelet and the smoothing function with the corresponding h array to re-transform into image space
    # We have to go in reverse order 
    temporary_signal = waveletTransform_coefficients[levels] # Initialize the coefficients for the convolution 
    for i in range(0,levels):
        signal_convolved = scipy.ndimage.filters.convolve1d(temporary_signal*1.0,h_coefficients[levels-1-i]*1.0, mode='wrap') # levels-1 because python begins counting at 0
        temporary_signal = signal_convolved + waveletTransform_coefficients[levels-1-i] # We add the corresponding coefficients for the next convolution
    signal = temporary_signal
    return numpy.array(signal)




# Filter an input signal by using the 1 standard deviation noise estimate and wavelets
# epsilon is the iteration-stop parameter for extracting signal from the
# residual signal
def cleanSignal(signal, noise, levels, sigmaCutoff = 5.0, epsilon = 0.05):
    signal = numpy.array(signal) # Tranform for speed to numpy array
    noise = numpy.array(noise)
    for i,element in enumerate(signal):
        if numpy.isnan(element) == True:
            signal[i] = 0.0			# If we have missing values, set the signal to 0 and the noise to 100000 standard deviations to 
            noise[i] = 100000.0*numpy.std(signal) # downweigh the signal
    for i,element in enumerate(noise):
        if numpy.isnan(element) == True:
            signal[i] = 0.0			# If we have missing values, set the signal to 0 and the noise to 100000 standard deviations to 
            noise[i] = 100000.0*numpy.std(signal) # downweigh the signal

    sigmaCutoff = float(sigmaCutoff)
    epsilon = float(epsilon)
    signalSize = signal.size

    # Test if the chosen levels are too many for our signal (sampling condition)
    levels = test_levels(signal, levels)

    diracSignal = [0.0 for x in range(signalSize)]
    diracSignal[signalSize/2] = 1.0 		# We have a dirac function, 0 everywhere except 1 in the central pixel (pixel value = integral over signal in area 
    diracSignal = numpy.array(diracSignal)	# of pixel = 1 for a dirac function)
    dirac_coefficients = wavelet_transform(diracSignal, levels) # We calculate the wavelet coefficients. We do not need the coefficients for the "smoothing function", thus
    dirac_coefficients = dirac_coefficients[0:dirac_coefficients.shape[0]-1] # we delete the last array (.shape[0] gives us the number of arrays)
    variance_coefficients = []			# We calculate the wavelet coefficients for the noise (= 1 sigma**2)
    for i,array in enumerate(dirac_coefficients):
        variance_array = scipy.ndimage.filters.convolve1d(dirac_coefficients[i]**2.0, noise**2.0, mode='wrap')   # We convolve the squared array elements to obtain the variance
        variance_coefficients.append(variance_array) 
    noise_coefficients = numpy.array(variance_coefficients)**(1/2.0)	# Get standard deviation array of arrays
    
    
    # Create the multiresolution support
    signal_coefficients = wavelet_transform(signal,levels)
    M_support = numpy.zeros((levels+1,signalSize)) # The support is 0 for every non-significant wavelet and 1 for every significant wavelet/smoothing function coefficient
    for i,array in enumerate(noise_coefficients):
        for j,element in enumerate(array):
            if i == 0:
                if numpy.abs(signal_coefficients[i,j]) < numpy.abs((sigmaCutoff+1.0)*element):	# If the coefficient is less than (x+1)*sigma detection, we consider it
                    M_support[i,j] = 0.0							# as noise. We increase the threshold for i = 0 = high frequencies,
                else:										# as typically noise has high frequencies and signal has lower frequencies
                    M_support[i,j] = 1.0
            else:
                if numpy.abs(signal_coefficients[i,j]) < numpy.abs(sigmaCutoff*element):	
                    M_support[i,j] = 0.0							
                else:										
                    M_support[i,j] = 1.0
    M_support[levels,:] = 1.0		# The smoothing function coefficients are always all significant
 					


    # Do the cleaning
    cleaned_signal = numpy.zeros(signalSize)	# Initialize clean signal
    residual_signal_sigma_old = 0.0		# Initialize the standard deviation of the residual signal
    residual_signal = signal			# Initialize the residual signal
    residual_signal_sigma = numpy.std(residual_signal)


    # We can still extract signal from the residual. We do this here after the first iteration until the epsilon condition is false and add it to the already extracted signal
    while numpy.abs((residual_signal_sigma_old - residual_signal_sigma)/residual_signal_sigma) > epsilon: # We continue to extract until the standard deviation doesn't change too much
        residual_coefficients = wavelet_transform(residual_signal, levels)

        for i, array in enumerate(residual_coefficients):	# We clean the non-significant wavelets
            for j, element in enumerate(array):
                if M_support[i,j] == 0:
                    residual_coefficients[i,j] = 0.0
        
        cleaned_signal = cleaned_signal + wavelet_backTransform(residual_coefficients)
        residual_signal = signal - cleaned_signal
        residual_signal_sigma_old = residual_signal_sigma
        residual_signal_sigma = numpy.std(residual_signal)        

    return cleaned_signal




# test function for the functions defined above
def test(stdDev=5.0, random='yes', levels=3, sigmaCutoff=5.0, epsilon=0.05):
    def gauss(x,amplitude, mu,sigma):
        return amplitude*numpy.exp(-(x-mu)**2 / (2.0*sigma**2))
    signal = [gauss(x,50.0,0.0,5.0) for x in range(-20,20)]
    xRange = [x for x in range(-20,20)]
    if random == 'yes':
        noise = numpy.random.normal(0,stdDev,numpy.size(signal))
    elif random == 'no':
        noise = numpy.array([  1.57439344,   3.25840322,  -2.5442794 ,   0.22431039,
         2.66510799,  -4.31975032,  -2.38717285,  -4.00639705,
        -3.73459199,   7.390269  ,  -6.76915999,   0.49809766,
         9.03401679,  -6.20273076,  -1.08940482,   1.40441121,
        10.27251409,  -0.77355778,  -4.31925802,  -2.38433115,
         5.43225019,   3.12188464,  -2.53535334,  -8.8736281 ,
        -2.88545833,   2.2022186 ,   2.6992551 ,   4.13129327,
        -1.13410893,  -2.58064627,   1.01763015,   4.97684862,
        -0.58867364,  -1.41797943,  -3.36680655,  -7.25776942,
         1.70606578,   2.10790981,   0.12066381,  -0.16763699])
    else:
        print "Error: random keyword must be yes or no!"
    stdDevList = [stdDev for x in range(-20,20)]
    signal_final = signal + noise
    wavelet_signal = wavelet_transform(signal_final,levels)
    reconstructed_signal = wavelet_backTransform(wavelet_signal)
    deNoisedSignal = cleanSignal(signal_final, stdDevList, levels, sigmaCutoff=sigmaCutoff, epsilon=epsilon)
    plt.plot(xRange,signal_final,color='blue')
    plt.plot(xRange, reconstructed_signal, 'r--')
    plt.plot(xRange, signal, 'b--')
    plt.plot(xRange, deNoisedSignal, color='black', ls='dashed')
    plt.show()







