# The data quality is used to set a data quality flag for each pixel.
# It serves two purposes:
# i) to mask out pixels that do not contain any data (e.g. in the case
# that not all spectra span the same wavelength range)
# ii) to flag any pixel affected by one or more anomalies (such as detector flaws, cosmic ray hits, ...).
#
# The Euro3D format uses the following convention. The higher the value, the more severe is the problem :
# Bit     Flag value      Quality condition
# 0       0               Good pixel - no flaw detected
# 1       1               Affected by telluric feature (corrected)
# 2       2               Affected by telluric feature (uncorrected)
# 3       4               Ghost/stray light at > 10% intensity level
# 4       8               Electronic pickup noise
# 5       16              Cosmic ray (removed)
# 6       32              Cosmic ray (unremoved)
# 7       64              low QE pixel (< 20% of the average sensitivity; e.g. defective CCD coating, vignetting...)
# 8       128             Calibration file defect (if pixel is flagged in any calibration file)
# 9       256             Hot pixel (> 5 sigma median dark)
# 10      512             Dark pixel (permanent CCD charge trap)
# 11      1024            Questionable pixel (lying above a charge trap which may have affected it)
# 12      2048            Detector potential well saturation (signal irrecoverable, but known to exceed the max. e-number)
# 13      4096            A/D converter saturation (signal irrecoverable, but known to exceed the A/D full scale signal)
# 14      8192            Permanent camera defect (such as blocked columns, dead pixels)
# 15      16384           Bad pixel not fitting into any other category
# 31      230             missing data (pixel was lost)
# 32      231             outside data range (outside of spectral range, inactive detector area, mosaic gap, ...)
#
#
# Since each condition is linked to a bit, several simultaneous conditions can be directly expressed
# as the sum of all corresponding flag values.
# Example: a pixel with calibration defects, known as a hot pixel and saturated would have a flag value of 128 + 256 + 4096 = 4480.

DQ_PIXEL = dict({'Good': 0, \
                 'TelluricCorrected': (1 << 0), \
                 'TelluricUnCorrected': (1 << 1), \
                 'GhostStrayLight': (1 << 2), \
                 'ElectronicNoise': (1 << 3), \
                 'CosmicRemoved': (1 << 4), \
                 'CosmicUnCorrected': (1 << 5), \
                 'LowQE': (1 << 6), \
                 'CalibrationFileDefect': (1 << 7), \
                 'HotPixel': (1 << 8), \
                 'Dark': (1 << 9), \
                 'Questionable': (1 << 10), \
                 'WellSaturation': (1 << 11), \
                 'ADSaturation': (1 << 12), \
                 'PermanentCameraDefect': (1 << 13), \
                 'BadOther': (1 << 14), \
                 'MissingData': (1 << 30), \
                 'OutsideDataRange': (1 << 31) })