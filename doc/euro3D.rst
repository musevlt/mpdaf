:orphan:

.. _euro3D:

**************************************
Python interface for euro3D convention
**************************************

The Euro3D Research Training Network introduces a common data format for
Integral Field Spectroscopy.

`~mpdaf.tools.euro3D` package contains Euro3D conventions about the pixel data quality.


The data quality is used to set a data quality flag for each pixel.
It serves two purposes:

 * to mask out pixels that do not contain any data (e.g. in the case that not
   all spectra span the same wavelength range)

 * to flag any pixel affected by one or more anomalies (such as detector flaws,
   cosmic ray hits, ...).

The Euro3D format links a condition to a bit and to a flag value. The higher
the value, the more severe is the problem.

euro3D.DQ_PIXEL dictionary contains these condition/value pairs:

+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| Key in DQ_PIXEL         | Flag value | Bit | Quality condition                                                                                |
+=========================+============+=====+==================================================================================================+
| 'Good'                  |    0       | 0   | Good pixel - no flaw detected                                                                    |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'TelluricCorrected'     |    1       | 1   | Affected by telluric feature (corrected)                                                         |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'TelluricUnCorrected'   |    2       | 2   | Affected by telluric feature (uncorrected)                                                       |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'GhostStrayLight'       |    4       | 3   | Ghost/stray light at > 10% intensity level                                                       |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'ElectronicNoise'       |    8       | 4   | Electronic pickup noise                                                                          |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'CosmicRemoved'         |    16      | 5   | Cosmic ray (removed)                                                                             |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'CosmicUnCorrected'     |    32      | 6   | Cosmic ray (unremoved)                                                                           |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'LowQE'                 |    64      | 7   | Low QE pixel (< 20% of the average sensitivity; e.g. defective CCD coating, vignetting...)       |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'CalibrationFileDefect' |    128     | 8   | Calibration file defect (if pixel is flagged in any calibration file)                            |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'HotPixel'              |    256     | 9   | Hot pixel (> 5 sigma median dark)                                                                |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'Dark'                  |    512     | 10  | Dark pixel (permanent CCD charge trap)                                                           |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'Questionable'          |    1024    | 11  | Questionable pixel (lying above a charge trap which may have affected it)                        |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'WellSaturation'        |    2048    | 12  | Detector potential well saturation (signal irrecoverable, but known to exceed the max. e-number) |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'ADSaturation'          |    4096    | 13  | A/D converter saturation (signal irrecoverable, but known to exceed the A/D full scale signal)   |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'PermanentCameraDefect' |    8192    | 14  | Permanent camera defect (such as blocked columns, dead pixels)                                   |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'BadOther'              |    16384   | 15  | Bad pixel not fitting into any other category                                                    |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'MissingData'           |    230     | 31  | Missing data (pixel was lost)                                                                    |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+
| 'OutsideDataRange'      |    231     | 32  | Outside data range (outside of spectral range, inactive detector area, mosaic gap, ...)          |
+-------------------------+------------+-----+--------------------------------------------------------------------------------------------------+


Since each condition is linked to a bit, several simultaneous conditions can be directly expressed as the sum of all corresponding flag values.
for example, a pixel with calibration defects, known as a hot pixel and saturated would have a flag value of 128 + 256 + 4096 = 4480:

.. ipython::

  In [1]: from mpdaf.tools import euro3D
  
  In [2]: flag = euro3D.DQ_PIXEL['CalibrationFileDefect'] + euro3D.DQ_PIXEL['HotPixel'] + euro3D.DQ_PIXEL['ADSaturation']
  
  In [3]: print(flag)
