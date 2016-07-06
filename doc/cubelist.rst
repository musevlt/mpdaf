**************************
Cube lists and combination
**************************

The `~mpdaf.obj.CubeList` and `~mpdaf.obj.CubeMosaic` classes manage
lists of cube FITS filenames, and allow one to combine cubes using
several methods (median, sigma clipping).


CubeList object format
======================

A cube object `O` consist of:

+------------+--------------------------------------------------------------------------------------------------+
| Component  | Description                                                                                      |
+============+==================================================================================================+
| O.files    | list of cubes FITS filenames                                                                     |
+------------+--------------------------------------------------------------------------------------------------+
| O.nfiles   | Number of files                                                                                  |
+------------+--------------------------------------------------------------------------------------------------+
| O.shape    | Array containing the 3 dimensions [nk,np,nq] of the cube: nk channels and np x nq spatial pixels |
+------------+--------------------------------------------------------------------------------------------------+
| O.fscale   | Scaling factor for the flux and variance values                                                  |
+------------+--------------------------------------------------------------------------------------------------+
| O.wcs      | Spatial world-coordinate information (`~mpdaf.obj.WCS` object)                                   |
+------------+--------------------------------------------------------------------------------------------------+
| O.wave     | Spectral world-coordinate information  (`~mpdaf.obj.WaveCoord` object)                           |
+------------+--------------------------------------------------------------------------------------------------+

All of the cubes must have the same dimensions, scaling factors and
world-coordinates.


Examples
========

54 exposures of MUSE were reduced with the muse standard pipeline. The result
was 54 data cubes with the same coordinates.  We create a CubeList object
containing these 54 cubes::

 >>> import glob
 >>> cubes = glob.glob('HDFS/DATACUBE-MUSE*')
 >>> from mpdaf.obj import CubeList
 >>> l = CubeList(cubes)

We merge these cubes into a single data-cube containing the median values of each voxel::

 >>> cube, expmap, statpix = l.median()
  1: HDFS/DATACUBE-MUSE.2014-07-26T05:08:59.778.fits
  2: HDFS/DATACUBE-MUSE.2014-07-28T05:28:57.448.fits
  3: HDFS/DATACUBE-MUSE.2014-08-01T05:08:41.575.fits
  4: HDFS/DATACUBE-MUSE.2014-08-04T07:11:55.173.fits
  5: HDFS/DATACUBE-MUSE.2014-08-03T07:35:28.847.fits
  6: HDFS/DATACUBE-MUSE.2014-07-26T08:29:45.020.fits
  7: HDFS/DATACUBE-MUSE.2014-07-29T07:08:17.444.fits
  8: HDFS/DATACUBE-MUSE.2014-07-27T04:58:50.964.fits
  9: HDFS/DATACUBE-MUSE.2014-08-04T07:43:43.175.fits
 10: HDFS/DATACUBE-MUSE.2014-08-03T07:00:03.821.fits
 11: HDFS/DATACUBE-MUSE.2014-07-28T04:16:36.813.fits
 12: HDFS/DATACUBE-MUSE.2014-07-29T04:18:53.033.fits
 13: HDFS/DATACUBE-MUSE.2014-08-01T08:36:16.992.fits
 14: HDFS/DATACUBE-MUSE.2014-07-27T07:16:47.912.fits
 15: HDFS/DATACUBE-MUSE.2014-08-01T06:15:10.080.fits
 16: HDFS/DATACUBE-MUSE.2014-07-28T08:25:06.578.fits
 17: HDFS/DATACUBE-MUSE.2014-07-26T06:16:21.251.fits
 18: HDFS/DATACUBE-MUSE.2014-07-29T05:58:59.010.fits
 19: HDFS/DATACUBE-MUSE.2014-07-30T05:40:22.949.fits
 20: HDFS/DATACUBE-MUSE.2014-07-28T06:06:39.475.fits
 21: HDFS/DATACUBE-MUSE.2014-08-01T06:53:24.986.fits
 22: HDFS/DATACUBE-MUSE.2014-07-26T05:44:30.479.fits
 23: HDFS/DATACUBE-MUSE.2014-08-03T08:07:17.846.fits
 24: HDFS/DATACUBE-MUSE.2014-07-27T06:08:00.081.fits
 25: HDFS/DATACUBE-MUSE.2014-08-03T04:45:53.021.fits
 26: HDFS/DATACUBE-MUSE.2014-07-29T05:27:09.308.fits
 27: HDFS/DATACUBE-MUSE.2014-07-28T06:38:29.498.fits
 28: HDFS/DATACUBE-MUSE.2014-07-30T04:56:39.095.fits
 29: HDFS/DATACUBE-MUSE.2014-07-29T07:43:07.457.fits
 30: HDFS/DATACUBE-MUSE.2014-08-01T04:36:53.069.fits
 31: HDFS/DATACUBE-MUSE.2014-07-28T07:15:07.546.fits
 32: HDFS/DATACUBE-MUSE.2014-07-27T04:22:08.024.fits
 33: HDFS/DATACUBE-MUSE.2014-07-27T03:50:16.539.fits
 34: HDFS/DATACUBE-MUSE.2014-07-26T07:57:54.539.fits
 35: HDFS/DATACUBE-MUSE.2014-07-26T04:37:08.541.fits
 36: HDFS/DATACUBE-MUSE.2014-08-01T05:43:21.522.fits
 37: HDFS/DATACUBE-MUSE.2014-08-04T08:18:56.903.fits
 38: HDFS/DATACUBE-MUSE.2014-07-27T06:39:51.353.fits
 39: HDFS/DATACUBE-MUSE.2014-08-04T08:50:44.862.fits
 40: HDFS/DATACUBE-MUSE.2014-07-26T07:22:57.504.fits
 41: HDFS/DATACUBE-MUSE.2014-08-03T05:21:36.275.fits
 42: HDFS/DATACUBE-MUSE.2014-08-01T07:28:34.951.fits
 43: HDFS/DATACUBE-MUSE.2014-08-03T04:10:18.845.fits
 44: HDFS/DATACUBE-MUSE.2014-07-28T07:46:57.935.fits
 45: HDFS/DATACUBE-MUSE.2014-07-28T04:57:07.504.fits
 46: HDFS/DATACUBE-MUSE.2014-08-01T04:00:03.074.fits
 47: HDFS/DATACUBE-MUSE.2014-08-03T06:28:12.994.fits
 48: HDFS/DATACUBE-MUSE.2014-08-03T05:53:26.091.fits
 49: HDFS/DATACUBE-MUSE.2014-07-30T06:12:13.628.fits
 50: HDFS/DATACUBE-MUSE.2014-07-29T06:36:25.596.fits
 51: HDFS/DATACUBE-MUSE.2014-07-26T06:51:07.539.fits
 52: HDFS/DATACUBE-MUSE.2014-07-29T04:50:42.874.fits
 53: HDFS/DATACUBE-MUSE.2014-08-01T08:00:23.925.fits
 54: HDFS/DATACUBE-MUSE.2014-07-27T05:30:41.420.fits
 nfiles: 54
 Read fits files
 naxes 326 331 3641
 Memory allocation
 Loop over all planes of the cube
 12/15/14 - 10:12AM 0.0%
 12/15/14 - 10:13AM 2.0%
 12/15/14 - 10:14AM 5.6%
 12/15/14 - 10:15AM 9.2%
 12/15/14 - 10:16AM 12.8%
 12/15/14 - 10:17AM 16.4%
 12/15/14 - 10:18AM 20.0%
 12/15/14 - 10:19AM 23.7%
 12/15/14 - 10:20AM 27.3%
 12/15/14 - 10:21AM 30.8%
 12/15/14 - 10:22AM 34.4%
 12/15/14 - 10:23AM 37.9%
 12/15/14 - 10:24AM 41.4%
 12/15/14 - 10:25AM 45.0%
 12/15/14 - 10:26AM 48.5%
 12/15/14 - 10:27AM 52.0%
 12/15/14 - 10:28AM 55.5%
 12/15/14 - 10:29AM 59.0%
 12/15/14 - 10:30AM 62.6%
 12/15/14 - 10:31AM 66.1%
 12/15/14 - 10:32AM 69.6%
 12/15/14 - 10:33AM 73.2%
 12/15/14 - 10:34AM 76.7%
 12/15/14 - 10:35AM 80.2%
 12/15/14 - 10:36AM 83.7%
 12/15/14 - 10:37AM 87.3%
 12/15/14 - 10:38AM 90.7%
 12/15/14 - 10:39AM 94.3%
 12/15/14 - 10:40AM 97.8%
 12/15/14 - 10:40AM 100%

In this example, the cube and expmap variables hold `mpdaf.obj.Cube`
objects that respectively contain the merged cube and an exposure map
data cube which counts the number of exposures used in the combination
of each pixel. The statpix variable holds an astropy.Table object of
pixel statistics.

This process is multithreaded. It needs 30 minutes on a machine with 32 cpus.

It is also possible to merge these cubes using a sigma clipped mean::

 >>> cube, expmap, statpix = l.combine(nmax=2, nclip=5.0, nstop=2, var='stat_mean')
  1: HDFS/DATACUBE-MUSE.2014-07-26T05:08:59.778.fits
  2: HDFS/DATACUBE-MUSE.2014-07-28T05:28:57.448.fits
  3: HDFS/DATACUBE-MUSE.2014-08-01T05:08:41.575.fits
  4: HDFS/DATACUBE-MUSE.2014-08-04T07:11:55.173.fits
  5: HDFS/DATACUBE-MUSE.2014-08-03T07:35:28.847.fits
  6: HDFS/DATACUBE-MUSE.2014-07-26T08:29:45.020.fits
  7: HDFS/DATACUBE-MUSE.2014-07-29T07:08:17.444.fits
  8: HDFS/DATACUBE-MUSE.2014-07-27T04:58:50.964.fits
  9: HDFS/DATACUBE-MUSE.2014-08-04T07:43:43.175.fits
 10: HDFS/DATACUBE-MUSE.2014-08-03T07:00:03.821.fits
 11: HDFS/DATACUBE-MUSE.2014-07-28T04:16:36.813.fits
 12: HDFS/DATACUBE-MUSE.2014-07-29T04:18:53.033.fits
 13: HDFS/DATACUBE-MUSE.2014-08-01T08:36:16.992.fits
 14: HDFS/DATACUBE-MUSE.2014-07-27T07:16:47.912.fits
 15: HDFS/DATACUBE-MUSE.2014-08-01T06:15:10.080.fits
 16: HDFS/DATACUBE-MUSE.2014-07-28T08:25:06.578.fits
 17: HDFS/DATACUBE-MUSE.2014-07-26T06:16:21.251.fits
 18: HDFS/DATACUBE-MUSE.2014-07-29T05:58:59.010.fits
 19: HDFS/DATACUBE-MUSE.2014-07-30T05:40:22.949.fits
 20: HDFS/DATACUBE-MUSE.2014-07-28T06:06:39.475.fits
 21: HDFS/DATACUBE-MUSE.2014-08-01T06:53:24.986.fits
 22: HDFS/DATACUBE-MUSE.2014-07-26T05:44:30.479.fits
 23: HDFS/DATACUBE-MUSE.2014-08-03T08:07:17.846.fits
 24: HDFS/DATACUBE-MUSE.2014-07-27T06:08:00.081.fits
 25: HDFS/DATACUBE-MUSE.2014-08-03T04:45:53.021.fits
 26: HDFS/DATACUBE-MUSE.2014-07-29T05:27:09.308.fits
 27: HDFS/DATACUBE-MUSE.2014-07-28T06:38:29.498.fits
 28: HDFS/DATACUBE-MUSE.2014-07-30T04:56:39.095.fits
 29: HDFS/DATACUBE-MUSE.2014-07-29T07:43:07.457.fits
 30: HDFS/DATACUBE-MUSE.2014-08-01T04:36:53.069.fits
 31: HDFS/DATACUBE-MUSE.2014-07-28T07:15:07.546.fits
 32: HDFS/DATACUBE-MUSE.2014-07-27T04:22:08.024.fits
 33: HDFS/DATACUBE-MUSE.2014-07-27T03:50:16.539.fits
 34: HDFS/DATACUBE-MUSE.2014-07-26T07:57:54.539.fits
 35: HDFS/DATACUBE-MUSE.2014-07-26T04:37:08.541.fits
 36: HDFS/DATACUBE-MUSE.2014-08-01T05:43:21.522.fits
 37: HDFS/DATACUBE-MUSE.2014-08-04T08:18:56.903.fits
 38: HDFS/DATACUBE-MUSE.2014-07-27T06:39:51.353.fits
 39: HDFS/DATACUBE-MUSE.2014-08-04T08:50:44.862.fits
 40: HDFS/DATACUBE-MUSE.2014-07-26T07:22:57.504.fits
 41: HDFS/DATACUBE-MUSE.2014-08-03T05:21:36.275.fits
 42: HDFS/DATACUBE-MUSE.2014-08-01T07:28:34.951.fits
 43: HDFS/DATACUBE-MUSE.2014-08-03T04:10:18.845.fits
 44: HDFS/DATACUBE-MUSE.2014-07-28T07:46:57.935.fits
 45: HDFS/DATACUBE-MUSE.2014-07-28T04:57:07.504.fits
 46: HDFS/DATACUBE-MUSE.2014-08-01T04:00:03.074.fits
 47: HDFS/DATACUBE-MUSE.2014-08-03T06:28:12.994.fits
 48: HDFS/DATACUBE-MUSE.2014-08-03T05:53:26.091.fits
 49: HDFS/DATACUBE-MUSE.2014-07-30T06:12:13.628.fits
 50: HDFS/DATACUBE-MUSE.2014-07-29T06:36:25.596.fits
 51: HDFS/DATACUBE-MUSE.2014-07-26T06:51:07.539.fits
 52: HDFS/DATACUBE-MUSE.2014-07-29T04:50:42.874.fits
 53: HDFS/DATACUBE-MUSE.2014-08-01T08:00:23.925.fits
 54: HDFS/DATACUBE-MUSE.2014-07-27T05:30:41.420.fits
 nfiles: 54
 Read fits files
 naxes 326 331 3641
 merging cube using mean with sigma clipping
 nmax = 2
 nclip = 5.000000
 nstop = 2
 Memory allocation
 Loop over all planes of the cube
 12/15/14 - 10:44AM 0.0%
 12/15/14 - 10:45AM 0.1%
 12/15/14 - 10:46AM 3.2%
 12/15/14 - 10:47AM 6.4%
 12/15/14 - 10:48AM 9.5%
 12/15/14 - 10:49AM 12.7%
 12/15/14 - 10:50AM 15.8%
 12/15/14 - 10:51AM 18.9%
 12/15/14 - 10:52AM 22.1%
 12/15/14 - 10:53AM 25.2%
 12/15/14 - 10:54AM 28.3%
 12/15/14 - 10:55AM 31.4%
 12/15/14 - 10:56AM 34.6%
 12/15/14 - 10:57AM 37.7%
 12/15/14 - 10:58AM 40.7%
 12/15/14 - 10:59AM 43.9%
 12/15/14 - 11:00AM 47.0%
 12/15/14 - 11:01AM 50.1%
 12/15/14 - 11:02AM 53.2%
 12/15/14 - 11:03AM 56.3%
 12/15/14 - 11:04AM 59.3%
 12/15/14 - 11:05AM 62.3%
 12/15/14 - 11:06AM 65.4%
 12/15/14 - 11:07AM 68.4%
 12/15/14 - 11:08AM 71.3%
 12/15/14 - 11:09AM 74.4%
 12/15/14 - 11:10AM 77.4%
 12/15/14 - 11:11AM 80.4%
 12/15/14 - 11:12AM 83.5%
 12/15/14 - 11:13AM 86.7%
 12/15/14 - 11:14AM 89.8%
 12/15/14 - 11:15AM 92.8%
 12/15/14 - 11:16AM 96.0%
 12/15/14 - 11:17AM 99.1%
 12/15/14 - 11:17AM 100%


The procedure prints the main parameters, which are:

 - nmax: The maximum number of clipping iterations
 - nclip: The number of sigma at which to clip.
 - nstop: If the number of none-rejected pixels is less than this number, the clipping iterations stop.

The resulting cube contains an additional extension for the variance.
Three options are available to compute the variance:

 - ``propagate``: The variance is the mean of the variances of the N individual
   exposures divided by N**2.

 - ``stat_mean``: The variance of each combined pixel is computed as the
   variance derived from the comparison of the N individual exposures divided
   by N-1.

 - ``stat_one``: The variance of each combined pixel is computed as the
   variance derived from the comparison of the N individual exposures.

`N` is the number of voxels left after the sigma-clipping.


Reference
=========

.. autosummary::

   mpdaf.obj.Cube
   mpdaf.obj.CubeList

- `mpdaf.obj.CubeList` is the constructor.
- `mpdaf.obj.CubeMosaic` is the constructor.
- `mpdaf.obj.CubeList.info` prints information.

Checking
--------

- `mpdaf.obj.CubeList.check_dim` checks if all cubes have same dimensions.
- `mpdaf.obj.CubeList.check_wcs` checks if all cubes have same world coordinates.
- `mpdaf.obj.CubeList.check_fscale` checks if all cubes have same scale factor.
- `mpdaf.obj.CubeList.check_compatibility` checks if all cubes are compatible.

Merging
-------

- `mpdaf.obj.CubeList.median` combines cubes in a single data cube using median.
- `mpdaf.obj.CubeList.combine` combines cubes in a single data cube using sigma clipped mean.
