Python interface for fusion software
************************************

.. warning::

   Draft version, still incomplete


HyperFusion is a C++ code for the Bayesian fusion of hyperspectral astronomical images.
fusion package lets the user to easily manage HyperFusion software :

  * create the fusion configuration file
  * launch the C++ software in background
  * stop / rerun
  * update inputs and rerun
  * fusion on sub-cubes
  * compute residuals 
  * view key parameters of each observations
  * view residuals
  * view progress 

Fusion class
============

This class interfaces the fusion software.

A Fusion object O consist of:

+------------+--------------------------------------------------------------------------------------------------------------------------------------+
| Component  | Description                                                                                                                          |
+============+======================================================================================================================================+
| O.obs      | List of :class:`Observation <mpdaf.fusion.Observation>` objects associated to the observations                                       |
+------------+--------------------------------------------------------------------------------------------------------------------------------------+
| O.sampling | :class:`HyperFSampling <mpdaf.fusion.HyperFSampling>` object describing the sampling of the reconstruction/fusion space              |
+------------+--------------------------------------------------------------------------------------------------------------------------------------+
| O.algo     | :class:`HyperFAlgo <mpdaf.fusion.HyperFAlgo>` object containing the fusion algorithm parameters                                      |
+------------+--------------------------------------------------------------------------------------------------------------------------------------+
| O.prior    | :class:`HyperFPrior <mpdaf.fusion.HyperFPrior>` object describing the quadratic prior model used during fusion estimation algorithms |
+------------+--------------------------------------------------------------------------------------------------------------------------------------+
| O.LSF      | :class:`LSFModel <mpdaf.fusion.LSFModel>` object containing the LSF model                                                            |
+------------+--------------------------------------------------------------------------------------------------------------------------------------+
| O.variance | :class:`Variance <mpdaf.fusion.Variance>` object containing parameters used to estimate variance                                     |
+------------+--------------------------------------------------------------------------------------------------------------------------------------+
| O.output   | Output path where the fusion outputs are stored                                                                                      |
+------------+--------------------------------------------------------------------------------------------------------------------------------------+

Observation class
=================

Observation class stores all parameter about an observation.

An Observation object O consist of:

+-------------+--------------------------------------------------------------------------------------------------------------+
| Component   | Description                                                                                                  |
+=============+==============================================================================================================+
| O.pixtable  | Pixtable filename associated to the observation                                                              |
+-------------+--------------------------------------------------------------------------------------------------------------+    
| O.type      | Type of the observation                                                                                      |
+-------------+--------------------------------------------------------------------------------------------------------------+    
| O.to_arcsec | Conversion factor between spatial unit in the pixtable and arcsec                                            |
+-------------+--------------------------------------------------------------------------------------------------------------+    
| O.to_nm     | Conversion factor between spectral unit in the pixtable and nm                                               |
+-------------+--------------------------------------------------------------------------------------------------------------+    
| O.dq_mask   | Data quality flag binary mask to determine whether a pixel must be taken into account in the pixtable or not |
+-------------+--------------------------------------------------------------------------------------------------------------+    
| O.FSF       | Object :class:`FSFModel <mpdaf.fusion.FSFModel>` containing the FSF model                                    |
+-------------+--------------------------------------------------------------------------------------------------------------+ 

Tutorials
=========

We can load the tutorial files with the command::

git clone http://urania1.univ-lyon1.fr/git/mpdaf_data.git


Tutorial 1
----------

In this tutorial we learn how to run a fusion of 3 observations.

First we create a Fusion object containing only 2 observations and using all default parameters. This object is created from a list of MUSE pixel tables.

  >>> from mpdaf.fusion import Fusion
  >>> pixtabs = ["small_pixtable1.fits", "small_pixtable2.fits"] # lists of observations
  >>> fus = Fusion(pixtables=pixtabs, output="test_fusion")
    Creating a new fusion session
    Reading pixtables ...
    0 small_pixtable1.fits
    1 small_pixtable2.fits
  >>> fus.info_obs()
    +----------------------+----------------------------+--------+---------+------+--------+---------+----------+
    |    Pixtable name     |         Date time          |   RA   |   Dec   | Rot  | seeing | airmass | exp time |
    +----------------------+----------------------------+--------+---------+------+--------+---------+----------+
    | small_pixtable1.fits | 2012-02-02 09:35:14.628682 | 19.999 | -30.000 | 0.00 |  0.73  |  1.173  |  3600.0  |
    | small_pixtable2.fits | 2012-02-02 09:40:52.022031 | 20.000 | -30.000 | 0.00 |  0.98  |  1.037  |  3600.0  |
    +----------------------+----------------------------+--------+---------+------+--------+---------+----------+
  
Now, we change the data quality mask of the first observation (corresponding to pixtable1.fits)::

  >>> from mpdaf.tools import euro3D
  >>> fus.obs[0].dq_mask = euro3D.DQ_PIXEL['CosmicUnCorrected'] + euro3D.DQ_PIXEL['HotPixel']
  
Note that the `euro3D package <../../../tools/build/html/euro3D.html>`_ is used to define the data quality binary mask.

It is also possible to remove and add observation::

  >>> fus.remove_observation(1)
   0 ../mpdaf_data/fusion/small_pixtable1.fits
  >>> from mpdaf.fusion import Observation
  >>> obs = Observation("small_pixtable2.fits", type = "MUSE_V1", dq_mask = euro3D.DQ_PIXEL['LowQE' ])
  >>> fus.add_observation(obs)
   0 ../mpdaf_data/fusion/small_pixtable1.fits
   1 ../mpdaf_data/fusion/small_pixtable2.fits
  >>> fus.info_obs()
   +----------------------+----------------------------+--------+---------+------+--------+---------+----------+
   |    Pixtable name     |         Date time          |   RA   |   Dec   | Rot  | seeing | airmass | exp time |
   +----------------------+----------------------------+--------+---------+------+--------+---------+----------+
   | small_pixtable1.fits | 2012-02-02 09:35:14.628682 | 19.999 | -30.000 | 0.00 |  0.73  |  1.173  |  3600.0  |
   | small_pixtable2.fits | 2012-02-02 09:40:52.022031 | 20.000 | -30.000 | 0.00 |  0.98  |  1.037  |  3600.0  |
   +----------------------+----------------------------+--------+---------+------+--------+---------+----------+
  
  
Before running the fusion, we re-estimate the hyperparameter omega for smoothness quadratic prior from a cube having the same size, pixel size, content and resolution as in the expected Bayesian fusion result:

  >>> fus.prior.compute_omega("cube.fits")
  >>> print fus.prior.omega_xy
   0.00143970991121
  
Then, we create the HyperFusion configuration file and lauch the HyperFusion code::

  >>> fus.create_config_file()
   Writing the fusion configuration file: test_fusion/fusion.cfg
  >>> fus.run_fit()
   start fusion_fit, use .info() to have the progress
  >>> fus.info()
   [fusion_LSF] Pre-sampling of LSF kernels (output in test_fusion/hyperf_res/LSF.fits) from the configuration file test_fusion/fusion.cfg...
  >>> fus.info()
   [fusion_LSF] Pre-sampling of LSF kernels (output in test_fusion/hyperf_res/LSF.fits) from the configuration file test_fusion/fusion.cfg...
   [fusion_FSF] Total number of observations to be processed: 2
   [fusion_FSF] Pre-sampling of FSF kernels (output in test_fusion/hyperf_res/Y1_FSF.fits) from the configuration section "observation_1" in test_fusion/fusion.cfg...
   [fusion_FSF] Pre-sampling of FSF kernels (output in test_fusion/hyperf_res/Y2_FSF.fits) from the configuration section "observation_2" in test_fusion/fusion.cfg...
   [fusion_fit] Total number of observations: 2
   [fusion_fit] Adding the observation "observation_1" to the fusion pipeline...
  >>> fus.info()
   [fusion_LSF] Pre-sampling of LSF kernels (output in test_fusion/hyperf_res/LSF.fits) from the configuration file test_fusion/fusion.cfg...
   [fusion_FSF] Total number of observations to be processed: 2
   [fusion_FSF] Pre-sampling of FSF kernels (output in test_fusion/hyperf_res/Y1_FSF.fits) from the configuration section "observation_1" in test_fusion/fusion.cfg...
   [fusion_FSF] Pre-sampling of FSF kernels (output in test_fusion/hyperf_res/Y2_FSF.fits) from the configuration section "observation_2" in test_fusion/fusion.cfg...
   [fusion_fit] Total number of observations: 2
   [fusion_fit] Adding the observation "observation_1" to the fusion pipeline...
   [fusion_fit] Adding the observation "observation_2" to the fusion pipeline...
   [fusion_fit] Bayesian fusion
         Checking observations...
         Computation of the initialization image (saved in test_fusion/hyperf_res/L_init.fits)...
         Undefined pixels have been found in the computed initialization image and are now replaced with the image mean...
         357211 undefined pixels have been updated in the computed initialization image
         Conjugate gradient algorithm (initialization)...
         Starting minimization. Maximum iteration: 75 - Stop criterion: 28.089402356755481804
         Iteration       Date    Current/Stop criterion
         1       Wed Sep 12 12:35:12 2012        280894.03066353488248/28.089402356755481804
         2       Wed Sep 12 12:35:18 2012        449255.27323976485059/28.089402356755481804
         3       Wed Sep 12 12:35:21 2012        244723.25024099647999/28.089402356755481804
         4       Wed Sep 12 12:35:25 2012        234089.12179201008985/28.089402356755481804



It is now possible to quit the ipython session.


Tutorial 2
----------

In this second tutorial we reconnect the an old fusion session from a new ipython terminal::

  >>> from mpdaf.fusion import Fusion
  >>> fus = Fusion()
  >>> fus.info()
   [fusion_LSF] Pre-sampling of LSF kernels (output in test_fusion/hyperf_res/LSF.fits) from the configuration file test_fusion/fusion.cfg...
   [fusion_FSF] Total number of observations to be processed: 2
   [fusion_FSF] Pre-sampling of FSF kernels (output in test_fusion/hyperf_res/Y1_FSF.fits) from the configuration section "observation_1" in test_fusion/fusion.cfg...
   [fusion_FSF] Pre-sampling of FSF kernels (output in test_fusion/hyperf_res/Y2_FSF.fits) from the configuration section "observation_2" in test_fusion/fusion.cfg...
   [fusion_fit] Total number of observations: 2
   [fusion_fit] Adding the observation "observation_1" to the fusion pipeline...
   [fusion_fit] Adding the observation "observation_2" to the fusion pipeline...
   [fusion_fit] Bayesian fusion
         Checking observations...
         Computation of the initialization image (saved in test_fusion/hyperf_res/L_init.fits)...
         Undefined pixels have been found in the computed initialization image and are now replaced with the image mean...
         357211 undefined pixels have been updated in the computed initialization image
         Conjugate gradient algorithm (initialization)...
         Starting minimization. Maximum iteration: 75 - Stop criterion: 28.089402356755481804
         Iteration       Date    Current/Stop criterion
         1       Wed Sep 12 12:35:12 2012        280894.03066353488248/28.089402356755481804
         
         ...
         
         58      Wed Sep 12 12:38:40 2012        268.29005405907611248/28.089402356755481804
  >>> fus.stop()
  >>> from mpdaf.fusion import remove_session
  >>> remove_session()
   Please choose an id corresponding to the session to remove
   0 - 2012-09-12 12:38:31.000664 - output:/home/piqueras/test_fusion 
  >>> 0
  >>> remove_session()
   no existing fusion session

  

Reference
=========

Create a Fusion object
----------------------

:func:`mpdaf.fusion.Fusion <mpdaf.fusion.Fusion>` is used to create/reconnect a fusion session.

:func:`mpdaf.fusion.HyperFSampling <mpdaf.fusion.HyperFSampling>` is the HyperFSampling constructor.

:func:`mpdaf.fusion.HyperFAlgo <mpdaf.fusion.HyperFAlgo>` is the HyperFAlgo constructor.

:func:`mpdaf.fusion.HyperFPrior <mpdaf.fusion.HyperFPrior>` is the HyperFPrior constructor.

:func:`mpdaf.fusion.LSFModel <mpdaf.fusion.LSFModel>` is the LSFModel constructor.

:func:`mpdaf.fusion.Variance <mpdaf.fusion.Variance>` is the Variance constructor.


Fusion method
-------------

:func:`mpdaf.fusion.Fusion.add_observation <mpdaf.fusion.Fusion.add_observation>` adds an observation.

:func:`mpdaf.fusion.Fusion.create_config_file <mpdaf.fusion.Fusion.create_config_file>` creates the HyperFusion configuration file.

:func:`mpdaf.fusion.Fusion.info <mpdaf.fusion.Fusion.info>` prints the fusion progress (or error).

:func:`mpdaf.fusion.Fusion.info_obs <mpdaf.fusion.Fusion.info_obs>` prints observations parameters.

:func:`mpdaf.fusion.Fusion.remove_observation <mpdaf.fusion.Fusion.remove_observation>` removes an observation.

:func:`mpdaf.fusion.Fusion.run_fit <mpdaf.fusion.Fusion.run_fit>` runs the Bayesian fusion of observations.

:func:`mpdaf.fusion.Fusion.run_residual <mpdaf.fusion.Fusion.run_residual>` runs the computation of fusion residuals.

:func:`mpdaf.fusion.Fusion.run_variance <mpdaf.fusion.Fusion.run_variance>` runs the computation of fusion variance.

:func:`mpdaf.fusion.Fusion.stop <mpdaf.fusion.Fusion.stop>` stops the fusion process.

:func:`mpdaf.fusion.HyperFPrior.compute_omega <mpdaf.fusion.HyperFPrior.compute_omega>` estimates the hyperparameter omega for smoothness quadratic prior. 


Create a Observation object
---------------------------

:func:`mpdaf.fusion.Observation <mpdaf.fusion.Observation>` is the Observation constructor.

:func:`mpdaf.fusion.FSFModel <mpdaf.fusion.FSFModel>` is the FSFModel constructor.


Remove a Fusion session
-----------------------

:func:`mpdaf.fusion.remove_session <mpdaf.fusion.remove_session>` lets the user to remove a fusion session.

