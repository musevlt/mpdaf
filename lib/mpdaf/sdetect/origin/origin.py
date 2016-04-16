"""ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

This software has been developped by Carole Clastres under the supervision of
David Mary (Lagrange institute, University of Nice) and ported to python by
Laure Piqueras (CRAL).

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL). Please contact
Carole for more info at carole.clastres@univ-lyon1.fr

origin.py contains an oriented-object interface to run the ORIGIN software
"""

import astropy.units as u
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy.io import loadmat

from ...obj import Cube, Image, Spectrum
from .lib_origin import Compute_PSF, Spatial_Segmentation, \
                        Compute_PCA_SubCube, Compute_Number_Eigenvectors_Zone, \
                        Compute_Proj_Eigenvector_Zone, Correlation_GLR_test, \
                        Compute_pval_correl_zone, Compute_pval_channel_Zone, \
                        Compute_pval_final, Compute_Connected_Voxel, \
                        Compute_Referent_Voxel, Narrow_Band_Test, \
                        Narrow_Band_Threshold, Estimation_Line, \
                        Spatial_Merging_Circle, Spectral_Merging, \
                        Add_radec_to_Cat, Construct_Object_Catalogue

class ORIGIN(object):
    """ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

       This software has been developped by Carole Clastres under the
       supervision of David Mary (Lagrange institute, University of Nice).

       The project is funded by the ERC MUSICOS (Roland Bacon, CRAL).
       Please contact Carole for more info at carole.clastres@univ-lyon1.fr

       An Origin object is mainly composed by:
        - cube data (raw data and covariance)
        - 1D dictionary of spectral profiles
        - MUSE PSF

       The class contains the methods that compose the ORIGIN software.

        Attributes
        ----------
        filename      : string
                        Cube FITS file name.
        cube_raw      : array (Nz, Ny, Nx)
                        Raw data.
        var           : array (Nz, Ny, Nx)
                        Covariance.
        Nx            : integer
                        Number of columns
        Ny            : integer
                        Number of rows
        Nz            : int
                        Number of spectral channels
        wcs           : `mpdaf.obj.WCS`
                        RA-DEC coordinates.
        wave          : `mpdaf.obj.WaveCoord`
                        Spectral coordinates.
        intx          : array
                        Limits in pixels of the columns for each zone
        inty          : array
                        Limits in pixels of the rows for each zone
        Edge_xmin     : int
                        Minimum limits along the x-axis in pixel
                        of the data cube taken to compute p-values
        Edge_xmax     : int
                        Maximum limits along the x-axis in pixel
                        of the data cube taken to compute p-values
        Edge_ymin     : int
                        Minimum limits along the y-axis in pixel
                        of the data cube taken to compute p-values
        Edge_ymax     : int
                        Maximum limits along the y-axis in pixel
                        of the data cube taken to compute p-values
        profiles      : array
                        Dictionary of spectral profiles to test
        FWHM_profiles : array
                        FWHM of the profiles in pixels.
        PSF           : array (Nz, Nfsf, Nfsf)
                        MUSE PSF
        FWHM_PSF      : float
                        Mean of the fwhm of the PSF in pixel
    """

    def __init__(self, cube, NbSubcube, Edge_xmin=None, Edge_xmax=None,
                 Edge_ymin=None, Edge_ymax=None, profiles=None, FWHM_profiles=None, PSF=None,
                 FWHM_PSF=None):
        """Create a ORIGIN object.

        An Origin object is composed by:
        - cube data (raw data and covariance)
        - 1D dictionary of spectral profiles
        - MUSE PSF
        - parameters used to segment the cube in different zones.


        Parameters
        ----------
        cube        : string
                      Cube FITS file name.
        NbSubcube   : integer
                      Number of sub-cubes for the spatial segmentation
        Edge_xmin   : int
                      Minimum limits along the x-axis in pixel
                      of the data cube taken to compute p-values
        Edge_xmax   : int
                      Maximum limits along the x-axis in pixel
                      of the data cube taken to compute p-values
        Edge_ymin   : int
                      Minimum limits along the y-axis in pixel
                      of the data cube taken to compute p-values
        Edge_ymax   : int
                      Maximum limits along the y-axis in pixel
                      of the data cube taken to compute p-values
        profiles    : array (Size_profile, N_profile)
                      Dictionary of spectral profiles
                      If None, a default dictionary of 20 profiles is used.
        FWHM_profiles : array (N_profile)
                        FWHM of the profiles in pixels.
        PSF         : string
                      Cube FITS filename containing a MUSE PSF per wavelength.
                      If None, PSF are computed with a Moffat function
                      (13x13 pixels, beta=2.6, fwhm1=0.76, fwhm2=0.66,
                      lambda1=4750, lambda2=7000)
        FWHM_PSF    : array (Nz)
                      FWHM of the PSFs in pixels.
        """
        self._logger = logging.getLogger(__name__)
        # create parameters dictionary
        self.param = {}
        self.param['cubename'] = cube
        self.param['nbsubcube'] = NbSubcube
        if Edge_xmin is not None:
            self.param['edgecube'] = [Edge_xmin,Edge_xmax,Edge_ymin,Edge_ymax]
        self.param['PSF'] = PSF 
        # Read cube
        self._logger.info('ORIGIN - Read the Data Cube')
        self.filename = cube
        cub = Cube(self.filename)
        # Raw data cube
        # Set to 0 the Nan
        self.cube_raw = cub.data.filled(fill_value=0)
        # variance
        self.var = cub.var
        # RA-DEC coordinates
        self.wcs = cub.wcs
        # spectral coordinates
        self.wave = cub.wave

        #Dimensions
        self.Nz, self.Ny, self.Nx = cub.shape

        del cub

        # Set to Inf the Nana
        self.var[np.isnan(self.var)] = np.inf

        self.NbSubcube = NbSubcube

        if Edge_xmin is None:
            self.Edge_xmin = 0
        else:
            self.Edge_xmin = Edge_xmin
        if Edge_xmax is None:
            self.Edge_xmax = self.Nx
        else:
            self.Edge_xmax = Edge_xmax
        if Edge_ymin is None:
            self.Edge_ymin = 0
        else:
            self.Edge_ymin = Edge_ymin
        if Edge_ymax is None:
            self.Edge_ymax = self.Ny
        else:
            self.Edge_ymax = Edge_ymax

        # Dictionary of spectral profile
        if profiles is None or FWHM_profiles is None:
            self._logger.info('ORIGIN - Load dictionary of spectral profile')
            DIR = os.path.dirname(__file__)
            self.profiles =  loadmat(DIR+'/Dico_FWHM_2_12.mat')['Dico']
            self.FWHM_profiles = np.linspace(2, 12, 20) #pixels
        else:
            self.profiles = profiles
            self.FWHM_profiles = FWHM_profiles


        # 1 pixel in arcsec
        step_arcsec = self.wcs.get_step(unit=u.arcsec)[0]
        if PSF is None or FWHM_PSF is None:
            self._logger.info('ORIGIN - Compute PSF')
            self.PSF, fwhm_pix ,fwhm_arcsec = Compute_PSF(self.wave, self.Nz,
                                                Nfsf=13, beta=2.6,
                                                fwhm1=0.76, fwhm2=0.66,
                                                lambda1=4750, lambda2=7000,
                                                step_arcsec=step_arcsec)
            # mean of the fwhm of the FSF in pixel
            self.FWHM_PSF = np.mean(fwhm_arcsec)/step_arcsec
        else:
            cubePSF = Cube(PSF)
            if cubePSF.shape[0] != self.Nz:
                raise IOError('PSF and data cube have not the same dimensions along the spectral axis.')
            if np.isclose(cubePSF.wcs.get_step(unit=u.arcsec)[0], step_arcsec):
                raise IOError('PSF and data cube have not the same pixel sizes.')

            self.PSF = cubePSF.data.data
            # mean of the fwhm of the FSF in pixel
            self.FWHM_PSF = np.mean(FWHM_PSF)

        #Spatial segmentation
        self._logger.info('ORIGIN - Spatial segmentation')
        self.inty, self.intx = Spatial_Segmentation(self.Nx, self.Ny,
                                                    NbSubcube)

    def compute_PCA(self, r0=0.67, plot_lambda=False):
        """ Loop on each zone of the data cube and compute the PCA,
        the number of eigenvectors to keep for the projection
        (with a linear regression and its associated determination
        coefficient) and return the projection of the data
        in the original basis keeping the desired number eigenvalues.

        Parameters
        ----------
        r0          : float
                      Coefficient of determination for projection during PCA
        plot_lambda : bool
                      If True, plot the eigenvalues and the separation point.

        Returns
        -------
        cube_faint : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the lower
                     eigenvalues of the data cube
                     (representing the faint signal)
        cube_cont  : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the higher
                     eigenvalues of the data cube
                     (representing the continuum)
        """
        # save paaremeters values in object
        self.param['r0PCA'] = r0
        # Weigthed data cube
        cube_std = self.cube_raw / np.sqrt(self.var)
        # Compute PCA results
        self._logger.info('ORIGIN - Compute the PCA on each zone')
        A, V, eig_val, nx, ny, nz = Compute_PCA_SubCube(self.NbSubcube,
                                                        cube_std,
                                                        self.intx, self.inty,
                                                        self.Edge_xmin,
                                                        self.Edge_xmax,
                                                        self.Edge_ymin,
                                                        self.Edge_ymax)

        # Number of eigenvectors for each zone
        # Parameter set to 1 if we want to plot the results
        # Parameters for projection during PCA
        self._logger.info('ORIGIN - Compute the number of eigenvectors to keep for the projection')
        list_r0  = np.resize(r0, self.NbSubcube**2)
        nbkeep = Compute_Number_Eigenvectors_Zone(self.NbSubcube, list_r0, eig_val,
                                              plot_lambda)
        # Adaptive projection of the cube on the eigenvectors
        self._logger.info('ORIGIN - Adaptive projection of the cube on the eigenvectors')
        cube_faint, cube_cont = Compute_Proj_Eigenvector_Zone(nbkeep,
                                                              self.NbSubcube,
                                                              self.Nx,
                                                              self.Ny,
                                                              self.Nz,
                                                              A, V,
                                                              nx, ny, nz,
                                                              self.inty, self.intx)
        cube_faint = Cube(data=cube_faint, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask)
        cube_cont = Cube(data=cube_cont, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask)
        return cube_faint, cube_cont

    def compute_TGLR(self, cube_faint):
        """Compute the cube of GLR test values obtained with the given
        PSF and dictionary of spectral profile.

        Parameters
        ----------
        cube_faint : mpdaf.obj.cube
                     data cube on test

        Returns
        -------
        correl  : `~mpdaf.obj.Cube`
                  cube of T_GLR values
        profile : `~mpdaf.obj.Cube` (type int)
                  Number of the profile associated to the T_GLR
                  profile = Cube('profile.fits', dtype=int)
        """
        # TGLR computing (normalized correlations)
        self._logger.info('ORIGIN - Compute the GLR test')
        correl, profile = Correlation_GLR_test(cube_faint.data.data, self.var,
                                               self.PSF, self.profiles)
        correl = Cube(data=correl, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask)
        profile = Cube(data=profile, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask, dtype=int)
        return correl, profile

    def compute_pvalues(self, correl, threshold=8):
        """Loop on each zone of the data cube and compute for each zone:

        - the p-values associated to the T_GLR values,
        - the p-values associated to the number of thresholded p-values
          of the correlations per spectral channel,
        - the final p-values which are the thresholded pvalues associated
          to the T_GLR values divided by twice the pvalues associated to the
          number of thresholded p-values of the correlations per spectral
          channel.

        Parameters
        ----------
        threshold : float
                    Threshold applied on pvalues.

        Returns
        -------
        cube_pval_correl  : `~mpdaf.obj.Cube`
                            Cube of thresholded p-values associated
                            to the T_GLR values
        cube_pval_channel : `~mpdaf.obj.Cube`
                            Cube of p-values associated to the number of
                            thresholded p-values of the correlations
                            per spectral channel for each zone
        cube_pval_final   : `~mpdaf.obj.Cube`
                            Cube of final thresholded p-values
        """
        # p-values of correlation values
        self._logger.info('ORIGIN - Compute p-values of correlation values')
        self.param['ThresholdPval'] = threshold
        cube_pval_correl = Compute_pval_correl_zone(correl.data.data, self.intx,
                                                    self.inty, self.NbSubcube,
                                                    self.Edge_xmin,
                                                    self.Edge_xmax,
                                                    self.Edge_ymin,
                                                    self.Edge_ymax,
                                                    threshold)
        # p-values of spectral channel
        # Estimated mean for p-values distribution related
        # to the Rayleigh criterium
        self._logger.info('ORIGIN - Compute p-values of spectral channel')
        mean_est = self.FWHM_PSF**2
        self.param['meanestPvalChan'] = mean_est
        cube_pval_channel = Compute_pval_channel_Zone(cube_pval_correl,
                                                      self.intx, self.inty,
                                                      self.NbSubcube, mean_est)

        # Final p-values
        self._logger.info('ORIGIN - Compute final p-values')
        cube_pval_final = Compute_pval_final(cube_pval_correl, cube_pval_channel,
                                             threshold)

        cube_pval_correl = Cube(data=cube_pval_correl, wave=self.wave, wcs=self.wcs,
                                mask=np.ma.nomask)
        cube_pval_channel = Cube(data=cube_pval_channel, wave=self.wave, wcs=self.wcs,
                                 mask=np.ma.nomask)
        cube_pval_final = Cube(data=cube_pval_final, wave=self.wave, wcs=self.wcs,
                               mask=np.ma.nomask)

        return cube_pval_correl, cube_pval_channel, cube_pval_final

    def compute_ref_pix(self, correl, profile, cube_pval_correl,
                                  cube_pval_channel, cube_pval_final,
                                  neighboors=26):
        """compute the groups of connected voxels with a flood-fill algorithm
        on the cube of final thresholded p-values. Then compute referent
        voxel of each group of connected voxels using the voxel with the
        higher T_GLR value.

        Parameters
        ----------
        correl            : `~mpdaf.obj.Cube`
                            Cube of T_GLR values
        profile           : `~mpdaf.obj.Cube` (type int)
                            Number of the profile associated to the T_GLR
        cube_pval_correl  : `~mpdaf.obj.Cube`
                           Cube of thresholded p-values associated
                           to the T_GLR values
        cube_pval_channel : `~mpdaf.obj.Cube`
                            Cube of spectral p-values
        cube_pval_final   : `~mpdaf.obj.Cube`
                            Cube of final thresholded p-values
        neighboors        : integer
                            Connectivity of contiguous voxels

        Returns
        -------
        Cat0 : astropy.Table
               Catalogue of the referent voxels for each group.
               Coordinates are in pixels.
               Columns of Cat_ref : x y z T_GLR profile pvalC pvalS pvalF
        """
        # connected voxels
        self._logger.info('ORIGIN - Compute connected voxels')
        self.param['neighboors'] = neighboors
        labeled_cube, Ngp = Compute_Connected_Voxel(cube_pval_final.data.data, neighboors)
        self._logger.info('ORIGIN - %d connected voxels detected'%Ngp)
        # Referent pixel
        self._logger.info('ORIGIN - Compute referent pixels')
        Cat0 = Compute_Referent_Voxel(correl.data.data, profile.data.data, cube_pval_correl.data.data,
                                  cube_pval_channel.data.data, cube_pval_final.data.data, Ngp,
                                  labeled_cube)
        return Cat0

    def compute_NBtests(self, Cat0, nb_ranges=3, plot_narrow=False):
        """compute the 2 narrow band tests for each detected emission line.

        Parameters
        ----------
        Cat0        : astropy.Table
                      Catalogue of parameters of detected emission lines.
                      Columns of the Catalogue Cat0 :
                      x y z T_GLR profile pvalC pvalS pvalF
        nb_ranges   : integer
                      Number of the spectral ranges skipped to compute the
                      controle cube
        plot_narrow : boolean
                      If True, plot the narrow bands images

        Returns
        -------
        Cat1 : astropy.Table
               Catalogue of parameters of detected emission lines.
               Columns of the Catalogue Cat1 :
               x y z T_GLR profile pvalC pvalS pvalF T1 T2
        """
        # Parameter set to 1 if we want to plot the results and associated folder
        plot_narrow = False
        self._logger.info('ORIGIN - Compute narrow band tests')
        self.param['NBranges'] = nb_ranges
        Cat1 = Narrow_Band_Test(Cat0, self.cube_raw, self.profiles,
                                self.PSF, nb_ranges,
                                plot_narrow, self.wcs)
        return Cat1

    def select_NBtests(self, Cat1, thresh_T1=0.2, thresh_T2=2):
        """select emission lines according to the 2 narrow band tests.

        Parameters
        ----------
        Cat1      : astropy.Table
                    Catalogue of detected emission lines.
        thresh_T1 : float
                    Threshold for the test 1
        thresh_T2 : float
                    Threshold for the test 2

        Returns
        -------
        Cat1_T1 : astropy.Table
                  Catalogue of parameters of detected emission lines selected with
                  the test 1
        Cat1_T2 : astropy.Table
                  Catalogue of parameters of detected emission lines selected with
                  the test 2

        Columns of the catalogues :
        x y z T_GLR profile pvalC pvalS pvalF T1 T2
        """
        self.param['threshT1'] = thresh_T1
        self.param['threshT2'] = thresh_T2
        # Thresholded narrow bands tests
        Cat1_T1, Cat1_T2 = Narrow_Band_Threshold(Cat1, thresh_T1, thresh_T2)
        self._logger.info('ORIGIN - %d emission lines selected with the test 1'%len(Cat1_T1))
        self._logger.info('ORIGIN - %d emission lines selected with the test 2'%len(Cat1_T2))
        return Cat1_T1, Cat1_T2

    def estimate_line(self, Cat1_T, profile, cube_faint,
                      grid_dxy=0, grid_dz=0):
        """compute the estimated emission line and the optimal coordinates
        for each detected lines in a spatio-spectral grid (each emission line
        is estimated with the deconvolution model :
        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))

        Parameters
        ----------
        Cat1_T     : astropy.Table
                     Catalogue of parameters of detected emission lines selected
                     with a narrow band test.
                     Columns of the Catalogue Cat1_T:
                     x y z T_GLR profile pvalC pvalS pvalF T1 T2
        profile    : `~mpdaf.obj.Cube`
                     Number of the profile associated to the T_GLR
        cube_faint : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the lower
                     eigenvalues
        grid_dxy   : integer
                     Maximum spatial shift for the grid
        grid_dz    : integer
                     Maximum spectral shift for the grid

        Returns
        -------
        Cat2_T           : astropy.Table
                           Catalogue of parameters of detected emission lines.
                           Columns of the Catalogue Cat2:
                           x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual
                           flux num_line
        Cat_est_line : list of `~mpdaf.obj.Spectrum`
                        Estimated lines
        """
        self._logger.info('ORIGIN - Lines estimation')
        self.param['grid_dxy'] = grid_dxy
        self.param['grid_dz'] = grid_dz
        Cat2_T, Cat_est_line_raw_T, Cat_est_line_std_T = \
        Estimation_Line(Cat1_T, profile.data.data, self.Nx, self.Ny, self.Nz, self.var, cube_faint.data.data,
                    grid_dxy, grid_dz, self.PSF, self.profiles)
        Cat_est_line = []
        for data, var in zip(Cat_est_line_raw_T, Cat_est_line_std_T):
            spe = Spectrum(data=data, var=var, wave=self.wave, mask=np.ma.nomask)
            Cat_est_line.append(spe)
        return Cat2_T, Cat_est_line

    def merge_spatialy(self, Cat2_T):
        """Construct a catalogue of sources by spatial merging of the
        detected emission lines in a circle with a diameter equal to
        the mean over the wavelengths of the FWHM of the FSF.

        Parameters
        ----------
        Cat2_T   : astropy.Table
                   catalogue
                   Columns of Cat2_T:
                   x y z T_GLR profile pvalC pvalS pvalF T1 T2
                   residual flux num_line

        Returns
        -------
        Cat3 : astropy.Table
               Columns of Cat3:
               ID x_circle y_circle x_centroid y_centroid nb_lines
               x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux num_line
        """
        self._logger.info('ORIGIN - Spatial merging')
        Cat3 = Spatial_Merging_Circle(Cat2_T, self.FWHM_PSF, self.Nx, self.Ny)
        return Cat3

    def merge_spectraly(self, Cat3, Cat_est_line, deltaz=1):
        """Merge the detected emission lines distants to less than deltaz
           spectral channel in each group.

        Parameters
        ----------
        Cat3         : astropy.Table
                       Catalogue of detected emission lines
                       Columns of Cat:
                       ID x_circle y_circle x_centroid y_centroid nb_lines
                       x y z T_GLR profile pvalC pvalS pvalF T1 T2
                       residual flux num_line
        Cat_est_line : list of `~mpdaf.obj.Spectrum`
                       List of estimated lines
        deltaz       : integer
                       Distance maximum between 2 different lines

        Returns
        -------
        Cat4 : astropy.Table
               Catalogue
               Columns of Cat4:
               ID x_circle y_circle x_centroid y_centroid nb_lines
               x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux num_line
        """
        self._logger.info('ORIGIN - Spectral merging')
        self.param['deltaz'] = deltaz
        Cat_est_line_raw = [spe.data.data for spe in Cat_est_line]
        Cat4 = Spectral_Merging(Cat3, Cat_est_line_raw, deltaz)
        return Cat4

    def get_sources(self, Cat4, Cat_est_line, correl):
        """add corresponding RA/DEC to each referent pixel of each group and
        create the final catalogue of sources with their parameters


        Parameters
        ----------
        Cat4             : astropy.Table
                           Catalogue of the detected emission lines:
                           ID x_circle y_circle x_centroid y_centroid
                           nb_lines x y z T_GLR profile pvalC pvalS pvalF
                           T1 T2 residual flux num_line
        Cat_est_line : list of `~mpdaf.obj.Spectrum`
                           List of estimated lines
        correl           : `~mpdaf.obj.Cube`
                           Cube of T_GLR values

        Returns
        -------
        sources : mpdaf.sdetect.SourceList
                  List of sources
        """
        # Add RA-DEC to the catalogue
        self._logger.info('ORIGIN - Add RA-DEC to the catalogue')
        CatF_radec = Add_radec_to_Cat(Cat4, self.wcs)

        # list of source objects
        self._logger.info('ORIGIN - Create the list of sources')
        sources = Construct_Object_Catalogue(CatF_radec, Cat_est_line,
                                         correl.data.data, self.wave,
                                         self.filename, self.FWHM_profiles)
        # save orig parameters in sources
        for src in sources:
            src.OP_THRES = (self.param['ThresholdPval'],'Orig Threshold Pval')
            src.OP_DZ = (self.param['deltaz'],'Orig deltaz')
            src.OP_R0 = (self.param['r0PCA'],'Orig PCA R0')
            src.OP_T1 = (self.param['threshT1'],'Orig T1 threshold')
            src.OP_T1 = (self.param['threshT2'],'Orig T2 threshold')
            src.OP_NG = (self.param['neighboors'],'Orig Neighboors')
            src.OP_MP = (self.param['meanestPvalChan'],'Orig Meanest PvalChan')
            src.OP_NS = (self.param['nbsubcube'],'Orig nb of subcubes')
            src.OP_DXY = (self.param['grid_dxy'],'Orig Grid Nxy')
            src.OP_DZ = (self.param['grid_dz'],'Orig Grid Nz')
            src.OP_FSF = (self.param['PSF'],'Orig FSF cube')
        return sources

    def plot(self, correl, x, y, circle=False, vmin=0, vmax=30, title=None, ax=None):
        """Plot detected emission lines on the 2D map of maximum of the T_GLR
        values over the spectral channels.

        Parameters
        ----------
        correl : `~mpdaf.obj.Cube`
                 Cube of T_GLR values
        x      : array
                 Coordinates along the x-axis of the estimated lines
                 in pixels (column).
        y      : array
                 Coordinates along the y-axis of the estimated lines
                 in pixels (column).
        circle  : bool
                  If true, plot circles with a diameter equal to the
                  mean of the fwhm of the PSF.
        vmin : float
                Minimum pixel value to use for the scaling.
        vmax : float
                Maximum pixel value to use for the scaling.
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        """
        carte_2D_correl = np.amax(correl.data.data, axis=0)
        carte_2D_correl_ = Image(data=carte_2D_correl, wcs=self.wcs)

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.plot(x, y, 'k+')
        if circle:
            for px, py in zip(x, y):
                c = plt.Circle((px, py), np.round(self.FWHM_PSF/2), color='k',
                               fill=False)
                ax.add_artist(c)
        carte_2D_correl_.plot(vmin=vmin, vmax=vmax, title=title, ax=ax)
