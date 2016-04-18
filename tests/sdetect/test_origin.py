"""Test interface on ORIGIN software."""

from __future__ import absolute_import, division

import nose.tools
from nose.plugins.attrib import attr
import shutil

from mpdaf.sdetect import ORIGIN


class TestORIGIN():

    @attr(speed='slow')
    def test_main(self):
        """ORIGIN: tests the process"""

        # name of the MUSE data cube
        filename = 'data/sdetect/minicube.fits'
        # Number of subcubes for the spatial segmentation
        NbSubcube = 1

        my_origin = ORIGIN(filename, NbSubcube, [2, 2, 1, 3])

        # Coefficient of determination for projection during PCA
        r0 = 0.67
        # PCA
        cube_faint, cube_cont = my_origin.compute_PCA(r0)

        # TGLR computing (normalized correlations)
        correl, profile = my_origin.compute_TGLR(cube_faint)

        # threshold applied on pvalues
        threshold = 8
        # compute pvalues
        cube_pval_correl, cube_pval_channel, cube_pval_final = \
                                   my_origin.compute_pvalues(correl, threshold)

        # Connectivity of contiguous voxels
        neighboors = 26
        # Compute connected voxels and their referent pixels
        Cat0 = my_origin.compute_ref_pix(correl, profile, cube_pval_correl,
                                  cube_pval_channel, cube_pval_final,
                                  neighboors)

        # Number of the spectral ranges skipped to compute the controle cube
        nb_ranges = 3
        # Narrow band tests
        Cat1 = my_origin.compute_NBtests(Cat0, nb_ranges)
        # Thresholded narrow bands tests
        thresh_T1 = .2
        thresh_T2 = 2

        Cat1_T1, Cat1_T2 = my_origin.select_NBtests(Cat1, thresh_T1,
                                                       thresh_T2)

        # Estimation with the catalogue from the narrow band Test number 2
        Cat2_T2, Cat_est_line = \
        my_origin.estimate_line(Cat1_T2, profile, cube_faint)

        # Spatial merging
        Cat3 = my_origin.merge_spatialy(Cat2_T2)

        # Distance maximum between 2 different lines (in pixels)
        deltaz = 1
        # Spectral merging
        Cat4 = my_origin.merge_spectraly(Cat3, Cat_est_line, deltaz)

        # list of source objects
        nsources = my_origin.write_sources(Cat4, Cat_est_line, correl)

        nose.tools.assert_equal(nsources, 2)
        shutil.rmtree('./origin')
