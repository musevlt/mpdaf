from mpdaf.obj import CubeDisk
from mpdaf.drs import PixTable
from mpdaf.MUSE import Slicer
import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import plt_zscale
import matplotlib.cm as cm


class DisplayPixTable(object):
    """DisplayPixTable class

    This class displays MUSE pixel table files

    :param pixtable: The FITS file name of MUSE pixel table.
    :type pixtable: string.

    :param cube: The FITS file name of MUSE cube.
    :type cube: string.

    Attributes
    ----------

    pixtable : string
    The FITS file name of MUSE pixel table.

    cube : string
    The FITS file name of MUSE cube.
    """

    def __init__(self, pixtable, cube):
        """creates a DisplayPixTable object and verifies
        that pixtable and cube are compatible"""

        self.pixtable = pixtable
        self.cube = cube

    def info(self):
        """Prints information.
        """
        cub = CubeDisk(self.cube)
        cub.info()
        print ''
        pix = PixTable(self.pixtable)
        pix.info()

    def _det_display(self, date, pix, ima, spe, ifu_limits, l, exp, sky, \
                     lbda, sky_scale, sky_cmap, det_scale, det_cmap, \
                     det_vmin, det_vmax):
        """display in detector mode for one exposure

        :param exp: exposure number.
        :type sky: integer or None

        :param sky: (y, x, size, shape) extract an aperture on the sky,
        defined by a center in degrees (y, x), a shape
        ('C' for circular, 'S' for square) and size in arcsec
        (radius or half side length).
        :type sky: (float, float, float, char)

        :param lbda: (min, max) wavelength range in Angstrom.
        :type lbda: (float, float)

        :param sky_scale: The stretch function to use for the scaling
        of the sky image (default is 'linear').
        :type sky_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param sky_cmap: color map used for the white image on the sky
        :type sky_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param det_scale: The stretch function to use for the scaling
        of the detector images (default is 'linear').
        :type det_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param det_cmap: color map used for the detectors images
        :type det_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param det_vmin: Minimum pixel value to use
        for the scaling of the detector images.
        If None, det_vmin is set with the IRAF zscale algorithm.
        :type det_vmin: float

        :param det_vmax: Maximum pixel value to use
        for the scaling of the detector images.
        If None, det_vmax is set with the IRAF zscale algorithm.
        :type det_vmax: float
        """
        plt.figure()
        plt.figtext(0.1, 0.05, 'Pixtable %s %s' % (self.pixtable, date), \
                    fontsize=10)
        plt.figtext(0.1, 0.03, 'Cube %s %s' % (self.cube, date), fontsize=10)

        # number of ifus in the aperture
        print 'extract sub-pixel table ...'
        subpix = pix.extract(lbda=lbda, sky=sky, exp=exp)
        if subpix is None:
            raise ValueError('pixel table extraction is not valid')
        # number of plots
        nplots = subpix.nifu

        # plot images on detectors
        list_ifu = subpix.origin2ifu(subpix.get_origin())
        list_ifu = np.unique(list_ifu)

        ypix = subpix.origin2ypix(subpix.get_origin())
        ystart = np.min(ypix)
        ystop = np.max(ypix)
        del ypix

        list_detima = []
        list_vmin = []
        list_vmax = []
        for ifu in list_ifu:
            print 'plot detector image of the CHAN%02d ...' % ifu
            subsubpix = subpix.extract(ifu=ifu)
            detima = subsubpix.reconstruct_det_image(ystart=ystart, \
                                                     ystop=ystop)
            del subsubpix
            list_detima.append(detima)
            if det_vmin is None or det_vmax is None:
                vmin, vmax = plt_zscale.zscale(detima.data.filled(0))
                list_vmin.append(vmin)
                list_vmax.append(vmax)

        if det_vmin is None:
            det_vmin = np.min(list_vmin)
        if det_vmax is None:
            det_vmax = np.max(list_vmax)

        iplot = 0
        for ifu, detima in zip(list_ifu, list_detima):
            limits = detima.wcs.get_range()
            plt.subplot2grid((5, nplots), (2, iplot), rowspan=2)
            im = detima.plot(title='%02d' % ifu, scale=det_scale, \
                             vmin=det_vmin, vmax=det_vmax, \
                             extent=[limits[0][1], limits[1][1], \
                                     limits[0][0], limits[1][0]], \
                             cmap=det_cmap)
            if iplot != 0:
                plt.ylabel('')
                plt.yticks([])
            else:
                plt.ylabel('y in pixels detector')
            plt.xticks([])
            plt.xlabel('%04d' % ((limits[0][1] + limits[1][1]) / 2))
            iplot += 1
        # colorbar
        plt.subplot2grid((5, nplots), (4, 0), colspan=nplots)
        plt.gca().set_visible(False)
        plt.colorbar(im, orientation='horizontal')

        # plot corresponding white image and spectrum
        print 'plot corresponding white image ...'
        if (nplots % 2 == 0):
            colspan_ima = nplots / 2
        else:
            colspan_ima = int(nplots / 2) + 1
        plt.subplot2grid((5, nplots), (0, 0), colspan=colspan_ima, rowspan=2)
        for ifu in list_ifu:  # plot ifus limits on image
            ymin, ymax = ifu_limits[ifu]
            ymin = int(ima.wcs.sky2pix((ymin, 0))[0][0])
            ymax = int(ima.wcs.sky2pix((ymax, 0))[0][0])
            plt.plot(np.arange(0, l), np.ones(l) * ymin, 'b-')
            plt.plot(np.arange(0, l), np.ones(l) * ymax, 'b-')
            ymin = max(0, ymin)
            ymax = min(ymax, ima.shape[0])
            plt.annotate('%02d' % ifu, xy=(0, (ymin + ymax) / 2.0),  \
                         xycoords='data', textcoords='data', color='b')
        im = ima.plot(colorbar='h', scale=sky_scale, cmap=sky_cmap)
        im.get_axes().set_axis_off()
        print 'plot corresponding spectrum ...'
        plt.subplot2grid((5, nplots), (0, colspan_ima), \
                         rowspan=2, colspan=int(nplots / 2))
        spe.plot()
        plt.xlim(lbda[0], lbda[1])

    def det_display(self, sky, lbda, sky_scale='linear', sky_cmap=cm.copper, \
                    det_scale='linear', det_cmap=cm.copper, \
                    det_vmin=None, det_vmax=None):
        """display in detector mode

        :param sky: (y, x, size, shape) extract an aperture on the sky,
        defined by a center in degrees (y, x), a shape
        ('C' for circular, 'S' for square) and size in arcsec
        (radius or half side length).
        :type sky: (float, float, float, char)

        :param lbda: (min, max) wavelength range in Angstrom.
        :type lbda: (float, float)

        :param sky_scale: The stretch function to use for the scaling
        of the sky image (default is 'linear').
        :type sky_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param sky_cmap: color map used for the white image on the sky
        :type sky_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param det_scale: The stretch function to use for the scaling
        of the detector images (default is 'linear').
        :type det_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param det_cmap: color map used for the detectors images
        :type det_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param det_vmin: Minimum pixel value to use for the scaling
        of the detector images.
        If None, det_vmin is set with the IRAF zscale algorithm.
        :type det_vmin: float

        :param det_vmax: Maximum pixel value to use for the scaling
        of the detector images.
        If None, det_vmax is set with the IRAF zscale algorithm.
        :type det_vmax: float
        """
        pix = PixTable(self.pixtable)
        cub = CubeDisk(self.cube)
        date = cub.primary_header['DATE-OBS'].split('T')[0]

        if (pix.wcs and cub.wcs.is_deg()):
            is_deg = True
        elif (not pix.wcs and not cub.wcs.is_deg()):
            is_deg = False
        else:
            raise ValueError('Cube and pixel table have different units')

        l1, l2 = lbda
        y, x, size, shape = sky

        # ifu grid
        ifu_low = \
        pix.get_keywords('HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU LOW')
        ifu_high = \
        pix.get_keywords('HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU HIGH')
        nifu = pix.nifu
        dp_ifu = float(cub.shape[1]) / nifu
        pmin = np.arange(nifu) * dp_ifu
        pmax = np.arange(1, nifu + 1) * dp_ifu
        c = np.zeros((nifu, 2))
        c[:, 0] = pmin
        ymin = cub.wcs.pix2sky(c)[:, 0]
        c[:, 0] = pmax
        ymax = cub.wcs.pix2sky(c)[:, 0]
        ifu_limits = dict((_ifu, (_ymin, _ymax)) for _ifu, _ymin, _ymax \
                          in zip(np.arange(ifu_high, ifu_low - 1, -1), \
                                 ymin, ymax))

        # plot corresponding white image and spectrum
        y_min = y - size
        y_max = y + size
        x_min = x - size
        x_max = x + size

        subcub = cub.truncate(l1, l2, y_min, y_max, x_min, x_max, mask=True)
        if subcub.shape[0] == 0 or subcub.shape[1] == 0 \
        or subcub.shape[2] == 0:
            raise ValueError('cube extraction is not valid')
        ima = subcub.sum(axis=0)
        spe = subcub.sum(axis=(1, 2))

        if shape == 'C':
            center = ima.wcs.sky2pix((y, x))[0]
            radius = size / np.abs(ima.wcs.get_step())[0]
            if is_deg:
                radius /= 3600.
            ima.mask(center=center, radius=radius, pix=True, inside=False)
        l = ima.shape[1]

        try:
            nexp = pix.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE COMBINED")
            col_exp = pix.get_exp()
            exposures = np.unique(col_exp)
            for exp in exposures:
                print 'exposure %02d' % exp
                self._det_display(date, pix, ima, spe, ifu_limits, l, exp, \
                                  sky, lbda, sky_scale, sky_cmap, det_scale, \
                                  det_cmap, det_vmin, det_vmax)
        except:
            self._det_display(date, pix, ima, spe, ifu_limits, l, None, \
                              sky, lbda, sky_scale, sky_cmap, det_scale, \
                              det_cmap, det_vmin, det_vmax)

    def _slice_display(self, date, pix, ima, spe, slice_limits, \
                       ima_center_p, ima_center_q, dp_slice, dq_slice, \
                       exp, sky, lbda, sky_scale, sky_cmap, slice_scale, \
                       slice_cmap, slice_vmin, slice_vmax):
        """display in slice mode

        :param exp: exposure number.
        :type sky: integer or None

        :param sky: (y, x, size, shape) extract an aperture on the sky,
        defined by a center in degrees (y, x), a shape
        ('C' for circular, 'S' for square) and size in arcsec
        (radius or half side length).
        :type sky: (float, float, float, char)

        :param lbda: (min, max) wavelength range in Angstrom.
        :type lbda: (float, float)

        :param sky_scale: The stretch function to use for the scaling
        of the sky image (default is 'linear').
        :type sky_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param sky_cmap: color map used for the white image on the sky.
        :type sky_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param slice_scale: The stretch function to use
        for the scaling of the slice images (default is 'linear').
        :type slice_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param slice_cmap: color map used for the slices images.
        :type slice_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param slice_vmin: Minimum pixel value to use
        for the scaling of the slice images.
        If None, det_vmin is set with the IRAF zscale algorithm.
        :type slice_vmin: float

        :param slice_vmax: Maximum pixel value to use
        for the scaling of the slice images.
        If None, det_vmax is set with the IRAF zscale algorithm.
        :type slice_vmax: float
        """
        # number of ifus
        print 'extract sub-pixel table ...'
        subpix = pix.extract(lbda=lbda, sky=sky, exp=exp)
        if subpix is None:
            raise ValueError('Pixel table extraction is not valid')

        ypix = subpix.origin2ypix(subpix.get_origin())
        ystart = np.min(ypix)
        ystop = np.max(ypix)
        del ypix

        # plot corresponding image
        plt.figure()
        plt.figtext(0.1, 0.05, 'Pixtable %s %s' % (self.pixtable, date), \
                    fontsize=10)
        plt.figtext(0.1, 0.03, 'Cube %s %s' % (self.cube, date), fontsize=10)

        list_ifu = np.unique(subpix.origin2ifu(subpix.get_origin()))

        distance = []
        for ifu in list_ifu:
            pixifu = subpix.extract(ifu=ifu)
            list_slice = np.unique(pixifu.origin2slice(pixifu.get_origin()))
            for slice_ccd in list_slice:
                sli = Slicer.ccd2sky(slice_ccd)
                p, q = slice_limits[(ifu, sli)]
                p, q = ima.wcs.sky2pix((p, q))[0]
                d = np.abs((p - ima_center_p) ** 2 + (q - ima_center_q) ** 2)
                distance.append((d, ifu, slice_ccd, p))

        distance.sort(key=lambda tup: tup[0])
        nplots = len(distance)
        distance = np.array(distance)

        print 'plot corresponding white image ...'
        if (nplots % 2 == 0):
            colspan_ima = nplots / 2
        else:
            colspan_ima = int(nplots / 2) + 1
        # colspan_spe = nplot - colspan_ima
        plt.subplot2grid((9, nplots), (0, 0), colspan=colspan_ima, rowspan=3)
        for ifu, sli, p in zip(distance[:, 1], distance[:, 2], \
                                 distance[:, 3]):
            if q < 0:
                q_label = 0
            elif q >= ima.shape[1]:
                q_label = ima.shape[1] - 1
            else:
                q_label = q
            plt.annotate('%02d/%02d' % (ifu, sli), xy=(q_label, p - 0.2), \
                         xycoords='data', textcoords='data', color='b')
            pmin = p - dp_slice / 2.0
            pmax = p + dp_slice / 2.0
            qmin = q - dq_slice / 2.0
            qmax = q + dq_slice / 2.0
            plt.plot(np.arange(qmin, qmax + 1, 1), \
                     np.ones(qmax - qmin + 2) * pmin, 'b-')
            plt.plot(np.arange(qmin, qmax + 1, 1), \
                     np.ones(qmax - qmin + 2) * pmax, 'b-')
            plt.plot(np.ones(pmax - pmin + 2) * qmin, \
                     np.arange(pmin, pmax + 1, 1), 'b-')
            plt.plot(np.ones(pmax - pmin + 2) * qmax, \
                     np.arange(pmin, pmax + 1, 1), 'b-')
        im = ima.plot(colorbar='h', scale=sky_scale, cmap=sky_cmap)
        im.get_axes().set_axis_off()

        # plot corresponding spectrum
        print 'plot corresponding spectrum ...'
        plt.subplot2grid((9, nplots), (0, colspan_ima), \
                         rowspan=3, colspan=nplots - colspan_ima)
        spe.plot()
        plt.xlim(lbda[0], lbda[1])

        # plot images on slices
        list_vmin = []
        list_vmax = []
        list_sliceima = []
        for ifu, sli in zip(distance[:, 1], distance[:, 2]):
            print 'plot slice image ifu=%02d slice=%02d ...' % (ifu, sli)
            pixslice = subpix.extract(ifu=ifu, sl=sli)
            sli = Slicer.ccd2sky(sli)
            sliceima = pixslice.reconstruct_det_image(ystart=ystart, ystop=ystop)
            del pixslice
            list_sliceima.append(sliceima)
            if slice_vmin is None or slice_vmax is None:
                vmin, vmax = plt_zscale.zscale(sliceima.data.filled(0))
                list_vmin.append(vmin)
                list_vmax.append(vmax)

        if slice_vmin is None:
            slice_vmin = np.min(list_vmin)
        if slice_vmax is None:
            slice_vmax = np.max(list_vmax)

        iplot = 0
        for ifu, sli, sliceima in zip(distance[:, 1], distance[:, 2], \
                                        list_sliceima):
            limits = sliceima.wcs.get_range()
            plt.subplot2grid((9, nplots), (4, iplot), rowspan=3)
            im = sliceima.plot(title='%02d/%02d' % (ifu, sli), \
                               scale=slice_scale, \
                               vmin=slice_vmin, vmax=slice_vmax, \
                               extent=[limits[0][1], limits[1][1], \
                                       limits[0][0], limits[1][0]], \
                               cmap=slice_cmap)
            if iplot != 0:
                plt.ylabel('')
                plt.yticks([])
            else:
                plt.ylabel('y in pixels detector')
            plt.xticks([])
            plt.xlabel('%04d' % ((limits[0][1] + limits[1][1]) / 2))
            iplot += 1
        # colorbar
        plt.subplot2grid((9, nplots), (7, 0), colspan=nplots, rowspan=2)
        plt.gca().set_visible(False)
        plt.colorbar(im, orientation='horizontal')

    def slice_display(self, sky, lbda, sky_scale='linear', \
                      sky_cmap=cm.copper, slice_scale='linear', \
                      slice_cmap=cm.copper, slice_vmin=None, slice_vmax=None):
        """display in slice mode

        :param sky: (y, x, size, shape) extract an aperture on the sky,
        defined by a center in degrees (y, x), a shape
        ('C' for circular, 'S' for square) and size in arcsec
        (radius or half side length).
        :type sky: (float, float, float, char)

        :param lbda: (min, max) wavelength range in Angstrom.
        :type lbda: (float, float)

        :param sky_scale: The stretch function to use
        for the scaling of the sky image (default is 'linear').
        :type sky_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param sky_cmap: color map used for the white image on the sky
        :type sky_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param slice_scale: The stretch function to use
        for the scaling of the slice images (default is 'linear').
        :type slice_scale: 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'

        :param slice_cmap: color map used for the slices images.
        :type slice_cmap: `matplotlib.cm <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_

        :param slice_vmin: Minimum pixel value to use for the scaling
        of the slice images. If None, det_vmin is set
        with the IRAF zscale algorithm.
        :type slice_vmin: float

        :param slice_vmax: Maximum pixel value to use
        for the scaling of the slice images.
        If None, det_vmax is set with the IRAF zscale algorithm.
        :type slice_vmax: float
        """
        cub = CubeDisk(self.cube)
        pix = PixTable(self.pixtable)
        date = cub.primary_header['DATE-OBS'].value.split('T')[0]

        if (pix.wcs and cub.wcs.is_deg()):
            is_deg = True
        elif (not pix.wcs and not cub.wcs.is_deg()):
            is_deg = False
        else:
            raise ValueError('Cube and pixel table have different units')

        l1, l2 = lbda
        y, x, size, shape = sky

        #ifu grid
        ifu_low = \
        pix.get_keywords('HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU LOW')
        ifu_high = \
        pix.get_keywords('HIERARCH ESO DRS MUSE PIXTABLE LIMITS IFU HIGH')
        nifu = pix.nifu
        dp_ifu = float(cub.shape[1]) / nifu
        dp_slice = dp_ifu / 12
        dq_slice = float(cub.shape[2]) / 4

        #pmin_ifu = np.arange(nifu) * dp_ifu
        #pmax_ifu = np.arange(1, nifu + 1) * dp_ifu

        #p_slice = np.arange(12) * dp_slice + dp_slice / 2.0
        #q_slice = np.arange(4) * dq_slice + dq_slice / 2.0

        list_ifu = np.empty((nifu, 48))
        for i, ifu in enumerate(range(ifu_high, ifu_low - 1, -1)):
            list_ifu[i, :] = ifu
        list_ifu = np.concatenate(list_ifu)

        list_slice = np.empty((nifu, 48))
        for i in range(nifu):
            list_slice[i, :] = np.arange(48, 0, -1)
        list_slice = np.concatenate(list_slice)

        list_p = np.empty((nifu, 48))
        for i in range(nifu):
            list_p[i, 0:12] = i * dp_ifu + np.arange(12) * dp_slice \
            + dp_slice / 2.0 - 0.5
            list_p[i, 12:24] = i * dp_ifu + np.arange(12) * dp_slice \
            + dp_slice / 2.0 - 0.5
            list_p[i, 24:36] = i * dp_ifu + np.arange(12) * dp_slice \
            + dp_slice / 2.0 - 0.5
            list_p[i, 36:48] = i * dp_ifu + np.arange(12) * dp_slice \
            + dp_slice / 2.0 - 0.5
        list_p = np.concatenate(list_p)

        list_q = np.empty((nifu, 48))
        for i in range(nifu):
            list_q[i, 36:48] = dq_slice / 2.0 - 0.5
            list_q[i, 24:36] = 3.0 / 2.0 * dq_slice - 0.5
            list_q[i, 12:24] = 5.0 / 2.0 * dq_slice - 0.5
            list_q[i, 0:12] = 7.0 / 2.0 * dq_slice - 0.5
        list_q = np.concatenate(list_q)

        coord_pix = np.zeros((nifu * 48, 2))
        coord_pix[:, 0] = list_p
        coord_pix[:, 1] = list_q
        coord_sky = cub.wcs.pix2sky(coord_pix)
        list_p = coord_sky[:, 0]
        list_q = coord_sky[:, 1]

        slice_limits = dict(((ifu, sli), (p, q)) for ifu, sli, p, q \
                            in zip(list_ifu, list_slice, list_p, list_q))

        # plot corresponding white image and spectrum
        y_min = y - size
        y_max = y + size
        x_min = x - size
        x_max = x + size

        subcub = cub.truncate(l1, l2, y_min, y_max, x_min, x_max, mask=True)
        if subcub.shape[0] == 0 or subcub.shape[1] == 0 \
        or subcub.shape[2] == 0:
            raise ValueError('cube extraction is not valid')
        ima = subcub.sum(axis=0)
        spe = subcub.sum(axis=(1, 2))

        if shape == 'C':
            center = ima.wcs.sky2pix((y, x))[0]
            radius = size / np.abs(ima.wcs.get_step())[0]
            if is_deg:
                radius /= 3600.
            ima.mask(center=center, radius=radius, pix=True, inside=False)
        ima_center_p = ima.shape[0] / 2.0
        ima_center_q = ima.shape[1] / 2.0

        try:
            nexp = pix.get_keywords("HIERARCH ESO DRS MUSE PIXTABLE COMBINED")
            col_exp = pix.get_exp()
            exposures = np.unique(col_exp)
            for exp in exposures:
                self._slice_display(date, pix, ima, spe, slice_limits, \
                                   ima_center_p, ima_center_q, dp_slice, \
                                   dq_slice, exp, sky, lbda, sky_scale, \
                                   sky_cmap, slice_scale, slice_cmap, \
                                   slice_vmin, slice_vmax)
        except:
            self._slice_display(date, pix, ima, spe, slice_limits, \
                                ima_center_p, ima_center_q, dp_slice, \
                                dq_slice, None, sky, lbda, sky_scale, \
                                sky_cmap, slice_scale, slice_cmap, \
                                slice_vmin, slice_vmax)
