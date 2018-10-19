"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2012-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from astropy.io import fits
from scipy.integrate import quad

from ..obj import Image, WCS

__all__ = ('RawFile', 'Channel')

NB_SUBSLICERS = 4  # number of sub-slicers
NB_SPEC_PER_SLICE = 75  # number of pixels per slice
NB_SLICES = 12  # number of slices per sub-slicer
INTERSPEC = 7  # inter-spectrum distance in pixel
OVERSCAN = 32  # overscan width in pixel
SLIT_POSITION = np.array([9, 8, 1, 10, 7, 2, 11, 6, 3, 12, 5, 4])


class Channel(object):

    """Channel object corresponds to an extension of a MUSE raw FITS file.

    Parameters
    ----------
    extname : str
        The extension name.
    filename : str
        The raw FITS file name.

    Attributes
    ----------
    extname : str
        The extension name
    header : `astropy.io.fits.Header`
        The extension header
    nx : int
        Lengths of data in X
    ny : int
        Lengths of data in Y
    mask : array of bool
        Arrays that contents TRUE for overscanned pixels, FALSE for the others.

    """

    def __init__(self, extname, filename):
        self._logger = logging.getLogger(__name__)
        self.extname = extname

        with fits.open(filename) as hdulist:
            self.header = hdr = hdulist[extname].header
            self.nx = hdr["NAXIS1"]
            self.ny = hdr["NAXIS2"]
            self.data = hdulist[extname].data

        self.mask = self._init_mask()

    def _get_detector_indices(self, idx, with_prescan=True):
        """Return the indices for one quadrant, with prescan or not."""

        if idx not in (1, 2, 3, 4):
            raise ValueError('Invalid quadrant index')

        # Physical active pixels in X, Y
        nx_data = self.header["ESO DET CHIP NX"]
        ny_data = self.header["ESO DET CHIP NY"]

        # Output data pixels in X, Y
        key = "ESO DET OUT%i" % idx
        nx = self.header["%s NX" % key]
        ny = self.header["%s NY" % key]

        # Output prescan pixels in X, Y
        prscx = self.header.get("%s PRSCX" % key, OVERSCAN)
        prscy = self.header.get("%s PRSCY" % key, OVERSCAN)

        # X, Y location of output
        x = self.header["%s X" % key]
        y = self.header["%s Y" % key]

        if with_prescan:
            if x < nx_data // 2:
                i1 = x - 1
                i2 = i1 + nx + 2 * prscx
            else:
                i2 = self.nx
                i1 = i2 - nx - 2 * prscx

            if y < ny_data // 2:
                j1 = y - 1
                j2 = j1 + ny + 2 * prscy
            else:
                j2 = self.ny
                j1 = j2 - ny - 2 * prscy
        else:
            if x < nx_data // 2:
                i1 = x - 1 + prscx
                i2 = i1 + nx
            else:
                i2 = self.nx - prscx
                i1 = i2 - nx

            if y < ny_data // 2:
                j1 = y - 1 + prscy
                j2 = j1 + ny
            else:
                j2 = self.ny - prscy
                j1 = j2 - ny

        return slice(j1, j2), slice(i1, i2)

    def _init_mask(self):
        """Create mask that invalidates over scanned pixels."""
        mask = np.ones((self.ny, self.nx), dtype=bool)
        for i in range(1, 5):
            sy, sx = self._get_detector_indices(i, with_prescan=False)
            mask[sy, sx] = False
        return mask

    def trimmed(self, copy=True):
        """Return a masked array where overscan pixels are masked."""
        return np.ma.MaskedArray(self.data, mask=self.mask, copy=copy)

    def overscan(self, copy=True):
        """Return a masked array where only overscan pixels are not masked."""
        return np.ma.MaskedArray(self.data, mask=~self.mask, copy=copy)

    def get_image(self, det_out=None, bias=False):
        """Return an Image object.

        Parameters
        ----------
        det_out : int in [1,4]
            Number of output detector. If None, all image is returned.
        bias : bool
            If True, median value of the overscanned pixels is subtracted.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        wcs = WCS(self.header)
        ima = Image(wcs=wcs, data=self.data, dtype=None)

        if det_out is not None:
            sy, sx = self._get_detector_indices(det_out, with_prescan=True)
            ima = ima[sy, sx]
            if bias:
                ima = ima - self.get_bias_level(det_out)

        if det_out is None and bias:
            for det in range(1, 5):
                sy, sx = self._get_detector_indices(det, with_prescan=True)
                ima[sy, sx] = ima[sy, sx] - self.get_bias_level(det)

        return ima

    def get_bias_level(self, det_out):
        """computes median value of the overscanned pixels.

        Parameters
        ----------
        det_out : int in [1,4]
            Number of detector taken into account.

        Returns
        -------
        out : float
        """
        ima = self.get_image_just_overscan(det_out)
        ksel = np.where(ima.data.mask == False)
        return np.median(ima.data.data[ksel])

    def get_trimmed_image(self, det_out=None, bias=False):
        """Return an Image object without over scanned pixels.

        Parameters
        ----------
        det_out : int in [1,4]
            Number of output detector. If None, all image is returned.
        bias : bool
            If True, median value of the overscanned pixels is subtracted.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        work = self.trimmed()

        if bias:
            det_list = [det_out] if isinstance(det_out, int) else range(1, 5)
            for det in det_list:
                sy, sx = self._get_detector_indices(det, with_prescan=True)
                ksel = np.where(self.mask[sy, sx] == True)
                bias_level = np.median(work.data[sy, sx][ksel])
                work[sy, sx] = work[sy, sx] - bias_level

        # Physical active pixels in X, Y
        nx_data = self.header["ESO DET CHIP NX"]
        ny_data = self.header["ESO DET CHIP NY"]

        data = np.ma.compressed(work)
        data = np.reshape(data, (ny_data, nx_data))
        wcs = WCS(crpix=(1.0, 1.0), shape=(ny_data, nx_data))
        ima = Image(wcs=wcs, data=data, dtype=None)

        if det_out is not None:
            sy, sx = self._get_detector_indices(det_out, with_prescan=False)
            ima = ima[sy, sx]

        return ima

    def get_image_mask_overscan(self, det_out=None):
        """Return an Image object in which overscanned pixels are masked.

        Parameters
        ----------
        det_out : int in [1,4]
                  Number of output detector.
                  If None, all image is returned.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        ima = Image(wcs=WCS(self.header), data=self.trimmed(), copy=False)
        if det_out is not None:
            sy, sx = self._get_detector_indices(det_out, with_prescan=True)
            ima = ima[sy, sx]
        return ima

    def get_image_just_overscan(self, det_out=None):
        """Return an Image object in which only overscanned pixels are not
        masked.

        Parameters
        ----------
        det_out : int in [1,4]
            Number of output detector. If None, all image is returned.

        Returns
        -------
        out : `~mpdaf.obj.Image`

        """
        ima = Image(wcs=WCS(self.header), data=self.overscan(), copy=False)
        if det_out is not None:
            sy, sx = self._get_detector_indices(det_out, with_prescan=True)
            ima = ima[sy, sx]
        return ima


class RawFile(object):

    """RawFile class manages input/output for raw FITS file.

    Parameters
    ----------
    filename : str
        The raw FITS file name.

    Attributes
    ----------
    filename : str
        The raw FITS file name. None if any.
    channels : dict
        List of extension (extname,Channel)
    primary_header : `astropy.io.fits.Header`
        The primary header
    nx : int
        Lengths of data in X
    ny : int
        Lengths of data in Y
    next : int
        Number of extensions
    """

    def __init__(self, filename):
        self._logger = logging.getLogger(__name__)
        self.filename = filename
        self.primary_header = fits.Header()
        self.channels = dict()
        self.nx = 0
        self.ny = 0
        self.next = 0

        with fits.open(self.filename) as hdulist:
            self.primary_header = hdulist[0].header
            for hdu in hdulist[1:]:
                extname = hdu.header["EXTNAME"]
                exttype = hdu.header["XTENSION"]
                if exttype == 'IMAGE' and hdu.header["NAXIS"] != 0:
                    nx = hdu.header["NAXIS1"]
                    ny = hdu.header["NAXIS2"]
                    if self.nx == 0:
                        self.nx = nx
                        self.ny = ny
                    if nx != self.nx and ny != self.ny:
                        self._logger.warning(
                            'image extensions %s not considered '
                            '(different sizes)', extname)
                    else:
                        self.channels[extname] = None
            self.next = len(self.channels)

    def info(self):
        """Print information."""
        if self.filename is not None:
            msg = self.filename
        else:
            msg = 'NoName'
        self._logger.info(msg)
        self._logger.info('Nb extensions:\t%i (loaded:%i %s)', self.next,
                          len(self.channels), list(self.channels.keys()))
        msg = 'format:\t(%i,%i)' % (self.nx, self.ny)
        self._logger.info(msg)

    def get_keywords(self, key):
        """Return the keyword value."""
        return self.primary_header[key]

    def get_channels_extname_list(self):
        """Return the list of existing channels names."""
        return list(self.channels.keys())

    def get_channel(self, extname):
        """Return a Channel object.

        Parameters
        ----------
        extname : str
            The extension name.

        Returns
        -------
        out : `mpdaf.drs.Channel`
        """
        if self.channels[extname] is None:
            self.channels[extname] = Channel(extname, self.filename)
        return self.channels[extname]

    def __len__(self):
        """Return the number of extensions."""
        return self.next

    def __getitem__(self, key):
        """Load the Channel object if relevant and returns it.

        Parameters
        ----------
        key : int
            The extension number.

        Returns
        -------
        out : `mpdaf.drs.Channel`
        """
        return self.get_channel("CHAN%02d" % key)

    def plot(self, title=None, channels="all", area=None, scale='linear',
             vmin=None, vmax=None, zscale=False, colorbar=None, figsize=None,
             **kwargs):
        """Plot the raw images.

        Parameters
        ----------
        title : str
            Figure title (None by default).
        channels : list or 'all'
            List of channel names. All by default.
        area : list
            list of pixels ``[pmin,pmax,qmin,qmax]`` to zoom.
        scale : str
            'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'
            The stretch function to use for the scaling
            (default is 'linear').
        vmin : float
            Minimum pixel value to use for the scaling. If None, vmin
            is set to min of data.
        vmax : float
            Maximum pixel value to use for the scaling. If None, vmax
            is set to max of data.
        zscale : bool
            If true, vmin and vmax are computed
            using the IRAF zscale algorithm.
        colorbar : str
            If 'h'/'v', a horizontal/vertical colorbar is added.
        figsize: tuple
            width, height of the figure.
        kwargs : dict
            Additional options passed to ``Image.plot``.

        """
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(wspace=0.02, hspace=0.01)

        if title is not None:
            plt.title(title)

        if channels == "all":
            channels = self.channels.keys()

        nchan = len(channels)
        nrows = int(np.sqrt(nchan))
        if nchan % nrows == 0:
            ncols = nchan // nrows
        else:
            ncols = int(nchan // nrows) + 1

        for i, name in enumerate(channels):
            chan = self.get_channel(name)
            ima = chan.get_trimmed_image(det_out=None, bias=False)
            if area is not None:
                ima = ima[area[0]:area[1], area[2]:area[3]]
            ima = ima.rebin(6)
            ichan = int(name[-2:])
            fig.add_subplot(nrows, ncols, i + 1)
            ima.plot(None, scale, vmin, vmax, zscale, colorbar, **kwargs)
            plt.axis('off')
            plt.text(ima.shape[0] - 30, ima.shape[1] - 30, '%i' % ichan,
                     style='italic',
                     bbox={'facecolor': 'red', 'alpha': 0.2, 'pad': 10})

    def reconstruct_white_image(self, mask=None, channels="all"):
        """Reconstructs the white image of the FOV using a mask file.

        Parameters
        ----------
        mask : str
            Mask used to extract the data for each slice. Must be a FITS file
            with one extension per channel, and extension names which match the
            ones from the raw file. Defaults to ``PAE_July2013.fits.gz`` which
            comes with Mpdaf.
        channels : list or 'all'
            List of channel names. All by default.

        Returns
        -------
        `~mpdaf.obj.Image`

        """
        if mask is None:
            mask = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                'mumdatMask_1x1', 'PAE_July2013.fits.gz')

        if channels == "all":
            channels = self.get_channels_extname_list()

        raw_mask = RawFile(mask)
        white_ima = np.zeros((12 * 24, 300))

        for chan in channels:
            ifu = int(chan[-2:])
            mask_chan = raw_mask.get_channel(chan)
            ima = self.get_channel(chan).get_trimmed_image(bias=True).data.data
            mask = mask_chan.get_trimmed_image(bias=False).data.data
            ima *= mask
            spe = ima.sum(axis=0)
            data = np.empty((48, NB_SPEC_PER_SLICE))
            mhdr = mask_chan.header

            for sli in range(1, 49):
                xstart = mhdr['ESO DET SLICE%d XSTART' % sli] - OVERSCAN
                xend = mhdr['ESO DET SLICE%d XEND' % sli] - OVERSCAN
                if xstart > (mhdr["ESO DET CHIP NX"] / 2.0):
                    xstart -= 2 * OVERSCAN
                if xend > (mhdr["ESO DET CHIP NX"] / 2.0):
                    xend -= 2 * OVERSCAN

                spe_slice = spe[xstart:xend + 1]
                n = spe_slice.shape[0]

                if n < NB_SPEC_PER_SLICE:
                    spe_slice_75pix = np.zeros(NB_SPEC_PER_SLICE)
                    spe_slice_75pix[:n] = spe_slice
                elif n == NB_SPEC_PER_SLICE:
                    spe_slice_75pix = spe_slice
                else:
                    spe_slice_75pix = np.empty(NB_SPEC_PER_SLICE,
                                               dtype=float)

                f = lambda x: spe_slice[int(x + 0.5)]
                pix = np.arange(NB_SPEC_PER_SLICE + 1, dtype=float)
                new_step = float(n) / NB_SPEC_PER_SLICE
                x = pix * new_step - 0.5 * new_step

                for i in range(NB_SPEC_PER_SLICE):
                    spe_slice_75pix[i] = quad(f, x[i], x[i + 1],
                                              full_output=1)[0] / new_step

                data[sli - 1, :] = spe_slice_75pix

            # For each subslicer 1-4
            for k in range(1, NB_SUBSLICERS + 1):
                # For each slice 1-12*/
                for l in range(1, NB_SLICES + 1):
                    noslice = (k - 1) * NB_SLICES + l
                    wr_row = NB_SLICES - SLIT_POSITION[l - 1] \
                        + 12 * (24 - ifu)
                    wr_col = k * NB_SPEC_PER_SLICE
                    white_ima[wr_row, wr_col - NB_SPEC_PER_SLICE:wr_col] = \
                        data[noslice - 1, :]

        return Image(data=white_ima, wcs=WCS())

    def _plot_ifu_slice_on_white_image(self, ifu, sli):
        # plot channel
        ymin = NB_SLICES * (24 - ifu) - 0.5
        ymax = ymin + NB_SLICES
        plt.plot(np.arange(-0.5, 299.5), np.ones(300) * ymin, 'b-')
        plt.plot(np.arange(-0.5, 299.5), np.ones(300) * ymax, 'b-')
        plt.annotate('%02d' % ifu, xy=(0, (ymin + ymax) / 2.0),
                     xycoords='data', textcoords='data', color='b')
        # plot slice
        k = np.floor((sli - 1) // NB_SLICES)
        l = np.mod(sli - 1, NB_SLICES) + 1
        wr_row = NB_SLICES - SLIT_POSITION[l - 1] + 12 * (24 - ifu) - 0.5
        wr_col = k * NB_SPEC_PER_SLICE - 0.5
        plt.plot(np.arange(wr_col, wr_col + 76), np.ones(76) * wr_row, 'r-')
        plt.plot(np.arange(wr_col, wr_col + 76),
                 np.ones(76) * (wr_row + 1), 'r-')
        plt.plot(np.ones(2) * wr_col, np.arange(wr_row, wr_row + 2), 'r-')
        plt.plot(np.ones(2) * (wr_col + 75),
                 np.arange(wr_row, wr_row + 2), 'r-')
        self.whiteima.plot(cmap='copper')

    def _plot_slice_on_raw_image(self, ifu, sli, same_raw=False):
        mask_raw = RawFile(self.mask_file)
        chan = 'CHAN%02d' % ifu
        mask_chan = mask_raw.get_channel(chan)
        mhdr = mask_chan.header

        self.x1 = mhdr['ESO DET SLICE1 XSTART'] - OVERSCAN
        self.x2 = mhdr['ESO DET SLICE48 XEND'] - 2 * OVERSCAN

        xstart = mhdr['ESO DET SLICE%d XSTART' % sli] - OVERSCAN
        xend = mhdr['ESO DET SLICE%d XEND' % sli] - OVERSCAN
        if xstart > (mhdr["ESO DET CHIP NX"] / 2.0):
            xstart -= 2 * OVERSCAN
        if xend > (mhdr["ESO DET CHIP NX"] / 2.0):
            xend -= 2 * OVERSCAN

        ystart = mhdr['ESO DET SLICE%d YSTART' % sli] - OVERSCAN
        yend = mhdr['ESO DET SLICE%d YEND' % sli] - OVERSCAN
        if ystart > (mhdr["ESO DET CHIP NY"] / 2.0):
            ystart -= 2 * OVERSCAN
        if yend > (mhdr["ESO DET CHIP NY"] / 2.0):
            yend -= 2 * OVERSCAN

        plt.plot(np.arange(xstart, xend + 1),
                 np.ones(xend + 1 - xstart) * ystart, 'r-')
        plt.plot(np.arange(xstart, xend + 1),
                 np.ones(xend + 1 - xstart) * yend, 'r-')
        plt.plot(np.ones(yend + 1 - ystart) * xstart,
                 np.arange(ystart, yend + 1), 'r-')
        plt.plot(np.ones(yend + 1 - ystart) * xend,
                 np.arange(ystart, yend + 1), 'r-')
        plt.annotate('%02d' % sli, xy=(xstart, yend + 1),
                     xycoords='data', textcoords='data', color='r')
        if same_raw is False:
            self.rawima = self.get_channel(chan).get_trimmed_image()
        self.rawima.plot(title='CHAN%02d' % ifu, cmap='copper')
        self.plotted_chan = ifu

    def _onclick(self, event):
        if event.button != 1:
            if event.inaxes is not None:
                if (event.x < self.fig.canvas.get_width_height()[0] // 2):
                    p = event.ydata
                    q = event.xdata
                    ifu = 24 - int(p + 0.5) // NB_SLICES
                    k = int(q + 0.5) // NB_SPEC_PER_SLICE
                    pos = NB_SLICES + 12 * (24 - ifu) - int(p + 0.5)
                    l = np.where(SLIT_POSITION == pos)[0][0] + 1
                    sli = k * NB_SLICES + l
                    ax = plt.subplot(1, 2, 1)
                    ax.clear()
                    self._plot_ifu_slice_on_white_image(ifu, sli)
                    ax = plt.subplot(1, 2, 2)
                    ax.clear()
                    if ifu == self.plotted_chan:
                        self._plot_slice_on_raw_image(ifu, sli,
                                                      same_raw=True)
                    else:
                        self._plot_slice_on_raw_image(ifu, sli)
                else:
                    p = event.ydata
                    q = event.xdata
                    nq = (self.x2 - self.x1) // 48
                    sli = int((q + 0.5 - self.x1) / nq) + 1
                    ax = plt.subplot(1, 2, 2)
                    ax.clear()
                    self._plot_slice_on_raw_image(self.plotted_chan,
                                                  sli, same_raw=True)
                    ax = plt.subplot(1, 2, 1)
                    ax.clear()
                    self._plot_ifu_slice_on_white_image(self.plotted_chan,
                                                        sli)

    def plot_white_image(self, mask=None, channels="all"):
        """Reconstructs the white image of the FOV using a mask file and plots
        this image.

        Parameters
        ----------
        mask : str
            Mask used to extract the data for each slice. Must be a FITS file
            with one extension per channel, and extension names which match the
            ones from the raw file. Defaults to ``PAE_July2013.fits.gz`` which
            comes with Mpdaf.
        channels : list or 'all'
            List of channel names. All by default.

        """
        if mask is None:
            mask = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                'mumdatMask_1x1', 'PAE_July2013.fits.gz')

        self.mask_file = mask
        # create image
        self.whiteima = self.reconstruct_white_image(self.mask_file,
                                                     channels=channels)
        # highlighted ifu
        selected_ifu = 12
        # plot white image
        self.fig = plt.figure()
        plt.subplot(1, 2, 1)
        self._plot_ifu_slice_on_white_image(selected_ifu, 1)
        # plot raw image
        plt.subplot(1, 2, 2)
        self._plot_slice_on_raw_image(selected_ifu, 1)
        self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        print('To select on other channel/slice, '
              'click on the images with the right mouse button.')
