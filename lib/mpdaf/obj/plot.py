"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2019 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
import astropy.units as u
import logging
import numpy as np

__all__ = ('FormatCoord', 'get_plot_norm', 'plot_rgb')


class FormatCoord:
    """Alter mouse-over coordinates displayed by plt.show()"""

    def __init__(self, image, data):
        self.image = image
        self.data = data

    def __call__(self, x, y):  # pragma: no cover
        """Tell the interactive plotting window how to display the sky
        coordinates and pixel values of an image.

        Parameters
        ----------
        x : float
            The X-axis pixel index of the mouse pointer.
        y : float
            The Y-axis pixel index of the mouse pointer.

        Returns
        -------
        out : str
            The string to be displayed when the mouse pointer is
            over pixel x,y.

        """
        # Find the pixel indexes closest to the specified position.
        col = int(x + 0.5)
        row = int(y + 0.5)

        # Is the mouse pointer within the image?
        im = self.image
        if (im.wcs is not None and row >= 0 and row < im.shape[0] and
                col >= 0 and col < im.shape[1]):
            yc, xc = im.wcs.pix2sky([row, col], unit=im._unit)[0]
            val = self.data[row, col]
            if np.isscalar(val):
                return 'y= %g x=%g p=%i q=%i data=%g' % (yc, xc, row, col, val)
            else:
                return 'y= %g x=%g p=%i q=%i data=%s' % (yc, xc, row, col, val)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)


def get_plot_norm(data, vmin=None, vmax=None, zscale=False, scale='linear'):
    from astropy import visualization as viz
    from astropy.visualization.mpl_normalize import ImageNormalize

    # Choose vmin and vmax automatically?
    if zscale:
        interval = viz.ZScaleInterval()
        if data.dtype == np.float64:
            try:
                vmin, vmax = interval.get_limits(data.filled(np.nan))
            except Exception:
                # catch failure on all NaN
                if np.all(np.isnan(data.filled(np.nan))):
                    vmin, vmax = (np.nan, np.nan)
                else:
                    raise
        else:
            vmin, vmax = interval.get_limits(data.filled(0))

    # How are values between vmin and vmax mapped to corresponding
    # positions along the colorbar?
    if scale == 'linear':
        stretch = viz.LinearStretch
    elif scale == 'log':
        stretch = viz.LogStretch
    elif scale in ('asinh', 'arcsinh'):
        stretch = viz.AsinhStretch
    elif scale == 'sqrt':
        stretch = viz.SqrtStretch
    else:
        raise ValueError('Unknown scale: {}'.format(scale))

    # Create an object that will be used to map pixel values
    # in the range vmin..vmax to normalized colormap indexes.
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch(), clip=False)

    return norm


def plot_rgb(images, title=None, scale='linear', vmin=None, vmax=None,
             zscale=False, show_xlabel=False, show_ylabel=False, ax=None,
             unit=u.deg, use_wcs=False, **kwargs):
    """Plot the RGB composite image with axes labeled in pixels.

    For each color, final intensity values are assigned to each pixel as
    follows. First each pixel value, ``pv``, is normalized over the range
    ``vmin`` to ``vmax``, to have a value ``nv``, that goes from 0 to 1, as
    follows::

        nv = (pv - vmin) / (vmax - vmin)

    This value is then mapped to another number between 0 and 1 which
    determines the final value to give the displayed pixel. The mapping from
    normalized values to final value can be chosen using the scale argument,
    from the following options:

    - 'linear': ``color = nv``
    - 'log': ``color = log(1000 * nv + 1) / log(1000 + 1)``
    - 'sqrt': ``color = sqrt(nv)``
    - 'arcsinh': ``color = arcsinh(10*nv) / arcsinh(10.0)``

    By default the image is displayed in its own plot. Alternatively
    to make it a subplot of a larger figure, a suitable
    ``matplotlib.axes.Axes`` object can be passed via the ``ax`` argument.
    Note that unless matplotlib interative mode has previously been enabled
    by calling ``matplotlib.pyplot.ion()``, the plot window will not appear
    until the next time that ``matplotlib.pyplot.show()`` is called. So to
    arrange that a new window appears as soon as ``plot_rgb`` is
    called, do the following before the first call to ``plot_rgb``::

        import matplotlib.pyplot as plt
        plt.ion()

    Parameters
    ----------
    images : [`~mpdaf.obj.Image`, `~mpdaf.obj.Image`, `~mpdaf.obj.Image`]
        The three [blue, green, red] images to be used. i.e. ordered by
        increasing wavelength.
    title : str
        An optional title for the figure (None by default).
    scale : 'linear' | 'log' | 'sqrt' | 'arcsinh'
        The stretch function to use mapping pixel values to
        final values (The default is 'linear'). The same scaling is applied to
        all three imasges. The pixel values are
        first normalized to range from 0 for values <= vmin,
        to 1 for values >= vmax, then the stretch algorithm maps
        these normalized values, nv, to a position p from 0 to 1
        along the colorbar, as follows:
        linear:  p = nv
        log:     p = log(1000 * nv + 1) / log(1000 + 1)
        sqrt:    p = sqrt(nv)
        arcsinh: p = arcsinh(10*nv) / arcsinh(10.0)
    vmin : [float, float, float]
        Lower limits corresponing to the [blue, green, red] images.
        Pixels that have values <= vmin are assigned a value of 0.
        Pixel values between vmin and vmax are scaled according
        to the mapping algorithm specified by the scale argument.
    vmax : [float, float, float]
        Upper limits corresponing to the [blue, green, red] images.
        Pixels that have values >= vmax are assigned a value of 1.
        Pixel values between vmin and vmax are scaled according
        to the mapping algorithm specified by the scale argument.
    zscale : bool
        If True, vmin and vmax are automatically computed
        using the IRAF zscale algorithm.
    ax : matplotlib.axes.Axes
        An optional Axes instance in which to draw the image,
        or None to have one created using ``matplotlib.pyplot.gca()``.
    unit : `astropy.units.Unit`
        The units to use for displaying world coordinates
        (degrees by default). In the interactive plot, when
        the mouse pointer is over a pixel in the image the
        coordinates of the pixel are shown using these units,
        along with the pixel value.
    use_wcs : bool
        If True, use `astropy.visualization.wcsaxes` to get axes
        with world coordinates.
    kwargs : matplotlib.artist.Artist
        Optional extra keyword/value arguments to be passed to
        the ``ax.imshow()`` function.

    Returns
    -------
    ax : matplotlib AxesImage
    images_aligned : `~mpdaf.obj.Image`, `~mpdaf.obj.Image`, `~mpdaf.obj.Image`
        The input images, but all aligned to that with the highest resolution.

    """
    if vmin is None:
        vmin = [None, None, None]

    if vmax is None:
        vmax = [None, None, None]

    # Default X and Y axes are labeled in pixels.
    xlabel = 'q (pixel)'
    ylabel = 'p (pixel)'

    if ax is None:
        import matplotlib.pyplot as plt
        if use_wcs:
            ax = plt.subplot(projection=images[0].wcs.wcs)
            xlabel = 'ra'
            ylabel = 'dec'
        else:
            ax = plt.gca()
    elif use_wcs:
        logging.getLogger(__name__).warning(
            'use_wcs does not work when giving also an axis (ax)')

    # find which image has the highest pixel resolution
    # also find bbox that encloses all 3 images
    steps = np.full([3, 2], np.nan, dtype=float)
    corners = np.full([3, 4, 2], np.nan, dtype=float)
    for i_im, im in enumerate(images):
        wcs = im.wcs
        step = wcs.get_axis_increments(unit=u.deg)
        corn = wcs.wcs.calc_footprint(axes=[wcs.naxis1, wcs.naxis2])

        steps[i_im] = step
        corners[i_im] = corn

    idx_best_res = np.argmin(np.mean(np.abs(steps), 1))
    im_best_res = images[idx_best_res]  # image with highest res

    # get bounding pixel coords in best image
    corners = np.vstack(corners)
    corners = im_best_res.wcs.wcs.all_world2pix(corners, 0)
    new_shape = np.array([[np.min(corners[:, 0]), np.max(corners[:, 0])],
                          [np.min(corners[:, 1]), np.max(corners[:, 1])]])
    new_shape = np.around(new_shape).astype(int)

    new_dim = new_shape[:, 1] - new_shape[:, 0] + 1
    new_start = new_shape[:, 0].reshape(1, 2)
    new_start = im_best_res.wcs.wcs.all_pix2world(new_start, 0)[0]

    new_dim = new_dim[::-1]  # naxis2, naxis1
    new_start = new_start[::-1]  # dec, ra
    old_inc = im_best_res.get_axis_increments(unit=u.deg)

    # expand the reference image so that it now covers the footprints of the
    # other 2 images
    im_best_res = im_best_res.resample(new_dim, new_start, old_inc,
                                       unit_step=u.deg)

    # create BGR stack
    data_stack = np.full(im_best_res.shape + (3,), np.nan, dtype=float)
    data_stack = np.ma.array(data_stack)

    images_aligned = []
    for i, im in enumerate(images):
        # align all images to image with best res
        im = im.align_with_image(im_best_res)
        images_aligned.append(im)
        data = im.data

        norm = get_plot_norm(data, vmin=vmin[i], vmax=vmax[i], zscale=zscale,
                             scale=scale)

        data = norm(data)

        data_stack[:, :, i] = data

    data_stack = np.ma.clip(data_stack, 0, 1)
    data_stack = data_stack.filled(np.nan)

    # reverse BGR to RGB order
    data_stack = data_stack[:, :, ::-1]

    # mask all NaNs and plot transparent
    mask = np.all(np.isnan(data_stack), axis=2)
    alpha = ~mask * 1.  # no transparency where data is good
    data_stack = np.concatenate([data_stack, alpha[..., np.newaxis]], axis=2)

    # Display the RGBA image.
    ax.imshow(data_stack, interpolation='nearest', origin='lower', **kwargs)

    # Keep the axis to allow other functions to overplot
    # the image with contours etc.
    for im in images_aligned:
        im._ax = ax

    # Label the axes if requested.
    if show_xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # Change the way that plt.show() displays coordinates when the pointer
    # is over the image, such that world coordinates are displayed with the
    # specified unit, and pixel values are displayed with their native
    # units.
    ax.format_coord = FormatCoord(images_aligned[0], data_stack)
    for im in images_aligned:
        im._unit = unit
    return ax, images_aligned
