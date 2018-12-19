"""
Helper functions to plot images and spectra with Bokeh.

"""

import numpy as np

from astropy.visualization import (ZScaleInterval, PercentileInterval,
                                   MinMaxInterval)
from bokeh.palettes import Category10
# from bokeh.io import export_png, reset_output, export_svgs, curdoc
# from bokeh.layouts import gridplot, layout, row, column
from bokeh.models import Range1d, Span, Label, LabelSet, Legend
from bokeh.models import ColorBar, LinearColorMapper
# from bokeh.palettes import viridis, Category10
from bokeh.plotting import figure, ColumnDataSource
# from bokeh.themes import built_in_themes
from mpdaf.sdetect import get_emlines


def plot_image(im, size=(350, 350), title=None, colorbar=True, axis=True,
               palette="Viridis256", scale='minmax', x_range=None,
               y_range=None, center=None, catalog=None, show_tooltip=False):
    """Plot an image.

    Parameters
    ----------
    im : `mpdaf.obj.Image`
        The image to plot.
    size : (int, int)
        Size of the figure.
    title : str
        Title of the figure.
    colorbar : bool
        If True, a colorbar is shown.

    """

    if scale == 'zscale':
        interval = ZScaleInterval()
    elif scale == 'percentile':
        interval = PercentileInterval(99)
    elif scale == 'minmax':
        interval = MinMaxInterval()

    vmin, vmax = interval.get_limits(im.data)
    color_mapper = LinearColorMapper(palette=palette, low=vmin, high=vmax)
    if x_range is None:
        x_range = (0, im.shape[0])
    if y_range is None:
        y_range = (0, im.shape[1])

    if show_tooltip and catalog:
        tooltips = [
            ("ID", "@ID"),
            # ("(x,y)", "($x, $y)"),
            ("pos", "(@RA, @DEC)"),
            ("mag F775W", "@MAG_F775W"),
        ]
    else:
        tooltips = None

    p = figure(plot_width=size[0], plot_height=size[1], tooltips=tooltips,
               x_range=x_range, y_range=y_range, title=title)

    p.image(image=[im.data.filled()], x=[0], y=[0], dw=[im.shape[0]],
            dh=[im.shape[1]], color_mapper=color_mapper)
    p.grid.visible = False
    p.axis.visible = axis

    if center:
        y, x = im.wcs.sky2pix(center)[0]
        span = Span(location=x, dimension='height', line_color='white',
                    line_width=1, line_alpha=0.5)
        p.add_layout(span)
        span = Span(location=y, dimension='width', line_color='white',
                    line_width=1, line_alpha=0.5)
        p.add_layout(span)
        p.circle(x, y, size=10, line_color='red', line_width=2, line_alpha=0.6,
                 fill_color=None)

    if catalog:
        p.circle('x', 'y', source=catalog, size=5, line_color='white',
                 line_width=1, line_alpha=0.6, fill_color=None)
        label = LabelSet(x='x', y='y', source=catalog, x_offset=2, y_offset=2,
                         text='ID', text_font_size='10px', text_color='white')
        p.add_layout(label)

    if colorbar:
        color_bar = ColorBar(color_mapper=color_mapper,  # ticker=LogTicker(),
                             label_standoff=12, border_line_color=None,
                             location=(0, 0))
        p.add_layout(color_bar, 'right')

    return p


def plot_src_images(src, params, size=(350, 350), **kwargs):
    """Plot a list of images from a Source.

    Images to plot must be defined in the ``params`` dict, where keys are the
    image names, and the values can be a dict of parameters to configure the
    plots:

    - scale:
    - palette:
    - link:

    Example::

        images = {
            'MUSE_WHITE': {'scale': 'zscale'},
            'MUSE_EXPMAP': {'scale': 'minmax', 'palette': 'Greys256',
                            'link': 'MUSE_WHITE'},
            'MASK_OBJ': {'scale': 'minmax', 'palette': 'Greys256',
                        'link': 'MUSE_WHITE'},
            'HST_F606W': {'scale': 'percentile'},
            'HST_F775W': {'scale': 'percentile', 'link': 'HST_F606W'}
        }

    Parameters
    ----------
    src : `mpdaf.sdetect.Source`
        The input Source.
    params : dict
        Dict of parameters to configure each plot.
    size : (int, int)
        Size of each plot.

    """
    plots = {}

    cat = src.tables['HST_CAT'].copy(copy_data=True)
    skypos = np.array([cat['DEC'], cat['RA']]).T
    data = ColumnDataSource(cat.to_pandas())

    for name, param in params.items():
        im = src.images[name]
        kw = {'size': size, 'title': name, 'colorbar': False, 'axis': False,
              'scale': param.get('scale', 'minmax'),
              'palette': param.get('palette', 'Viridis256'), **kwargs}
        if param.get('link') is not None:
            kw['x_range'] = plots[param['link']].x_range
            kw['y_range'] = plots[param['link']].y_range

        cat['y'], cat['x'] = im.wcs.sky2pix(skypos).T
        data = ColumnDataSource(cat.to_pandas())

        if name.startswith(('HST_', 'MUSE_WHITE')):
            p = plot_image(im, center=(src.DEC, src.RA), catalog=data, **kw)
        else:
            p = plot_image(im, center=(src.DEC, src.RA), **kw)

        plots[name] = p

    return list(plots.values())


def plot_lines(p, src, lines_from_src=True):
    """Plot lines."""
    if lines_from_src:
        lines = src.lines
    else:
        z = src.z[src.z['Z_DESC'] == 'MUSE'][0]['Z']
        sp = next(src.spectra.values())
        lines = get_emlines(z=z, lbrange=sp.wave.get_range(), table=True)

    for line in lines:
        span = Span(location=line['LBDA_OBS'], dimension='height',
                    line_color='black', line_width=1, line_alpha=0.6,
                    line_dash='dashed')
        p.add_layout(span)
        label = Label(x=line['LBDA_OBS'], y=p.plot_height - 120,
                      y_units='screen', angle=90, angle_units='deg',
                      text=line['LINE'], text_font_size='10px')
        p.add_layout(label)


def plot_spectrum(src, size=(800, 350), axis_labels=True, lbrange=None,
                  show_legend=True, lines_from_src=True,
                  spectra_names=('MUSE_TOT_SKYSUB', 'MUSE_PSF_SKYSUB',
                                 'MUSE_WHITE_SKYSUB')):
    """Plot spectra from a Source.

    Parameters
    ----------
    src : `mpdaf.sdetect.Source`
        The input Source.
    size : (int, int)
        Size of the plot.
    axis_labels : bool
        If True, show axis labels.
    lbrange : (float, float)
        lmin and lmax values to extract a part of a spectrum.
    show_legend : bool
        If True, show the legend.
    lines_from_src : bool
        If True, plot lines from the Source.lines table, otherwise the lines
        are computed from the Z_MUSE redshift.
    spectra_names : list of str
        List of spectra names to plot.

    """
    p = figure(plot_width=size[0], plot_height=size[1])
    palette = Category10[8]

    # plot lines
    plot_lines(p, src, lines_from_src=lines_from_src)

    # plot spectra
    legend_items = []
    smin, smax = np.inf, -np.inf
    for i, sname in enumerate(spectra_names):
        sp = src.spectra[sname]
        smin = min(smin, sp.data.min())
        smax = max(smax, sp.data.max())
        if lbrange:
            sp = sp.subspec(lbrange[0], lbrange[1])
        line = p.line(sp.wave.coord(), sp.data, color=palette[i])
        legend_items.append((sname.lstrip('MUSE_'), [line]))
        if i > 0:
            line.visible = False

    # add variance with an extra axis
    p.extra_y_ranges = {"var": Range1d(start=0, end=sp.var.max())}
    p.y_range = Range1d(smin - 20, smax + 20)
    p.line(sp.wave.coord(), sp.var, line_color='gray', line_alpha=0.6,
           y_range_name="var")
    # p.add_layout(LinearAxis(y_range_name="var"), 'left')

    # customize legend, plotted outside of the graph
    legend = Legend(items=legend_items, location=(0, 0))
    p.add_layout(legend, 'above')
    p.legend.location = "top_left"
    p.legend.visible = show_legend
    p.legend.label_text_font_size = '12px'
    p.legend.padding = 0
    p.legend.background_fill_alpha = 0.5
    p.legend.orientation = "horizontal"
    p.legend.click_policy = "hide"

    p.yaxis.major_label_orientation = "vertical"
    if axis_labels:
        p.xaxis.axis_label = f'Wavelength ({sp.wave.unit})'
        p.yaxis.axis_label = f'Flux ({sp.unit})'

    return p


def plot_spectrum_lines(src, spname='MUSE_TOT_SKYSUB', size=(250, 250),
                        lines=None, nlines=1, lbda_width=20):
    palette = Category10[8]
    sp = src.spectra[spname]

    if lines is None:
        src.lines.sort('FLUX_REF')
        lines = src.lines[-nlines:]

    figures = []
    for line in lines:
        p = figure(plot_width=size[0], plot_height=size[1], title=line['LINE'])
        lbda = line['LBDA_OBS']
        subsp = sp.subspec(lbda - lbda_width // 2, lbda + lbda_width // 2)
        line = p.line(subsp.wave.coord(), subsp.data, color=palette[0])
        figures.append(p)

    return figures
