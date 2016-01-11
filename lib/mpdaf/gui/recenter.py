# -*- coding: utf-8 -*-

import matplotlib.pylab as plt


def plot_muse_field(image, centers=None):
    centers = centers or []
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    bckg = image.background()[0]
    image.plot(vmin=bckg-2, vmax=bckg+2, colorbar='v')
    for center in centers:
        y, x = image.wcs.sky2pix(center)[0]
        plt.axhline(y, color='k')
        plt.axvline(x, color='k')
    plt.subplot(1, 2, 2)
    image.plot(zscale=True)
    for center in centers:
        y, x = image.wcs.sky2pix(center)[0]
        plt.axhline(y, color='k')
        plt.axvline(x, color='k')


def plot_subimages(zmuse, zhst, center):
    nl = 1
    nc = 3
    plt.figure(figsize=(15, 5 * nl))

    plt.subplot2grid((nl, nc), (0, 0))
    zmuse.plot()
    y, x = zmuse.wcs.sky2pix(center)[0]
    plt.axhline(y, color='k')
    plt.axvline(x, color='k')

    y, x = zhst.wcs.sky2pix(center)[0]
    plt.subplot2grid((nl, nc), (0, 1))
    zhst.plot()
    plt.axhline(y, color='k')
    plt.axvline(x, color='k')

    plt.subplot2grid((nl, nc), (0, 2))
    zhst.plot(zscale=True)
    plt.axhline(y, color='k')
    plt.axvline(x, color='k')


def plot_convolved(zhst, chst, rhst):
    nl = 1
    nc = 3
    plt.figure(figsize=(15, 5 * nl))
    # Display result
    plt.subplot2grid((nl, nc), (0, 0))
    zhst.plot()
    plt.subplot2grid((nl, nc), (0, 1))
    chst.plot()
    plt.subplot2grid((nl, nc), (0, 2))
    rhst.plot()


def plot_correlation(auto, auto_center, xcorr, xcorr_center):
    nl = 1
    nc = 2
    plt.figure(figsize=(10, 5 * nl))
    # plot results
    plt.subplot2grid((nl, nc), (0, 0))
    auto.plot()
    y, x = auto_center
    plt.axhline(y, color='w')
    plt.axvline(x, color='w')
    plt.subplot2grid((nl, nc), (0, 1))
    xcorr.plot()
    y, x = xcorr_center
    plt.axhline(y, color='w')
    plt.axvline(x, color='w')


def plot_recentered(zmuse, zhst, center, offpix):
    nl = 1
    nc = 2
    plt.figure(figsize=(15, 5 * nl))
    wcs = zmuse.wcs.copy()
    wcs.info()
    wcs.set_crpix1(wcs.get_crpix1() + offpix[1])
    wcs.set_crpix2(wcs.get_crpix2() + offpix[0])
    wcs.info()
    zoffmuse = zmuse.copy()
    zoffmuse.set_wcs(wcs)
    # HST ref center
    zmusepix = zmuse.wcs.sky2pix(center)[0]
    zoffmusepix = zoffmuse.wcs.sky2pix(center)[0]
    # now perform plots
    plt.subplot2grid((nl, nc), (0, 0))
    zmuse.plot()
    y, x = zmusepix
    plt.axhline(y, color='w')
    plt.axvline(x, color='w')
    plt.subplot2grid((nl, nc), (0, 1))
    zoffmuse.plot()
    y, x = zoffmusepix
    plt.axhline(y, color='w')
    plt.axvline(x, color='w')


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='[%(levelname)s] %(message)s')
    # EXP = 'IMAGE-MUSE.2014-10-26T08:27:12.752.fits'
    EXP = 'IMAGE-MUSE.2014-09-27T05:32:52.054.fits'
    musefile = ('/muse/UDF/private/datared/0.2/scipost/'
                'IMAGE-MUSE.2014-11-21T06:03:16.330.fits')
    cat = '/muse/UDF/private/datared/0.1/ref-files/UDF-RefObject-Centering.dat'
    hstfile = '/muse/UDF/private/HST/h_udf_wfc_i_drz_img.fits'
    # find_wcs_offsets(musefile, hstfile, cat)
