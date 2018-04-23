
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tqdm

from astropy.modeling import fitting, custom_model
from astropy.modeling.models import Moffat2D, Const2D, Linear1D
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.table import vstack
from matplotlib import gridspec
from photutils.detection import find_peaks
from scipy.special import j1, jn_zeros

from ..obj import Image, Cube
from ..obj.nfm import (SmoothDisk2D, SmoothOuterDisk2D, EllipticalMoffat2D,
                       AiryDisk2D)

# Magic parameters from the IDL code
ACT = .19    # [m] Inter-actuator distance in the pupil plane
FOC = 1800.  # [m] Focal length
PIX = 30.*1e-6/7.*5.24*4.8*2.*1.1  # [m] Pixel size
OCC = 1.12   # [m] Occultation
D = 8.       # [m] Aperture

BETA_H = 11./6.

RZ = jn_zeros(1, 1)[0] / np.pi


def compute_airy_radius(lbda):
    """
    omega = pix*D/lambda/f

    PIX     - Size of pixels, in meter per pix
    D       - Diameter of pupil, in meter
    LAMBDA  - Wavelength, in meter
    F       - Focal length, in meter

    """
    omega = PIX * D / (lbda * 1e-10) / FOC
    return omega
    # return (
    #     lbda * 1e-10  # lbda in meter
    #     / 8           # / D (telescop diameter)
    #     * 206265000   # radian to mas
    #     / 25          # mas to pixel
    #     * 1.22        # radius for astropy
    # )


def correction_radius(lbda):
    """Compute RC (OA cutoff radius in the focal plane) from ACT"""

    Rc = .5 / ACT * lbda * 1e-10 * FOC / PIX
    return Rc

    # previous formula wass derived from R0 - Lbda: 685 - 11.3, 870 - 14.3
    # a = (14.3 - 11.3) / (8700 - 6850)
    # b = 11.3 - a * 6850
    # return a*lbda + b


def mphd_model(shape, lbda, with_ellipticity=False, R_0=None, verbose=False,
               with_bounds=False, peak=1):
    # center
    x_0, y_0 = np.array(shape)/2

    R_c = correction_radius(lbda) if R_0 is None else R_0
    if verbose:
        print(f'Initial Rc for $\lambda$ {lbda} is {R_c:.2f}')

    if with_ellipticity:
        Mpeak = EllipticalMoffat2D(x_0=x_0, y_0=y_0, alpha=1., name='Mpeak',
                                   amplitude=peak/2)
        Mhalo = EllipticalMoffat2D(x_0=x_0, y_0=y_0, alpha=2., name='Mhalo')
        # Mpeak.amplitude.min = 0
    else:
        Mpeak = Moffat2D(x_0=x_0, y_0=y_0, alpha=1., name='Mpeak',
                         amplitude=peak)
        Mhalo = Moffat2D(x_0=x_0, y_0=y_0, alpha=2., name='Mhalo',
                         amplitude=peak/2)

    W = SmoothDisk2D(x_0=x_0, y_0=y_0, R_0=R_c, name='W')
    Winv = SmoothOuterDisk2D(x_0=x_0, y_0=y_0, R_0=R_c, name='Winv')
    W.amplitude.fixed = True
    Winv.amplitude.fixed = True

    if R_0 is not None:
        W.R_0.fixed = True
        Winv.R_0.fixed = True
    else:
        Winv.R_0.tied = lambda g: g['W'].R_0

    if with_bounds:
        Mpeak.alpha.min = 0
        Mhalo.alpha.min = 0
        W.R_0.min = R_c - 4
        W.R_0.max = R_c + 4

    R_airy = compute_airy_radius(lbda)
    # airy = Gaussian2D(x_mean=x_0, y_mean=y_0, x_stddev=R_airy,
    #                   y_stddev=R_airy, name='Airy')
    # airy.theta.fixed = True
    airy = AiryDisk2D(x_0=x_0, y_0=y_0, radius=R_airy*2, name='Airy',
                      amplitude=peak)
    # airy.radius.fixed = True
    # airy.amplitude.min = 1e3
    if verbose:
        print(f'Initial $R_airy$ for $\lambda$ {lbda} is {R_airy:.2f}')

    Bp = Const2D(name='Bp', amplitude=0)

    model = (((Mpeak + Bp) * W + airy)) + (Mhalo * Winv)

    # centers must be tied together
    for mod in model[1:]:
        # if mod.name == 'Airy':
        #     mod.x_mean.tied = lambda g: g['Mpeak'].x_0
        #     mod.y_mean.tied = lambda g: g['Mpeak'].y_0
        if mod.name != 'Bp':
            mod.x_0.tied = lambda g: g['Mpeak'].x_0
            mod.y_0.tied = lambda g: g['Mpeak'].y_0

    return model


@custom_model
def mphd(x, y, x_0=0, y_0=0, R_c=1, Airy_radius=1, Airy_amplitude=1,
         Mpeak_amplitude=1, Mpeak_alpha=1, Mpeak_gamma=1, Bp=0,
         Mhalo_amplitude=1, Mhalo_gamma=1):

    # print(x_0, y_0, R_c, Airy_radius, Airy_amplitude,
    #       Mpeak_amplitude, Mpeak_alpha, Mpeak_gamma, Bp,
    #       Mhalo_amplitude, BETA_H, Mhalo_gamma)

    rr = ((x - x_0) ** 2 + (y - y_0) ** 2)
    rr2 = np.sqrt(rr)

    # Peak Moffat
    rr_gg = rr / Mpeak_gamma ** 2
    Mpeak = Bp + Mpeak_amplitude * (1 + rr_gg) ** (-Mpeak_alpha)

    # Halo Moffat
    rr_gg = rr / Mhalo_gamma ** 2
    Mhalo = Mhalo_amplitude * (1 + rr_gg) ** (-BETA_H)

    W = 1 / (1 + np.exp((rr2 - R_c)))

    Airy = np.ones(rr2.shape)
    mask = rr2 > 0
    rt = rr2[mask] * (np.pi * RZ / Airy_radius)
    Airy[mask] = (2.0 * j1(rt) / rt) ** 2
    Airy *= Airy_amplitude

    return Mpeak * W + Airy + Mhalo * (1 - W)


def mphd_model2(shape, lbda, R_0=None, verbose=False, with_bounds=False,
                peak=1):
    # initial parameters
    x_0, y_0 = np.array(shape)/2
    R_c = (1.22 * correction_radius(lbda)) if R_0 is None else R_0
    R_airy = compute_airy_radius(lbda)

    r0 = 0.25      # [m] Fried parameter
    # strehl = 0.22  # Strehl ratio (between 0 and 1)
    # κ known conversion factor from the atmospheric turbulence to
    # intensity in the focal plane
    kappa = lbda * 1e-10 * FOC / PIX
    ah = 0.2301 * kappa / r0
    Mhalo_amplitude = (BETA_H - 1) / np.pi  # 0.2652

    if verbose:
        print(f'Initial R_c for lambda {lbda} is {R_c:.2f}')
        print(f'Initial R_airy for lambda {lbda} is {R_airy:.2f}')

    model = mphd(x_0=x_0, y_0=y_0, R_c=R_c, Airy_radius=R_airy*2,
                 Airy_amplitude=peak,
                 Mpeak_amplitude=peak, Mpeak_alpha=1, Mpeak_gamma=1, Bp=0,
                 Mhalo_amplitude=Mhalo_amplitude, Mhalo_gamma=ah)

    if R_0 is not None:
        model.R_c.fixed = True

    if with_bounds:
        model.Airy_radius.min = 0.1
        model.Mpeak_gamma.min = 0.1
        model.Mhalo_gamma.min = 0.1
        model.Mpeak_alpha.min = 0.1
        model.Bp.min = 0.
        model.R_c.min = R_c - 4
        model.R_c.max = R_c + 4

    return model


def get_fit_params(fit):
    if fit.n_submodels() > 1:
        params = {}
        for model in fit:
            pars = dict(zip((f'{model.name}_{n}' for n in model.param_names),
                            model.parameters))
            if model.name in ('Mpeak', 'Mhalo'):
                pars[f'{model.name}_fwhm'] = model.fwhm
            if model.name != 'Mpeak':
                pars = {k: v for k, v in pars.items()
                        if not k.endswith(('x_0', 'y_0'))}
            params.update(pars)
    else:
        params = dict(zip(fit.param_names, fit.parameters))
        params['Mpeak_fwhm'] = moffat_to_fwhm(fit.Mpeak_alpha.value,
                                              fit.Mpeak_gamma.value)
        params['Mhalo_fwhm'] = moffat_to_fwhm(BETA_H,
                                              fit.Mhalo_gamma.value)

    return params


def moffat_to_fwhm(alpha, gamma):
    return 2.0 * gamma * np.sqrt(2.0 ** (1.0 / alpha) - 1.0)


def fit_mphd(img, lbda, use_var=False, maxiter=2000, with_ellipticity=False,
             R_c=None, verbose=True, savefig=None, return_all=False,
             custom_model=True):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    fitter = fitting.LevMarLSQFitter()
    if custom_model:
        model = mphd_model2(img.shape, lbda, R_0=R_c, verbose=verbose,
                            peak=img.data.max())
    else:
        model = mphd_model(img.shape, lbda, with_ellipticity=with_ellipticity,
                           R_0=R_c, verbose=verbose, peak=img.data.max())
    kwargs = dict(maxiter=maxiter)
    if use_var:
        kwargs['weights'] = 1 / np.sqrt(img.var.filled(np.inf))

    fit = fitter(model, xx, yy, img._prepare_data(), **kwargs)
    fit_info = fitter.fit_info

    if verbose:
        print(f"{fit_info['nfev']} iterations, {fit_info['message']}")

    if savefig is not None:
        title = (f"slice ${int(lbda)} \AA$ - "
                 f"ierr:{fit_info['ierr']}, nfev:{fit_info['nfev']}\n"
                 f"{fit!r}")
        if fit_info['ierr'] not in (1, 2, 3, 4):
            title += f", msg:{fit_info['message']}"
        plot_residual(img, fit, title=title, figsize=(12, 9))
        plt.savefig(savefig, transparent=False)
        plt.close()

    if return_all:
        return fit, fit_info
    else:
        params = get_fit_params(fit)
        params.update({k: fit_info[k] for k in ('ierr', 'nfev')})
        return params, fit(xx, yy)


def fit_mphd_cube(cube, center, size, ind=None, use_var=True, verbose=False,
                  maxiter=2000, outdir=None, R_c=None):
    subcube = cube.subcube(center, size, unit_center=None, unit_size=None)
    lbda = subcube.wave.coord()
    yy, xx = np.mgrid[:subcube.shape[1], :subcube.shape[2]]

    if ind is not None:
        lbda = lbda[ind]
        subcube = subcube[ind]

    res = []
    for im, l in tqdm.tqdm(zip(subcube, lbda), total=subcube.shape[0]):
        if not np.all(np.isnan(im._data)):
            r = (R_c[0] * l + R_c[1]) if R_c is not None else None
            figname = f'{outdir}/{center[0]}-{center[1]}-{int(l)}.png'
            params, fitres = fit_mphd(im, l, use_var=use_var, verbose=verbose,
                                      maxiter=maxiter, savefig=figname, R_c=r)
            params['lbda'] = l
            res.append((params, fitres))

    params, images = zip(*res)
    cube_res = Cube.new_from_obj(subcube, data=np.array(images), var=False)

    t = Table(data=params)
    for col in t.columns.values():
        if col.info.dtype == np.float64:
            col.format = '.2f'

    return cube_res, t


def fit_mphd_cube_all_stars(cube_name, img_name, outdir, nstars=1, R_c=None,
                            fitsize=(80, 80), maxiter=2000, step=None):
    white = Image(img_name)
    mean, std = white.background()
    peaks = find_peaks(white.data, mean + 15*std, npeaks=nstars)
    peaks['starid'] = np.arange(len(peaks))

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    cube = Cube(cube_name)
    ind = list(range(0, cube.shape[0], step)) if step is not None else None

    res = []
    for peak in peaks:
        center = (peak['y_peak'], peak['x_peak'])
        cube_res, params = fit_mphd_cube(cube, center, fitsize, ind=ind,
                                         maxiter=maxiter, outdir=outdir,
                                         R_c=R_c)
        cube_res.write(f'{outdir}/cube_residual_s{peak["starid"]}.fits')
        params['starid'] = peak["starid"]
        res.append(params)

    params = vstack(res)
    params.write(f'{outdir}/fit_table.fits', overwrite=True)


def plot_residual(im, fit, figsize=(12, 8), title=None):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    ima_fit = Image(data=fit(xx, yy), copy=False)

    for ax, img in zip([ax1, ax2, ax3], [im, ima_fit, im - ima_fit]):
        img.plot(ax=ax, zscale=True, colorbar='v')

    compound = fit.n_submodels() > 1
    if compound:
        fy = int(fit['Mpeak'].x_0.value + 0.5)
    else:
        fy = int(fit.x_0.value + 0.5)

    ax = fig.add_subplot(gs[1, :])

    ax.plot(im.data[fy], label='data')
    ax.plot(ima_fit.data[fy], label='sum', lw=2)

    kwargs = dict(alpha=0.6, linestyle='--')
    if compound:
        peak_fit = fit['Mpeak'] * fit['W']
        airy_fit = fit['Airy'] * fit['W']
        halo_fit = fit['Mhalo'] * fit['Winv']
        ax.plot(peak_fit(xx, yy)[fy], label='core', **kwargs)
        ax.plot(airy_fit(xx, yy)[fy], label='airy', **kwargs)
        ax.plot(halo_fit(xx, yy)[fy], label='halo', **kwargs)
    else:
        x_0, y_0 = fit.x_0.value, fit.y_0.value
        rr = ((xx - fit.x_0.value) ** 2 + (yy - y_0) ** 2)
        rr2 = np.sqrt(rr)
        W = 1 / (1 + np.exp((rr2 - fit.R_c.value)))
        m = Moffat2D(x_0=x_0, y_0=y_0, amplitude=fit.Mpeak_amplitude.value,
                     alpha=fit.Mpeak_alpha.value, gamma=fit.Mpeak_gamma.value)
        ax.plot((W * (m(xx, yy) + fit.Bp.value))[fy], label='core', **kwargs)

        m = AiryDisk2D(amplitude=fit.Airy_amplitude.value, x_0=x_0,
                       y_0=y_0, radius=fit.Airy_radius.value)
        ax.plot(m(xx, yy)[fy], label='airy', **kwargs)

        m = Moffat2D(x_0=x_0, y_0=y_0,
                     amplitude=fit.Mhalo_amplitude.value,
                     alpha=BETA_H, gamma=fit.Mhalo_gamma.value)
        ax.plot(((1 - W) * m(xx, yy))[fy], label='halo', **kwargs)

    ax.set_yscale('log')
    ax.set_ylim((1, ax.get_ylim()[1]))
    ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()


def fit_line_with_outliers(x, y, niter=3, sigma=3.0, plot=False):
    model = Linear1D()
    fitter = fitting.LinearLSQFitter()
    fit = fitter(model, x, y)
    fitter_out = fitting.FittingWithOutlierRemoval(fitter, sigma_clip,
                                                   niter=niter, sigma=sigma)
    filt_data, fit2 = fitter_out(model, x, y)

    if plot:
        plt.plot(x, y, 'o', label='data')
        plt.plot(x, fit(x), label='fit')
        plt.plot(x, filt_data, 'ro', label='data filt')
        plt.plot(x, fit2(x), 'r', label='fit filt')
        plt.legend()

    return fit2


def plot_all_results(filename):
    t = Table.read(filename)
    starids = np.unique(t['starid'])
    valid = np.in1d(t['ierr'], [1, 2, 3, 4])
    good = t[valid]
    bad = t[~valid]
    print(f'{np.count_nonzero(~valid)} invalid fits out of {len(t)}')

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)

    axit = iter(axes.ravel())

    # R_c
    ax = next(axit)
    key = 'W_R_0' if 'W_R_0' in t.colnames else 'R_c'
    for i in starids:
        res = t[t['starid'] == i]
        ax.scatter(res['lbda'], res[key], s=10 + valid * 50, marker='o',
                   alpha=0.6, label=str(i))
    ax.legend()
    ax.set_title('$R_c$')

    # peak_fwhm
    ax = next(axit)
    for i in starids:
        res = t[t['starid'] == i]
        ax.plot(res['lbda'], res['Mpeak_fwhm'], '--o', alpha=0.6, label=str(i))
    ax.legend()
    ax.set_title('$M_{peak}\ FWHM$')

    # halo fwhm
    ax = next(axit)
    for i in starids:
        res = t[t['starid'] == i]
        ax.plot(res['lbda'], res['Mhalo_fwhm'], '--o', alpha=0.6, label=str(i))
    ax.legend()
    ax.set_title('$M_{halo}\ FWHM$')

    # amplitudes
    ax = next(axit)
    for name in res.colnames:
        if name.endswith('amplitude') and not name.startswith('W'):
            ax.plot(good['lbda'], good[name], 'o', label=name.split('_')[0])
    ax.legend()
    ax.set_title('ampltiudes')

    # niter
    ax = next(axit)
    ax.plot(good['lbda'], good['nfev'], 'o')
    ax.plot(bad['lbda'], bad['nfev'], 'ro')
    ax.set_title('niter')

    # beta peak
    ax = next(axit)
    for i in starids:
        res = t[t['starid'] == i]
        ax.plot(res['lbda'], res['Mpeak_alpha'], '--o', alpha=0.6,
                label=str(i))
    ax.legend()
    ax.set_ylim((0, 3))
    ax.set_title('$beta_{peak}$')

    # beta halo
    # ax = next(axit)
    # for i in starids:
    #     res = t[t['starid'] == i]
    #     ax.plot(res['lbda'], BETA_H, '-.^', alpha=0.6,
    #             label=str(i))
    # ax.legend()
    # ax.set_ylim((0, 3))
    # ax.set_title('$beta_{halo}$')

    # airy
    ax = next(axit)
    for i in starids:
        res = t[t['starid'] == i]
        ax.plot(res['lbda'], res['Airy_radius'], '-.^', alpha=0.6,
                label=str(i))
    ax.legend()
    ax.set_title('$Airy_{radius}$')
