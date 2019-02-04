import logging
import numpy as np
import warnings
from astropy.stats import sigma_clipped_stats

from .image import Image

__all__ = ('mask_sources', )


def mask_sources(image, sigma=3., iterations=2, opening_iterations=0,
                 outfile=None, plot=False):
    """Create a mask of sources, using photutils.

    Parameters
    ----------
    image : `Image` or str
        Input image.
    sigma : int, optional
        Number of sigma for the detection threshold.
    iterations : int
        Number of iterations for the binary dilatation.
    opening_iterations : int
        Number of iterations for the binary opening.
    outfile : str, optional
        Output mask filename.
    plot : bool
        Plot the image and mask.

    """
    logger = logging.getLogger(__name__)
    from scipy import ndimage as ndi
    try:
        import photutils
    except ImportError:
        logger.critical('photutils is required and was not found.')
        raise

    logger.info('Reading image %s', image)
    im = image if isinstance(image, Image) else Image(image)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean, median, std = sigma_clipped_stats(im.data, sigma=3.0)
        logger.info('mean: %s, median: %s, std: %s', mean, median, std)
        threshold = median + (std * sigma)
        segm_img = photutils.detect_sources(im.data, threshold, npixels=5)

    # turn segm_img into a mask
    mask = segm_img.data.astype(np.bool)

    if opening_iterations > 0:
        struct = ndi.generate_binary_structure(2, 2)
        mask = ndi.binary_opening(mask, structure=struct,
                                  iterations=opening_iterations)

    if iterations > 0:
        struct = ndi.generate_binary_structure(2, 2)
        mask = ndi.binary_dilation(mask, structure=struct,
                                   iterations=iterations)

    im_mask = Image(data=mask, dtype=int, wcs=im.wcs, copy=False)
    if outfile:
        im_mask.write(outfile, savemask='none')

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)
        ax = ax.ravel()
        vmin, vmax = mean - 5 * std, mean + 5 * std
        im.plot(ax=ax[0], scale='linear', vmin=vmin, vmax=vmax, colorbar='v')
        ax[1].imshow(segm_img, origin='lower')
        ax[1].set_title('Segmentation map')
        ax[2].imshow(mask, cmap='binary', origin='lower')
        ax[2].set_title('Mask')
        im_masked = im.copy()
        im_masked.mask_selection(mask)
        im_masked.plot(ax=ax[3], scale='linear', vmin=vmin, vmax=vmax,
                       title='Masked image')

    return im_mask
