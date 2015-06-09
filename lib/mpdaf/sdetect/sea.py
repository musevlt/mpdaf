"""sea.py contains SpectExtractAnd[nothing],
the first part of SpecExtractAndWeb software developed by Jarle

This software has been developed by Jarle Brinchmann (University of Leiden)
and ported to python by Laure Piqueras (CRAL).
It takes a MUSE data cube and a catalogue of objects and extracts small
sub-cubes around each object. From this it creates narrow-band images and
eventually spectra. To do the latter it is necessary to run an external
routine which runs sextractor on the images to define spectrum
extraction apertures.
Please contact Jarle for more info at jarle@strw.leidenuniv.nl
"""
from astropy.io import fits as pyfits

import logging
import numpy as np
import os
import subprocess

from ..obj import Image
from ..sdetect import Source, SourceList

__version__ = 1.0


def setup_config_files():
    DIR = os.path.dirname(__file__) + '/sea_data/'
    files = ['default.nnw',    'default.param', 'default.sex', 'gauss_5.0_9x9.conv']
    for f in files:
        try:
            os.symlink(DIR+f, './'+f)
        except:
            pass
        

def remove_config_files():
    files = ['default.nnw',    'default.param', 'default.sex', 'gauss_5.0_9x9.conv']
    for f in files:
        os.unlink(f)
     
   
def findCentralDetection(images, tolerance=1):
    """
    Determine which image has a detection close to the centre. We start with the centre for
    all. If all have a value zero there we continue.
    """ 
    logger = logging.getLogger('mpdaf corelib')
    d = {'class': 'SEA', 'method': 'findCentralDetection'}
    min_distances = {}
    min_values = {}
    global_min = 1e30
    global_ix_min = -1
    global_iy_min = -1
    
    count = 0
    bad = {}
    for key, im in images.items():
        logger.info('Doing %s'%key, extra=d)
        if (count == 0):
            nx, ny = im.shape
            ixc = nx/2
            iyc = ny/2 
        
        # Find the parts of the segmentation map where there is an object.
        ix, iy = np.where(im > 0)
        # Find the one closest to the centre.
        
        if (len(ix) > 0):
            # At least one object detected!
            dist = np.abs(ix-ixc)+ np.abs(iy-iyc)
            min_dist = np.min(dist)
            i_min = np.argmin(dist)
            ix_min = ix[i_min]
            iy_min = iy[i_min]
            val_min = im[ix_min, iy_min]
        
            # Record the essential information
            min_distances[key] = min_dist
            min_values[key] = val_min
            bad[key] = 0
            
            if (min_dist < global_min):
                global_min = min_dist
                global_ix_min = ix_min
                global_iy_min = iy_min
                global_im_index_min = key
                global_value = val_min
        else:
            bad[key] = 1
            min_distances[key] = -1e30
            min_values[key] = -1
            
        count = count+1
        
    # We have now looped through. Time to take stock. First let us check that there
    # was at least one detection.
    n_useful = 0
    segmentation_maps = {}
    isUseful = {}
    if (global_ix_min >= 0):
        # Ok, we are good. We have now at least one good segmentation map.
        # So we can make one simple one here.
        ref_map = np.where(images[global_im_index_min] == global_value, 1, 0)
        
        # Then check the others as well and if they do have a map at this position 
        # get another simple segmentation map.
        for key in images:
            if bad[key] == 1:
                logger.info('Image %s has no objects'%key, extra=d)
                this_map = np.zeros(ref_map.shape)
            else:
                # Has at least one object - let us see.
                if np.abs(min_distances[key] - global_min) <= tolerance:
                    # Create simple map
                    logger.info('Image %s has one useful objects'%key, extra=d)
                    this_map = np.where(images[key] == min_values[key], 1, 0)
                    n_useful = n_useful + 1
                    isUseful[key] = True
                else:
                    # Ok, this is too far away, I do not want to use this.
                    this_map = np.zeros(ref_map.shape)
                    
            segmentation_maps[key] = this_map
            
    else:
        # No objects found. Let us create a list of empty images.
        keys = images.keys()
        segmentation_maps = {key: np.zeros(images[keys[0]].shape) for key in keys}
        isUseful = {key: 0 for key in keys}
        n_useful = 0
    
    result = {'N_useful': n_useful, 'seg': segmentation_maps, 'isUseful': isUseful}
    
    return result


def union(seg):
    """
    Given a list of segmentation maps, create a segmentation map
    """
    first = True
    for im in seg.values():
        if first:
            mask = im
            first = False  
        else:
            mask += im
        
        mask = np.where(mask > 0, 1, 0)
        
    return mask

def intersection(seg):
    """
    Given a list of segmentation maps, create a segmentation map
    """
    first = True
    for im in seg.values():
        if (np.max(im) > 0): 
            if first:
                mask = im
                first = False
            else:
                mask *= im
                
    return mask


def findSkyMask(images):
    """
Loop over all segmentation images and use the region where no object is
detected in any segmentation map as our sky image.
    """

    first = True
    for im in images.values():
        if first:
            # Define the sky mask to have ones everywhere. 
            # For every segmentation map i will set the regions 
            # where an object is detected to zero.
            skymask = np.ones(im.shape, dtype=np.int)
            first = False
            
        isObj = np.where(im > 0)
        skymask[isObj] = 0

    return skymask

def segmentation(source):
    # suppose that MUSE_WHITE image exists
    try:
        subprocess.check_call(['sex'])
        cmd_sex = 'sex'
    except OSError:
        try:
            subprocess.check_call(['sextractor'])
            cmd_sex = 'sextractor'
        except OSError:
            raise OSError('SExtractor not found')
        
    dim = source.images['MUSE_WHITE'].shape
    start = source.images['MUSE_WHITE'].wcs.pix2sky([0,0])[0]
    step = source.images['MUSE_WHITE'].get_step()
    wcs = source.images['MUSE_WHITE'].wcs
    
    maps = {}
    nobj = {}
    setup_config_files()
    # size in arcsec
    for tag, ima in source.images.iteritems():
        tag2 = tag.replace('[','').replace(']','')
        
        fname = '%04d-%s.fits'%(source.id, tag2)
        start_ima = ima.wcs.pix2sky([0,0])[0]
        step_ima = ima.get_step()
        prihdu = pyfits.PrimaryHDU()
        hdulist = [prihdu]
        if ima.shape[0]==dim[0] and  ima.shape[1]==dim[1] and \
           start_ima[0]==start[0] and start_ima[1]==start[1] and \
           step_ima[0]==step[0] and step_ima[1]==step[1]:
            data_hdu = ima.get_data_hdu(name='DATA', savemask='nan')
        else:
            ima2 = ima.rebin(dim, start, step, flux=True)
            data_hdu = ima2.get_data_hdu(name='DATA', savemask='nan')
        hdulist.append(data_hdu)
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(fname, clobber=True, output_verify='fix')
            
        catalogFile = 'cat-' + fname
        segFile = 'seg-'+ fname
            
        command = [cmd_sex, "-CHECKIMAGE_NAME", segFile, '-CATALOG_NAME',
                       catalogFile, fname]
        subprocess.call(command)
        # remove source file
        os.remove(fname)
        try:
            hdul = pyfits.open(segFile)
            maps[tag] = hdul[0].data
            nobj[tag] = np.max(maps[tag])
            hdul.close()
        except:
            print "Something went wrong!"
        # remove seg file
        os.remove(segFile)
        # remove catalog file
        os.remove(catalogFile)
    remove_config_files()
           
    #make master segmentation
    # Allow for a tiny margin.
    if len(maps) > 0:
        r = findCentralDetection(maps, tolerance=3)
        
        object_mask = union(r['seg'])
        small_mask = intersection(r['seg'])
        sky_mask = findSkyMask(maps)
        
        ima = Image(wcs=wcs, data=object_mask)
        source.images['MASK_UNION'] = ima
        ima = Image(wcs=wcs, data=sky_mask)
        source.images['MASK_SKY'] = ima
        ima = Image(wcs=wcs, data=small_mask)
        source.images['MASK_INTER'] = ima


def SEA(cat, cube, hst=None, size=5, psf=None):
    """
    
    Parameters
    ----------
    cat : astropy.Table
          Tables containing positions and names of the objects.
          It needs to have at minimum these columns: ID, Z, RA, DEC
          for the name, redshift & position of the object.
    cube : :class:`mpdaf.obj.Cube`
           Data cube.
    hst : :class:`dict`
          Dictionary containing one or more HST images of the field
          which you want to extract stamps.

          Keys gives the filter ('SRC_WHITE' for white image, TBC)
              
          Values are :class:`mpdaf.obj.Image` object
    size : float
           The size to extract in arcseconds.
           By default 5x5arcsec
    psf  : np.array
           The PSF to use for PSF-weighted extraction.
           This can be a vector of length equal to the wavelength
           axis to give the FWHM of the Gaussian PSF at each
           wavelength (in arcsec) or a cube with the PSF to use.
          
    Returns
    -------
    out : :class:`mpdaf.sdetect.SourceList`
    """

    if hst is None:
        hst = {}
        
    # create source objects
    sources = []
    origin = ('sea', __version__, os.path.basename(cube.filename))
    
    for obj in cat:
        
        cen = cube.wcs.sky2pix([obj['DEC'], obj['RA']])[0]
        if cen[0] >= 0 and cen[0] <= cube.wcs.naxis1 and \
        cen[1] >= 0 and cen[1] <= cube.wcs.naxis2:
        
            source = Source.from_data(obj['ID'], obj['RA'], obj['DEC'], origin)
            try:
                z = obj['Z']
            except:
                z = -9999
            try:
                errz = obj['Z_ERR']
            except:
                errz = -9999
            source.add_z('CAT', z, errz)
            
            # create white image
            source.add_white_image(cube, size)
            
            # create narrow band images
            source.add_narrow_band_images(cube, 'CAT')
            
            # extract hst stamps
            newdim = source.images['MUSE_WHITE'].shape
            newstep = source.images['MUSE_WHITE'].get_step()
            # size in arcsec
            cdelt = np.abs(newstep*3600)
            newsize = np.max(cdelt*newdim)
            for tag, ima in hst.iteritems():
                source.add_image(ima, 'HST_'+tag, newsize)
                    
            # segmentation maps
            source.add_masks()
                
            # extract spectra
            source.extract_spectra(cube, skysub=True, psf=psf)
            source.extract_spectra(cube, skysub=False, psf=psf)
        
            sources.append(source)
        
    # return list of sources
    return SourceList(sources)