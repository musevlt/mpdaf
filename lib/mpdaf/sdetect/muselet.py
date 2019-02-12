"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2018 Johan Richard <jrichard@univ-lyon1.fr>
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

from ctypes import c_float, c_bool
import logging
import os
from os.path import join
from pathlib import Path
import multiprocessing as mp
import shutil
import subprocess
import stat
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy import table
import astropy.units as u

from ..obj import Cube, Image
from ..sdetect import Source, SourceList
from ..tools import chdir

__version__ = 3.0

DATADIR = Path(__file__).parent.resolve() / 'muselet_data'
CONFIG_FILES = {'sex': 'default.sex', 'conv': 'default.conv',
                'nnw': 'default.nnw', 'param': 'default.param'}
CONFIG_FILES_NB = {'sex': 'nb_default.sex', 'conv': 'nb_default.conv',
                    'nnw': 'nb_default.nnw', 'param': 'nb_default.param'}
DEFAULT_CONFIG = {
        'detect_minarea' : 4,
        'detect_thresh' : 2.0,
        'deblend_nthresh' : 50,
        'deblend_mincont' : 0.000000001,
        'back_size' :  11,
        'back_filtersize' : 1,
        }
DEFAULT_CONFIG_NB = {
        'detect_minarea' : 3,
        'detect_thresh' : 2.5,
        'deblend_nthresh' : 50,
        'deblend_mincont' : 0.000000001,
        'back_size' :  11,
        'back_filtersize' : 1,
        }

shared_args = {} #global variable storage needed for multiprocessing


class ProgressCounter(object):

    def __init__(self, total, msg='', every=1):
        self.count = 0
        self.total = total
        self.msg = msg
        self.every = every
        self.update_display()

    def __del__(self):
        self.close()

    def increment(self):
        self.count += 1
        self.update_display()

    def update_display(self):
       if ((self.count % self.every) == 0) or (self.count == self.total):
           sys.stdout.write("{}{}/{}\r".format(self.msg, self.count,
                                                self.total))

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def setup_config_files(dir_, config, nb=False):

    logger = logging.getLogger(__name__)

    #choose whether working for NB files or BB files
    if nb:
        config_files = CONFIG_FILES_NB
        default_config = DEFAULT_CONFIG_NB
        name = 'narrow-band'
    else:
        config_files = CONFIG_FILES
        default_config = DEFAULT_CONFIG
        name = 'broad-band'
    

    #copy nnw file
    f1 = DATADIR / config_files['nnw']
    f2 = dir_ / 'default.nnw'
    try:
        f2.unlink() #remove existing
    except FileNotFoundError:
        pass
    shutil.copy(f1, f2)

    #copy param file
    f1 = DATADIR / config_files['param']
    f2 = dir_ / 'default.param'
    try:
        f2.unlink() #remove existing
    except FileNotFoundError:
        pass
    shutil.copy(f1, f2)

    #copy conv file
    if 'conv' not in config:
        #use default
        f1 = DATADIR / config_files['conv']
    else:
        #use user supplied file
        f1 = Path(config['conv'])
        msg = "{} detection with user supplied convolution kernel: {}"
        logger.info(msg.format(name, f1))
    f2 = dir_ / 'default.conv'
    try:
        f2.unlink() #remove existing
    except FileNotFoundError:
        pass
    shutil.copy(f1, f2)

    #write sex configuration
    with open(DATADIR / config_files['sex'], 'r') as fh:
        template = fh.read()

    config_out = default_config.copy()
    for k in default_config.keys():
        try:
            v = config[k]
            config_out[k] = v
            msg = "{} detection with {}: {}"
            logger.debug(msg.format(name, k, v))
        except KeyError:
            v = config_out[k]
            msg = "{} detection with {}: {} (default)"
            logger.debug(msg.format(name, k, v))

    with open(dir_ / 'default.sex', 'w') as fh:
        fh.write(template.format(**config_out))
    

def remove_files():

    files = ['default.sex', 'default.conv', 'default.nnw', 'default.param',
             'emlines', 'emlines_small', 'BGR.cat', 'inv_variance.fits',
             'segment.fits', 'white.fits', 'whiteb.fits', 'whiteg.fits',
             'whiter.fits', 'detect.cat']

    for f in files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    shutil.rmtree('nb')


def write_white_image(cube, dir_):

    logger = logging.getLogger(__name__)

    data = np.ma.average(cube.data, weights=1./cube.var, axis=0)
    image = Image(data=data, wcs=cube.wcs, unit=cube.unit, copy=False)

    file_ = dir_ / 'im_white.fits'
    logger.debug('writing white light image: {}'.format(file_))
    image.write(file_, savemask='nan')


def write_weight_image(cube, cube_exp, dir_):

    logger = logging.getLogger(__name__)

    if cube_exp is None:
        logger.info("calculating weight image from variance cube")
        data = 1. / np.ma.mean(cube.var, axis=0)
        unit = u.Unit(1) / cube.unit ** 2.
        image = Image(data=data, wcs=cube.wcs, unit=unit, copy=False)

    else:
        logger.info("calculating weight image from exposure cube")
        cube_exp.data = cube_exp.data.astype(np.float32)
        image = cube_exp.mean(axis=0)

    file_ = dir_ / 'im_weight.fits'
    logger.debug("writing weight image: {}".format(file_))
    image.write(file_, savemask='nan')


def write_rgb_images(cube, dir_):

    logger = logging.getLogger(__name__)

    #split datacube into 3 equal regions
    idx = np.round(np.linspace(0, cube.shape[0], 4)).astype(int)
    cube_b = cube[:idx[1]]
    cube_g = cube[idx[1]:idx[2]]
    cube_r = cube[idx[2]:]

    data_b = np.ma.average(cube_b.data, weights=1./cube_b.var, axis=0)
    data_g = np.ma.average(cube_g.data, weights=1./cube_g.var, axis=0)
    data_r = np.ma.average(cube_r.data, weights=1./cube_r.var, axis=0)

    im_b = Image(data=data_b, wcs=cube.wcs, unit=cube.unit, copy=False)
    im_g = Image(data=data_g, wcs=cube.wcs, unit=cube.unit, copy=False)
    im_r = Image(data=data_r, wcs=cube.wcs, unit=cube.unit, copy=False)

    file_ = dir_ / 'im_b.fits'
    logger.debug("writing blue image: {}".format(file_))
    im_b.write(file_, savemask='nan')

    file_ = dir_ / 'im_g.fits'
    logger.debug("writing green image: {}".format(file_))
    im_g.write(file_, savemask='nan')

    file_ = dir_ / 'im_r.fits'
    logger.debug("writing red image: {}".format(file_))
    im_r.write(file_, savemask='nan')



def write_bb_images(cube, cube_exp, dir_, n_cpu=1, limit_ram=False):

    logger = logging.getLogger(__name__)

    t0_create = time.time()
    if (n_cpu == 1) or limit_ram: #enables nicer traceback for debugging
        logger.info("creating broad-band images using 1 CPU")
            
        write_white_image(cube, dir_)
        write_weight_image(cube, cube_exp, dir_)
        write_rgb_images(cube, dir_)


    else:
        use_cpu = np.clip(n_cpu, None, 3) #at most 3CPUs
        logger.info("creating broad-band images using "
                    "{} CPUs".format(use_cpu))
        pool = mp.Pool(use_cpu)
        r1 = pool.apply_async(write_white_image, (cube, dir_))
        r2 = pool.apply_async(write_weight_image, (cube, cube_exp, dir_))
        r3 = pool.apply_async(write_rgb_images, (cube, dir_))
        pool.close()

        [r.get(999999) for r in [r1, r2, r3]]

    t_create = time.time() - t0_create
    logger.debug("Broad-bands created in {0:.1f} seconds".format(t_create))


def write_nb(i, data, var, left, right, exp, fw, hdr, dir_):

    weight = fw[:,np.newaxis,np.newaxis] / var
    im_center = np.ma.average(data, weights=weight, axis=0)

    im_left = np.median(left, axis=0)
    im_right = np.median(right, axis=0)

    im_mask = im_center.mask
    im_center = np.ma.filled(im_center, np.nan)

    n_l = len(left)
    n_r = len(right)

    if (n_l + n_r):
        im_cont = (n_l*im_left + n_r*im_right) / (n_l + n_r)
        im_nb = im_center - im_cont
    else:
        im_nb = im_center

    hdu = fits.ImageHDU(im_nb, header=hdr, name='DATA')
    file_ = dir_ / 'nb/nb{:04d}.fits'.format(i)
    hdu.writeto(file_, overwrite=True)

    #expand mask by two pixels

    #im_mask = binary_dilation(im_mask, iterations=2, border_value=1)

    hdu = fits.ImageHDU(im_mask.astype(np.uint8), header=hdr, name='DATA')
    file_ = dir_ / 'nb/nb{:04d}-mask.fits'.format(i)
    hdu.writeto(file_, overwrite=True)

    if exp is not None:
        if exp.ndim == 3:
            im_exp  = np.average(exp, weights=fw, axis=0)
        else:
            im_exp = exp
        hdu = fits.ImageHDU(im_exp, header=hdr, name='DATA')
    else:
        im_weight = np.ma.sum(weight, axis=0).filled(0)
        hdu = fits.ImageHDU(im_weight, header=hdr)

    file_ = dir_ / 'nb/nb{:04d}-weight.fits'.format(i)
    hdu.writeto(file_, overwrite=True)

    return im_nb


def init_write_nb_multi(data, data_shape, var, var_shape,
        mask, mask_shape, exp, exp_shape, cube_nb, cube_nb_shape):

    shared_args['data'] = data
    shared_args['data_shape'] = data_shape
    shared_args['var'] = var
    shared_args['var_shape'] = var_shape
    shared_args['mask'] = mask
    shared_args['mask_shape'] = mask_shape
    shared_args['exp'] = exp
    shared_args['exp_shape'] = exp_shape
    shared_args['cube_nb'] = cube_nb
    shared_args['cube_nb_shape'] = cube_nb_shape


def write_nb_multi(i, delta, fw, hdr, dir_):

    # "load" shared arrays

    data = np.frombuffer(shared_args['data'], dtype='=f4')
    data = data.reshape(shared_args['data_shape'])

    var = np.frombuffer(shared_args['var'], dtype='=f4')
    var = var.reshape(shared_args['var_shape'])

    mask = np.frombuffer(shared_args['mask'], dtype='|b1')
    mask = mask.reshape(shared_args['mask_shape'])

    if shared_args['exp'] is not None:
        exp = np.frombuffer(shared_args['exp'], dtype='=f4')
        exp = exp.reshape(shared_args['exp_shape'])
    else:
        exp = None

    cube_nb = np.frombuffer(shared_args['cube_nb'], dtype='=f4')
    cube_nb = cube_nb.reshape(shared_args['cube_nb_shape'])

    n_w, n_y, n_x = data.shape
    c_min, c_max = np.clip([i-2, i+3], 0, n_w)
    l_min, l_max = np.clip([i-2-delta, i-2], 0, n_w)
    r_min, r_max = np.clip([i+3, i+3+delta], 0, n_w)

    d = np.ma.MaskedArray(data[c_min:c_max], mask=mask[c_min:c_max])
    v = np.ma.MaskedArray(var[c_min:c_max], mask=mask[c_min:c_max])

    if exp is not None:
        e = exp[i]
        #e = exp[c_min:c_max]
    else:
        e = None

    if l_max == 0:
        l = data[0].reshape([1, n_y, n_x])
        l[:, mask[0]] = 0.
    else:
        l = data[l_min:l_max]
        l[mask[l_min:l_max]] = 0.

    if r_min == n_w:
        r = data[-1].reshape([1, n_y, n_x])
        r[:,mask[-1]] = 0.
    else:
        r = data[r_min:r_max]
        r[mask[r_min:r_max]] = 0.

    im_nb = write_nb(i, d, v, l, r, e, fw, hdr, dir_)

    cube_nb[i] = im_nb #output


def write_nb_images(cube, cube_exp, delta, fw, dir_, n_cpu=1,
            limit_ram=False):

    logger = logging.getLogger(__name__)

    os.makedirs(dir_ / 'nb', exist_ok=True)

    n_w = cube.shape[0]
    hdr = cube.wcs.to_header()
    data = cube.data
    var = cube.var
    if cube_exp is not None:
        exp = cube_exp.data.filled(0).astype(np.float32)
    else:
        exp = None

    t0_create = time.time()

    if n_cpu == 1:
        logger.info("creating narrow-band images using 1 CPU")


        cube_nb = np.zeros(cube.shape, dtype=np.float32)
        if not limit_ram:
            data0 = data.filled(0.) #for computing continuum
        else:
            data0 = data #delay filling until have sliced data

        progress = ProgressCounter(n_w-5, msg='Created:', every=1)
        for i in range(2, n_w-3):

            n_w, n_y, n_x = data.shape
            c_min, c_max = np.clip([i-2, i+3], 0, n_w)
            l_min, l_max = np.clip([i-2-delta, i-2], 0, n_w)
            r_min, r_max = np.clip([i+3, i+3+delta], 0, n_w)

            d = data[c_min:c_max]
            v = var[c_min:c_max]

            if exp is not None:
                e = exp[i]
                #e = exp[c_min:c_max]
            else:
                e = None

            if l_max == 0:
                l = data0[0].reshape([1, n_y, n_x])
            else:
                l = data0[l_min:l_max]

            if r_min == n_w:
                r = data0[-1].reshape([1, n_y, n_x])
            else:
                r = data0[r_min:r_max]

            if limit_ram:
                l = l.filled(0.)
                r = r.filled(0.)

            im_nb = write_nb(i, d, v, l, r, e, fw, hdr, dir_)
            cube_nb[i] = im_nb

            progress.increment()

    else: #parallel
        logger.info("creating narrow-band images using {} CPUs".format(n_cpu))

        #setup shared arrays (no locks needed!), this can take some time
        logger.debug("allocating shared arrays for multiprocessing")
        t0_allocate = time.time()

        #shared inputs
        logger.debug("allocating shared data cube (input)")
        shape = data.shape
        data_raw = mp.RawArray(c_float, int(np.prod(shape)))
        data_np = np.frombuffer(data_raw, dtype='=f4').reshape(shape)
        data_shape = shape
        data_np[:] = data.data

        logger.debug("allocating shared variance cube (input)")
        shape = var.shape
        var_raw = mp.RawArray(c_float, int(np.prod(shape)))
        var_np = np.frombuffer(var_raw, dtype='=f4').reshape(shape)
        var_shape = shape
        var_np[:] = var.data

        logger.debug("allocating shared mask cube (input)")
        shape = data.mask.shape
        mask_raw = mp.RawArray(c_bool, int(np.prod(shape)))
        mask_np = np.frombuffer(mask_raw, dtype='|b1').reshape(shape)
        mask_shape = shape
        mask_np[:] = data.mask

        if exp is not None:
            logger.debug("allocating shared exposure cube (input)")
            shape = exp.shape
            exp_raw = mp.RawArray(c_float, int(np.prod(shape)))
            exp_np = np.frombuffer(exp_raw, dtype='=f4').reshape(shape)
            exp_np[:] = exp
            exp_shape = shape
        else:
            exp_raw = None
            exp_shape = None

        #shared output
        logger.debug("allocating shared nb cube (output)")
        shape = data.shape
        cube_nb_raw = mp.RawArray(c_float, int(np.prod(shape)))
        cube_nb = np.frombuffer(cube_nb_raw, dtype='=f4').reshape(shape)
        cube_nb_shape = shape

        logger.debug("initializing process pool")
        initargs = (data_raw, data_shape, var_raw, var_shape, mask_raw,
                mask_shape, exp_raw, exp_shape, cube_nb_raw, cube_nb_shape)

        pool = mp.Pool(n_cpu, initializer=init_write_nb_multi,
                        initargs=initargs)
        t_allocate = time.time() - t0_allocate
        logger.debug("all data allocated in {0:.1f} seconds".format(t_allocate))

        logger.debug("starting multiprocessing")

        results = []
        progress = ProgressCounter(n_w-5, msg='Created:', every=10)
        for i in range(2, n_w-3):

            res = pool.apply_async(write_nb_multi, (i, delta, fw, hdr, dir_),
                        callback=lambda x: progress.increment())
          #  for debugging
          #  init_write_nb_multi(*initargs)
          #  res = write_nb_multi(i, delta, fw, hdr, dir_)
            results.append(res)

        pool.close()

        [res.get(999999) for res in results]

    progress.close()

    t_create = time.time() - t0_create
    logger.debug("Narrow-bands created in {0:.1f} seconds".format(t_create))


    hdr = cube.wcs.to_cube_header(cube.wave)
    cube_nb = fits.ImageHDU(cube_nb, header=hdr, name='DATA')

    return cube_nb


def step1(cubename, expmapcube, fw, delta, dir_=None, nbcube=False, n_cpu=1,
        limit_ram=False):

    logger = logging.getLogger(__name__)
    logger.info("Opening: %s", cubename)

    if dir_ is None:
        dir_ = Path.cwd()

    cube = Cube(str(cubename))

#    #mvar=c.var.filled(np.inf)
#    mvar=c.var
#    #mvar[mvar <= 0] = np.inf
#    c._var = None

    if expmapcube is None:
        cube_exp = None
    else:
        logger.info("Opening exposure map cube: %s", expmapcube)
        cube_exp = Cube(str(expmapcube))

    logger.info("STEP 1: create white light, variance, RGB and "
                "narrow-band images")

    write_bb_images(cube, cube_exp, dir_, n_cpu=n_cpu, limit_ram=limit_ram)
    cube_nb = write_nb_images(cube, cube_exp, delta, fw, dir_, n_cpu=n_cpu,
                        limit_ram=limit_ram)

    if nbcube:
        file_ = dir_ / ('NB_' + cubename.name)
        logger.debug("writing narrow-band cube: {}".format(file_))
        cube_nb.writeto(file_, overwrite=True)


def run_sex(cmd, dir_):
    p = subprocess.Popen(cmd, cwd=dir_, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode != 0:
        stdout = p.communicate()[0].decode('utf-8', errors='ignore')
        logger = logging.getLogger(__name__)
        logger.error("Dumping output from SExtractor:" + stdout)
        msg = "Error in subprocess call: '{}'".format(' '.join(cmd))
        raise Exception(msg)


def step2(cubename, cmd_sex, config=None, config_nb=None, dir_=None, n_cpu=1):

    if config is None:
        config = {}

    if config_nb is None:
        config_nb = {}

    if dir_ is None:
        dir_ = Path.cwd()

    logger = logging.getLogger(__name__)
    logger.info("STEP 2: run SExtractor on broad-band and narrow-band images")

    # replace existing config files in work dir
    setup_config_files(dir_, config)


    logger.info("running SExtractor on white light and RGB images") 

    run_sex([cmd_sex, 'im_white.fits'], dir_)
    for band in ['b', 'g', 'r']:
        cmd = [cmd_sex,
                '-CATALOG_NAME', 'cat_{}.dat'.format(band),
                'im_white.fits,im_{}.fits'.format(band)]
        run_sex(cmd, dir_)

    logger.debug("combining catalogs") 
    cat_b = table.Table.read(dir_ / 'cat_b.dat', format='ascii.sextractor')
    cat_g = table.Table.read(dir_ / 'cat_g.dat', format='ascii.sextractor')
    cat_r = table.Table.read(dir_ / 'cat_r.dat', format='ascii.sextractor')

    names = ('NUMBER',
             'RA', 'DEC',
             'I_X', 'I_Y',
             'MAG_APER_B', 'MAGERR_APER_B',
             'MAG_APER_G', 'MAGERR_APER_G',
             'MAG_APER_R', 'MAGERR_APER_R')
    cat_bgr = table.Table([cat_b['NUMBER'],
                  cat_b['ALPHA_J2000'], cat_b['DELTA_J2000'],
                  cat_b['X_IMAGE']-1, cat_b['Y_IMAGE']-1,
                  cat_b['MAG_APER'], cat_b['MAGERR_APER'],
                  cat_g['MAG_APER'], cat_g['MAGERR_APER'],
                  cat_r['MAG_APER'], cat_r['MAGERR_APER']], names=names)
    logger.info("{} continiuum objects detected".format(len(cat_bgr)))

    file_ = dir_ / 'cat_bgr.dat'
    logger.debug("writing catalog: {}".format(file_))
    cat_bgr.write(file_, format='ascii.fixed_width_two_line')

    for band in ['b', 'g', 'r']:
        file_ = dir_ / 'cat_{}.dat'.format(band)
        logger.debug("removing file {}".format(file_))
        os.remove(file_)

    # replace existing config files in work dir
    setup_config_files(dir_ / 'nb', config_nb, nb=True)

    cube = Cube(str(cubename))
    n_w = cube.shape[0]

    #generate sextractor commands
    commands = []
    for i in range(2, n_w-3):
        cmd = [cmd_sex,
                '-CATALOG_NAME', 'cat{:04d}.dat'.format(i),
                '-WEIGHT_IMAGE', 'nb{:04d}-weight.fits'.format(i),
                '-FLAG_IMAGE', 'nb{:04d}-mask.fits'.format(i),
                '-CHECKIMAGE_NAME', 'seg{:04d}.fits'.format(i),
                'nb{:04d}.fits'.format(i)]
        commands.append(cmd)

    t0_run = time.time()
    if n_cpu == 1:
        logger.info("running SExtractor on narrow-band images using 1 CPU")

        progress = ProgressCounter(len(commands), msg='SExtractor:', every=1)

        for cmd in commands:
            run_sex(cmd, dir_ / 'nb')
            progress.increment()

    else:
        logger.info("running SExtractor on narrow-band images using "
                    "{} CPUs".format(n_cpu))

        progress = ProgressCounter(len(commands), msg='SExtractor:', every=10)

        pool = mp.Pool(n_cpu)
        results = []
        for cmd in commands:
            res = pool.apply_async(run_sex, (cmd, dir_ / 'nb'),
                            callback=lambda x: progress.increment())
            results.append(res)
        pool.close()

        [res.get(999999) for res in results]

    progress.close()

    t_run = time.time() - t0_run
    logger.debug("running SExtractor took {0:.1f} seconds".format(t_run))


def load_cat(i, dir_):

    file_ = dir_ / 'nb/cat{:04d}.dat'.format(i)
    data = table.Table.read(file_, format='ascii.sextractor')

    n_data = len(data)

    data.rename_column('NUMBER', 'ID_SLICE')
    data.rename_column('ALPHA_J2000', 'RA')
    data.rename_column('DELTA_J2000', 'DEC')
    data.rename_column('X_IMAGE', 'I_X')
    data.rename_column('Y_IMAGE', 'I_Y')

    #make image coords 0-indexed
    data['I_X'] -= 1
    data['I_Y'] -= 1

    #add extra columns
    #new ID unqiue in whole cube, to be filled later
    id_cube = np.full(n_data, -1, dtype=int)
    c1 = table.Column(id_cube, name='ID_CUBE') 

    #Z index (i.e. slice+1)
    z_image = np.full(n_data, i, dtype=int)
    c2 = table.Column(z_image, name='I_Z') 

    #wavelength, to be filled later
    wave = np.full(n_data, np.nan, dtype=float)
    c3 = table.Column(wave, name='WAVE')

    mag = data['MAG_APER'].astype(float)
    mag_err = data['MAGERR_APER'].astype(float)
    mag[np.isclose(mag, 99.)] = np.nan
    mag_err[np.isclose(mag_err, 99.)] = np.nan

    flux = 10. ** ((25. - mag) / -2.5)
    flux_err = np.abs(flux * np.log(10) * mag_err / -2.5)

    c4 = table.Column(flux, name='FLUX')
    c5 = table.Column(flux_err, name='FLUX_ERR')

    data.add_columns([c1, c2, c3, c4, c5], indexes=[0, 5, 3, 5, 5])

    return data


#def assign_friends(coords, kdTree, ids, i, max_dist, id_, n_cpu=1):
#
#    #assign index of current
#    ids[i] = id_
#
#    #find index of nearest
#    idx_near = kdTree.query_ball_point(coords[i], max_dist, n_jobs=n_cpu)
#
#    #for each index in nearest
#    for i_test in idx_near:
#        if ids[i_test] == 0: #otherwise skip current and previously checked
#            assign_friends(coords, kdTree, ids, i_test, max_dist, id_,
#                            n_cpu=n_cpu)


def merge_raw_3D(x, y, z, dist_spatial, dist_spectral, n_cpu=1):

    coords_spatial = np.column_stack([x, y])
    coords_spectral = z.reshape([-1, 1])

    tree_spatial = cKDTree(coords_spatial) 
    tree_spectral = cKDTree(coords_spectral) 

    idx_unassigned = set(range(len(coords_spatial)))
    ids = np.zeros([len(coords_spatial)], dtype=int)
    id_group = 1
    while len(idx_unassigned): #while still unassigned 

        i_init = next(iter(idx_unassigned)) #get first
        new_friends = set([i_init])

        while len(new_friends):
            
            i = new_friends.pop() #pick one of the new friends
            ids[i] = id_group #assign it the ids
            idx_unassigned.remove(i) #remove 

            idx_1 = tree_spatial.query_ball_point(coords_spatial[i],
                            dist_spatial, n_jobs=n_cpu)

            idx_2 = tree_spectral.query_ball_point(coords_spectral[i],
                            dist_spectral, n_jobs=n_cpu)
            
            #near in all dimensions, and not already assigned
            idx_new = set(idx_1) & set(idx_2) & idx_unassigned
            
            #extend list of indicies to check
            new_friends |= idx_new

        id_group += 1

    return ids



def step3(cubename, ima_size, clean, skyclean, radius, nlines_max, dir_=None,
        n_cpu=1):

    if dir_ is None:
        dir_ = Path.cwd()

    logger = logging.getLogger(__name__)
    logger.info("STEP 3: merge SExtractor catalogs and measure redshifts")

    cube = Cube(str(cubename))
    n_w = cube.shape[0]

    #remove wavelengths with sky lines
    idx_nb = np.arange(2, 500, dtype=int)
#    idx_nb = np.arange(2, n_w-3, dtype=int)
    wave = cube.wave.coord(idx_nb, unit=u.Unit('Angstrom'))
    mask = np.zeros_like(idx_nb, dtype=bool)
    for wave_range in skyclean:
        msg = "excluding wavelengths between {:.02f}\u212B and {:.02f}\u212B"
        logger.debug(msg.format(*wave_range))
        m = (wave >= wave_range[0]) & (wave <= wave_range[1])
        mask |= m
    idx_nb = idx_nb[~mask]

    msg = "{} narrow-bands will be excluded"
    logger.debug(msg.format(np.sum(mask)))

    #load NB catalogs
    cat_all = []
    t0_load = time.time()

    if n_cpu == 1:
        logger.info("loading narrow-band catalogs using 1 CPU")

        progress = ProgressCounter(len(idx_nb), msg='Loaded:', every=1)
        for i in idx_nb:
            cat = load_cat(i, dir_)
            cat_all.append(cat)
            progress.increment()

    else:
        use_cpu = np.clip(n_cpu, None, 16) #no point using more than 16 CPUs
        msg = "loading narrow-band catalogs using {} CPUs".format(use_cpu)
        logger.info(msg)
        progress = ProgressCounter(len(idx_nb), msg='Loaded:', every=10)
        pool = mp.Pool(use_cpu)
        results = []
        for i in idx_nb:
            res = pool.apply_async(load_cat, (i, dir_),
                    callback=lambda x:progress.increment())
            results.append(res)
        pool.close()

        for res in results:
            cat = res.get(999999)
            cat_all.append(cat)

    progress.close()

    #combine into one large table
    cat = table.vstack(cat_all)

    cat['ID_CUBE'] = np.arange(len(cat), dtype=int) + 1
    cat['WAVE'] = cube.wave.coord(cat['I_Z'])

    t_load = time.time() - t0_load
    logger.debug("catalogs loaded in {0:.1f} seconds".format(t_load))

    msg = "{} raw detections found".format(len(cat))
    logger.info(msg)


    logger.info("cleaning detections with fluxes below 5\u03C3")
    #remove insigficant fluxes

    flux = cat['FLUX']
    flux_err = cat['FLUX_ERR']
    with np.errstate(invalid='ignore', divide='ignore'):
        mask = (flux / flux_err) < 5.
        mask |= np.isnan(flux)

    cat = cat[~mask]
    msg = "{} detections remain ({} removed)"
    logger.info(msg.format(np.sum(~mask), np.sum(mask)))
    

    logger.info("cleaning detections at edge of cube")
    #clean detections partially (>10%) outside data 

    area_tot = cat['ISOAREA_IMAGE'].astype(float)
    area_bad = cat['NIMAFLAGS_ISO'].astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mask = (area_bad / area_tot) > 0.1
        mask |= area_tot == 0

    cat = cat[~mask]
    msg = "{} detections remain ({} removed)"
    logger.info(msg.format(np.sum(~mask), np.sum(mask)))


    file_weight = str(dir_ / 'im_weight.fits')
    im_weight = Image(file_weight)
    clean_thresh = clean * np.ma.median(im_weight.data)
    logger.info("cleaning below image weight %s", clean_thresh)

    i_y = np.round(cat['I_Y']).astype(int)
    i_x = np.round(cat['I_X']).astype(int)
    mask = im_weight.data[i_y, i_x] < clean_thresh

    cat = cat[~mask]
    msg = "{} detections remain ({} removed)"
    logger.info(msg.format(np.sum(~mask), np.sum(mask)))


    max_sep_spatial = 0.8
    max_sep_spectral = 3.75
    msg = "merging raw detections using a friends-of-friends algorithm"
    logger.info(msg)
    msg = "spatial link-length \u0394r < {:.2f}\u2033"
    logger.debug(msg.format(max_sep_spatial))
    msg = "spectral link-length \u0394\u03BB < {:.2f}\u212B"
    logger.debug(msg.format(max_sep_spectral))


    #merge detections when close in wavelength and keep the brightest

    x0 = cube.wcs.get_crval1(unit=u.deg)
    y0 = cube.wcs.get_crval2(unit=u.deg)
    z0 = cube.wave.get_crval(unit=u.Unit('Angstrom'))

    x = (cat['RA'] - x0) * 3600. * np.cos(np.radians(y0))
    y = (cat['DEC'] - y0) * 3600.
    z = (cat['WAVE'] - z0)

    ids_line = merge_raw_3D(x, y, z, max_sep_spatial, max_sep_spectral,
                    n_cpu=n_cpu)

    #loop over groups and keep brightest line
    mask = np.zeros([len(cat)], dtype=bool)
    for id_line in np.unique(ids_line):
        m_line = (ids_line == id_line)
        fluxes = cat['FLUX'][m_line] 
        idx = np.argmax(fluxes)
        id_keep = cat['ID_CUBE'][m_line][idx]
        mask |= (cat['ID_CUBE'] == id_keep)

    cat = cat[mask]
    #reassign ID numbers
    cat['ID_CUBE'] = np.arange(len(cat), dtype=int) + 1

    msg = "{} lines found ({} duplicate detections discarded)"
    logger.info(msg.format(np.sum(mask), np.sum(~mask)))

    logger.info("grouping detections")

    #group detections close to continuum sources
    file_ = dir_ / 'cat_bgr.dat'
    cat_bgr = table.Table.read(file_, format='ascii.fixed_width_two_line')

    x0 = cube.wcs.get_crval1(unit=u.deg)
    y0 = cube.wcs.get_crval2(unit=u.deg)

    x_raw = (cat['RA'] - x0) * 3600. * np.cos(np.radians(y0))
    y_raw = (cat['DEC'] - y0) * 3600.

    x_cont = (cat_bgr['RA'] - x0) * 3600. * np.cos(np.radians(y0))
    y_cont = (cat_bgr['DEC'] - y0) * 3600.

    coords_raw = np.column_stack([x_raw, y_raw])
    coords_group = np.column_stack([x_cont, y_cont])

    tree_raw = cKDTree(coords_raw)
    tree_group = cKDTree(coords_group)

    ids = np.zeros([len(cat)], dtype=int)
    dist, ids = tree_group.query(coords_raw, 1, distance_upper_bound=1.2)

    mask_nogroup = ~np.isfinite(dist)
    ids[mask_nogroup] = -1

    id_group = np.max(ids) + 1

    while np.sum(mask_nogroup):

        i = np.where(mask_nogroup)[0][0]
        
        idx = tree_raw.query_ball_point(coords_raw[i], 1.2)
        idx = np.array(idx)

        mask_nogroup = ids == -1
        idx = idx[mask_nogroup[idx]]

        ids[idx] = id_group
        
        id_group +=1 

#assign_to_group
#for each raw find nearest cont #label centers fixed
#for each unassined group nearest
#if no group make new group
#recalc group centers of non-fixed
#for each raw assign to nearest group within radius
#if no group make new group
#recalc group centers of non-fixed
#repeat untill IDs don't change or max iter limit

    #iterate on this! TODO

    n_obj = len(np.unique(ids))
    logger.info("{} objects found".format(n_obj))

    c = table.Column(ids, name='ID_OBJ')
    cat.add_column(c, index=1)
    


    # Sources list
    cube_version = str(cube.primary_header.get('CUBE_V', ''))
    origin = ('muselet', __version__, cubename, cube_version)

    dw = cube.wave.get_step(unit=u.angstrom)

    logger.info("building raw sources".format(n_obj))

    raw_cat = SourceList()

    progress = ProgressCounter(len(cat), msg='Built:', every=1)
    for row in cat:
        src = Source.from_data(ID=row['ID_CUBE'], ra=row['RA'], dec=row['DEC'],
                        origin=origin)

        file_nb = (dir_ / 'nb/nb{:04d}.fits'.format(row['I_Z']))
        file_seg = (dir_ / 'nb/seg{:04d}.fits'.format(row['I_Z']))
        im_nb = Image(str(file_nb))
        im_seg = Image(str(file_seg))
        im_seg.data = (im_seg.data == row['ID_SLICE']) * 1
      #  im_seg.data.mask = np.zeros_like(im_seg.data, dtype=bool)

        src.add_image(im_nb, 'NB{:04.0f}'.format(row['WAVE']), ima_size)
        import pdb; pdb.set_trace()
        src.add_image(im_seg, 'MASK_OBJ', ima_size)

        wave = row['WAVE']
        flux = row['FLUX']
        flux_err = row['FLUX_ERR']

        lines = table.Table([[wave], [dw], [flux], [flux_err]],
                      names=['LBDA_OBS', 'LBDA_OBS_ERR', 'FLUX', 'FLUX_ERR'],
                      dtype=['<f8', '<f8', '<f8', '<f8'])
        lines['LBDA_OBS'].format = '.2f'
        lines['LBDA_OBS'].unit = u.angstrom
        lines['LBDA_OBS_ERR'].format = '.2f'
        lines['LBDA_OBS_ERR'].unit = u.angstrom
        lines['FLUX'].format = '.4f'
        lines['FLUX'].unit = cube.unit
        lines['FLUX_ERR'].format = '.4f'
        lines['FLUX_ERR'].unit = cube.unit
        src.lines = lines
        raw_cat.append(src)

        progress.increment()

    progress.close()
    

    logger.info("building object sources".format(n_obj))

    obj_cat = SourceList()

    ids = np.unique(cat['ID_OBJ'])
    progress = ProgressCounter(len(ids), msg='Built:', every=1)
    for id_ in ids:

        rows = cat[cat['ID_OBJ'] == id_]

        ra = np.mean(rows['RA'])
        dec = np.mean(rows['DEC'])
        src = Source.from_data(ID=id_, ra=ra, dec=dec, origin=origin)

        sort = np.argsort(rows['FLUX'])[::-1]
        rows = rows[sort][:nlines_max]

        images_nb = [] 
        images_seg = [] 

        for row in rows:
            file_nb = (dir_ / 'nb/nb{:04d}.fits'.format(row['I_Z']))
            file_seg = (dir_ / 'nb/seg{:04d}.fits'.format(row['I_Z']))
            im_nb = Image(str(file_nb))
            im_seg = Image(str(file_seg))
            im_seg.data = (im_seg.data == row['ID_SLICE']) * 1
            im_seg.data.mask = np.zeros_like(im_seg.data, dtype=bool)

            src.add_image(im_nb, 'NB{:04.0f}'.format(row['WAVE']), ima_size)

            images_nb.append(im_nb)
            images_seg.append(im_seg)


        im_nb_coadd = images_nb[0].copy()
        im_seg_coadd = images_seg[0].copy()

        for im_nb, im_seg in zip(images_nb[1:], images_seg[1:]):
            im_nb_coadd.data += im_nb
            im_seg_coadd += im_seg

        im_seg_coadd.data = (im_seg_coadd.data != 0) * 1
        src.add_image(im_nb_coadd, 'NB_COADD', ima_size)
        src.add_image(im_seg_coadd, 'MASK_OBJ', ima_size)

        wave = rows['WAVE']
        flux = rows['FLUX']
        flux_err = rows['FLUX_ERR']

        lines = table.Table([wave, [dw]*len(rows), flux, flux_err],
                      names=['LBDA_OBS', 'LBDA_OBS_ERR', 'FLUX', 'FLUX_ERR'],
                      dtype=['<f8', '<f8', '<f8', '<f8'])
        lines['LBDA_OBS'].format = '.2f'
        lines['LBDA_OBS'].unit = u.angstrom
        lines['LBDA_OBS_ERR'].format = '.2f'
        lines['LBDA_OBS_ERR'].unit = u.angstrom
        lines['FLUX'].format = '.4f'
        lines['FLUX'].unit = cube.unit
        lines['FLUX_ERR'].format = '.4f'
        lines['FLUX_ERR'].unit = cube.unit
        src.lines = lines

        obj_cat.append(src)

        progress.increment()

    progress.close()

    return raw_cat, obj_cat
#
#
#
#
#    import matplotlib.pyplot as plt
#    plt.plot(cat['I_X'], cat['I_Y'], 'o')
#    plt.plot(cat_bgr['I_X'], cat_bgr['I_Y'], 'o')
#    for id_ in np.unique(ids):
#        m = ids == id_
#        plt.plot(cat['I_X'][m], cat['I_Y'][m])
#    plt.show()
#
#
#  #  cat.write('b.fit', overwrite=True)
#    import pdb; pdb.set_trace()


#improve edge cleaning
#find centroid of groups and link to continuum objects
#build raw sources of each
#include segmentiation maps
#build combined sources (store only the 25 brights lines)
#include union+intersection segmentiation maps
#run crackz on combined sources

     


    cube_version = str(cube.primary_header.get('CUBE_V', ''))

    wlmin = cube.wave.get_start(unit=u.angstrom)
    dw = cube.wave.get_step(unit=u.angstrom)

    ima_size = ima_size * c.wcs.get_step(unit=u.arcsec)[0]

    fullvar = Image('inv_variance.fits')

    cleanlimit = clean * np.ma.median(fullvar.data)

    logger.info("cleaning below inverse variance %s", cleanlimit)

    tBGR = Table.read('BGR.cat', format='ascii.fixed_width_two_line')

    maxidc = 0
    # Continuum lines
    C_ll = []
    C_idmin = []
    C_fline = []
    C_eline = []
    C_xline = []
    C_yline = []
    C_magB = []
    C_magG = []
    C_magR = []
    C_emagB = []
    C_emagG = []
    C_emagR = []
    C_catID = []
    # Single lines
    S_ll = []
    S_fline = []
    S_eline = []
    S_xline = []
    S_yline = []
    S_catID = []

    for i in range(3, nslices - 14):
        ll = wlmin + dw * i
        flagsky = 0

        if len(skyclean) > 0:
            for (skylmin, skylmax) in skyclean:
                if ll > skylmin and ll < skylmax:
                    flagsky = 1

        if flagsky == 0:
            slicename = "nb/nb%04d.cat" % i
            t = Table.read(slicename, format='ascii.sextractor')
            for line in t:
                xline = line['X_IMAGE']
                yline = line['Y_IMAGE']
                if fullvar.data[int(yline - 1), int(xline - 1)] > cleanlimit:
                    fline = 10.0 ** (0.4 * (25. - float(line['MAG_APER'])))
                    eline = float(line['MAGERR_APER']) * fline * (2.3 / 2.5)
                    flag = 0
                    distmin = -1
                    distlist = ((xline - tBGR['X_IMAGE']) ** 2.0 +
                                (yline - tBGR['Y_IMAGE']) ** 2.0)
                    ksel = np.where(distlist < radius ** 2.0)

                    for j in ksel[0]:
                        if(fline > 5.0 * eline):
                            if flag <= 0 or distlist[j] < distmin:
                                idmin = tBGR['NUMBER'][j]
                                distmin = distlist[j]
                                magB = tBGR['MAG_APER_B'][j]
                                magG = tBGR['MAG_APER_G'][j]
                                magR = tBGR['MAG_APER_R'][j]
                                emagB = tBGR['MAGERR_APER_B'][j]
                                emagG = tBGR['MAGERR_APER_G'][j]
                                emagR = tBGR['MAGERR_APER_R'][j]
                                xline = tBGR['X_IMAGE'][j]
                                yline = tBGR['Y_IMAGE'][j]
                                flag = 1
                        else:
                            if fline < -5 * eline:
                                idmin = tBGR['NUMBER'][j]
                                distmin = distlist[j]
                                flag = -2
                            else:
                                flag = -1

                    if flag == 1:
                        C_ll.append(ll)
                        C_idmin.append(idmin)
                        C_fline.append(fline)
                        C_eline.append(eline)
                        C_xline.append(xline)
                        C_yline.append(yline)
                        C_magB.append(magB)
                        C_magG.append(magG)
                        C_magR.append(magR)
                        C_emagB.append(emagB)
                        C_emagG.append(emagG)
                        C_emagR.append(emagR)
                        C_catID.append(i)
                        if(idmin > maxidc):
                            maxidc = idmin

                    if flag == 0 and ll < 9300.0:
                        S_ll.append(ll)
                        S_fline.append(fline)
                        S_eline.append(eline)
                        S_xline.append(xline)
                        S_yline.append(yline)
                        S_catID.append(i)

    nC = len(C_ll)
    nS = len(S_ll)

    flags = np.ones(nC)
    for i in range(nC):
        fl = 0
        for j in range(nC):
            if ((i != j) and (C_idmin[i] == C_idmin[j]) and
                    (np.abs(C_ll[j] - C_ll[i]) < (3.00))):
                if(C_fline[i] < C_fline[j]):
                    flags[i] = 0
                fl = 1
        if fl == 0:  # identification of single line emissions
            flags[i] = 2

    # Sources list
    continuum_lines = SourceList()
    origin = ('muselet', __version__, cubename, cubevers)

    # write all continuum lines here:
    raw_catalog = SourceList()
    idraw = 0
    for i in range(nC):
        if flags[i] == 1:
            idraw = idraw + 1
            dec, ra = c.wcs.pix2sky([C_yline[i] - 1, C_xline[i] - 1],
                                    unit=u.deg)[0]
            s = Source.from_data(ID=idraw, ra=ra, dec=dec, origin=origin)
            s.add_mag('MUSEB', C_magB[i], C_emagB[i])
            s.add_mag('MUSEG', C_magG[i], C_emagG[i])
            s.add_mag('MUSER', C_magR[i], C_emagR[i])
            lbdas = [C_ll[i]]
            fluxes = [C_fline[i]]
            err_fluxes = [C_eline[i]]
            ima = Image('nb/nb%04d.fits' % C_catID[i])
            s.add_image(ima, 'NB%04d' % int(C_ll[i]), ima_size)
            lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                          names=['LBDA_OBS', 'LBDA_OBS_ERR',
                                 'FLUX', 'FLUX_ERR'],
                          dtype=['<f8', '<f8', '<f8', '<f8'])
            lines['LBDA_OBS'].format = '.2f'
            lines['LBDA_OBS'].unit = u.angstrom
            lines['LBDA_OBS_ERR'].format = '.2f'
            lines['LBDA_OBS_ERR'].unit = u.angstrom
            lines['FLUX'].format = '.4f'
            # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lines['FLUX_ERR'].format = '.4f'
            s.lines = lines
            raw_catalog.append(s)

    for r in range(maxidc + 1):
        lbdas = []
        fluxes = []
        err_fluxes = []
        for i in range(nC):
            if (C_idmin[i] == r) and (flags[i] == 1):
                if len(lbdas) == 0:
                    dec, ra = c.wcs.pix2sky([C_yline[i] - 1, C_xline[i] - 1],
                                            unit=u.deg)[0]
                    s = Source.from_data(ID=r, ra=ra, dec=dec, origin=origin)
                    s.add_mag('MUSEB', C_magB[i], C_emagB[i])
                    s.add_mag('MUSEG', C_magG[i], C_emagG[i])
                    s.add_mag('MUSER', C_magR[i], C_emagR[i])
                lbdas.append(C_ll[i])
                fluxes.append(C_fline[i])
                err_fluxes.append(C_eline[i])
                ima = Image('nb/nb%04d.fits' % C_catID[i])
                s.add_image(ima, 'NB%04d' % int(C_ll[i]), ima_size)
        if len(lbdas) > 0:
            lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                          names=['LBDA_OBS', 'LBDA_OBS_ERR',
                                 'FLUX', 'FLUX_ERR'],
                          dtype=['<f8', '<f8', '<f8', '<f8'])
            lines['LBDA_OBS'].format = '.2f'
            lines['LBDA_OBS'].unit = u.angstrom
            lines['LBDA_OBS_ERR'].format = '.2f'
            lines['LBDA_OBS_ERR'].unit = u.angstrom
            lines['FLUX'].format = '.4f'
            # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lines['FLUX_ERR'].format = '.4f'
            s.lines = lines
            continuum_lines.append(s)

    if len(continuum_lines) > 0:
        logger.info("%d continuum lines detected",
                    len(continuum_lines))
    else:
        logger.info("no continuum lines detected")

    singflags = np.ones(nS)
    S2_ll = []
    S2_fline = []
    S2_eline = []
    S2_xline = []
    S2_yline = []
    S2_catID = []

    for i in range(nS):
        fl = 0
        xref = S_xline[i]
        yref = S_yline[i]
        ksel = np.where((xref - S_xline) ** 2.0 + (yref - S_yline) ** 2.0 <
                        (radius / 2.0) ** 2.0)  # spatial distance
        for j in ksel[0]:
            if (i != j) and (np.abs(S_ll[j] - S_ll[i]) < 3.0):
                if S_fline[i] < S_fline[j]:
                    singflags[i] = 0
                fl = 1
        if fl == 0:
            singflags[i] = 2
        if singflags[i] == 1:
            S2_ll.append(S_ll[i])
            S2_fline.append(S_fline[i])
            S2_eline.append(S_eline[i])
            S2_xline.append(S_xline[i])
            S2_yline.append(S_yline[i])
            S2_catID.append(S_catID[i])

    # output single lines catalogs here:S2_ll,S2_fline,S2_eline,S2_xline,
    # S2_yline,S2_catID
    nlines = len(S2_ll)
    for i in range(nlines):
        idraw = idraw + 1
        dec, ra = c.wcs.pix2sky([S2_yline[i] - 1, S2_xline[i] - 1],
                                unit=u.deg)[0]
        s = Source.from_data(ID=idraw, ra=ra, dec=dec, origin=origin)
        lbdas = [S2_ll[i]]
        fluxes = [S2_fline[i]]
        err_fluxes = [S2_eline[i]]
        ima = Image('nb/nb%04d.fits' % S2_catID[i])
        s.add_image(ima, 'NB%04d' % int(S2_ll[i]), ima_size)
        lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                      names=['LBDA_OBS', 'LBDA_OBS_ERR',
                             'FLUX', 'FLUX_ERR'],
                      dtype=['<f8', '<f8', '<f8', '<f8'])
        lines['LBDA_OBS'].format = '.2f'
        lines['LBDA_OBS'].unit = u.angstrom
        lines['LBDA_OBS_ERR'].format = '.2f'
        lines['LBDA_OBS_ERR'].unit = u.angstrom
        lines['FLUX'].format = '.4f'
        # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        lines['FLUX_ERR'].format = '.4f'
        s.lines = lines
        raw_catalog.append(s)

    # List of single lines
    # Merging single lines of the same object
    single_lines = SourceList()
    flags = np.zeros(nlines)
    ising = 0
    for i in range(nlines):
        if(flags[i] == 0):
            ising = ising + 1
            lbdas = []
            fluxes = []
            err_fluxes = []

            dec, ra = c.wcs.pix2sky([S2_yline[i] - 1, S2_xline[i] - 1],
                                    unit=u.deg)[0]
            s = Source.from_data(ID=ising, ra=ra, dec=dec, origin=origin)
            lbdas.append(S2_ll[i])
            fluxes.append(S2_fline[i])
            err_fluxes.append(S2_eline[i])
            ima = Image('nb/nb%04d.fits' % S2_catID[i])
            s.add_image(ima, 'NB%04d' % int(S2_ll[i]), ima_size)
            ksel = np.where(((S2_xline[i] - S2_xline) ** 2.0 +
                             (S2_yline[i] - S2_yline) ** 2.0 <
                             radius ** 2.0) & (flags == 0))
            for j in ksel[0]:
                if(j != i):
                    lbdas.append(S2_ll[j])
                    fluxes.append(S2_fline[j])
                    err_fluxes.append(S2_eline[j])
                    ima = Image('nb/nb%04d.fits' % S2_catID[j])
                    s.add_image(ima, 'NB%04d' % int(S2_ll[j]), ima_size)
                    flags[j] = 1
            lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                          names=['LBDA_OBS', 'LBDA_OBS_ERR',
                                 'FLUX', 'FLUX_ERR'],
                          dtype=['<f8', '<f8', '<f8', '<f8'])
            lines['LBDA_OBS'].format = '.2f'
            lines['LBDA_OBS'].unit = u.angstrom
            lines['LBDA_OBS_ERR'].format = '.2f'
            lines['LBDA_OBS_ERR'].unit = u.angstrom
            lines['FLUX'].format = '.4f'
            # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lines['FLUX_ERR'].format = '.4f'
            s.lines = lines
            single_lines.append(s)
            flags[i] = 1

    if len(single_lines) > 0:
        logger.info("%d single lines detected", len(single_lines))
    else:
        logger.info("no single lines detected")

    # redshift of continuum objects
    if not os.path.isfile('emlines'):
        shutil.copy(join(DATADIR, 'emlines'), 'emlines')
    if not os.path.isfile('emlines_small'):
        shutil.copy(join(DATADIR, 'emlines_small'), 'emlines_small')

    eml = dict(np.loadtxt("emlines",
                          dtype={'names': ('lambda', 'lname'),
                                 'formats': ('f', 'S20')}))
    eml2 = dict(np.loadtxt("emlines_small",
                           dtype={'names': ('lambda', 'lname'),
                                  'formats': ('f', 'S20')}))

    logger.info("estimating the best redshift")
    for source in continuum_lines:
        if(len(source.lines) > 3):
            source.crack_z(eml)
        else:
            source.crack_z(eml2)
        source.sort_lines(nlines_max)
    for source in single_lines:
        if(len(source.lines) > 3):
            source.crack_z(eml, 20)
        else:
            source.crack_z(eml2, 20)
        source.sort_lines(nlines_max)

#         logger.info("cleaning narrow-band images")
#         if(nbcube):
#             filelist = glob('nb/nb*.fits')
#             for f in filelist:
#                 os.remove(f)

    # sort single line sources by wavelength
    lsource = lambda s: s.lines[0][0]
    single_lines.sort(key=lsource)

    return continuum_lines, single_lines, raw_catalog


def muselet(cubename, step=1, delta=20, fw=(0.26, 0.7, 1., 0.7, 0.26),
            sex_config=None, sex_config_nb=None,
            radius=4.0, ima_size=21, nlines_max=25, clean=0.5, nbcube=True,
            expmapcube=None, skyclean=((5573.5, 5578.8), (6297.0, 6300.5)),
            del_sex=False, workdir=None, n_cpu=1):
    """MUSELET (for MUSE Line Emission Tracker) is a simple SExtractor-based
    python tool to detect emission lines in a datacube. It has been developed
    by Johan Richard (johan.richard@univ-lyon1.fr)

    Parameters
    ----------
    cubename : str
        Name of the MUSE cube.
    step : int in {1,2,3}
        Starting step for MUSELET to run:

        - (1) produces the narrow-band images
        - (2) runs SExtractor
        - (3) merges catalogs and measure redshifts

    delta : int
        Size of the two median continuum estimates to be taken
        on each side of the narrow-band image (in MUSE wavelength planes).
        Default is 20 planes, or 25 Angstroms.
    fw : list of 5 floats
        Define the weights on the 5 central wavelength planes when estimated
        the line-profile-weighted flux in the narrow-band images.
    radius : float
        Radius in spatial pixels (default=4) within which emission lines
        are merged spatially into the same object.
    ima_size : int
        Size of the extracted images in pixels.
    nlines_max : int
        Maximum number of lines detected per object.
    clean : float
        Removing sources at a fraction of the the max_weight level.
    nbcube : bool
        Flag to produce an output datacube containing all narrow-band images
    expmapcube : str
        Name of the associated exposure map cube (to be used as a weight map
        for SExtractor)
    skyclean : array of float tuples
        List of wavelength ranges to exclude from the raw detections
    del_sex : bool
        If True, configuration files and intermediate files used by sextractor
        are removed.
    workdir : str
        Working directory, default is the current directory.
    n_cpu : int
        max number of CPU cores to use in parallel

    Returns
    -------
    continuum, single, raw : `~mpdaf.sdetect.SourceList`, `~mpdaf.sdetect.SourceList`, `~mpdaf.sdetect.SourceList`
        - continuum : List of detected sources that contains emission lines
            associated with continuum detection
        - single : List of detected sources that contains emission lines not
            associated with continuum detection
        - raw : List of detected sources  before the merging procedure.

    """
    logger = logging.getLogger(__name__)

    if step not in (1, 2, 3):
        logger.error("ERROR: step must be 1, 2 or 3")
        logger.error("STEP 1: create images from cube")
        logger.error("STEP 2: run SExtractor")
        logger.error("STEP 3: merge catalogs and measure redshifts")
        return

    if len(fw) != 5:
        logger.error("len(fw) != 5")

    try:
        fw = np.array(fw, dtype=np.float32)
    except Exception:
        logger.error('fw is not an array of float')

    try:
        subprocess.check_call(['sex', '-v'])
        cmd_sex = 'sex'
    except OSError:
        try:
            subprocess.check_call(['sextractor', '-v'])
            cmd_sex = 'sextractor'
        except OSError:
            raise OSError('SExtractor not found')

    if workdir is None:
        workdir = Path.cwd()
    else:
        workdir = Path(workdir)
        os.makedirs(workdir, exist_ok=True)

    cubename = Path(cubename)
    if expmapcube is not None:
        expmapcube = Path(expmapcube)
    else:
        logger.debug("No exposure cube provided")


    if step == 1:
        step1(cubename, expmapcube, fw, delta, dir_=workdir, nbcube=nbcube,
                n_cpu=n_cpu)
    if step <= 2:
        step2(cubename, cmd_sex, config=sex_config, config_nb=sex_config_nb,
                dir_=workdir, n_cpu=n_cpu)

    if step <= 3:
        raw, obj = step3(cubename, ima_size, clean, skyclean,
                            radius, nlines_max, dir_=workdir, n_cpu=n_cpu)

    return raw, obj

   # with chdir(workdir):
   #     if del_sex:
   #         remove_files()

   # return continuum_lines, single_lines, raw_catalog

#             t = Catalog.from_sources(continuum_lines)
#             t.write('continuum_lines_z2.cat', format='ascii')
#             t = Catalog.from_sources(single_lines)
#             t.write('single_lines_z2.cat', format='ascii')
