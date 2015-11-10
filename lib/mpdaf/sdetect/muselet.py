from astropy.table import Table
import astropy.units as u
import logging
import numpy as np
import os
import shutil
import subprocess
import stat
import sys
from glob import glob


from ..obj import Cube, Image
from ..sdetect import Source, SourceList

__version__ = 2.1


def setup_config_files():
    DIR = os.path.dirname(__file__) + '/muselet_data/'
    files = ['default.sex', 'default.conv', 'default.nnw', 'default.param']
    for f in files:
        try:
            if not os.path.isfile(f):
                shutil.copy(DIR + f, './' + f)
        except:
            pass


def setup_config_files_nb():
    DIR = os.path.dirname(__file__) + '/muselet_data/'
    files = ['nb_default.sex', 'nb_default.conv', 'nb_default.nnw', 'nb_default.param']
    for f in files:
        try:
            if not os.path.isfile(f[3:]):
                shutil.copy(DIR + f, './' + f[3:])
        except:
            pass


def remove_files():
    files = ['default.sex', 'default.conv', 'default.nnw', 'default.param',
             'emlines', 'emlines_small', 'BGR.cat', 'inv_variance.fits', 'segment.fits',
             'white.fits', 'whiteb.fits', 'whiteg.fits', 'whiter.fits', 'detect.cat']

    for f in files:
        try:
            os.remove(f)
        except:
            pass
    for f in glob('nb/*'):
        os.remove(f)
    os.removedirs('nb')


def muselet(cubename, step=1, delta=20, fw=[0.26, 0.7, 1., 0.7, 0.26], radius=4.0, ima_size=21, nlines_max=25, clean=0.5, nbcube=True, expmapcube=None, skyclean=[(5573.5, 5578.8), (6297.0, 6300.5)], del_sex=False):
    """MUSELET (for MUSE Line Emission Tracker) is a simple SExtractor-based python tool
    to detect emission lines in a datacube. It has been developed by Johan Richard
    (johan.richard@univ-lyon1.fr)

    Parameters
    ----------
    cubename : string
               Name of the MUSE cube.
    step   : integer in {1,2,3}
             Starting step for MUSELET to run.
             (1) produces the narrow-band images
             (2) runs SExtractor
             (3) merges catalogs and measure redshifts,
    delta  : integer
             Size of the two median continuum estimates to be taken 
             on each side of the narrow-band image (in MUSE wavelength planes).
             Default is 20 planes, or 25 Angstroms.
    fw     : list of 5 floats
             Define the weights on the 5 central wavelength planes
             when estimated the line-profile-weighted flux
             in the narrow-band images
    radius : double
             Radius in spatial pixels (default=4) within which emission lines
             are merged spatially into the same object.
    ima_size : integer
               Size of the extracted images in pixels.
    nlines_max : integer
                 Maximum number of lines detected per object.
    clean  : Float
             Removing sources at a fraction of the the max_weight level.
    nbcube : Boolean
             Flag to produce an output datacube containing all narrow-band images
    expmapcube : string
             Name of the associated exposure map cube (to be used as a weight map for SExtractor)
    skyclean : array of float tuples
             List of wavelength ranges to exclude from the raw detections
    del_sex  : boolean
              If True, configuration files and intermediate files used by sextractor are removed.

    Returns
    -------
    continuum,single,raw : :class:`mpdaf.sdetect.SourceList`, :class:`mpdaf.sdetect.SourceList`, :class:`mpdaf.sdetect.SourceList`
    continuum             : List of detected sources that contains emission lines associated with continuum detection
    single                : List of detected sources that contains emission lines not associated with continuum detection
    raw                   : List of detected sources  before the merging procedure.
    """
    logger = logging.getLogger(__name__)

    if(step != 1 and step != 2 and step != 3):
        logger.error("muselet - ERROR: step must be 1, 2 or 3")
        logger.error("muselet - STEP 1: creates images from cube")
        logger.error("muselet - STEP 2: runs SExtractor")
        logger.error("muselet - STEP 3: merge catalogs and measure redshifts")
        return
    if len(fw) != 5:
        logger.error("muselet - len(fw) != 5")
    try:
        fw = np.array(fw, dtype=np.float)
    except:
        logger.error('muselet - fw is not an array of float')

    try:
        subprocess.check_call(['sex'])
        cmd_sex = 'sex'
    except OSError:
        try:
            subprocess.check_call(['sextractor'])
            cmd_sex = 'sextractor'
        except OSError:
            raise OSError('SExtractor not found')

    path = os.path.dirname(__file__) + '/muselet_data/'
    c = None
    fullvar = None

    if(step == 1):
        logger.info("muselet - Opening: " + cubename)
        c = Cube(cubename)

        mvar = np.ma.masked_invalid(c.var)

        imsum = c[0, :, :]
        size1 = c.shape[0]
        size2 = c.shape[1]
        size3 = c.shape[2]

        mcentralvar = np.ma.masked_invalid(c[2: size1 - 3, :, :].var)

        nsfilter = int(size1 / 3.0)

        logger.info("muselet - STEP 1: creates white light, variance, RGB and narrow-band images")
        weight_data = np.ma.average(c.data[1:size1 - 1, :, :], weights=1. / mvar[1:size1 - 1, :, :], axis=0)
        weight = Image(wcs=imsum.wcs, data=np.ma.filled(weight_data, np.nan), unit=imsum.unit)
        weight.write('white.fits', savemask='nan')

        if not expmapcube:
            fullvar_data = np.ma.masked_invalid(1.0 / mcentralvar.mean(axis=0))
            fullvar = Image(wcs=imsum.wcs, data=np.ma.filled(fullvar_data, np.nan), unit=u.Unit(1) / (imsum.unit**2))
            fullvar.write('inv_variance.fits', savemask='nan')
        else:
            logger.info("muselet - Opening exposure map cube: " + expmapcube)
            expmap = Cube(expmapcube)
            fullvar = expmap.mean(axis=0)
            fullvar.write('inv_variance.fits', savemask='nan')

        bdata = np.ma.average(c.data[1:nsfilter, :, :], weights=1. / mvar[1:nsfilter, :, :], axis=0)
        gdata = np.ma.average(c.data[nsfilter:2 * nsfilter, :, :], weights=1. / mvar[nsfilter:2 * nsfilter, :, :], axis=0)
        rdata = np.ma.average(c.data[2 * nsfilter:size1 - 1, :, :], weights=1. / mvar[2 * nsfilter:size1 - 1, :, :], axis=0)
        r = Image(wcs=imsum.wcs, data=np.ma.filled(rdata, np.nan), unit=imsum.unit)
        g = Image(wcs=imsum.wcs, data=np.ma.filled(gdata, np.nan), unit=imsum.unit)
        b = Image(wcs=imsum.wcs, data=np.ma.filled(bdata, np.nan), unit=imsum.unit)
        r.write('whiter.fits', savemask='nan')
        g.write('whiteg.fits', savemask='nan')
        b.write('whiteb.fits', savemask='nan')

        fwcube = np.ones((5, size2, size3)) * fw[:, np.newaxis, np.newaxis]

        try:
            os.mkdir("nb")
        except:
            pass

        if(nbcube):
            outnbcubename = 'NB_' + os.path.basename(cubename)
            outnbcube = c.clone(data_init=np.empty)
            outnbcube.var = None

        f2 = open("nb/dosex", 'w')
        for k in range(2, size1 - 3):
            sys.stdout.write("Narrow band:%d/%d" % (k, size1 - 3) + "\r")
            leftmin = max(0, k - 2 - delta)
            leftmax = k - 2
            rightmin = k + 3
            rightmax = min(size1, k + 3 + delta)
            imslice = np.ma.average(c.data[k - 2:k + 3, :, :], weights=fwcube / mvar[k - 2:k + 3, :, :], axis=0)
            if(leftmax == 1):
                contleft = c.data.data[0, :, :]
            elif(leftmax > leftmin + 1):
                contleft = np.median(c.data.data[leftmin:leftmax, :, :], axis=0)
            elif(rightmax == size1):
                contleft = c.data.data[-1, :, :]
            else:
                contleft = c.data.data[0, :, :]
            if(rightmax > rightmin):
                contright = np.median(c.data.data[rightmin:rightmax, :, :], axis=0)
            else:
                contright = c.data.data[0, :, :]
            sizeleft = leftmax - leftmin
            sizeright = rightmax - rightmin
            contmean = (sizeleft * contleft + sizeright * contright) / (sizeleft + sizeright)
            imnb = Image(wcs=imsum.wcs, data=np.ma.filled(imslice - contmean, np.nan),
                         unit=imsum.unit)
            kstr = "%04d" % k
            imnb.write('nb/nb' + kstr + '.fits', savemask='nan')
            if(nbcube):
                outnbcube.data[k, :, :] = imnb.data[:, :]
            if(expmapcube):
                expmap[k, :, :].write('nb/exp' + kstr + '.fits', savemask='nan')
                f2.write(cmd_sex + ' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME nb' + kstr + '.cat -WEIGHT_IMAGE exp' + kstr + '.fits nb' + kstr + '.fits\n')
            else:
                f2.write(cmd_sex + ' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME nb' + kstr + '.cat nb' + kstr + '.fits\n')

        sys.stdout.write("\n")
        sys.stdout.flush()
        f2.close()
        if(nbcube):
            outnbcube.write(outnbcubename)

    if(step <= 2):
        logger.info("muselet - STEP 2: runs SExtractor on white light, RGB and narrow-band images")
        # tests here if the files default.sex, default.conv, default.nnw and default.param exist.
        # Otherwise copy them
        setup_config_files()

        subprocess.Popen(cmd_sex + ' white.fits', shell=True).wait()
        subprocess.Popen(cmd_sex + ' -CATALOG_NAME R.cat -CATALOG_TYPE ASCII_HEAD white.fits,whiter.fits', shell=True).wait()
        subprocess.Popen(cmd_sex + ' -CATALOG_NAME G.cat -CATALOG_TYPE ASCII_HEAD white.fits,whiteg.fits', shell=True).wait()
        subprocess.Popen(cmd_sex + ' -CATALOG_NAME B.cat -CATALOG_TYPE ASCII_HEAD white.fits,whiteb.fits', shell=True).wait()

        tB = Table.read('B.cat', format='ascii.sextractor')
        tG = Table.read('G.cat', format='ascii.sextractor')
        tR = Table.read('R.cat', format='ascii.sextractor')

        names = ('NUMBER', 'X_IMAGE', 'Y_IMAGE',
                 'MAG_APER_B', 'MAGERR_APER_B',
                 'MAG_APER_G', 'MAGERR_APER_G',
                 'MAG_APER_R', 'MAGERR_APER_R')
        tBGR = Table([tB['NUMBER'], tB['X_IMAGE'], tB['Y_IMAGE'],
                      tB['MAG_APER'], tB['MAGERR_APER'],
                      tG['MAG_APER'], tG['MAGERR_APER'],
                      tR['MAG_APER'], tR['MAGERR_APER']], names=names)
        tBGR.write('BGR.cat', format='ascii.fixed_width_two_line')

        os.remove('B.cat')
        os.remove('G.cat')
        os.remove('R.cat')

        try:
            os.chdir("nb")
        except:
            logger.error("muselet - ERROR: missing nb directory")
            return

        # tests here if the files default.sex, default.conv, default.nnw and default.param exist.
        # Otherwise copy them
        setup_config_files_nb()
        shutil.copy('../inv_variance.fits', 'inv_variance.fits')
        st = os.stat('dosex')
        os.chmod('dosex', st.st_mode | stat.S_IEXEC)
        subprocess.Popen('./dosex', shell=True).wait()
        os.chdir("..")

#         file = open('dosex', 'r')
#         for line in file:
#             p = subprocess.Popen(line, shell=True).wait()

    if(step <= 3):
        logger.info("muselet - STEP 3: merge SExtractor catalogs and measure redshifts")

        if c is None:
            logger.info("muselet - Opening: " + cubename)
            c = Cube(cubename)

        wlmin = c.wave.get_start(unit=u.angstrom)
        dw = c.wave.get_step(unit=u.angstrom)
        nslices = c.shape[0]

        ima_size = ima_size * c.wcs.get_step(unit=u.arcsec)[0]

        if fullvar is None:
            fullvar = Image('inv_variance.fits')

        cleanlimit = clean * np.median(fullvar.data)

        logger.info("muselet - cleaning below inverse variance " + str(cleanlimit))

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
                    if(ll > skylmin and ll < skylmax):
                        flagsky = 1
            if(flagsky == 0):
                slicename = "nb/nb%04d.cat" % i
                t = Table.read(slicename, format='ascii.sextractor')
                for line in t:
                    xline = line['X_IMAGE']
                    yline = line['Y_IMAGE']
                    if(fullvar.data[int(yline - 1), int(xline - 1)] > cleanlimit):
                        fline = 10.0 ** (0.4 * (25. - float(line['MAG_APER'])))
                        eline = float(line['MAGERR_APER']) * fline * (2.3 / 2.5)
                        flag = 0
                        distmin = -1
                        distlist = (xline - tBGR['X_IMAGE']) ** 2.0 + (yline - tBGR['Y_IMAGE']) ** 2.0
                        ksel = np.where(distlist < radius ** 2.0)
                        for j in ksel[0]:
                            if(fline > 5.0 * eline):
                                if((flag <= 0)or(distlist[j] < distmin)):
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
                                if(fline < -5 * eline):
                                    idmin = tBGR['NUMBER'][j]
                                    distmin = distlist[j]
                                    flag = -2
                                else:
                                    flag = -1
                        if(flag == 1):
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
                        if (flag == 0) and (ll < 9300.0):
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
                if((i != j) and (C_idmin[i] == C_idmin[j]) and (np.abs(C_ll[j] - C_ll[i]) < (3.00))):
                    if(C_fline[i] < C_fline[j]):
                        flags[i] = 0
                    fl = 1
            if(fl == 0):  # identification of single line emissions
                flags[i] = 2

        # Sources list
        continuum_lines = SourceList()
        origin = ('muselet', __version__, cubename)

        # write all continuum lines here:
        raw_catalog = SourceList()
        idraw = 0
        for i in range(nC):
            if (flags[i] == 1):
                idraw = idraw + 1
                dec, ra = c.wcs.pix2sky([C_yline[i] - 1, C_xline[i] - 1], unit=u.deg)[0]
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
                        dec, ra = c.wcs.pix2sky([C_yline[i] - 1, C_xline[i] - 1], unit=u.deg)[0]
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
            logger.info("muselet - %d continuum lines detected" % len(continuum_lines))
        else:
            logger.info("muselet - no continuum lines detected")

        #
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
            ksel = np.where((xref - S_xline) ** 2.0 + (yref - S_yline) ** 2.0 < (radius / 2.0) ** 2.0)  # spatial distance
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

        # output single lines catalogs here:S2_ll,S2_fline,S2_eline,S2_xline,S2_yline,S2_catID
        nlines = len(S2_ll)
        for i in range(nlines):
            idraw = idraw + 1
            dec, ra = c.wcs.pix2sky([S2_yline[i] - 1, S2_xline[i] - 1], unit=u.deg)[0]
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

                dec, ra = c.wcs.pix2sky([S2_yline[i] - 1, S2_xline[i] - 1], unit=u.deg)[0]
                s = Source.from_data(ID=ising, ra=ra, dec=dec, origin=origin)
                lbdas.append(S2_ll[i])
                fluxes.append(S2_fline[i])
                err_fluxes.append(S2_eline[i])
                ima = Image('nb/nb%04d.fits' % S2_catID[i])
                s.add_image(ima, 'NB%04d' % int(S2_ll[i]), ima_size)
                ksel = np.where(((S2_xline[i] - S2_xline) ** 2.0 + (S2_yline[i] - S2_yline) ** 2.0 < radius ** 2.0) & (flags == 0))
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
            logger.info("muselet - %d single lines detected" % len(single_lines))
        else:
            logger.info("muselet - no single lines detected")

        # redshift of continuum objects

        if not os.path.isfile('emlines'):
            shutil.copy(path + 'emlines', 'emlines')
        if not os.path.isfile('emlines_small'):
            shutil.copy(path + 'emlines_small', 'emlines_small')
        eml = dict(np.loadtxt("emlines", dtype={'names': ('lambda', 'lname'), 'formats': ('f', 'S20')}))
        eml2 = dict(np.loadtxt("emlines_small", dtype={'names': ('lambda', 'lname'), 'formats': ('f', 'S20')}))

        logger.info("muselet - estimating the best redshift")
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

        logger.info("muselet - cleaning narrow-band images")
        if(nbcube):
            filelist = glob('nb/nb*.fits')
            for f in filelist:
                os.remove(f)

        # sort single line sources by wavelength
        lsource = lambda s: s.lines[0][0]
        single_lines.sort(key=lsource)

        if del_sex:
            remove_files()

        return continuum_lines, single_lines, raw_catalog

#             t = Catalog.from_sources(continuum_lines)
#             t.write('continuum_lines_z2.cat', format='ascii')
#             t = Catalog.from_sources(single_lines)
#             t.write('single_lines_z2.cat', format='ascii')
