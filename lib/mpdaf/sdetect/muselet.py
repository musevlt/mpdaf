from astropy.table import Table
import numpy as np
from ..obj import Cube, Image
import os
import sys
import shutil
import subprocess
import stat


def catalog(nb, xline, yline, line):
    if len(line[0][0]) == 2:
        names= ('NUMBER', 'X_IMAGE', 'Y_IMAGE', 'LINE (LAMBDA, FLUX)')
        t = Table([nb, xline, yline, line], names=names)
        t['NUMBER'].format = '%04d'
        t['X_IMAGE'].format = '%.1f'
        t['Y_IMAGE'].format = '%.1f'
        #t['LINE (LAMBDA, FLUX)'].format = lambda v : '%.2f %.2e'%(v[0][0], v[0][1])
        return t
    
#t = Table.read('continuum_lines.cat', format='ascii.fixed_width_two_line')
        #for line in t:
        #    l = line['LINE (LAMBDA, FLUX)']
        #    a = np.array(l.replace('[',' ').replace(']',' ').replace('(',' ').replace(')',' ').replace(',',' ').split(), dtype=np.float)
        #    wl = a[::2]
        #    flux = a[1::2]
        

def matchlines(nlines, wl, z, eml, eml2):
    """ try to match all the lines given : for each line computes the distance
     in Angstroms to the closest line. Add the errors
    """
    jfound = np.zeros(nlines, dtype=np.int)
    if(nlines > 3):
        listwl = eml
    else:
        listwl = eml2
    error = 0
    for i in range(nlines):
        # finds closest emline to this line
        jfound[i] = np.argmin((wl[i] / (1 + z) - listwl['lambda']) ** 2.0)
        error += (wl[i] / (1 + z) - listwl['lambda'][jfound[i]]) ** 2.0
    error = np.sqrt(error / nlines)
    if((nlines >= 2)and(jfound[0] == jfound[1])):
        error = 15.
    return(error, jfound)


def crackz(nlines, wl, flux, eml, eml2):
    errmin = 3.0
    zmin = 0.0
    zmax = 7.0
    if(nlines == 0):
        return 0
    if(nlines == 1):
        return(1, "%f %f Lya z=%f or [OII] z=%f" % (wl[0], flux[0], wl[0] / 1216.0 - 1.0, wl[0] / 3727. - 1.0))
    if(nlines > 1):
        found = 0
        if(nlines > 3):
            listwl = eml
        else:
            listwl = eml2
        for z in np.arange(zmin, zmax, 0.001):
            (error, jfound) = matchlines(nlines, wl, z, eml, eml2)
            if(error < errmin):
                errmin = error
                found = 1
                zfound = z
                jfinal = jfound.copy()
        if(found == 1):
            jfinal = np.array(jfinal).astype(int)
            returnstr = "%f %f " % (zfound, errmin / np.min(listwl['lambda'][jfinal]))
            for i in range(nlines):
                returnstr = returnstr + "%f %s " % (wl[i], listwl['lname'][jfinal[i]])
            return(nlines, returnstr)
        else:
            if(nlines > 3):
                # keep the three brightest
                ksel = np.argsort(flux)[-1:-4:-1]
                return crackz(3, wl[ksel], flux[ksel], eml, eml2)
            if(nlines == 3):
                # keep the two brightest
                ksel = np.argsort(flux)[-1:-3:-1]
                return crackz(2, wl[ksel], flux[ksel], eml, eml2)
            if(nlines == 2):
                # keep the brightest
                ksel = np.argsort(flux)[-1]
                return(1, "%f %f Lya z=%f or [OII] z=%f" % (wl[ksel], flux[ksel], wl[ksel] / 1216.0 - 1.0, wl[ksel] / 3727. - 1.0))



def muselet(cubename, step=1, delta=20, fw=[0.26, 0.7, 1., 0.7, 0.26],radius=4.0):
    """
    delta : size of the two median continuum estimates near the emission line (in MUSE wavelength planes)
    fw: list of 5 floats
    
    """
    if(step != 1 and step != 2 and step != 3):
        print "ERROR: step must be 1, 2 or 3"
        print "STEP 1: creates images from cube"
        print "STEP 2: runs SExtractor"
        print "STEP 3: merge catalogs and measure redshifts"
        return
    if len(fw) != 5:
        print 'error'
    try:
        fw = np.array(fw, dtype=np.float)
    except:
        print 'error'
        
    try:
        subprocess.check_call(['sex'])
        cmd_sex = 'sex'
    except OSError:
        try:
            subprocess.check_call(['sextractor'])
            cmd_sex = 'sextractor'
        except OSError:
            raise OSError('SExtractor not found')
           
    path = os.path.dirname(__file__)+'/muselet_data/'    
        
    try:
        print "Opening: " + cubename
        c = Cube(cubename)
    except:
        print "Cannot open cube named: " + cubename + " !!!"
        return
    imsum = c[0, :, :]
    size1 = c.shape[0]
    size2 = c.shape[1]
    size3 = c.shape[2]

    nsfilter = int(size1 / 3.0)

    #filter = [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]

    if(step == 1):
        print "STEP 1: creates white light, variance, RGB and narrow-band images"
        weight_data = np.ma.average(c.data[1:size1 - 1, :, :], weights=1. / c.var[1:size1 - 1, :, :], axis=0)
        weight = Image(wcs=imsum.wcs, data=np.ma.filled(weight_data, np.nan), shape=imsum.shape, fscale=imsum.fscale)
        weight.write('white.fits', savemask='nan')

        fullvar_data = np.ma.masked_invalid(1.0 / c[2: size1 - 3, :, :].var.mean(axis=0))
        fullvar = Image(wcs=imsum.wcs, data=np.ma.filled(fullvar_data, np.nan), shape=imsum.shape, fscale=imsum.fscale ** 2.0)
        fullvar.write('inv_variance.fits', savemask='nan')

        bdata = np.ma.average(c.data[1:nsfilter, :, :], weights=1. / c.var[1:nsfilter, :, :], axis=0)
        gdata = np.ma.average(c.data[nsfilter:2 * nsfilter, :, :], weights=1. / c.var[nsfilter:2 * nsfilter, :, :], axis=0)
        rdata = np.ma.average(c.data[2 * nsfilter:size1 - 1, :, :], weights=1. / c.var[2 * nsfilter:size1 - 1, :, :], axis=0)
        r = Image(wcs=imsum.wcs, data=np.ma.filled(rdata, np.nan), shape=imsum.shape, fscale=imsum.fscale)
        g = Image(wcs=imsum.wcs, data=np.ma.filled(gdata, np.nan), shape=imsum.shape, fscale=imsum.fscale)
        b = Image(wcs=imsum.wcs, data=np.ma.filled(bdata, np.nan), shape=imsum.shape, fscale=imsum.fscale)
        r.write('whiter.fits', savemask='nan')
        g.write('whiteg.fits', savemask='nan')
        b.write('whiteb.fits', savemask='nan')

        
        fwcube = np.ones((5, size2, size3)) * fw[:, np.newaxis, np.newaxis]

        try:
            os.mkdir("nb")
        except:
            pass

        f2 = open("nb/dosex", 'w')
        for k in range(2, size1 - 3):
            sys.stdout.write("Narrow band:%d/%d" % (k, size1 - 3) + "\r");
            leftmin = max(0, k - 2 - delta)
            leftmax = k - 2
            rightmin = k + 3
            rightmax = min(size1, k + 3 + delta)
            imslice = np.ma.average(c.data[k - 2:k + 3, :, :], weights=fwcube / c.var[k - 2:k + 3, :, :], axis=0)
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
            imnb = Image(wcs=imsum.wcs, fscale=imsum.fscale, data=np.ma.filled(imslice - contmean, np.nan), shape=imsum.shape)
            kstr = "%04d" % k
            imnb.write('nb/nb' + kstr + '.fits', savemask='nan')
            f2.write(cmd_sex + ' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME nb' + kstr + '.cat nb' + kstr + '.fits\n')

        f2.close()

    if(step <= 2):
        print "STEP 2: runs SExtractor on white light, RGB and narrow-band images"
        # tests here if the files default.sex, default.conv, default.nnw and default.param exist. Otherwise copy them
        if not os.path.isfile('default.sex'):
            shutil.copy(path+'default.sex', 'default.sex')
        if not os.path.isfile('default.conv'):
            shutil.copy(path+'default.conv', 'default.conv')
        if not os.path.isfile('default.nnw'):
            shutil.copy(path+'default.nnw', 'default.nnw')
        if not os.path.isfile('default.param'):
            shutil.copy(path+'default.param', 'default.param')
        
        subprocess.Popen(cmd_sex +' white.fits', shell=True).wait()
        subprocess.Popen(cmd_sex + ' -CATALOG_NAME R.cat -CATALOG_TYPE ASCII_HEAD white.fits,whiter.fits', shell=True).wait()
        subprocess.Popen(cmd_sex + ' -CATALOG_NAME G.cat -CATALOG_TYPE ASCII_HEAD white.fits,whiteg.fits', shell=True).wait()
        subprocess.Popen(cmd_sex + ' -CATALOG_NAME B.cat -CATALOG_TYPE ASCII_HEAD white.fits,whiteb.fits', shell=True).wait()
        
        tB = Table.read('B.cat', format='ascii.sextractor')
        tG = Table.read('G.cat', format='ascii.sextractor')
        tR = Table.read('R.cat', format='ascii.sextractor')
        
        names= ('NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_APER_B', 'MAG_APER_G', 'MAG_APER_R')
        tBGR = Table([tB['NUMBER'], tB['X_IMAGE'], tB['Y_IMAGE'], tB['MAG_APER'], tG['MAG_APER'], tR['MAG_APER']], names=names)
        tBGR.write('BGR.cat', format='ascii.fixed_width_two_line')
        
        os.remove('B.cat')
        os.remove('G.cat')
        os.remove('R.cat')
        
        try:
            os.chdir("nb")
        except:
            print "ERROR: missing nb directory"
            return
        # tests here if the files default.sex, default.conv, default.nnw and default.param exist. Otherwise copy them
        if not os.path.isfile('default.sex'):
            shutil.copy(path+'nb_default.sex', 'default.sex')
        if not os.path.isfile('default.conv'):
            shutil.copy(path+'nb_default.conv', 'default.conv')
        if not os.path.isfile('default.nnw'):
            shutil.copy(path+'nb_default.nnw', 'default.nnw')
        if not os.path.isfile('default.param'):
            shutil.copy(path+'nb_default.param', 'default.param')
        shutil.copy('../inv_variance.fits', 'inv_variance.fits')
        st = os.stat('dosex')
        os.chmod('dosex', st.st_mode | stat.S_IEXEC)
        subprocess.Popen('./dosex', shell=True).wait()
        os.chdir("..")

    if(step <= 3):
        print "STEP 3: merge SExtractor catalogs and measure redshifts"
        wlmin = c.wave.crval
        dw = c.wave.cdelt
        nslices = c.shape[0]
        
        tBGR = Table.read('BGR.cat', format='ascii.fixed_width_two_line')

        maxidc = 0
        # C
        C_ll = []
        C_idmin = []
        C_fline = []
        C_xline = []
        C_yline = []
        # S
        S_ll = []
        S_fline = []
        S_xline = []
        S_yline = []
        for i in range(3, nslices - 14):
            ll = wlmin + dw * i
            slicename = "nb/nb%04d.cat" % i
            t = Table.read(slicename, format='ascii.sextractor')
            for line in t:
                xline = line['X_IMAGE']
                yline = line['Y_IMAGE']
                fline = 10.0 ** (0.4 * (25. - float(line['MAG_APER'])))
                eline = float(line['MAGERR_APER']) * fline * (2.3 / 2.5)

                flag = 0
                distmin = -1
                distlist = (xline - tBGR['X_IMAGE']) ** 2.0 + (yline - tBGR['Y_IMAGE']) ** 2.0
                ksel = np.where(distlist < radius**2.0)
                for j in ksel[0]:
                    if(fline > 5.0 * eline):
                        if((flag <= 0)or(distlist[j] < distmin)):
                            idmin = tBGR['NUMBER'][j]
                            distmin = distlist[j]
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
                    C_xline.append(xline)
                    C_yline.append(yline)
                    if(idmin > maxidc):
                        maxidc = idmin
                if (flag == 0) and (ll < 9300.0):
                    S_ll.append(ll)
                    S_fline.append(fline)
                    S_xline.append(xline)
                    S_yline.append(yline)

        nC = len(C_ll)
        nS = len(S_ll)

        # C2 -> continuum_lines
        C2_id = []
        C2_xline = []
        C2_yline = []
        C2_line = []
     
        flags = np.ones(nC)
        for i in range(nC):
            fl = 0
            for j in range(nC):
                if( (i != j) and (C_idmin[i] == C_idmin[j]) and (np.abs(C_ll[j] - C_ll[i]) < (3.00))):
                    if(C_fline[i] < C_fline[j]):
                        flags[i] = 0
                    fl = 1
            if(fl == 0):  # identification of single line emissions
                flags[i] = 2
 
        for r in range(maxidc + 1):
            lines = []
            for i in range(nC):
                if (C_idmin[i] == r) and (flags[i] == 1):
                    if len(lines) == 0:
                        C2_id.append(r)
                        C2_xline.append(C_xline[i])
                        C2_yline.append(C_yline[i])
                    lines.append((round(C_ll[i],2), round(C_fline[i],4)))
            if len(lines) > 0:
                C2_line.append(lines)
                
             
        # write continuum_lines.cat
        t = catalog(C2_id, C2_xline, C2_yline, C2_line)
        t.write('continuum_lines.cat', format='ascii.fixed_width_two_line')

        #S2
        singflags = np.ones(nS)
        S2_ll = []
        S2_fline = []
        S2_xline = []
        S2_yline = []

        for i in range(nS):
            fl = 0
            xref = S_xline[i]
            yref = S_yline[i]
            ksel = np.where((xref - S_xline) ** 2.0 + (yref - S_yline) ** 2.0 < (radius/2.0)**2.0)  # spatial distance
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
                S2_xline.append(S_xline[i])
                S2_yline.append(S_yline[i])

        # Merging single lines of the same object

        # S3
        S3_xline = []
        S3_yline = []
        S3_line = []
        
        nlines = len(S2_ll)
        flags = np.zeros(nlines)

        for i in range(nlines):
            if(flags[i] == 0):
                lines = []
                S3_xline.append(S2_xline[i])
                S3_yline.append(S2_yline[i])
                lines.append((round(S2_ll[i],2), round(S2_fline[i],4)))
                ksel = np.where(((S2_xline[i] - S2_xline) ** 2.0 + (S2_yline[i] - S2_yline) ** 2.0 < radius**2.0) & (flags == 0))
                for j in ksel[0]:
                    if(j != i):
                        lines.append((round(S2_ll[j],2), round(S2_fline[j],4)))
                        flags[j] = 1
                S3_line.append(lines)
                flags[i] = 1
                
        nS3 = len(S3_xline)
        S3_id = range(1, nS3+1)
        
        t = catalog(S3_id, S3_xline, S3_yline, S3_line)
        t.write('single_lines.cat', format='ascii.fixed_width_two_line')
        
#         # S3
#         S3_xline = []
#         S3_yline = []
#         S3_lline = []
#         S3_fline = []
#         
#         nlines = len(S2_ll)
#         flags = np.zeros(nlines)
# 
#         for i in range(nlines):
#             if(flags[i] == 0):
#                 lbdas = []
#                 fluxes = []
#                 S3_xline.append(S2_xline[i])
#                 S3_yline.append(S2_yline[i])
#                 lbdas.append(S2_ll[i])
#                 fluxes.append(S2_fline[i])
#                 ksel = np.where(((S2_xline[i] - S2_xline) ** 2.0 + (S2_yline[i] - S2_yline) ** 2.0 < radius**2.0) & (flags == 0))
#                 for j in ksel[0]:
#                     if(j != i):
#                         lbdas.append(S2_ll[j])
#                         fluxes.append(S2_fline[j])
#                         flags[j] = 1
#                 S3_lline.append(np.array_str(np.array(lbdas), precision=1))
#                 S3_fline.append(np.array_str(np.array(fluxes), precision=3))
#                 flags[i] = 1
#                 
#         nS3 = len(S3_xline)
#         S3_id = range(1, nS3+1)
#         names= ('NUMBER', 'X_IMAGE', 'Y_IMAGE', 'LAMBDA', 'FLUX')
#         t = Table([S3_id, S3_xline, S3_yline, S3_lline, S3_fline], names=names)
#         t.write('single_lines_2.cat', format='ascii.fixed_width_two_line')    
# 
#     

        ########################################################################################################################
        # redshift of continuum objects

        if not os.path.isfile('emlines'):
            shutil.copy(path+'emlines', 'emlines')
        if not os.path.isfile('emlines_small'):
            shutil.copy(path+'emlines_small', 'emlines_small')
        eml = np.loadtxt("emlines", dtype={'names': ('lambda', 'lname'), 'formats': ('f', 'S20')})
        eml2 = np.loadtxt("emlines_small", dtype={'names': ('lambda', 'lname'), 'formats': ('f', 'S20')})

        fout = open("continuum_lines_z.cat", 'w')
        for i, x, y, line in zip(C2_id, C2_xline, C2_yline, C2_line):
            line = np.array(line).ravel()
            wl = line[::2]
            flux = line[1::2]
            (flag, returnstr) = crackz(len(wl), wl, flux, eml, eml2)
            if(flag > 0):
                fout.write("%d %f %f %s\n" % (i, x, y, returnstr))             
        fout.close()


        fout = open("single_lines_z.cat", 'w')
        for i, x, y, line in zip(S3_id, S3_xline, S3_yline, S3_line):
            line = np.array(line).ravel()
            wl = line[::2]
            flux = line[1::2]
            nlines = len(wl)
            if((nlines > 0) & (nlines < 20)):
                (flag, returnstr) = crackz(nlines, wl, flux, eml, eml2)
                if(flag > 0):
                    fout.write("%d %f %f %s\n" % (i, x, y, returnstr))
        fout.close()
