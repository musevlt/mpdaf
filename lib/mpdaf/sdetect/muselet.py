from astropy.table import Table
import numpy as np
from ..obj import Cube, Image
import os
import sys
import shutil
import subprocess
import stat

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


def muselet(cubename, step=1, delta=20, fw=[0.26, 0.7, 1., 0.7, 0.26]):
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

        maxidc = 0
        S = []
        C = []
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
                distlist = (xline - tB['X_IMAGE']) ** 2.0 + (yline - tB['Y_IMAGE']) ** 2.0
                ksel = np.where(distlist < 16.0)
                for j in ksel[0]:
                    if(fline > 5.0 * eline):
                        if((flag <= 0)or(distlist[j] < distmin)):
                            idmin = tB['NUMBER'][j]
                            distmin = distlist[j]
                            flag = 1
                    else:
                        if(fline < -5 * eline):
                            idmin = tB['NUMBER'][j]
                            distmin = distlist[j]
                            flag = -2
                        else:
                            flag = -1
                if(flag == 1):
                    C.append([ll, idmin, fline, xline, yline])
                    if(idmin > maxidc):
                        maxidc = idmin
                if((flag == 0)and(ll < 9300.0)):
                    S.append([ll, xline, yline, fline])

        k = np.array(C).shape[0]
        l = np.array(S).shape[0]

        fout = open("continuum_nb.lines_clean", 'w')
        n = 0
        x = 0
        flags = np.ones(k)
        for i in range(k):
            fl = 0
            for j in range(k):
                if((i != j)and(C[i][1] == C[j][1])and(np.abs(C[j][0] - C[i][0]) < (3.00))):
                    if(C[i][2] < C[j][2]):
                        flags[i] = 0
                    fl = 1
            if(fl == 0):  # identification des emissions spontanee isolee
                flags[i] = 2
 
        C2 = []
        column = []
        for r in range(maxidc + 1):
            firstline = 1
            for i in range(k):
                if((C[i][1] == r)and(flags[i] == 1)):
                    if(firstline == 1):
                        C2.append([r, C[i][3], C[i][4]]) #id, xpix, ypix
                        firstline = 0
                        fout.write("%d %f %f " % (r, C[i][3], C[i][4]))
                    C2[n].append(C[i][0])
                    C2[n].append(C[i][2])
                    fout.write("%f %f " % (C[i][0], C[i][2])) #lline, fline
                    x = x + 1  # index on the number of emission lines
            if(firstline == 0):
                column.append(x)
                n = n + 1
                x = 0
                fout.write("\n")
        fout.close()

#         lid = []
#         xpix = []
#         ypix = []
#         lline = []
#         fline = []
#     
#         flags = np.ones(k)
#         for i in range(k):
#             fl = 0
#             for j in range(k):
#                 if((i != j)and(C[i][1] == C[j][1])and(np.abs(C[j][0] - C[i][0]) < (3.00))):
#                     if(C[i][2] < C[j][2]):
#                         flags[i] = 0
#                     fl = 1
#             if(fl == 0):  # identification des emissions spontanee isolee
#                 flags[i] = 2
# 
#         for r in range(maxidc + 1):
#             lbdas = []
#             fluxes = []
#             for i in range(k):
#                 if((C[i][1] == r)and(flags[i] == 1)):
#                     if len(lbdas) == 0:
#                         lid.append(r)
#                         xpix.append(C[i][3])
#                         ypix.append(C[i][4])
#                     lbdas.append(C[i][0])
#                     fluxes.append(C[i][2])
#             lline.append(lbdas)
#             fline.append(fluxes)
#             
#         #write continuum_nb.lines_clean
#         print len(lid)
#         print len(lline)
#         t = Table([lid, xpix, ypix, lline, fline], names=('NUMBER', 'X_IMAGE', 'Y_IMAGE', 'LAMBDA', 'FLUX'))
#         t.write('continuum_nb.lines_clean.dat', format='ascii')

        p = 0
        nlines = l
        singflags = np.ones(nlines)
        S2 = []
        S = np.array(S)

        for i in range(nlines):
            fl = 0
            xref = S[i][1]
            yref = S[i][2]
            ksel = np.where((xref - S[:, 1]) ** 2.0 + (yref - S[:, 2]) ** 2.0 < 4.0)  # spatial distance
            for j in ksel[0]:
                if((i != j)and(np.abs(S[j, 0] - S[i, 0]) < 2.50)):
                    if(S[i, 3] < S[j, 3]):
                        singflags[i] = 0
                    fl = 1
            if(fl == 0):
                singflags[i] = 2
            if(singflags[i] == 1):
                S2.append(S[i, 0:4])
                p = p + 1

        # Merging single lines of the same object
        fout = open("single_nb.lines_clean", 'w')

        nlines = p
        pp = 0
        x = 0

        flags = np.zeros(nlines)

        S3 = []
        column = []

        S2 = np.array(S2)

        for i in range(nlines):
            if(flags[i] == 0):
                S3.append([S2[i][1], S2[i][2], S2[i][0], S2[i][3]])
                column.append(1)
                nobj = pp + 1
                fout.write("%d %f %f %f %f " % (nobj, S3[pp][0], S3[pp][1], S3[pp][2], S3[pp][3]))
                ksel = np.where(((S2[i, 1] - S2[:, 1]) ** 2.0 + (S2[i, 2] - S2[:, 2]) ** 2.0 < 16) & (flags == 0))
                for j in ksel[0]:
                    if(j != i):
                        x = x + 1
                        S3[pp].append(S2[j, 0])
                        S3[pp].append(S2[j, 3])
                        column[pp] = column[pp] + 1
                        fout.write("%f %f " % (S3[pp][2 + 2 * x], S3[pp][3 + 2 * x]))
                        flags[j] = 1
                x = 0
                pp = pp + 1
                fout.write("\n")
                flags[i] = 1

        fout.close()

        ########################################################################################################################
        # redshift of continuum objects

        if not os.path.isfile('emlines'):
            shutil.copy(path+'emlines', 'emlines')
        if not os.path.isfile('emlines_small'):
            shutil.copy(path+'emlines_small', 'emlines_small')
        eml = np.loadtxt("emlines", dtype={'names': ('lambda', 'lname'), 'formats': ('f', 'S20')})
        eml2 = np.loadtxt("emlines_small", dtype={'names': ('lambda', 'lname'), 'formats': ('f', 'S20')})
#        nem = np.size(eml['lambda'])

        fout = open("continuum_nb.lines_clean2", 'w')

        with open("continuum_nb.lines_clean", 'r') as fin:
            for line in fin.readlines():
                line = line.strip()
                Fld = line.split()
                nlines = (np.size(Fld) - 3) / 2
                if(nlines > 0):
                    objid = int(float(Fld[0]))
                    x = float(Fld[1])
                    y = float(Fld[2])
                    wl = np.array([float(i) for i in Fld[3::2]])
                    flux = np.array([float(i) for i in Fld[4::2]])
                    (flag, returnstr) = crackz(nlines, wl, flux, eml, eml2)
                    if(flag > 0):
                        fout.write("%d %f %f %s\n" % (objid, x, y, returnstr))
        fin.close()
        fout.close()

        fout = open("single_nb.lines_clean2", 'w')

        with open("single_nb.lines_clean", 'r') as fin:
            for line in fin.readlines():
                line = line.strip()
                Fld = line.split()
                objid = int(float(Fld[0]))
                nlines = (np.size(Fld) - 3) / 2
                if((nlines > 0) & (nlines < 20)):
                    x = float(Fld[1])
                    y = float(Fld[2])
                    wl = np.array([float(i) for i in Fld[3::2]])
                    flux = np.array([float(i) for i in Fld[4::2]])
                    (flag, returnstr) = crackz(nlines, wl, flux, eml, eml2)
                    if(flag > 0):
                        fout.write("%d %f %f %s\n" % (objid, x, y, returnstr))
        fin.close()
