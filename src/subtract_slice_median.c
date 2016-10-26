#include "tools.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct Pix {
    int ifu;
    int sli;
    double data;
    int quad;
} Pix;

// typ=0 -> mpdaf_divide_slice_median
// typ=1 -> mpdaf_subtract_slice_median
int initPix(Pix * array, int npix, int* ifu, int* sli, double* data, double* lbda, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n, int* quad, int typ) {
    int i, count=0;
    if (typ==0)
    {
        for (i = 0; i < npix; i++)
        {
            if ((mask[i] == 0) && (lbda[i] >= 4800) && (lbda[i] <= 9300))
            {
                array[count].ifu = ifu[i];
                array[count].sli = sli[i];
                array[count].quad = quad[i];
                array[count].data = data[i] / mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]);
                count = count + 1;
            }
        }
    }
    else
    {
        for (i = 0; i < npix; i++)
        {
            if ((mask[i] == 0) && (lbda[i] >= 4800) && (lbda[i] <= 9300))
            {
                array[count].ifu = ifu[i];
                array[count].sli = sli[i];
                array[count].quad = quad[i];
                array[count].data = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]) - data[i];
                count = count + 1;
            }
        }
    }
    return count;
}


int comparePix(const void * elem1, const void * elem2) {
    Pix * i1, *i2;
    i1 = (Pix*)elem1;
    i2 = (Pix*)elem2;
    if (i1->ifu == i2->ifu)
    {
        if (i1->sli == i2->sli)
        {
            if (i1->quad == i2->quad)
            {
                if (i1->data == i2->data)
                    return 0;
                else
                    return (i1->data > i2->data) ? 1 : -1;
            }
            else
                return (i1->quad > i2->quad) ? 1 : -1;
        }
        else
            return (i1->sli > i2->sli) ? 1 : -1;
    }
    else
        return (i1->ifu > i2->ifu) ? 1 : -1;
}


void sortPix(Pix * array, int npix) {
    qsort((void *)array, npix, sizeof(Pix), comparePix);
}


void mpdaf_sky_ref(double* data, double* lbda, int* mask, int npix, double lmin, double dl, int n, int nmax, double nclip_low, double nclip_up, int nstop, double* result)
{
    double x[3];
    int i, ii, l, count=0;
    double l0, l1;
    int* indx = (int*) malloc(npix * sizeof(int));
    int* work = (int *) malloc(npix * sizeof(int));

    for (i=0; i<npix; i++)
        indx[i] = i;
    indexx(npix, lbda, indx);

    l = 0;
    l0 = lmin;
    l1 = lmin + dl;

    for (i=0; i<npix; i++)
    {
        ii = indx[i];
        if ((lbda[ii] >= l0) && (lbda[ii] < l1)) {
            if (mask[ii]==0) {
                work[count++] = ii;
            }
        } else {
            // Compute median for the current bin
            mpdaf_median_sigma_clip(data, count, x, nmax, nclip_low, nclip_up, nstop, work);
            // printf("Median bin %04d : %f, %f, %f\n", l, x[0], x[1], x[2]);
            result[l] = x[0];

            // New bin
            count = 0;
            if (++l >= n) {
                printf("Not enough wavelength bins for the lbda array");
                printf("Stopping at %f, bin %d/%d", lbda[ii], l, n);
                break;
            }
            l0 += dl;
            l1 += dl;
            if (mask[ii]==0) {
                work[count++] = ii;
            }
        }
    }
    // Last bin
    if (count > 0) {
        mpdaf_median_sigma_clip(data, count, x, nmax, nclip_low, nclip_up, nstop, work);
        result[l] = x[0];
    }

    free(work);
    free(indx);
}


void mpdaf_sky_ref_indx(double* data, double* lbda, int* mask, int npix, double lmin, double dl, int n, int nmax, double nclip_low, double nclip_up, int nstop, double* result, int* indx, int* bincount)
{
    double x[3];
    int i, ii, l, count=0;
    double l0, l1;
    int* work = (int *) malloc(npix * sizeof(int));

    indexx(npix, lbda, indx);

    l = 0;
    l0 = lmin;
    l1 = lmin + dl;

    for (i=0; i<npix; i++)
    {
        ii = indx[i];
        if ((lbda[ii] >= l0) && (lbda[ii] < l1)) {
            if (mask[ii]==0) {
                work[count++] = ii;
            }
        } else {
            // Compute median for the current bin
            mpdaf_median_sigma_clip(data, count, x, nmax, nclip_low, nclip_up, nstop,work);
            // printf("Median bin %04d : %f, %f, %f\n", l, x[0], x[1], x[2]);
            result[l] = x[0];
            bincount[l] = x[2];

            // New bin
            count = 0;
            if (++l >= n) {
                printf("Not enough wavelength bins for the lbda array");
                printf("Stopping at %f, bin %d/%d", lbda[ii], l, n);
                break;
            }
            l0 += dl;
            l1 += dl;
            if (mask[ii]==0) {
                work[count++] = ii;
            }
        }
    }
    // Last bin
    if (count > 0) {
        mpdaf_median_sigma_clip(data, count, x, nmax, nclip_low, nclip_up, nstop, work);
        result[l] = x[0];
    }

    free(work);
}

void compute_quad(int* xpix, int* ypix, int* quad, int npix) {
    int i = 0;
    #pragma omp parallel for
    for (i = 0; i < npix; i++) {
        if (xpix[i] < 2048) {
            if (ypix[i] < 2056)
                quad[i] = 1;
            else
                quad[i] = 2;
        } else {
            if (ypix[i] < 2056)
                quad[i] = 4;
            else
                quad[i] = 3;
        }
    }
}

#define NQUAD 10
#define NIFUS 24
#define NSLICES 48
#define MIN_PTS_PER_SLICE 100
#define MAPIDX(i, s, q) (int)(NQUAD*(i-1)*NSLICES + NQUAD*(s-1) + q - 1)
#define IFUIDX(k) (k / (NQUAD*NSLICES) + 1)
#define SLIIDX(k) ((k % (NQUAD*NSLICES)) / NQUAD + 1)
#define QUADIDX(k) ((k % (NQUAD*NSLICES)) % NQUAD + 1)

void mpdaf_slice_median(
        double* result,
        double* corr,
        int* npts,
        int* ifu,
        int* sli,
        double* data,
        double* lbda,
        int npix,
        int* mask,
        double* skyref_flux,
        double* skyref_lbda,
        int skyref_n,
        int* xpix,
        int* ypix
) {
    size_t i, j, k, n, s, q;
    int index, sky_count;
    double x[3];

    double *slice_sky = (double*) malloc(skyref_n*sizeof(double));
    double *slice_diff = (double*) malloc(skyref_n*sizeof(double));
    int *indx = (int*) malloc(skyref_n*sizeof(int));
    int *bincount = (int*) malloc(skyref_n*sizeof(int));
    int* quad = (int*) malloc(npix*sizeof(int));
    int *indmap[NQUAD * NIFUS * NSLICES];

    double lmin = skyref_lbda[0];
    double dl = skyref_lbda[1] - skyref_lbda[0];
    printf("- Using lmin=%f, dlbda=%f ...\n", lmin, dl);

    int skyseg[] = {0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8731, 9275, 10000};

    // Clipping parameters
    int nmax=5, nstop=2;
    double nclip_low=3.0, nclip_up=3.0;

    printf("- Using %d lambda slices\n", NQUAD);
    printf("- Compute lambda indexes ... ");
    #pragma omp parallel for private(q)
    for (i = 0; i < (size_t)npix; i++) {
        for (q = 1; q < NQUAD+1; q++) {
            if ((lbda[i] >= skyseg[q-1]) && (lbda[i] < skyseg[q])) {
                quad[i] = q;
                break;
            }
        }
    }
    printf("OK\n\n");

    for (k=0; k<NIFUS*NSLICES*NQUAD; k++) {
        npts[k] = 0;
        corr[k] = 0.0;
        indmap[k] = (int*) malloc(npix/(NIFUS*NSLICES) * sizeof(int));
    }

    for (n=0; n < (size_t)npix; n++) {
        index = MAPIDX(ifu[n], sli[n], quad[n]);
        indmap[index][npts[index]++] = n;
    }

    //#pragma omp parallel for private(bincount,indx,slice_sky,x)
    for (k=0; k<NIFUS*NSLICES*NQUAD; k++) {
        /* k = MAPIDX(i, s, q); */
        i = IFUIDX(k);
        s = SLIIDX(k);
        q = QUADIDX(k);
        if ((s == 1) && (q == 1))
            printf("=================================\n");
        if (q == 1)
            printf("\n");
        /* printf("\n- IFU %02zu SLICE %02zu QUAD %02zu\n", i, s, q); */

        if (q == NQUAD) {
            corr[k] = corr[k-1];
            printf("IFU %02zu SLICE %02zu QUAD %02zu : %f \n", i, s, q, corr[k]);
            continue;
        }
        if (npts[k] > MIN_PTS_PER_SLICE) {
            for (j=0; j < (size_t)skyref_n; j++) {
                bincount[j] = 0;
                slice_sky[j] = 0.0;
                slice_diff[j] = 0.0;
            }

            mpdaf_sky_ref_indx(data, lbda, mask, npts[k], lmin, dl,
                               skyref_n, nmax, nclip_low, nclip_up, nstop,
                               slice_sky, indmap[k], bincount);

            sky_count = 0;
            for (j=0; j < (size_t)skyref_n; j++) {
                if (bincount[j] > 50) {
                    indx[sky_count++] = j;
                    slice_diff[j] = slice_sky[j] - skyref_flux[j];
                }
            }

            if (sky_count > 0) {
                corr[k] = mpdaf_median(slice_diff, sky_count, indx);
                mpdaf_mean_sigma_clip(slice_diff, sky_count, x, 5, 3.0, 3.0, nstop, indx);
                printf("IFU %02zu SLICE %02zu QUAD %02zu : %f (%f, %.2f, %d)\n",
                       i, s, q, corr[k], x[0], x[1], (int)x[2]);
                /* corr[k] = x[0]; */
            }
        }
    }
            /* corr[MAPIDX(i, s, 10)] = corr[MAPIDX(i, s, 9)]; */
        /* } */
        /* printf("\n"); */
    /* } */

    for (k=0; k<NIFUS*NSLICES*NQUAD; k++)
        free(indmap[k]);

    printf("Apply corrections ...\n");
    #pragma omp parallel for private(index)
    for (n=0; n < (size_t)npix; n++) {
        index = MAPIDX(ifu[n], sli[n], quad[n]);
        result[n] =  data[n] - corr[index];
    }

    free(bincount);
    free(indx);
    free(quad);
    free(slice_sky);
    free(slice_diff);
}

// typ=0 -> mpdaf_divide_slice_median
// typ=1 -> mpdaf_subtract_slice_median
void mpdaf_slice_median2(double* result, double* result_stat, double* corr, int* npts, int* ifu, int* sli, double* data,  double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n, int* xpix, int* ypix, int typ)
{
    Pix* pixels = (Pix*) malloc(npix * sizeof(Pix));
    int npix2;
    int* quad = (int*) malloc(npix*sizeof(int));
    compute_quad(xpix, ypix, quad, npix);
    npix2 = initPix(pixels, npix, ifu, sli, data, lbda, mask, skyref_flux,
                    skyref_lbda, skyref_n, quad, typ);
    sortPix(pixels, npix2);

    int i=0, n, index;
    int q=pixels[0].quad;
    int chan=pixels[0].ifu;
    int sl=pixels[0].sli;
    int imin=0, imax=-1;
    int num, imed;

    for(i=0;i<npix2;i++)
    {
        if ((pixels[i].quad == q) && (pixels[i].sli == sl) && (pixels[i].ifu==chan))
            imax = imax +1;
        else
        {
            if(imin!=imax)
            {
                index = 4*48*(chan-1)+4*(sl-1)+q-1;
                num = imax-imin+1;
                imed = imin + (int)(num/2);
                if (num%2 == 0)
                    corr[index] = (pixels[imed].data + pixels[imed-1].data)/2;
                else
                    corr[index] = pixels[imed].data;
                npts[index] = num;
                //printf("chan=%d sl=%d q=%d n=%d d=%0.2f\n",chan, sl, q, npts[index], corr[index]);
            }
            q=pixels[i].quad;
            chan=pixels[i].ifu;
            sl = pixels[i].sli;
            imin = i;
            imax = i;
        }
    }
    if(imin!=imax)
    {
        index = 4*48*(chan-1)+4*(sl-1)+q-1;
        num = imax-imin+1;
        imed = imin + (int)(num/2);
        if (num%2 == 0)
            corr[index] = (pixels[imed].data + pixels[imed-1].data)/2;
        else
            corr[index] = pixels[imed].data;
        npts[index] = num;  // a reprendre avec quad et a initialiser a 0
        //printf("chan=%d sl=%d q=%d n=%d d=%0.2f\n",chan, sl, q, npts[index], corr[index]);
    }

    free(pixels);

    if(typ==0)
    {

        #pragma omp parallel shared(result, data, corr, ifu, sli, result_stat, npix, quad) private(index)
        {
            #pragma omp for
            for(n=0; n<npix; n++)
            {
                index = 4*48*(ifu[n]-1)+4*(sli[n]-1)+quad[n]-1;
                result[n] =  data[n] / corr[index];
                result_stat[n] =  result_stat[n] / corr[index] / corr[index];
            }
        }
    }
    else
    {
        #pragma omp parallel shared(result, data, corr, ifu, sli, npix, quad) private(index)
        {
            #pragma omp for
            for(n=0; n<npix; n++)
            {
                index = 4*48*(ifu[n]-1)+4*(sli[n]-1)+quad[n]-1;
                result[n] =  data[n] + corr[index];
            }
        }
    }

    free(quad);
}
