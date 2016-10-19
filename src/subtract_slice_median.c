#include "tools.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
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


void mpdaf_sky_ref_indx(double* data, double* lbda, int* mask, int npix, double lmin, double dl, int n, int nmax, double nclip_low, double nclip_up, int nstop, double* result, int* indx)
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

#define NIFUS 24
#define NSLICES 48
#define MIN_PTS_PER_SLICE 100
#define MAPIDX(i, s) (int)((i-1)*NSLICES + (s-1))
/* index = 4*48*(chan-1)+4*(sl-1)+q-1; */

void mpdaf_slice_median(double* result, double* result_stat, double* corr, int* npts, int* ifu, int* sli, double* data,  double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n, int* xpix, int* ypix, int typ)
{
    int index;
    int *indmap[NIFUS * NSLICES];
    /* double x[3]; */

    double *slice_sky = (double*) malloc(skyref_n*sizeof(double));
    int *indx = (int*) malloc(skyref_n*sizeof(int));

    double lmin = skyref_lbda[0];
    double dl = skyref_lbda[1] - skyref_lbda[0];

    int nmax=2, nstop=2;
    double nclip_low=5.0, nclip_up=5.0;

    for (size_t k=0; k<NIFUS*NSLICES; k++) {
        npts[k] = 0;
        indmap[k] = (int*) malloc(npix/NIFUS * sizeof(int));
    }

    for (size_t n=0; n < (size_t)npix; n++) {
        index = MAPIDX(ifu[n], sli[n]);
        indmap[index][npts[index]++] = n;
    }

    for (size_t k=0; k<NIFUS*NSLICES; k++) {
        if (npts[k] > MIN_PTS_PER_SLICE) {
            mpdaf_sky_ref_indx(data, lbda, mask, npts[k], lmin, dl, skyref_n,
                    nmax, nclip_low, nclip_up, nstop, slice_sky, indmap[k]);

            for (size_t j=0; j < (size_t)skyref_n; j++) {
                indx[j] = j;
                slice_sky[j] -= skyref_flux[j];
            }
            corr[4*k] = mpdaf_median(slice_sky, skyref_n, indx);
            /* mpdaf_median_sigma_clip(slice_sky, skyref_n, x, nmax, */
            /*                         nclip_low, nclip_up, nstop, indx); */
            /* corr[4*k] = x[0]; */
        } else {
            corr[4*k] = 0.0;
        }
    }

    for (size_t k=0; k<NIFUS*NSLICES; k++)
        free(indmap[k]);

    #pragma omp parallel shared(result, data, corr, ifu, sli, result_stat, npix) private(index)
    {
        #pragma omp for
        for (size_t n=0; n < (size_t)npix; n++)
        {
            index = MAPIDX(ifu[n], sli[n]);
            result[n] =  data[n] - corr[4*index];
        }
    }
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
