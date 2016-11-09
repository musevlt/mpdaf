#include "tools.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NIFUS 24
#define NSLICES 48
#define MIN_PTS_PER_SLICE 100
#define MAPIDX(i, s, q) (NIFUS*NSLICES*(q-1) + NSLICES*(i-1) + s - 1)
/* #define MAPIDX(i, s, q) (NQUAD*NSLICES*(i-1) + NQUAD*(s-1) + q - 1) */
/* #define IFUIDX(k) (k / (NQUAD*NSLICES) + 1) */
/* #define SLIIDX(k) ((k % (NQUAD*NSLICES)) / NQUAD + 1) */
/* #define QUADIDX(k) ((k % (NQUAD*NSLICES)) % NQUAD + 1) */


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


double mpdaf_slice_mean(double* data, int* xpix, int n, int* indx)
{
    int i, j, count, meancount=0, minmax[2];
    double x[3];

    mpdaf_minmax_int(xpix, n, indx, minmax);

    int nx = (minmax[1] - minmax[0] + 1);
    int *ind = (int*) malloc(n*sizeof(int));
    double *meanarr = (double*) malloc(nx*sizeof(double));

    for (i=minmax[0]; i <= minmax[1]; i++) {
        count = 0;
        for (j=0; j<n; j++) {
            if (xpix[indx[j]] == i)
                ind[count++] = indx[j];
        }
        if (count>50) {
            mpdaf_mean(data, count, x, ind);
            meanarr[meancount] = x[0];
            meancount++;
        }
    }
    for (j=0; j<meancount; j++)
        ind[j] = j;
    mpdaf_mean_sigma_clip(meanarr, meancount, x, 15, 3, 3, 15, ind);
    free(ind);
    free(meanarr);
    return x[0];
}

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
        int* xpix,
        int lbdabins_n,
        int *lbdabins
) {
    size_t nlbin = (size_t)lbdabins_n;
    size_t i, k, n, s, q, slidx;
    double tot_flux, x[3], minmax[2];
    int slice_count, tot_count;

    double *slice_flux = (double*) malloc(NSLICES*NIFUS*sizeof(double));
    double *ifu_flux = (double*) malloc(NIFUS*sizeof(double));
    int *slice_ind = (int*) malloc(NSLICES*sizeof(int));
    int *tot_ind = (int*) malloc(NIFUS*NSLICES*sizeof(int));
    int* quad = (int*) malloc(npix*sizeof(int));
    int *indmap[nlbin * NIFUS * NSLICES];

    // Clipping parameters
    int nmax=15, nstop=20;
    double nclip_low=3.0, nclip_up=3.0;

    for (k=0; k<NIFUS*NSLICES*nlbin; k++) {
        npts[k] = 0;
        corr[k] = 1.0;
        indmap[k] = (int*) malloc(npix/(NIFUS*NSLICES) * sizeof(int));
    }

    printf("Using %zu lambda slices\n", nlbin);
    printf("Computing lambda indexes ...\n");
    #pragma omp parallel for private(k,q)
    for (n = 0; n < (size_t)npix; n++) {
        for (q = 1; q < nlbin+1; q++) {
            if ((lbda[n] >= lbdabins[q-1]) && (lbda[n] < lbdabins[q])) {
                quad[n] = q;
                break;
            }
        }
        if (mask[n] == 0) {
            k = MAPIDX(ifu[n], sli[n], quad[n]);
            indmap[k][npts[k]++] = n;
        }
    }

    for (q = 0; q < nlbin; q++) {
        printf("\n\nLambda bin: %d - %d\n", lbdabins[q], lbdabins[q+1]);
        printf("Computing reference levels ...\n\n");
        tot_count = 0;
        for (i = 0; i < NIFUS; i++) {
            printf("- IFU %02zu\n", i+1);
            slice_count = 0;
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                slidx = NSLICES*i + s;
                if (npts[k] > 100) {
                    slice_flux[slidx] = mpdaf_slice_mean(data, xpix, npts[k], indmap[k]);
                    slice_ind[slice_count++] = slidx;
                    tot_ind[tot_count++] = slidx;
                } else {
                    slice_flux[slidx] = NAN;
                }
                /* printf("  - SLICE %02zu : %f (%d)\n", s+1, slice_flux[slidx], npts[k]); */
            }
            /* ifu_flux[i] = mpdaf_median(slice_flux, slice_count, slice_ind); */
            /* printf("  - Median flux : %f (%d pts)\n", ifu_flux[i], slice_count); */

            mpdaf_minmax(slice_flux, slice_count, slice_ind, minmax);
            printf("  - Min max : %f %f\n", minmax[0], minmax[1]);

            mpdaf_mean_sigma_clip(slice_flux, slice_count, x, nmax, nclip_low,
                    nclip_up, nstop, slice_ind);
            printf("  - Mean flux : %f (%f, %d)\n", x[0], x[1], (int)x[2]);
            ifu_flux[i] = x[0];
            if (isnan(x[0])) {
                printf("rrrrhhhhhhaaaaaaaaaaaaaaaaaa !!!!\n");
            }

            // Use mean ifu flux for slices without useful values
            for (s = 0; s < NSLICES; s++) {
                slidx = NSLICES*i + s;
                if (isnan(slice_flux[slidx]))
                    slice_flux[slidx] = ifu_flux[i];
            }
        }
        mpdaf_minmax(slice_flux, tot_count, tot_ind, minmax);
        printf("\n- Min max : %f %f\n", minmax[0], minmax[1]);

        /* tot_flux = mpdaf_median(slice_flux, tot_count, tot_ind); */
        /* printf("- Total flux : %f (%d pts)\n", tot_flux, tot_count); */

        mpdaf_mean_sigma_clip(slice_flux, tot_count, x, nmax, nclip_low,
                nclip_up, nstop, tot_ind);
        printf("- Total flux (clipped) : %f (%f, %d)\n", x[0], x[1], (int)x[2]);
        tot_flux = x[0];

        printf("\nComputing corrections ...\n\n");

        for (i = 0; i < NIFUS; i++) {
            printf("- IFU %02zu\n", i+1);
            slice_count = 0;
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                slidx = NSLICES*i + s;
                if (isnan(slice_flux[slidx])) {
                    printf("nnnnnoooooooooooooooooooooooooo\n");
                } else if (npts[k] == 0) {
                    printf("oioioioioioioioi\n");
                }
                slice_ind[slice_count++] = slidx;
                /* corr[k] = tot_flux / ifu_flux[i]; */
                corr[k] = tot_flux / slice_flux[slidx];
                /* printf("  - SLICE %02zu : %f\n", s+1, corr[k]); */
            }
            mpdaf_minmax(corr, slice_count, slice_ind, minmax);
            printf("  - Min max : %f %f\n", minmax[0], minmax[1]);

            mpdaf_mean_sigma_clip(corr, slice_count, x, nmax, nclip_low,
                    nclip_up, nstop, slice_ind);
            printf("  - Mean correction (clipped) : %f (%f, %d)\n",
                   x[0], x[1], (int)x[2]);

            printf("  - Checking slice corrections ...\n");
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                if (fabs(corr[k] - x[0]) > 3*x[1]) {
                    printf("    - SLICE %02zu : %f\n", s+1, corr[k]);
                    corr[k] = tot_flux / ifu_flux[i];
                    printf("      Using IFU Mean : %f\n", corr[k]);
                }
            }
        }

        /* if (q == nlbin) { */
        /*     corr[k] = corr[k-1]; */
        /*     printf("IFU %02zu SLICE %02zu QUAD %02zu : %f \n", i, s, q, corr[k]); */
        /*     continue; */
        /* } */
    }

    for (k=0; k<NIFUS*NSLICES*nlbin; k++)
        free(indmap[k]);

    printf("\nApply corrections ...\n");
    #pragma omp parallel for private(k)
    for (n=0; n < (size_t)npix; n++) {
        k = MAPIDX(ifu[n], sli[n], quad[n]);
        result[n] =  data[n] * corr[k];
    }
    printf("\nOK, done.\n");

    free(slice_flux);
    free(ifu_flux);
    free(slice_ind);
    free(tot_ind);
    free(quad);
}
