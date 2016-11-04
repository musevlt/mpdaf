#include "tools.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NQUAD 14
#define NIFUS 24
#define NSLICES 48
#define MIN_PTS_PER_SLICE 100
#define MAPIDX(i, s, q) (NQUAD*(i-1)*NSLICES + NQUAD*(s-1) + q - 1)
#define IFUIDX(k) (k / (NQUAD*NSLICES) + 1)
#define SLIIDX(k) ((k % (NQUAD*NSLICES)) / NQUAD + 1)
#define QUADIDX(k) ((k % (NQUAD*NSLICES)) % NQUAD + 1)


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


void mpdaf_slice_median(
        double* result,
        double* corr,
        int* npts,
        int* ifu,
        int* sli,
        double* data,
        double* lbda,
        int npix,
        int* mask
) {
    size_t i, k, n, s, q, slidx;
    double tot_flux, x[3], minmax[2];
    int slice_count, tot_count;

    double *slice_flux = (double*) malloc(NSLICES*NIFUS*sizeof(double));
    double *ifu_flux = (double*) malloc(NIFUS*sizeof(double));
    int *slice_ind = (int*) malloc(NSLICES*sizeof(int));
    int *tot_ind = (int*) malloc(NIFUS*NSLICES*sizeof(int));
    int* quad = (int*) malloc(npix*sizeof(int));
    int *indmap[NQUAD * NIFUS * NSLICES];

    int skyseg[] = {0, 5100, 5400, 5800, 6120, 6440, 6760, 7200, 7450, 7700,
                    8170, 8565, 8731, 9275, 10000};

    // Clipping parameters
    int nmax=15, nstop=20;
    double nclip_low=3.0, nclip_up=3.0;

    printf("Using %d lambda slices\n", NQUAD);
    printf("Computing lambda indexes ...\n");
    #pragma omp parallel for private(i,q)
    for (i = 0; i < (size_t)npix; i++) {
        for (q = 1; q < NQUAD+1; q++) {
            if ((lbda[i] >= skyseg[q-1]) && (lbda[i] < skyseg[q])) {
                quad[i] = q;
                break;
            }
        }
    }

    for (k=0; k<NIFUS*NSLICES*NQUAD; k++) {
        npts[k] = 0;
        corr[k] = 1.0;
        indmap[k] = (int*) malloc(npix/(NIFUS*NSLICES) * sizeof(int));
    }

    for (n=0; n < (size_t)npix; n++) {
        if (mask[n]==0) {
            k = MAPIDX(ifu[n], sli[n], quad[n]);
            indmap[k][npts[k]++] = n;
        }
    }

    /* #pragma omp parallel for private(k,sky_count,skyind) */
    for (q = 0; q < NQUAD; q++) {
        printf("\n\nLambda bin: %d - %d\n", skyseg[q], skyseg[q+1]);
        printf("Computing reference levels ...\n");
        tot_count = 0;
        for (i = 0; i < NIFUS; i++) {
            printf("\n- IFU %02zu\n", i+1);
            slice_count = 0;
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                slidx = NSLICES*i + s;
                mpdaf_mean(data, npts[k], x, indmap[k]);
                /* med = mpdaf_median(data, npts[k], indmap[k]); */
                slice_flux[slidx] = x[0];
                if (npts[k] > 0) {
                    slice_ind[slice_count++] = slidx;
                    tot_ind[tot_count++] = slidx;
                }
                /* printf("  - SLICE %02zu : %f (%f, %d)\n", s+1, x[0], x[1], npts[k]); */
            }
            ifu_flux[i] = mpdaf_median(slice_flux, slice_count, slice_ind);
            printf("  - Median flux : %f (%d pts)\n", ifu_flux[i], slice_count);

            mpdaf_mean_sigma_clip(slice_flux, slice_count, x, nmax, nclip_low,
                    nclip_up, nstop, slice_ind);
            printf("  - Mean flux (clipped) : %f (%f, %d)\n", x[0], x[1], (int)x[2]);
            ifu_flux[i] = x[0];

            mpdaf_minmax(slice_flux, slice_count, slice_ind, minmax);
            printf("  - Min max : %f %f\n", minmax[0], minmax[1]);
        }
        tot_flux = mpdaf_median(slice_flux, tot_count, tot_ind);
        printf("\n- Total flux : %f (%d pts)\n", tot_flux, tot_count);

        mpdaf_mean_sigma_clip(slice_flux, tot_count, x, nmax, nclip_low,
                nclip_up, nstop, tot_ind);
        printf("- Total flux (clipped) : %f (%f, %d)\n", x[0], x[1], (int)x[2]);
        tot_flux = x[0];

        mpdaf_minmax(slice_flux, tot_count, tot_ind, minmax);
        printf("- Min max : %f %f\n", minmax[0], minmax[1]);

        printf("\nComputing corrections ...\n");

        for (i = 0; i < NIFUS; i++) {
            printf("\n- IFU %02zu\n", i+1);
            slice_count = 0;
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                slidx = NSLICES*i + s;
                if (npts[k] > 0) {
                    slice_ind[slice_count++] = slidx;

                    /* corr[k] = tot_flux / ifu_flux[i]; */
                    if (npts[k] > MIN_PTS_PER_SLICE) {
                        corr[k] = tot_flux / slice_flux[slidx];
                        /* printf("  - SLICE %02zu : %f\n", s+1, corr[k]); */
                    } else {
                        corr[k] = ifu_flux[i] / slice_flux[slidx];
                        /* printf("  - SLICE %02zu : %f (using IFU mean)\n", s+1, corr[k]); */
                    }
                }
            }
            mpdaf_mean_sigma_clip(corr, slice_count, x, nmax, nclip_low,
                    nclip_up, nstop, slice_ind);
            printf("  - Mean correction (clipped) : %f (%f, %d)\n", x[0], x[1], (int)x[2]);

            mpdaf_minmax(corr, slice_count, slice_ind, minmax);
            printf("  - Min max : %f %f\n", minmax[0], minmax[1]);
        }

        /* if (q == NQUAD) { */
        /*     corr[k] = corr[k-1]; */
        /*     printf("IFU %02zu SLICE %02zu QUAD %02zu : %f \n", i, s, q, corr[k]); */
        /*     continue; */
        /* } */
    }

    for (k=0; k<NIFUS*NSLICES*NQUAD; k++)
        free(indmap[k]);

    printf("Apply corrections ...\n");
    #pragma omp parallel for private(k)
    for (n=0; n < (size_t)npix; n++) {
        k = MAPIDX(ifu[n], sli[n], quad[n]);
        result[n] =  data[n] * corr[k];
    }
    printf("\nOK, done.\n");

    free(quad);
    free(tot_ind);
    free(slice_ind);
    free(slice_flux);
    free(ifu_flux);
}
