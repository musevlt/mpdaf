#include "tools.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_PTS_PER_SLICE 2e5
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


void mpdaf_slice_mean(double* data, int* xpix, int n, double x[3], int* indx,
                      int min_pix)
{
    int i, j, count, meancount=0, minmax[2];

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
    if (meancount < min_pix) {
        x[0] = 0;
        x[1] = 0;
        x[2] = 0;
    } else {
        for (j=0; j<meancount; j++)
            ind[j] = j;
        mpdaf_mean_madsigma_clip(meanarr, meancount, x, 15, 3, 3, 15, ind);
    }

    free(ind);
    free(meanarr);
}

void mpdaf_slice_median(
        double* result,
        double* result_stat,
        double* corr,
        int* npts,
        int* ifu,
        int* sli,
        double* data,
        double* stat,
        double* lbda,
        int npix,
        int* mask,
        int* xpix,
        int lbdabins_n,
        int *lbdabins,
        double corr_clip,
        char* logfile
) {
    size_t nlbin = (size_t)lbdabins_n;
    size_t i, k, n, s, q, slidx, prev, next;
    double tot_flux, x[3], x1[3], x2[3], minmax[2], meanq, threshq, tmp;
    double *refx, refflux;
    int slice_count1, slice_count2, tot_count;

    double *slice_flux = (double*) malloc(NSLICES*NIFUS*sizeof(double));
    double *ifu_flux = (double*) malloc(NIFUS*2*sizeof(double));
    int *slice_ind1 = (int*) malloc(NSLICES*sizeof(int));
    int *slice_ind2 = (int*) malloc(NSLICES*sizeof(int));
    int *tot_ind = (int*) malloc(NIFUS*NSLICES*sizeof(int));
    int* quad = (int*) malloc(npix*sizeof(int));
    int *indmap[nlbin * NIFUS * NSLICES];

    // Minimum number of pixels for which we have a flux, in one slice
    int min_pix_per_slice=20;

    // Clipping parameters
    int nmax=15, nstop=20;
    double nclip_low=3.0, nclip_up=3.0;

    FILE *fp=NULL;
    if (strlen(logfile) > 0) {
        printf("Output redirected to %s", logfile);
        fp = fopen(logfile, "w");
    } else {
        fp = stdout;
    }

    for (k=0; k<NIFUS*NSLICES*nlbin; k++) {
        npts[k] = 0;
        corr[k] = 1.0;
        indmap[k] = (int*) malloc(MAX_PTS_PER_SLICE * sizeof(int));
    }

    fprintf(fp, "Using %zu lambda slices\n", nlbin);
    fprintf(fp, "Computing lambda indexes ...\n");
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
    for (k=0; k<NIFUS*NSLICES*nlbin; k++) {
        if (npts[k] > MAX_PTS_PER_SLICE)
            abort();
    }

    for (q = 0; q < nlbin; q++) {
        fprintf(fp, "\n\nLambda bin: %d - %d\n", lbdabins[q], lbdabins[q+1]);
        fprintf(fp, "Computing reference levels ...\n\n");
        tot_count = 0;
        for (i = 0; i < NIFUS; i++) {
            fprintf(fp, "- IFU %02zu\n", i+1);
            slice_count1 = 0;
            slice_count2 = 0;
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                slidx = NSLICES*i + s;
                if (npts[k] > 100) {
                    mpdaf_slice_mean(data, xpix, npts[k], x, indmap[k], min_pix_per_slice);
                    if (x[2] > min_pix_per_slice) {
                        slice_flux[slidx] = x[0];
                        if (s < 24) {
                            slice_ind1[slice_count1++] = slidx;
                        } else {
                            slice_ind2[slice_count2++] = slidx;
                        }
                        tot_ind[tot_count++] = slidx;
                    } else {
                        slice_flux[slidx] = NAN;
                    }
                } else {
                    slice_flux[slidx] = NAN;
                }
                /* fprintf(fp, "  - SLICE %02zu : %f (%d)\n", s+1, slice_flux[slidx], npts[k]); */
            }

            if (!slice_count1) {
                ifu_flux[2*i] = 0.0;
            } else {
                mpdaf_minmax(slice_flux, slice_count1, slice_ind1, minmax);
                mpdaf_mean_madsigma_clip(slice_flux, slice_count1, x, nmax,
                        nclip_low, nclip_up, nstop, slice_ind1);
                fprintf(fp, "  - 1: Min max = %f %f / Mean = %f (%f, %d)\n",
                        minmax[0], minmax[1], x[0], x[1], (int)x[2]);
                ifu_flux[2*i] = x[0];

                if (isnan(x[0]))
                    fprintf(fp, "ERROR: Mean IFU flux is NAN\n");
            }

            if (!slice_count2) {
                ifu_flux[2*i+1] = 0.0;
            } else {
                mpdaf_minmax(slice_flux, slice_count2, slice_ind2, minmax);
                mpdaf_mean_madsigma_clip(slice_flux, slice_count2, x, nmax,
                        nclip_low, nclip_up, nstop, slice_ind2);
                fprintf(fp, "  - 2: Min max = %f %f / Mean = %f (%f, %d)\n",
                        minmax[0], minmax[1], x[0], x[1], (int)x[2]);
                ifu_flux[2*i+1] = x[0];

                if (isnan(x[0]))
                    fprintf(fp, "ERROR: Mean IFU flux is NAN\n");
            }

            // Use mean ifu flux for slices without useful values
            for (s = 0; s < NSLICES; s++) {
                slidx = NSLICES*i + s;
                if (isnan(slice_flux[slidx])) {
                    if (s < 24) {
                        slice_flux[slidx] = ifu_flux[2*i];
                    } else {
                        slice_flux[slidx] = ifu_flux[2*i+1];
                    }
                }
            }
        }
        if (!tot_count) {
            fprintf(fp, "WARNING: No values in this lambda bin\n");
            continue;
        }
        mpdaf_minmax(slice_flux, tot_count, tot_ind, minmax);
        fprintf(fp, "\n- Min max : %f %f\n", minmax[0], minmax[1]);

        mpdaf_mean_sigma_clip(slice_flux, tot_count, x, nmax, nclip_low,
                nclip_up, nstop, tot_ind);
        fprintf(fp, "- Total flux : %f (%f, %d)\n", x[0], x[1], (int)x[2]);
        tot_flux = x[0];

        fprintf(fp, "\nComputing corrections ...\n\n");

        for (i = 0; i < NIFUS; i++) {
            fprintf(fp, "- IFU %02zu\n", i+1);
            slice_count1 = 0;
            slice_count2 = 0;
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                slidx = NSLICES*i + s;
                if (npts[k] != 0) {
                    if (s < 24) {
                        slice_ind1[slice_count1++] = k;
                    } else {
                        slice_ind2[slice_count2++] = k;
                    }
                    corr[k] = tot_flux / slice_flux[slidx];
                    /* fprintf(fp, "  - SLICE %02zu : %f\n", s+1, corr[k]); */
                }
            }
            if ((slice_count1 + slice_count2) == 0) {
                fprintf(fp, "WARNING: No values in this IFU\n");
                continue;
            }

            if (slice_count1) {
                mpdaf_minmax(corr, slice_count1, slice_ind1, minmax);
                mpdaf_mean_madsigma_clip(corr, slice_count1, x1, nmax, nclip_low,
                        nclip_up, nstop, slice_ind1);
                fprintf(fp, "  - 1: Min max = %f %f / Mean = %f (%f, %d)\n",
                        minmax[0], minmax[1], x1[0], x1[1], (int)x1[2]);
            }

            if (slice_count2) {
                mpdaf_minmax(corr, slice_count2, slice_ind2, minmax);
                mpdaf_mean_madsigma_clip(corr, slice_count2, x2, nmax, nclip_low,
                        nclip_up, nstop, slice_ind2);
                fprintf(fp, "  - 2: Min max = %f %f / Mean = %f (%f, %d)\n",
                        minmax[0], minmax[1], x2[0], x2[1], (int)x2[2]);
            }

            fprintf(fp, "  - Checking slice corrections (%.1f sigma clip)...\n",
                   corr_clip);
            for (s = 0; s < NSLICES; s++) {
                k = MAPIDX(i+1, s+1, q+1);
                if (s < 24) {
                    if (!slice_count1)
                        continue;
                    refx = x1;
                    refflux = ifu_flux[2*i];
                } else {
                    if (!slice_count2)
                        continue;
                    refx = x2;
                    refflux = ifu_flux[2*i+1];
                }
                if (fabs(corr[k] - refx[0]) > corr_clip*refx[1]) {
                    tmp = corr[k];
                    corr[k] = tot_flux / refflux;
                    fprintf(fp, "    - SLICE %02zu : %f -> Using IFU Mean : %f\n",
                           s+1, tmp, corr[k]);
                }
            }
        }
    }

    fprintf(fp, "\nCleaning corrections ...\n\n");
    for (i = 1; i <= NIFUS; i++) {
        for (s = 1; s <= NSLICES; s++) {
            for (q = 1; q <= nlbin; q++) {
                k = MAPIDX(i, s, q);
                if (q == 1) {
                    prev = MAPIDX(i, s, q+1);
                    next = MAPIDX(i, s, q+2);
                } else if (q == nlbin) {
                    prev = MAPIDX(i, s, q-1);
                    next = MAPIDX(i, s, q-2);
                } else {
                    prev = MAPIDX(i, s, q-1);
                    next = MAPIDX(i, s, q+1);
                }
                meanq = (corr[prev] + corr[next]) / 2.;
                threshq = MAX(0.03, 3*fabs(corr[prev] - corr[next]));
                if (fabs(corr[k] - meanq) > threshq) {
                    fprintf(fp, "- %02zu / %02zu / %02zu : %f -> %f \n",
                            i, s, q, corr[k], meanq);
                    corr[k] = meanq;
                }
            }
        }
    }

    for (k=0; k<NIFUS*NSLICES*nlbin; k++) {
        free(indmap[k]);
    }

    fprintf(fp, "\nApply corrections ...\n");
    #pragma omp parallel for private(k)
    for (n=0; n < (size_t)npix; n++) {
        k = MAPIDX(ifu[n], sli[n], quad[n]);
        result[n] =  data[n] * corr[k];
        result_stat[n] =  stat[n] * corr[k] * corr[k];
    }

    if (strlen(logfile) > 0) {
        fclose(fp);
    }
    printf("\nOK, done.\n");

    free(slice_flux);
    free(ifu_flux);
    free(slice_ind1);
    free(slice_ind2);
    free(tot_ind);
    free(quad);
}
