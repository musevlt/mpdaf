#include "tools.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif


void mpdaf_old_subtract_slice_median(double* result, double* corr, int* npts, int* ifu, int* sli, double* data, double* lbda, int npix, int* mask, int skysub)
{
    double med[24 * 48];
    double median;

    //omp_set_num_threads

    #pragma omp parallel shared(ifu,sli,data,npix,med,mask,npts, skysub, median,corr,result)
    {
         int n,chan;
         #pragma omp for
         for (chan=1; chan<=24; chan++)
         {
            int sl, i, count;
            double m;
	    int* work;
	    work = (int *) malloc(npix * sizeof(int));
	    for (sl=1; sl<=48; sl++)
	    {
	        count = 0;
                for (i=0; i<npix; i++)
	        {
		  if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
		     {
			  work[count] = i;
			  count += 1;
		     }
	        }
	        m = mpdaf_median(data,count,work);
                med[48*(chan-1)+sl-1] = m;
		npts[48*(chan-1)+sl-1] = count;
	    }
	    free(work);
        }
  
	if (skysub==0)
	{
	  median = 0;
	}
	else
	{
	  #pragma omp single
	  {
	    int j;
	    int* work;
	    work = (int *) malloc(24*48 * sizeof(int));
	    for (j=0;j<24*48;j++) work[j]=j;
	    median = mpdaf_median(med, 24*48, work);
	    free(work);
	  }
	}

      #pragma omp for
      for(chan=0; chan<24; chan++)
	{
	  int sl;
	  for (sl=0; sl<48; sl++)
	    {
	      corr[48*chan+sl] = median-med[48*chan+sl];
	    }
	}


      #pragma omp for
      for(n=0; n<npix; n++)
      {
         result[n] =  data[n] - med[48*(ifu[n]-1)+sli[n]-1] + median;
             //result[i] =  data[i] - med[48*(((origin[i] >> 6) & 0x1f)-1)+(origin[i] & 0x3f)-1];
      }
    }
}

void mpdaf_sky_ref(double* data, double* lbda, int* mask, int npix, double lmin, double dl, int n, int nmax, double nclip_low, double nclip_up, int nstop, double* result)
{
  #pragma omp parallel shared(npix, mask, data, lbda, dl, lmin, nmax, nclip_low, nclip_up, nstop, result)
  {
    int l;
    #pragma omp for
    for (l=0; l<n; l++)
    {
      double x[3];
      //double med;
      int i, count;
      
      int* work;
      double l0, l1;
      l0 = lmin + l*dl;
      l1 = l0 + dl;

      work = (int *) malloc(npix * sizeof(int));
      count = 0;
      for (i=0; i<npix; i++)
      {
          if ((mask[i]==0) && (lbda[i] >= l0) && (lbda[i] < l1))
          {
              work[count] = i;
              count += 1;
          }
      }

      mpdaf_median_sigma_clip(data, count, x, nmax, nclip_low, nclip_up, nstop,work);
      free(work);

      result[l] = x[0];
    }
  }
}

void mpdaf_subtract_slice_median(double* result, double* corr, int* npts, int* ifu, int* sli, double* data, double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n)
{
    #pragma omp parallel shared(ifu,sli,data,npix,corr,mask,lbda,skyref_lbda,skyref_flux,skyref_n, npts,result)
    {   
        int n,chan;
        #pragma omp for
        for (chan=1; chan<=24; chan++)
        {
            int sl, i, count;
            double m;
            double *temp;
            double skyref;
            temp = (double *) malloc(npix * sizeof(double));
	    int *work;
	    work = (int *) malloc(npix * sizeof(int));
            for (sl=1; sl<=48; sl++)
            {
                count = 0;
                for (i=0; i<npix; i++)
                {
                    if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
                    {
                        skyref = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]);
                        temp[count] = skyref - data[i];
			work[count] = count;
                        count += 1;
                    }
                }
                if (count > 0)
		    m = mpdaf_median(temp,count,work);
                else
                    m = 0.0;

               
                corr[48*(chan-1)+sl-1] = m;
                npts[48*(chan-1)+sl-1] = count;
            }
            free(temp);
	    free(work);
        }

        #pragma omp for
        for(n=0; n<npix; n++)
        {
            result[n] =  data[n] + corr[48*(ifu[n]-1)+sli[n]-1];
        }
    }
}


void mpdaf_divide_slice_median(double* result, double* result_stat ,double* corr, int* npts, int* ifu, int* sli, double* data,  double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n)
{
   
    #pragma omp parallel shared(ifu,sli,data,npix,corr,mask,lbda,skyref_lbda,skyref_flux,skyref_n,npts,result, result_stat)
    {
         int n, chan;
         #pragma omp for
         for (chan=1; chan<=24; chan++)
         {
            int sl, i, count;
            double m;
            double *temp;
            double skyref;
            temp = (double *) malloc(npix * sizeof(double));
	    int *work;
	    work = (int *) malloc(npix * sizeof(int));
            for (sl=1; sl<=48; sl++)
            {
                count = 0;
                for (i=0; i<npix; i++)
                {
                    if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
                    {
                        skyref = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]);
                        temp[count] = data[i] / skyref;
			work[count] = count;
                        count += 1;
                    }
                }
                if (count > 0)
		    m = mpdaf_median(temp,count,work);
                else
                    m = 0.0;
                corr[48*(chan-1)+sl-1] = m;
                npts[48*(chan-1)+sl-1] = count;
            }
            free(temp);
	    free(work);
         }
    
         #pragma omp for
         for(n=0; n<npix; n++)
         {
	     result[n] =  data[n] / corr[48*(ifu[n]-1)+sli[n]-1];
	     result_stat[n] =  result_stat[n] / corr[48*(ifu[n]-1)+sli[n]-1] / corr[48*(ifu[n]-1)+sli[n]-1];
	 }
    }
}
