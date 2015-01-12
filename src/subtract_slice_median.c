#include "tools.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>


void mpdaf_old_subtract_slice_median(double* result, int* ifu, int* sli, double* data, double* lbda, int npix, int* mask, int skysub)
{
    int n,chan;
    double med[24 * 48];
    double median;

    //omp_set_num_threads

    #pragma omp parallel shared(ifu,sli,data,npix,med,mask) private(chan)
    {
         #pragma omp for 
         for (chan=1; chan<=24; chan++)
         {
            int sl, i, count;
            double m;
            double *temp;
	    temp = (double *) malloc(npix * sizeof(double));
	    for (sl=1; sl<=48; sl++)
	    {
	        count = 0;
                for (i=0; i<npix; i++)
	        {
		  if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
		     {
			  temp[count] = data[i];
			  count += 1;
		     }
	        }
	        m = mpdaf_median(temp,count);
                #pragma omp critical //#pragma omp atomic write
		{
                     med[48*(chan-1)+sl-1] = m;
	             //printf("%i %i %i %g \n",chan,sl,count,m);
	        }
	    }
	    free(temp);
        }
    }

    #pragma omp barrier

    if (skysub==0)
    {
	median = 0;
    }
    else
    {
        median = mpdaf_median(med, 24*48);
    }

    #pragma omp parallel shared(ifu,sli,data,npix,med, result) private(n)
    {
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
  int l;
  

#pragma omp parallel shared(npix, mask, data, lbda, dl, lmin, nmax, nclip_low, nclip_up, nstop, result) private(l)
  {
    #pragma omp for
    for (l=0; l<n; l++)
    {
      double x[3];
      //double med;
      int i, count;
      double *temp;
      double l0, l1;
      l0 = lmin + l*dl;
      l1 = l0 + dl;
      
      temp = (double *) malloc(npix * sizeof(double));
      count = 0;
      for (i=0; i<npix; i++)
      {
	if ((mask[i]==0) && (lbda[i] >= l0) && (lbda[i] < l1))
	  {
	      temp[count] = data[i];
	      count += 1;
	  }
      }

      mpdaf_median_sigma_clip(temp, count, x, nmax, nclip_low, nclip_up, nstop);
      //med = mpdaf_median(temp, count);
      free(temp);
      
      #pragma omp critical //#pragma omp atomic write
      {
	//printf("%g %g %i %g \n",l0,l1,count,med);
	//result[l] = med;
	result[l] = x[0];
       }
    }
   }
}

void mpdaf_subtract_slice_median(double* result, int* ifu, int* sli, double* data, double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n)
{
    int n,chan;
    double med[24 * 48];
    double skyref;

    //omp_set_num_threads

    #pragma omp parallel shared(ifu,sli,data,npix,med,mask) private(chan)
    {
         #pragma omp for 
         for (chan=1; chan<=24; chan++)
         {
            int sl, i, count;
            double m;
            double *temp;
	    temp = (double *) malloc(npix * sizeof(double));
	    for (sl=1; sl<=48; sl++)
	    {
	        count = 0;
                for (i=0; i<npix; i++)
	        {
		  if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
		     {
			  temp[count] = data[i];
			  count += 1;
		     }
	        }
	        m = mpdaf_median(temp,count);
                #pragma omp critical //#pragma omp atomic write
		{
                     med[48*(chan-1)+sl-1] = m;
	             //printf("%i %i %i %g \n",chan,sl,count,m);
	        }
	    }
	    free(temp);
        }
    }

    #pragma omp barrier

    #pragma omp parallel shared(ifu,sli,data,npix,med, result) private(n)
    {
          #pragma omp for
          for(n=0; n<npix; n++)
          {
	     skyref = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[n]);
             result[n] =  data[n] - med[48*(ifu[n]-1)+sli[n]-1] + skyref;
	  }
    }
}


void mpdaf_divide_slice_median(double* result, int* ifu, int* sli, double* data, double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n)
{
    int n,chan;
    double med[24 * 48];
    double skyref;

    //omp_set_num_threads

    #pragma omp parallel shared(ifu,sli,data,npix,med,mask) private(chan)
    {
         #pragma omp for 
         for (chan=1; chan<=24; chan++)
         {
            int sl, i, count;
            double m;
            double *temp;
	    temp = (double *) malloc(npix * sizeof(double));
	    for (sl=1; sl<=48; sl++)
	    {
	        count = 0;
                for (i=0; i<npix; i++)
	        {
		  if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
		     {
			  temp[count] = data[i];
			  count += 1;
		     }
	        }
	        m = mpdaf_median(temp,count);
                #pragma omp critical //#pragma omp atomic write
		{
                     med[48*(chan-1)+sl-1] = m;
	             //printf("%i %i %i %g \n",chan,sl,count,m);
	        }
	    }
	    free(temp);
        }
    }

    #pragma omp barrier

    #pragma omp parallel shared(ifu,sli,data,npix,med, result) private(n)
    {
          #pragma omp for
          for(n=0; n<npix; n++)
          {
	     skyref = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[n]);
             result[n] =  data[n] / med[48*(ifu[n]-1)+sli[n]-1] * skyref;
	  }
    }
}
