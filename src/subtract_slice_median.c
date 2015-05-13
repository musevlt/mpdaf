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
    int n,chan;
    double med[24 * 48];
    double median;

    //omp_set_num_threads

    #pragma omp parallel shared(ifu,sli,data,npix,med,mask,npts) private(chan)
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
		     npts[48*(chan-1)+sl-1] = count;
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
        double* cpy;
        cpy = (double *) malloc(24*48 * sizeof(double));
	memcpy(cpy, med, 24*48 * sizeof(double));
        median = mpdaf_median(cpy, 24*48);
	free(cpy);
    }

    #pragma omp parallel shared(corr, median, med) private(chan)
    {
      #pragma omp for
      for(chan=0; chan<24; chan++)
	{
	  int sl;
	  for (sl=0; sl<48; sl++)
	    {
	      corr[48*chan+sl] = median-med[48*chan+sl];
	    }
	}
    }


    #pragma omp parallel shared(ifu,sli,data,npix,med,result, median) private(n)
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

void mpdaf_subtract_slice_median(double* result, double* corr, int* npts, int* ifu, int* sli, double* data, double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n)
{
    int n,chan;

    //omp_set_num_threads

    #pragma omp parallel shared(ifu,sli,data,npix,corr,mask,lbda,skyref_lbda,skyref_flux,skyref_n, npts) private(chan)
    {
        #pragma omp for
        for (chan=1; chan<=24; chan++)
        {
            int sl, i, count;
            double m;
            double *temp;
            double skyref;
            temp = (double *) malloc(npix * sizeof(double));
            for (sl=1; sl<=48; sl++)
            {
                count = 0;
                for (i=0; i<npix; i++)
                {
                    if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
                    {
                        skyref = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]);
                        temp[count] = skyref - data[i];
                        count += 1;
                    }
                }
                if (count > 0)
                    m = mpdaf_median(temp,count);
                else
                    m = 0.0;

                #pragma omp critical //#pragma omp atomic write
                {
                    corr[48*(chan-1)+sl-1] = m;
                    npts[48*(chan-1)+sl-1] = count;
                    //printf("%i %i %i %g \n",chan,sl,count,m);
                }
            }
            free(temp);
        }
    }

    #pragma omp barrier

    #pragma omp parallel shared(ifu,sli,data,npix,corr,result) private(n)
    {
        #pragma omp for
        for(n=0; n<npix; n++)
        {
            result[n] =  data[n] + corr[48*(ifu[n]-1)+sli[n]-1];
        }
    }
}


void mpdaf_divide_slice_median(double* result, double* result_stat ,double* corr, int* npts, int* ifu, int* sli, double* data,  double* stat, double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n)
{
    int n, chan;

    //omp_set_num_threads

#pragma omp parallel shared(ifu,sli,data,npix,corr,mask,lbda,skyref_lbda,skyref_flux,skyref_n,npts) private(chan)
    {
         #pragma omp for
         for (chan=1; chan<=24; chan++)
         {
            int sl, i, count;
            double m;
            double *temp;
            double skyref;
	    temp = (double *) malloc(npix * sizeof(double));
	    for (sl=1; sl<=48; sl++)
	    {
	        count = 0;
                for (i=0; i<npix; i++)
	        {
		  if ((mask[i]==0) && (ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
		     {
		          skyref = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]);
			  temp[count] = data[i] / skyref;
			  count += 1;
		     }
	        }
	        m = mpdaf_median(temp,count);
                #pragma omp critical //#pragma omp atomic write
		{
                     corr[48*(chan-1)+sl-1] = m;
		     npts[48*(chan-1)+sl-1] = count;
	             //printf("%i %i %i %g \n",chan,sl,count,m);
	        }
	    }
	    free(temp);
        }
    }

    #pragma omp barrier

#pragma omp parallel shared(ifu,sli,data,npix,corr, result, stat, result_stat) private(n)
    {
          #pragma omp for
          for(n=0; n<npix; n++)
          {
	     result[n] =  data[n] / corr[48*(ifu[n]-1)+sli[n]-1];
	     result_stat[n] =  result_stat[n] / corr[48*(ifu[n]-1)+sli[n]-1] / corr[48*(ifu[n]-1)+sli[n]-1];
	  }
    }
}
