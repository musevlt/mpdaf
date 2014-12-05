#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

int qsort_compare (const void * a, const void * b)
{
        return ( *(double*)a > *(double*)b );
}

double med_value(double* data, int n)
{
  qsort(data, n, sizeof(double), qsort_compare);
  return data[n / 2];
}

void C_slice_correction(double* result, int* ifu, int* sli, double* data, double* lbda, int npix, int nmask, double* xpos, double* ypos, double* xmin, double* ymin, double* xmax, double* ymax, int skysub)
{
    int n,chan;
    double med[24 * 48];
    double median;

    #pragma omp parallel shared(ifu,sli,data,npix,med,nmask,xpos,ypos,xmin,ymin,xmax,ymax) private(chan)
    {
         #pragma omp for 
         for (chan=1; chan<=24; chan++)
         {
            int sl, i, p, count, insource;
            double m;
            double *temp;
	    temp = (double *) malloc(npix * sizeof(double));
	    for (sl=1; sl<=48; sl++)
	    {
	        count = 0;
                for (i=0; i<npix; i++)
	        {
		     if ((ifu[i]==chan) && (sli[i]==sl) && (lbda[i] > 4800) && (lbda[i] < 9300))
		     {
		          insource = 0;
		          for (p=0; p<nmask; p++)
		          {
			       if((xpos[i]>=xmin[p]) && (xpos[i]<=xmax[p]) && (ypos[i]>=ymin[p]) && (ypos[i]<=ymax[p]))
			       {
			          insource = 1;
			       }
		          }
		          if (insource == 0)
		          {
			      temp[count] = data[i];
			      count += 1;
		          }
		     }
	        }
	        m = med_value(temp,count);
                #pragma omp critical
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
        median = med_value(med, 24*48);
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
