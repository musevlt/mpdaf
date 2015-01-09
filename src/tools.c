#include "tools.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// Compare two elements
static int qsort_compare (const void * a, const void * b)
{
        return ( *(double*)a > *(double*)b );
}

// Compute the arithmetic mean
void mpdaf_mean(double* data, int n, double x[3])
{
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;i++)
    {
      mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;i++)
      {
	sum_deviation+=(data[i]-mean)*(data[i]-mean);
      }
    x[0] = mean;
    x[1] = sqrt(sum_deviation/n);           
}

// Compute the median
double mpdaf_median(double* data, int n)
{
  qsort(data, n, sizeof(double), qsort_compare);
  return data[n / 2];
}

// Iterative sigma-clipping of array elements
// return x[0]=mean, x[1]=std, x[2]=n, id
void mpdaf_mean_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop, int* id)
{
  double clip_lo, clip_up;
  mpdaf_mean(data, n, x);
  x[2] = n;
  double med;
  med =  mpdaf_median(data,n);
  clip_lo = med - (nclip_low*x[1]);
  clip_up = med + (nclip_up*x[1]);

  int i, ni = 0; 
  for (i=0; i<n; i++)
    {
      if ((data[i]<clip_up) && (data[i]>clip_lo))
        {
	  data[ni]=data[i];
	  id[ni]=id[i];
	  ni = ni + 1;
	}
    }
  if (ni<nstop || ni==n)
    {
      return;
    }
   if ( nmax > 0 )
   {
     nmax = nmax - 1;
     mpdaf_mean_sigma_clip(data, ni, x, nmax, nclip_low, nclip_up, nstop, id);
   }
}

// Iterative sigma-clipping of array elements
// return x[0]=median, x[1]=std, x[2]=n
void mpdaf_median_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop)
{
  //printf("n = %i med=%g\n",n,mpdaf_median(data,n));
  double clip_lo, clip_up;
  mpdaf_mean(data, n, x);
  x[2] = n;
  double med;
  med =  mpdaf_median(data,n);
  x[0] = med;
  clip_lo = med - (nclip_low*x[1]);
  clip_up = med + (nclip_up*x[1]);

  int i, ni = 0; 
  for (i=0; i<n; i++)
    {
      if ((data[i]<clip_up) && (data[i]>clip_lo))
        {
	  data[ni]=data[i];
	  ni = ni + 1;
	}
    }
  
  if (ni<nstop || ni==n)
    {
      return;
    }
   if ( nmax > 0 )
   {
     nmax = nmax - 1;
     mpdaf_median_sigma_clip(data, ni, x, nmax, nclip_low, nclip_up, nstop);
   }
}

// Given a value x, return a value j such that x is in the subrange data[j,j+1]
// data must be increasing
int mpdaf_locate(double* xx, int n, double x)
{
  int ju, jm, jl;
  jl = 0;
  ju = n-1;
  while(ju-jl > 1)
  {
    jm = (ju+jl) >> 1;
    if(x >= xx[jm])
      jl = jm;
    else
      ju = jm;
  }
  return fmax(0, fmin(n-2, jl));
  
}

double mpdaf_linear_interpolation(double* xx, double* yy, int n, double x)
{
  int jl = mpdaf_locate(xx, n, x);
  double x0 =  xx[jl];
  double x1 = xx[jl+1];
  double y0 = yy[jl];
  double y1 = yy[jl+1];
  double a = (y1 - y0) / (x1 - x0);
  double b = -a*x0 + y0;
  double y = a * x + b;
  return y;
}


