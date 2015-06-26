#include "tools.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>


// Compute the arithmetic mean
void mpdaf_mean(double* data, int n, double x[3], int* indx)
{
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;i++)
    {
      mean+=data[indx[i]];
    }
    mean=mean/n;
    for(i=0; i<n;i++)
      {
	sum_deviation+=(data[indx[i]]-mean)*(data[indx[i]]-mean);
      }
    x[0] = mean;
    x[1] = sqrt(sum_deviation/n);           
}

// Compute the sum
double mpdaf_sum(double* data, int n, int* indx)
{
  double sum=0;
  int i;
  for(i=0; i<n;i++)
    {
      sum+=data[indx[i]];
    }
  return sum;
}

// Compute median
double mpdaf_median(double *data, int  n, int *indx)
{
        int npts=n;
	double med;

	indexx(npts,data,indx);
	if (npts%2 == 0) 
		med = (data[indx[(int)(npts/2)]] + data[indx[(int)(npts/2)-1]])/2;
	else
		med = data[indx[(int)(npts/2)]];

	return(med);
}

// Compute the arithmetic mean and MAD sigma
void mpdaf_mean_mad(double* data, int n, double x[3], int *indx, double* work)
{
    double mean=0.0, median=0.0;
    int i;
    for(i=0; i<n;i++)
    {
      mean+=data[indx[i]];
    }
    mean=mean/n;

    median=mpdaf_median(data,n,indx);
    for(i=0; i<n;i++)
      {
	work[indx[i]]=fabs(data[indx[i]]-median);
      }
    x[0] = mean;
    x[1] = mpdaf_median(work,n,indx)*1.4826;
}

// Iterative sigma-clipping of array elements
// return x[0]=mean, x[1]=std, x[2]=n
// index must be initialized.
void mpdaf_mean_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop, int* indx)
{
  double clip_lo, clip_up;
  mpdaf_mean(data, n, x, indx);
  x[2] = n;
  double med;
  med =  mpdaf_median(data,n, indx);
  clip_lo = med - (nclip_low*x[1]);
  clip_up = med + (nclip_up*x[1]);

  int i, ni = 0; 
  for (i=0; i<n; i++)
    {
      if ((data[indx[i]]<clip_up) && (data[indx[i]]>clip_lo))
	{
	  ni = ni+1;
	}
    }
  if (ni<nstop || ni==n)
    {
      return;
    }
   if ( nmax > 0 )
   {
     ni = 0;
     for (i=0; i<n; i++)
     {
      if ((data[indx[i]]<clip_up) && (data[indx[i]]>clip_lo))
	{
	  indx[ni]=indx[i];
	  ni = ni+1;
	}
     }
     nmax = nmax - 1;
     mpdaf_mean_sigma_clip(data, ni, x, nmax, nclip_low, nclip_up, nstop, indx);
   }
}

// Iterative MAD sigma-clipping of array elements
// return x[0]=median, x[1]=MAD std, x[2]=n
void mpdaf_mean_madsigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop, int* indx, double* work)
{
  double clip_lo, clip_up;
  mpdaf_mean_mad(data, n, x, indx, work);
  x[2] = n;
  double med;
  med =  mpdaf_median(data,n, indx);
  clip_lo = med - (nclip_low*x[1]);
  clip_up = med + (nclip_up*x[1]);

  int i, ni = 0; 
  for (i=0; i<n; i++)
    {
      if ((data[indx[i]]<clip_up) && (data[indx[i]]>clip_lo))
	{
	  ni = ni+1;
	}
    }
  if (ni<nstop || ni==n)
    {
      return;
    }
   if ( nmax > 0 )
   {
     ni = 0;
     for (i=0; i<n; i++)
     {
      if ((data[indx[i]]<clip_up) && (data[indx[i]]>clip_lo))
	{
	  indx[ni]=indx[i];
	  ni = ni+1;
	}
     }
     nmax = nmax - 1;
     mpdaf_mean_madsigma_clip(data, ni, x, nmax, nclip_low, nclip_up, nstop, indx, work);
   }
}

// Iterative sigma-clipping of array elements
// return x[0]=median, x[1]=std, x[2]=n
void mpdaf_median_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop, int* indx)
{
  double clip_lo, clip_up;
  mpdaf_mean(data, n, x, indx);
  x[2] = n;
  double med;
  med =  mpdaf_median(data,n,indx);
  x[0] = med;
  clip_lo = med - (nclip_low*x[1]);
  clip_up = med + (nclip_up*x[1]);

  int i, ni = 0; 
  for (i=0; i<n; i++)
    {
      if ((data[indx[i]]<clip_up) && (data[indx[i]]>clip_lo))
        {
	  ni = ni + 1;
	}
    }
  
  if (ni<nstop || ni==n)
    {
      return;
    }
   if ( nmax > 0 )
   {
     ni = 0; 
     for (i=0; i<n; i++)
       {
	 if ((data[indx[i]]<clip_up) && (data[indx[i]]>clip_lo))
	   {
	     indx[ni]=indx[i];
	     ni = ni + 1;
	   }
       }
     nmax = nmax - 1;
     mpdaf_median_sigma_clip(data, ni, x, nmax, nclip_low, nclip_up, nstop,indx);
   }
}

// Given a value x, return a value j such that x is in the subrange xx[j,j+1]
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


/*-----------------------------------------------------------------------------
!
!.func                            indexx()
!
!.purp     indexes x[] in indx[] such as x[indx[]] is in ascending order
!.desc
! int indexx(npts,x,indx)
!
! int    npts;                             number of values
! double *x;                               array of values
! int    *indx;                            allocated array (size >= npts)
!.ed
------------------------------------------------------------------------------*/
#define SWAP(a,b) itemp=(a);(a)=(b);(b)=itemp;
#define M 7
#define NSTACK 50
#define NR_END 1

int indexx(int n, double *arr, int *indx)
{
	int long i,indxt,ir=n-1,itemp,j,k,l=0;
	int jstack=0, *istack;
    int *v;
	double a;

    v=(int *)malloc((size_t)((NSTACK+NR_END)*sizeof(int)));
    istack = v-1+NR_END;
    
    
    //for (j=l;j<=ir;j++) indx[j]=j;
	for (;;) {
		if (ir-l < M) {
			for (j=l+1;j<=ir;j++) {
				indxt=indx[j];
				a=arr[indxt];
				for (i=j-1;i>=0;i--) {
					if (arr[indx[i]] <= a) break;
					indx[i+1]=indx[i];
				}
				indx[i+1]=indxt;
			}
			if (jstack == 0) break;
			ir=istack[jstack--];
			l=istack[jstack--];
		} else {
			k=(l+ir) >> 1;
			SWAP(indx[k],indx[l+1]);
			if (arr[indx[l+1]] > arr[indx[ir]]) {
				SWAP(indx[l+1],indx[ir])
			}
			if (arr[indx[l]] > arr[indx[ir]]) {
				SWAP(indx[l],indx[ir])
			}
			if (arr[indx[l+1]] > arr[indx[l]]) {
				SWAP(indx[l+1],indx[l])
			}
			i=l+1;
			j=ir;
			indxt=indx[l];
			a=arr[indxt];
			for (;;) {
				do i++; while (arr[indx[i]] < a);
				do j--; while (arr[indx[j]] > a);
				if (j < i) break;
				SWAP(indx[i],indx[j])
			}
			indx[l]=indx[j];
			indx[j]=indxt;
			jstack += 2;
			if (jstack > NSTACK) return(-1);
			if (ir-i+1 >= j-l) {
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} else {
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
    free((char *)v);
	return(0);
}
