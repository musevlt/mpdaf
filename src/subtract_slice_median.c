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
}Pix;

//void initPix(Pix * array, int npix, int* ifu, int* sli, double* data, double* lbda, int* mask, int* quad, double* skyref_flux, double* skyref_lbda, int skyref_n) {
// typ=0 -> mpdaf_divide_slice_median
// typ=1 -> mpdaf_subtract_slice_median
int initPix(Pix * array, int npix, int* ifu, int* sli, double* data, double* lbda, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n, int* xpix, int*ypix, int* quad, int typ) {
  int i, count=0;
  if (typ==0)
  {
      for (i = 0; i < npix; i++)
      {
	  if ((mask[i] == 0) && (lbda[i] >= 4800) && (lbda[i] <= 9300))
	  {
	      array[count].ifu = ifu[i];
	      array[count].sli = sli[i];
	      array[count].data = data[i] / mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]);
	      if (xpix[i] < 2048)
	      {
		  if (ypix[i]<2056)
		  {
		      array[count].quad = 1;
		      quad[i] = 1;
		  }
		  else
		  {
		      array[count].quad = 2;
		      quad[i] = 2;
		  }
	      }
	      else
	      {
		  if (ypix[i]<2056)
		  {
		      array[count].quad = 4;
		      quad[i] = 4;
		  }
		  else
		  {
		      array[count].quad = 3;
		      quad[i] = 3;
		  }
	      }
	      count = count + 1;
	  }
	  else
	  {
	      if (xpix[i] < 2048)
	      {
	          if (ypix[i]<2056)
		      quad[i] = 1;
		  else
		      quad[i] = 2;
	      }
	      else
	      {
		  if (ypix[i]<2056)
		      quad[i] = 4;
		  else
		      quad[i] = 3;
	      }
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
	      array[count].data = mpdaf_linear_interpolation(skyref_lbda, skyref_flux, skyref_n , lbda[i]) - data[i];
	      if (xpix[i] < 2048)
	      {
		  if (ypix[i]<2056)
		  {
		      array[count].quad = 1;
		      quad[i] = 1;
		  }
		  else
		  {
		      array[count].quad = 2;
		      quad[i] = 2;
		  }
	      }
	      else
	      {
		  if (ypix[i]<2056)
		  {
		      array[count].quad = 4;
		      quad[i] = 4;
		  }
		  else
		  {
		      array[count].quad = 3;
		      quad[i] = 3;
		  }
	      }
	      count = count + 1;
	  }
	  else
	  {
	      if (xpix[i] < 2048)
	      {
	          if (ypix[i]<2056)
		      quad[i] = 1;
		  else
		      quad[i] = 2;
	      }
	      else
	      {
		  if (ypix[i]<2056)
		      quad[i] = 4;
		  else
		      quad[i] = 3;
	      }
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

// typ=0 -> mpdaf_divide_slice_median
// typ=1 -> mpdaf_subtract_slice_median
void mpdaf_slice_median(double* result, double* result_stat ,double* corr, int* npts, int* ifu, int* sli, double* data,  double* lbda, int npix, int* mask, double* skyref_flux, double* skyref_lbda, int skyref_n, int* xpix, int* ypix, int typ)
{
    Pix* pixels = (Pix*) malloc(npix * sizeof(Pix));
    int npix2;
    int* quad = (int*) malloc(npix*sizeof(int));
    npix2 = initPix(pixels, npix, ifu, sli, data, lbda, mask, skyref_flux, skyref_lbda, skyref_n, xpix, ypix, quad, typ);
    sortPix(pixels, npix2);
    
    int i=0, n, index;
    int q=pixels[0].quad;
    int chan=pixels[0].ifu;
    int sl=pixels[0].sli;
    int imin=0, imax=-1;
    for(i=0;i<npix2;i++)
      {
    	if ((pixels[i].quad == q) && (pixels[i].sli == sl) && (pixels[i].ifu==chan))
    	  imax = imax +1;
    	else
    	  {
    	    if(imin!=imax)
    	      {
		index = 4*48*(chan-1)+4*(sl-1)+q-1;
    		int num=imax-imin+1;
    		if (num%2 == 0)
    		  corr[index] = (pixels[imin+(int)(num/2)].data + pixels[imin+(int)(num/2)-1].data)/2;
    		else
    		  corr[index] = pixels[imin+(int)(num/2)].data;
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
    	int num=imax-imin+1;
    	if (num%2 == 0)
    	    corr[index] = (pixels[imin+(int)(num/2)].data + pixels[imin+(int)(num/2)-1].data)/2;
    	else
    	    corr[index] = pixels[imin+(int)(num/2)].data;
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
