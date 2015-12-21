#include "tools.h"

#include <string.h>
#include <stdio.h>
#include <fitsio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h> /* for exit */
#include <sys/resource.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))

int mpdaf_merging_median(char* input, double* data, int* expmap, int* valid_pix)
{
    char* filenames[500];
    char buffer[80], begin[80];
    int nfiles;
    time_t now;
    struct tm *info;

    // read input files list
    nfiles = 0;
    const char s[2] = "\n";
    char *token;
    token = strtok(input, s);
    while( token != NULL )
    {
        filenames[nfiles] = strdup(token);
        nfiles++;
        printf("%3d: %s\n", nfiles, filenames[nfiles-1]);
        token = strtok(NULL, s);
    }
    printf("nfiles: %d\n",nfiles);

    struct rlimit limit;
    /* Get max number of files. */
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
        printf("getrlimit() failed");
        return 1;
    }

    int num_nthreads = limit.rlim_cur/nfiles * 0.9;
    if (1000/nfiles < num_nthreads) //limit of cfitsio
      {
	num_nthreads = 1000/nfiles;
      }

    int nthreads;
    #pragma omp parallel
    {
      nthreads = omp_get_num_threads();
    }
    if (nthreads < num_nthreads)
      {
	num_nthreads=nthreads;
      }

    // create threads
    #pragma omp parallel shared(filenames, nfiles, data, expmap, valid_pix, buffer, begin) num_threads(num_nthreads)
    {
        int rang = omp_get_thread_num(); //current thread number
        int nthreads = omp_get_num_threads(); //number of threads

        char filename[500];
        fitsfile *fdata[100];
        int naxis;
        int status = 0;  // CFITSIO status value MUST be initialized to zero!
        long naxes[3] = {1,1,1}, bnaxes[3] = {1,1,1};

        int i, ii, n;
        long firstpix[3] = {1,1,1};

        int valid[nfiles];

	// read first file
        #pragma omp master
        {
            printf("Read fits files\n");
        }
        sprintf(filename, "%s[data]", filenames[0]);
        /* strcpy(filename, filenames[0]); */
        /* strcat(filename,"[data]\0" ); */
        fits_open_file(&(fdata[0]), filename, READONLY, &status); // open DATA extension
        if (status)
        {
            fits_report_error(stderr, status);
	    exit(EXIT_FAILURE);
        }
        fits_get_img_dim(fdata[0], &naxis, &status);  // read dimensions
        if (naxis != 3)
        {
            printf("Error: %s not a cube\n", filename);
	    exit(EXIT_FAILURE);
        }
        fits_get_img_size(fdata[0], 3, naxes, &status); // read shape
        #pragma omp master
        {
            printf("naxes %zu %zu %zu\n", naxes[0], naxes[1], naxes[2]);
        }

       // read other files
       for (i=1; i<nfiles; i++)
       {
          sprintf(filename, "%s[data]", filenames[i]);
          /* strcpy(filename, filenames[i]); */
          /* strcat(filename,"[data]\0" ); */
          fits_open_file(&(fdata[i]), filename, READONLY, &status); // open data extension
          if (status)
          {
	      fits_report_error(stderr, status);
	      exit(EXIT_FAILURE);
          }
          fits_get_img_dim(fdata[i], &naxis, &status);  // read dimensions
          if (naxis != 3)
          {
	      printf("Error: %s not a cube\n", filename);
              exit(EXIT_FAILURE);
          }
          fits_get_img_size(fdata[i], 3, bnaxes, &status); //compare that the shape is the same
          if ( naxes[0] != bnaxes[0] ||
	       naxes[1] != bnaxes[1] ||
	       naxes[2] != bnaxes[2] )
          {
	      printf("Error: %s don't have same size\n", filename);
	      exit(EXIT_FAILURE);
	  }
       }

       // start and end of the loop for the current thread
       int start, end;
       if (nthreads<naxes[2])
       {
	   int nloops = (int) naxes[2]/nthreads +1;
	   start = rang*nloops + 1;
	   end = MIN((rang+1)*nloops, naxes[2]);
       }
       else
       {
	   start = rang+1;
	   end = MIN(rang+2, naxes[2]);
       }

       firstpix[0] = 1;

       //initialization
       int *indx;
       double *pix[100], *wdata;
       long npixels = naxes[0];
       for (i=0; i<nfiles; i++)
       {
           pix[i] = (double *) malloc(npixels * sizeof(double));
	   if (pix[i] == NULL) {
               printf("Memory allocation error\n");
	       exit(EXIT_FAILURE);
	   }
	   valid[i] = 0;
       }
       wdata = (double *) malloc(nfiles * sizeof(double));
       indx = (int *) malloc(nfiles * sizeof(int));

       for (firstpix[2] = start; firstpix[2] <= end; firstpix[2]++)
       {
           for (firstpix[1] = 1; firstpix[1] <= naxes[1]; firstpix[1]++)
	   {
	       int index0 = (firstpix[1]-1)*naxes[0] + (firstpix[2]-1)*naxes[0]*naxes[1];

	       for (i=0; i<nfiles; i++)
	       {
	           if (fits_read_pix(fdata[i], TDOUBLE, firstpix, npixels, NULL, pix[i],
                        NULL, &status))
		       break;
	       }
	       for(ii=0; ii< npixels; ii++)
	       {
	           n = 0;
		   for (i=0; i<nfiles; i++)
	           {
		       if (!isnan(pix[i][ii]))
		       {
		           wdata[n] = pix[i][ii];
			   indx[n] = n;
		           n = n + 1;
			   valid[i] = valid[i] + 1;
		       }
		   }
		   int index = ii + index0;
		   if (n==0)
		   {
		       data[index] = NAN; //mean value
		       expmap[index] = 0; //exp map
		   }
	           else if (n==1)
	           {
		       data[index] = wdata[0]; //mean value
		       expmap[index] = 1; //exp map
		   }
		   else
	           {
		       data[index] = mpdaf_median(wdata,n,indx);
		       expmap[index] = n;
		   }
	       }
	   }
	   #pragma omp master
	   {
           time(&now);
	   info = localtime(&now);
	   strftime(buffer,80,"%x - %I:%M%p", info);
	   if(strcmp(buffer,begin) != 0)
	   {
	     printf("%s %3.1f%%\n", buffer, (firstpix[2]-start)*100.0/(end-start));
	     fflush(stdout);
	     strcpy(begin, buffer);
	   }
	   }
    }
    for (i=0; i<nfiles; i++)
    {
        #pragma omp atomic
        valid_pix[i] += valid[i];
    }
    free(wdata);
    free(indx);
    for (i=0; i<nfiles; i++)
    {
        free(pix[i]);
        fits_close_file(fdata[i], &status);
    }

    if (status)
    {
        fits_report_error(stderr, status);
        exit(EXIT_FAILURE);
    }
  }
  printf("%s 100%%\n", buffer);
  fflush(stdout);
  return(1);
}


// var=0: 'propagate'
// var=1:  'stat_mean'
// var=2:  'stat_one'
int mpdaf_merging_sigma_clipping(char* input, double* data, double* var, int* expmap, double* scale, int* selected_pix, int* valid_pix, int nmax, double nclip_low, double nclip_up, int nstop, int typ_var, int mad)
{
    char* filenames[500];
    char buffer[80], begin[80];
    int nfiles;

    time_t now;
    struct tm *info;

    printf("merging cube using mean with sigma clipping\n");
    printf("nmax = %d\n", nmax);
    printf("nclip_low = %f\n", nclip_low);
    printf("nclip_high = %f\n", nclip_up);
    printf("nstop = %d\n", nstop);

    // read input files list
    nfiles = 0;
    const char s[2] = "\n";
    char *token;
    token = strtok(input, s);
    while( token != NULL )
    {
        filenames[nfiles] = strdup(token);
        nfiles++;
        printf("%3d: %s\n", nfiles, filenames[nfiles-1]);
        token = strtok(NULL, s);
    }
    printf("nfiles: %d\n",nfiles);

    struct rlimit limit;
    /* Get max number of files. */
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
        printf("getrlimit() failed");
        return 1;
    }

    int num_nthreads = limit.rlim_cur/nfiles * 0.9;
    if (1000/nfiles < num_nthreads) //limit of cfitsio
      {
	num_nthreads = 1000/nfiles;
      }
    printf("num_nthreads: %d\n", num_nthreads);

    if (typ_var==0)
    {
      num_nthreads = num_nthreads/2;
    }

    int nthreads;
    #pragma omp parallel
    {
      nthreads = omp_get_num_threads();
    }
    printf("omp_get_num_threads: %d\n", nthreads);
    if (nthreads < num_nthreads)
    {
        num_nthreads=nthreads;
    }
    printf("Using %d threads\n", num_nthreads);

    // create threads
    #pragma omp parallel shared(filenames, nfiles, data, var, expmap, scale, valid_pix, buffer, begin, nmax, nclip_low, nclip_up, nstop, selected_pix, typ_var, mad) num_threads(num_nthreads)
    {
        int rang = omp_get_thread_num(); // current thread number
        int nthreads = omp_get_num_threads(); // number of threads

        char filename[500];
	fitsfile *fdata[100], *fvar[100];
	int naxis;
	int status = 0;  // CFITSIO status value MUST be initialized to zero!
	long naxes[3] = {1,1,1}, bnaxes[3] = {1,1,1};

	int i, ii, n;
	long firstpix[3] = {1,1,1};

	int valid[nfiles], select[nfiles];

        #pragma omp master
	{
	  printf("Read fits files\n");
	}
	// read first file
    sprintf(filename, "%s[data]", filenames[0]);
	/* strcpy(filename, filenames[0]); */
	/* strcat(filename,"[data]\0" ); */
	fits_open_file(&(fdata[0]), filename, READONLY, &status); // open data extension
	if (status)
	{
	    fits_report_error(stderr, status);
	    exit(EXIT_FAILURE);
	}
	fits_get_img_dim(fdata[0], &naxis, &status);  /* read dimensions */
	if (naxis != 3)
	{
	    printf("Error: %s not a cube\n", filename);
	    exit(EXIT_FAILURE);
	}
	fits_get_img_size(fdata[0], 3, naxes, &status);
        #pragma omp master
	{
            printf("naxes %zu %zu %zu\n", naxes[0], naxes[1], naxes[2]);
	}

	// read other files
	for (i=1; i<nfiles; i++)
        {
            sprintf(filename, "%s[data]", filenames[i]);
            /* strcpy(filename, filenames[i]); */
            /* strcat(filename,"[data]\0" ); */
	    fits_open_file(&(fdata[i]), filename, READONLY, &status); // open data extension
	    if (status)
	    {
	      fits_report_error(stderr, status);
	      exit(EXIT_FAILURE);
	    }
	    fits_get_img_dim(fdata[i], &naxis, &status);  // read dimensions
	    if (naxis != 3)
            {
	        printf("Error: %s not a cube\n", filename);
		exit(EXIT_FAILURE);
            }
	    fits_get_img_size(fdata[i], 3, bnaxes, &status);
	    if ( naxes[0] != bnaxes[0] ||
		 naxes[1] != bnaxes[1] ||
		 naxes[2] != bnaxes[2] )
	    {
	        printf("Error: %s don't have same size\n", filename);
		exit(EXIT_FAILURE);
	    }
	}

	if (typ_var==0)
	{
	    // read variance extension
	    for (i=0; i<nfiles; i++)
	    {
            sprintf(filename, "%s[stat]", filenames[i]);
	        /* strcpy(filename, filenames[i]); */
	        /* strcat(filename,"[stat]\0" ); */
	        fits_open_file(&(fvar[i]), filename, READONLY, &status);
	        if (status)
	        {
	            fits_report_error(stderr, status);
		    exit(EXIT_FAILURE);
	        }
	        fits_get_img_dim(fvar[i], &naxis, &status);
	        if (naxis != 3)
	        {
	            printf("Error: %s not a cube\n", filename);
		    exit(EXIT_FAILURE);
	        }
	        fits_get_img_size(fvar[i], 3, bnaxes, &status);
	        if ( naxes[0] != bnaxes[0] ||
		     naxes[1] != bnaxes[1] ||
		     naxes[2] != bnaxes[2] )
	        {
	            printf("Error: %s don't have same size\n", filename);
		    exit(EXIT_FAILURE);
	        }
	    }
	}

	int start, end;
	if (nthreads<naxes[2])
        {
	  int nloops = (int) naxes[2]/nthreads +1;
	  start = rang*nloops + 1;
	  end = MIN((rang+1)*nloops, naxes[2]);
	}
	else
	{
	  start = rang+1;
	  end = MIN(rang+2, naxes[2]);
	}

	firstpix[0] = 1;

	//initialization
	double *pix[100], *pixvar[100], *wdata, *wvar=NULL, *wmad=NULL;
	int *indx, *files_id;
	double x[3];
	long npixels = naxes[0];
	for (i=0; i<nfiles; i++)
        {
            pix[i] = (double *) malloc(npixels * sizeof(double));
	    if (pix[i] == NULL) {
	      printf("Memory allocation error\n");
	      exit(EXIT_FAILURE);
	    }
	    valid[i] = 0;
	    select[i] = 0;
	}
	if (typ_var==0)
	{
	    for (i=0; i<nfiles; i++)
	    {
	        pixvar[i] = (double *) malloc(npixels * sizeof(double));
		if (pix[i] == NULL) {
		  printf("Memory allocation error\n");
		  exit(EXIT_FAILURE);
		}
	    }
	    wvar = (double *) malloc(nfiles * sizeof(double));
	}
	wdata = (double *) malloc(nfiles * sizeof(double));
	indx = (int *) malloc(nfiles * sizeof(int));
	files_id = (int *) malloc(nfiles * sizeof(int));
	if (mad==1)
	{
	    wmad = (double *) malloc(nfiles * sizeof(double));
	}

	for (firstpix[2] = start; firstpix[2] <= end; firstpix[2]++)
        {
            for (firstpix[1] = 1; firstpix[1] <= naxes[1]; firstpix[1]++)
            {
	        int index0 = (firstpix[1]-1)*naxes[0] + (firstpix[2]-1)*naxes[0]*naxes[1];

		for (i=0; i<nfiles; i++)
	        {
		    if (fits_read_pix(fdata[i], TDOUBLE, firstpix, npixels, NULL, pix[i],
                        NULL, &status))
		        break;
		}
		if (typ_var==0)
		{
		    for (i=0; i<nfiles; i++)
		    {
		        if (fits_read_pix(fvar[i], TDOUBLE, firstpix, npixels, NULL, pixvar[i],
                                          NULL, &status))
		            break;
		    }
		}
		for(ii=0; ii< npixels; ii++)
        {
            n = 0;
            for (i=0; i<nfiles; i++)
            {
                if (!isnan(pix[i][ii]))
                {
                    wdata[n] = pix[i][ii]*scale[i];
                    files_id[n] = i;
                    indx[n] = n;
                    if (typ_var==0)
                    {
                        wvar[n] = pixvar[i][ii]*scale[i]*scale[i];
                    }
                    n = n + 1;
                    valid[i] = valid[i] + 1;
                }
            }
            int index = ii + index0;
            if (n==0)
            {
                data[index] = NAN; //mean value
                expmap[index] = 0; //exp map
                var[index] = NAN;  //var
            }
            else if (n==1)
            {
                data[index] = wdata[0]; //mean value
                expmap[index] = 1;      //exp map
                if (typ_var==0)         //var
                    var[index] = wvar[0];
                else
                    var[index] = NAN;
                select[files_id[0]] += 1;
            }
            else
            {
                if (mad==1)
                {
                    mpdaf_mean_madsigma_clip(wdata, n, x, nmax, nclip_low, nclip_up, nstop, indx, wmad);
                }
                else
                {
                    mpdaf_mean_sigma_clip(wdata, n, x, nmax, nclip_low, nclip_up, nstop, indx);
                }
                data[index] = x[0];//mean value
                expmap[index] = x[2];//exp map
                if (typ_var==0)
                {
                    var[index] = mpdaf_sum(wvar,x[2],indx)/x[2]/x[2];
                }
                else
                {
                    if (x[2]>1)
                    {
                        var[index] = (x[1]*x[1]);//var
                        if (typ_var==1)
                        {
                            var[index] /= (x[2]-1);
                        }
                    }
                    else
                    {
                        var[index] = NAN;//var
                    }
                }
                for (i=0; i<x[2]; i++)
                {
                    select[files_id[indx[i]]] += 1;
                }
            }
        }
	    }
            #pragma omp master
	    {
	      time(&now);
	      info = localtime(&now);
	      strftime(buffer,80,"%x - %I:%M%p", info);
	      if(strcmp(buffer,begin) != 0)
	      {
		  printf("%s %3.1f%%\n", buffer, firstpix[2]*100.0/(end-start));
		  fflush(stdout);
		  strcpy(begin, buffer);
	      }
	    }
	}
	for (i=0; i<nfiles; i++)
	{
            #pragma omp atomic
	    valid_pix[i] += valid[i];
            #pragma omp atomic
	    selected_pix[i] += select[i];
	}
	free(wdata);
	free(indx);
	free(files_id);
	for (i=0; i<nfiles; i++)
	{
	    free(pix[i]);
	    fits_close_file(fdata[i], &status);
	}
	if (typ_var==0)
	{
	  free(wvar);
	  for (i=0; i<nfiles; i++)
	  {
	      free(pixvar[i]);
	      fits_close_file(fvar[i], &status);
	  }
	}
	if (mad==1)
	{
	  free(wmad);
	}

	if (status)
	{
	    fits_report_error(stderr, status);
	    exit(EXIT_FAILURE);
	}
    }
    printf("%s 100%%\n", buffer);
    fflush(stdout);
    return(1);
}
















//gcc merging.c -o merging -L/usr/local/lib -lcfitsio -O2 -march=native -Wall
//./merging cubes.txt mean.fits expmap.fits novalid.txt mean 2 5 1





