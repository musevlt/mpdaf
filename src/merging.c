#include "tools.h"

#include <string.h>
#include <stdio.h>
#include <fitsio.h>
#include <time.h>
#include <math.h>

int mpdaf_merging_median(char* input, double* data, int* expmap)
{
  int status = 0;  /* CFITSIO status value MUST be initialized to zero! */
  
  char filename[500], buffer[80], begin[80];
  char* filenames[500];
  
  int i, ii, n, naxis, nfiles, index;
  long naxes[3] = {1,1,1}, bnaxes[3] = {1,1,1}, npixels = 1, firstpix[3] = {1,1,1};
  double *pix[100], *wdata;

  fitsfile *fdata[100];

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
  
  printf("Read fits files\n");
  // read first file
  strcpy(filename, filenames[0]);
  strcat(filename,"[data]\0" );
  fits_open_file(&(fdata[0]), filename, READONLY, &status); /* open input images */
  if (status)
  {
      fits_report_error(stderr, status); /* print error message */
      return(status);
  }
  fits_get_img_dim(fdata[0], &naxis, &status);  /* read dimensions */
  if (naxis != 3)
  {
      printf("Error: %s not a cube\n", filename);
  }
  fits_get_img_size(fdata[0], 3, naxes, &status);
  printf("naxes %zu %zu %zu\n", naxes[0], naxes[1], naxes[2]);

  // read other files
  for (i=1; i<nfiles; i++)
  {
      strcpy(filename, filenames[i]);
      strcat(filename,"[data]\0" );
      fits_open_file(&(fdata[i]), filename, READONLY, &status); /* open input images */
      if (status)
      {
          fits_report_error(stderr, status); /* print error message */
          return(status);
      }
      fits_get_img_dim(fdata[i], &naxis, &status);  /* read dimensions */
      if (naxis != 3)
      {
	  printf("Error: %s not a cube\n", filename);
      }
      fits_get_img_size(fdata[i], 3, bnaxes, &status);
      if ( naxes[0] != bnaxes[0] || 
           naxes[1] != bnaxes[1] || 
           naxes[2] != bnaxes[2] )
      {
	  printf("Error: %s don't have same size\n", filename);
          return 1;
      } 
  }

  
  //initialization
  printf("Memory allocation and creation of the new empty output file\n");
  npixels = naxes[0];  /* no. of pixels to read in each row */
  for (i=0; i<nfiles; i++)
  {
      pix[i] = (double *) malloc(npixels * sizeof(double));
      if (pix[i] == NULL) {
          printf("Memory allocation error\n");
          return(1);
      }
  }
  wdata = (double *) malloc(nfiles * sizeof(double));

  /* create the new empty output file if */
  firstpix[0] = 1;

  printf("Loop over all planes of the data cube\n");
  for (firstpix[2] = 1; firstpix[2] <= naxes[2]; firstpix[2]++)
  {
      for (firstpix[1] = 1; firstpix[1] <= naxes[1]; firstpix[1]++)
      {
          /* Read both cubes as doubles, regardless of actual datatype.  */
          /* Give starting pixel coordinate and no. of pixels to read.   */
          /* This version does not support undefined pixels in the cube. */
	  for (i=0; i<nfiles; i++)
	  {
	       if (fits_read_pix(fdata[i], TDOUBLE, firstpix, npixels, NULL, pix[i],
                        NULL, &status))
		    break;   /* jump out of loop on error */
	  }
          for(ii=0; ii< npixels; ii++)
	  {
	       n = 0;
               for (i=0; i<nfiles; i++)
	       {
		  if (!isnan(pix[i][ii]))
		  {
		      wdata[n] = pix[i][ii];
		      n = n + 1;
		   }
               }
	       index = ii + (firstpix[1]-1)*naxes[0] + (firstpix[2]-1)*naxes[0]*naxes[1];
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
		 data[index] = mpdaf_median(wdata,n);
		 expmap[index] = n;
       	       }
	  }     
      }
      time(&now);
      info = localtime(&now);
      strftime(buffer,80,"%x - %I:%M%p", info);
      if(strcmp(buffer,begin) != 0)
      {
          printf("%s %3.1f%%\n", buffer, firstpix[2]*100.0/naxes[2]);
	  fflush(stdout);
	  strcpy(begin, buffer);
      }
  }
  printf("%s 100%%\n", buffer);
  fflush(stdout);

  for (i=0; i<nfiles; i++)
  {
      free(pix[i]);
      fits_close_file(fdata[i], &status);
  } 
   
  if (status)
  {
     /* print any error messages */
     fits_report_error(stderr, status);
     return(status);
  }

  return (1);
}


int mpdaf_merging_sigma_clipping(char* input, double* data, double* var, int* expmap, int* valid_pix, int nmax, double nclip_low, double nclip_up, int nstop, int var_mean)
{
    int status = 0;  //CFITSIO status value MUST be initialized to zero!
    int naxis, i, ii, n, index;
    long npixels = 1, firstpix[3] = {1,1,1};
    long naxes[3] = {1,1,1}, bnaxes[3] = {1,1,1};

    double *wdata;
    int nfiles = 2;
    int *files_id;
    char filename[500];
    char* filenames[500];
    double* pix[100];
    fitsfile* fptr[100]; 
    double x[3];

    char buffer[80], begin[80];
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
  
    printf("Read fits files\n");
    // read first file
    strcpy(filename, filenames[0]);
    strcat(filename,"[data]\0" );
    fits_open_file(&(fptr[0]), filename, READONLY, &status); //open input images
    if (status)
    {
        fits_report_error(stderr, status);
	return(status);
    }
    fits_get_img_dim(fptr[0], &naxis, &status);  // read dimensions
    if (naxis != 3)
    {
        printf("Error: %s not a cube\n", filename);
    }
    fits_get_img_size(fptr[0], 3, naxes, &status);
    printf("naxes %zu %zu %zu\n", naxes[0], naxes[1], naxes[2]);

    // read other files
    for (i=1; i<nfiles; i++)
    {
        strcpy(filename, filenames[i]);
	strcat(filename,"[data]\0" );
	fits_open_file(&(fptr[i]), filename, READONLY, &status); // open input images
	if (status)
	{
            fits_report_error(stderr, status);
	    return(status);
	}
	fits_get_img_dim(fptr[i], &naxis, &status);  // read dimensions
	if (naxis != 3)
	{
	    printf("Error: %s not a cube\n", filename);
	}
	fits_get_img_size(fptr[i], 3, bnaxes, &status);
	if ( naxes[0] != bnaxes[0] || 
	     naxes[1] != bnaxes[1] || 
	     naxes[2] != bnaxes[2] )
	{
	    printf("Error: %s don't have same size\n", filename);
	    return 1;
	}
    }

    printf("merging cube using mean with sigma clipping\n");
    printf("nmax = %d\n", nmax);
    printf("nclip_low = %f\n", nclip_low);
    printf("nclip_high = %f\n", nclip_up);
    printf("nstop = %d\n", nstop);

    //initialization
    printf("Memory allocation and creation of the new empty output file\n");
    files_id = (int *) malloc(nfiles * sizeof(int));
    npixels = naxes[0];  // no. of pixels to read in each row
    for (i=0; i<nfiles; i++)
    {
        files_id[i] = i;
	valid_pix[i] = 0;
	pix[i] = (double *) malloc(npixels * sizeof(double));
	if (pix[i] == NULL)
        {
            printf("Memory allocation error\n");
            return(1);
	}
    }
    wdata = (double *) malloc(nfiles * sizeof(double));

    firstpix[0] = 1;

    printf("Loop over all planes of the cube\n");
    // loop over all planes of the cube (2D images have 1 plane)
    for (firstpix[2] = 1; firstpix[2] <= naxes[2]; firstpix[2]++)
    {
        // loop over all rows of the plane
        for (firstpix[1] = 1; firstpix[1] <= naxes[1]; firstpix[1]++)
	{
	    // Read both cubes as doubles, regardless of actual datatype.
	    // Give starting pixel coordinate and no. of pixels to read.
	    // This version does not support undefined pixels in the cube.
	    for (i=0; i<nfiles; i++)
	    {
	        if (fits_read_pix(fptr[i], TDOUBLE, firstpix, npixels, NULL, pix[i],
                        NULL, &status))
		    break;   // jump out of loop on error
	    }
	    for(ii=0; ii< npixels; ii++)
	    {
	        n = 0;
		for (i=0; i<nfiles; i++)
		{
	            if (!isnan(pix[i][ii]))
		    {
		        wdata[n] = pix[i][ii];
			files_id[n] = i;
			n = n + 1;
		    }
		}
		index = ii + (firstpix[1]-1)*naxes[0] + (firstpix[2]-1)*naxes[0]*naxes[1];
		if (n==0)
		{
		    data[index] = NAN; //mean value
		    expmap[index] = 0; //exp map
		    var[index] = NAN;//var
		} 
		else if (n==1)
		{
		    data[index] = wdata[0]; //mean value
		    expmap[index] = 1; //exp map
		    var[index] = NAN;//var
		}
		else
		{
                    mpdaf_mean_sigma_clip(wdata, n, x, nmax, nclip_low, nclip_up, nstop, files_id);
		    data[index] = x[0];//mean value
		    expmap[index] = x[2];//exp map
		    if (x[2]>1)
		    {
		        var[index] = (x[1]*x[1]);//var
			if (var_mean==1)
			{
			    var[index] /= (x[2]-1);
			}
		    } else
		    {
		        var[index] = NAN;//var
		    }
		    for (i=0; i<x[2]; i++)
		    {
		        valid_pix[files_id[i]] += 1;
		    }
		}
	    }
	}
	time(&now);
	info = localtime(&now);
	strftime(buffer,80,"%x - %I:%M%p", info);
	if(strcmp(buffer,begin) != 0)
	{
            printf("%s %3.1f%%\n", buffer, firstpix[2]*100.0/naxes[2]);
	    fflush(stdout);
	    strcpy(begin, buffer);
	}
    }
    printf("%s 100%%\n", buffer);
    fflush(stdout);
 
    for (i=0; i<nfiles; i++)
    {
        free(pix[i]);
	fits_close_file(fptr[i], &status);
    } 
   
    if (status)
    {
        // print any error messages
      fits_report_error(stderr, status);
      return(status);
    }

    return (1);
}

int mpdaf_merging_sigma_clipping_var(char* input, double* data, double* var, int* expmap, int* valid_pix, int nmax, double nclip_low, double nclip_up, int nstop)
{
    int status = 0;  //CFITSIO status value MUST be initialized to zero!
    int naxis, i, ii, n, index;
    long npixels = 1, firstpix[3] = {1,1,1};
    long naxes[3] = {1,1,1}, bnaxes[3] = {1,1,1};

    double *wdata, *wvar;
    int nfiles = 2;
    int *files_id;
    char filename[500];
    char* filenames[500];
    double *pix[100], *pixvar[100];
    fitsfile *fdata[100], *fvar[100]; 
    double x[3];

    char buffer[80], begin[80];
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
  
    printf("Read fits files\n");
    // read first file
    strcpy(filename, filenames[0]);
    strcat(filename,"[data]\0" );
    fits_open_file(&(fdata[0]), filename, READONLY, &status); //open input images
    if (status)
    {
        fits_report_error(stderr, status);
	return(status);
    }
    fits_get_img_dim(fdata[0], &naxis, &status);  // read dimensions
    if (naxis != 3)
    {
        printf("Error: %s not a cube\n", filename);
    }
    fits_get_img_size(fdata[0], 3, naxes, &status);
    printf("naxes %zu %zu %zu\n", naxes[0], naxes[1], naxes[2]);

    // read other files
    for (i=1; i<nfiles; i++)
    {
        strcpy(filename, filenames[i]);
	strcat(filename,"[data]\0" );
	fits_open_file(&(fdata[i]), filename, READONLY, &status); // open input images
	if (status)
	{
            fits_report_error(stderr, status);
	    return(status);
	}
	fits_get_img_dim(fdata[i], &naxis, &status);  // read dimensions
	if (naxis != 3)
	{
	    printf("Error: %s not a cube\n", filename);
	}
	fits_get_img_size(fdata[i], 3, bnaxes, &status);
	if ( naxes[0] != bnaxes[0] || 
	     naxes[1] != bnaxes[1] || 
	     naxes[2] != bnaxes[2] )
	{
	    printf("Error: %s don't have same size\n", filename);
	    return 1;
	}
    }
    
    // read variance extension
    for (i=0; i<nfiles; i++)
    {
        strcpy(filename, filenames[i]);
	strcat(filename,"[stat]\0" );
	fits_open_file(&(fvar[i]), filename, READONLY, &status); /* open input images */
	if (status)
	{
	    fits_report_error(stderr, status); /* print error message */
	    return(status);
	}
	fits_get_img_dim(fvar[i], &naxis, &status);  /* read dimensions */
	if (naxis != 3)
	{
	    printf("Error: %s not a cube\n", filename);
	}
	fits_get_img_size(fvar[i], 3, bnaxes, &status);
	if ( naxes[0] != bnaxes[0] || 
	     naxes[1] != bnaxes[1] || 
	     naxes[2] != bnaxes[2] )
	{
	    printf("Error: %s don't have same size\n", filename);
	    return 1;
	} 
    }

    printf("merging cube using mean with sigma clipping\n");
    printf("nmax = %d\n", nmax);
    printf("nclip_low = %f\n", nclip_low);
    printf("nclip_high = %f\n", nclip_up);
    printf("nstop = %d\n", nstop);

    //initialization
    printf("Memory allocation and creation of the new empty output file\n");
    files_id = (int *) malloc(nfiles * sizeof(int));
    npixels = naxes[0];  // no. of pixels to read in each row
    for (i=0; i<nfiles; i++)
    {
        files_id[i] = i;
	valid_pix[i] = 0;
	pix[i] = (double *) malloc(npixels * sizeof(double));
	if (pix[i] == NULL)
        {
            printf("Memory allocation error\n");
            return(1);
	}
	pixvar[i] = (double *) malloc(npixels * sizeof(double));
	if (pixvar[i] == NULL)
        {
            printf("Memory allocation error\n");
            return(1);
	}
    }
    wdata = (double *) malloc(nfiles * sizeof(double));
    wvar = (double *) malloc(nfiles * sizeof(double));

    firstpix[0] = 1;

    printf("Loop over all planes of the cube\n");
    // loop over all planes of the cube (2D images have 1 plane)
    for (firstpix[2] = 1; firstpix[2] <= naxes[2]; firstpix[2]++)
    {
        // loop over all rows of the plane
        for (firstpix[1] = 1; firstpix[1] <= naxes[1]; firstpix[1]++)
	{
	    // Read both cubes as doubles, regardless of actual datatype.
	    // Give starting pixel coordinate and no. of pixels to read.
	    // This version does not support undefined pixels in the cube.
	    for (i=0; i<nfiles; i++)
	    {
	        if (fits_read_pix(fdata[i], TDOUBLE, firstpix, npixels, NULL, pix[i],
                        NULL, &status))
		    break;   // jump out of loop on error
		if (fits_read_pix(fvar[i], TDOUBLE, firstpix, npixels, NULL, pixvar[i],
                        NULL, &status))
		    break;   // jump out of loop on error
	    }
	    for(ii=0; ii< npixels; ii++)
	    {
	        n = 0;
		for (i=0; i<nfiles; i++)
		{
	            if (!isnan(pix[i][ii]))
		    {
		        wdata[n] = pix[i][ii];
			wvar[n] = pixvar[i][ii];
			files_id[n] = i;
			n = n + 1;
		    }
		}
		index = ii + (firstpix[1]-1)*naxes[0] + (firstpix[2]-1)*naxes[0]*naxes[1];
		if (n==0)
		{
		    data[index] = NAN; //mean value
		    expmap[index] = 0; //exp map
		    var[index] = NAN;//var
		} 
		else if (n==1)
		{
		    data[index] = wdata[0]; //mean value
		    expmap[index] = 1; //exp map
		    var[index] = NAN;//var
		}
		else
		{
                    mpdaf_mean_sigma_clip(wdata, n, x, nmax, nclip_low, nclip_up, nstop, files_id);
		    data[index] = x[0];//mean value
		    expmap[index] = x[2];//exp map
		    var[index] = mpdaf_sum(wvar,n)/n/n;
		    for (i=0; i<x[2]; i++)
		    {
		        valid_pix[files_id[i]] += 1;
		    }
		}
	    }
	}
	time(&now);
	info = localtime(&now);
	strftime(buffer,80,"%x - %I:%M%p", info);
	if(strcmp(buffer,begin) != 0)
	{
            printf("%s %3.1f%%\n", buffer, firstpix[2]*100.0/naxes[2]);
	    fflush(stdout);
	    strcpy(begin, buffer);
	}
    }
    printf("%s 100%%\n", buffer);
    fflush(stdout);
 
    for (i=0; i<nfiles; i++)
    {
        free(pix[i]);
	fits_close_file(fdata[i], &status);
    } 
   
    if (status)
    {
        // print any error messages
      fits_report_error(stderr, status);
      return(status);
    }

    return (1);
}

//gcc merging.c -o merging -L/usr/local/lib -lcfitsio -O2 -march=native -Wall
//./merging cubes.txt mean.fits expmap.fits novalid.txt mean 2 5 1





