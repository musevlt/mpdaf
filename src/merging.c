#include <string.h>
#include <stdio.h>
#include <fitsio.h>
#include <time.h>
#include <math.h>
//#include <omp.h>


static int qsort_compare (const void * a, const void * b)
{
        return ( *(double*)a > *(double*)b );
}

static void mean_st(double* data, int n, double x[3])
{
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;i++)
    {
      mean+=data[i];
      //printf("%g, ",data[i]);
    }
    //printf("\n");
    mean=mean/n;
    for(i=0; i<n;i++)
      {
	sum_deviation+=(data[i]-mean)*(data[i]-mean);
      }
    x[0] = mean;
    x[1] = sqrt(sum_deviation/n);           
}

static double med_value(double* data, int n)
{
  qsort(data, n, sizeof(double), qsort_compare);
  return data[n / 2];
}

static void sigma_clip(double* data, int n, double x[3], int nmax, double nclip, int nstop, int* files_id)
{
  double clip_lo, clip_up;
  mean_st(data, n, x);
  x[2] = n;
  double med;
  med =  med_value(data,n);
  clip_lo = med - (nclip*x[1]);
  clip_up = med + (nclip*x[1]);

  int i, ni = 0; 
  for (i=0; i<n; i++)
    {
      if ((data[i]<clip_up) && (data[i]>clip_lo))
        {
	  data[ni]=data[i];
	  files_id[ni]=files_id[i];
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
     sigma_clip(data, ni, x, nmax, nclip, nstop, files_id);
   }
}

double median(double* data, int n)
{
  qsort(data, n, sizeof(double), qsort_compare);
  return data[n / 2];
}

int merging_median(char* input, char* output, char* output_path)
{
  int status = 0;  /* CFITSIO status value MUST be initialized to zero! */
  
  char cube_filename[500], expmap_filename[500], filename[500], buffer[80], begin[80];
  char* filenames[500];
  
  int i, ii, n, naxis, nfiles;
  long naxes[3] = {1,1,1}, bnaxes[3] = {1,1,1}, npixels = 1, firstpix[3] = {1,1,1};
  double *pix[100], *work;

  fitsfile* fptr[100], *output_cube_fptr, *output_expmap_fptr;

  time_t now;
  struct tm *info;
 
  // output filenames
  strcpy(cube_filename, output_path);
  strcat(cube_filename, "/DATACUBE_");
  strcat(cube_filename, output);
  strcat(cube_filename,".fits\0" );
  
  
  strcpy(expmap_filename, output_path);
  strcat(expmap_filename, "/EXPMAP_");
  strcat(expmap_filename, output);
  strcat(expmap_filename,".fits\0" );

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
  fits_open_file(&(fptr[0]), filename, READONLY, &status); /* open input images */
  if (status)
  {
      fits_report_error(stderr, status); /* print error message */
      return(status);
  }
  fits_get_img_dim(fptr[0], &naxis, &status);  /* read dimensions */
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
      fits_open_file(&(fptr[i]), filename, READONLY, &status); /* open input images */
      if (status)
      {
          fits_report_error(stderr, status); /* print error message */
          return(status);
      }
      fits_get_img_dim(fptr[i], &naxis, &status);  /* read dimensions */
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

  work = (double *) malloc(nfiles * sizeof(double));

  /* create the new empty output file if */
  if (!fits_create_file(&output_cube_fptr, cube_filename, &status) && !fits_create_file(&output_expmap_fptr, expmap_filename, &status)) 
  {
      long one = 1;
      fits_create_img(output_cube_fptr, 8 ,0, &one, &status);
      fits_copy_header(fptr[0], output_cube_fptr, &status);
      fits_create_img(output_expmap_fptr, 8 ,0, &one, &status);
      fits_copy_header(fptr[0], output_expmap_fptr, &status);

      firstpix[0] = 1;

      printf("Loop over all planes of the cube\n");
      /* loop over all planes of the cube (2D images have 1 plane) */
      for (firstpix[2] = 1; firstpix[2] <= naxes[2]; firstpix[2]++)
      {
          /* loop over all rows of the plane */
          for (firstpix[1] = 1; firstpix[1] <= naxes[1]; firstpix[1]++)
          {
               /* Read both cubes as doubles, regardless of actual datatype.  */
               /* Give starting pixel coordinate and no. of pixels to read.   */
               /* This version does not support undefined pixels in the cube. */
	       for (i=0; i<nfiles; i++)
	       {
		    if (fits_read_pix(fptr[i], TDOUBLE, firstpix, npixels, NULL, pix[i],
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
			  work[n] = pix[i][ii];
			  n = n + 1;
			}
                    }
		    if (n==0)
		    {
		      pix[0][ii] = NAN; //mean value
		      pix[1][ii] = 0; //exp map
                    } 
		    else if (n==1)
		    {
		      pix[0][ii] = work[0]; //mean value
		      pix[1][ii] = 1; //exp map
		    }
		    else
		    {
                      pix[0][ii] = median(work,n);
		      pix[1][ii] = n;//exp map
       		    }
	       }
             
	       fits_write_pix(output_cube_fptr, TDOUBLE, firstpix, npixels,
                       pix[0], &status); /* write new values to output image */
	       fits_write_pix(output_expmap_fptr, TDOUBLE, firstpix, npixels,
                       pix[1], &status);
               if (status)
	       {
                    fits_report_error(stderr, status); /* print error message */
                    return(status);
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
     
     fits_close_file(output_cube_fptr, &status);
     printf("%s created\n", cube_filename);
     free(work);
     printf("%s created\n", expmap_filename);
     fits_close_file(output_expmap_fptr, &status);

     for (i=0; i<nfiles; i++)
     {
         free(pix[i]);
         fits_close_file(fptr[i], &status);
     } 
   
  }
  if (status)
  {
     /* print any error messages */
     fits_report_error(stderr, status);
     return(status);
  }

  return (1);
}


int merging_sigma_clipping(char* input, char* output, char* output_path, int nmax, double nclip, int nstop, int var_mean)
{
  int status = 0;  //CFITSIO status value MUST be initialized to zero!
  int naxis, i, ii, n;
  long npixels = 1, firstpix[3] = {1,1,1};
  long naxes[3] = {1,1,1}, bnaxes[3] = {1,1,1};
  fitsfile *output_cube_fptr, *output_expmap_fptr;

  double *work, *var;
  int nfiles = 2;
  int *files_id, *valid_pix;
  char buff[500], filename[500], cube_filename[500], expmap_filename[500], novalid_filename[500];
  char* filenames[500];
  double* pix[100];
  fitsfile* fptr[100]; 
  FILE *fpix;
  double x[3];

  char buffer[80], begin[80];
  time_t now;
  struct tm *info;

  // output filenames
  strcpy(cube_filename, output_path);
  strcat(cube_filename, "/DATACUBE_");
  strcat(cube_filename, output);
  strcat(cube_filename,".fits\0" );
  
  strcpy(expmap_filename, output_path);
  strcat(expmap_filename, "/EXPMAP_");
  strcat(expmap_filename, output);
  strcat(expmap_filename,".fits\0" );

  strcpy(novalid_filename, output_path);
  strcat(novalid_filename, "/NOVALID_");
  strcat(novalid_filename, output);
  strcat(novalid_filename,".txt\0" );

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
  printf("nclip = %f\n", nclip);
  printf("nstop = %d\n", nstop);

  //initialization
  printf("Memory allocation and creation of the new empty output file\n");
  files_id = (int *) malloc(nfiles * sizeof(int));
  valid_pix = (int *) malloc(nfiles * sizeof(int));
  npixels = naxes[0];  // no. of pixels to read in each row
  for (i=0; i<nfiles; i++)
  {
      files_id[i] = i;
      valid_pix[i] = 0;
      pix[i] = (double *) malloc(npixels * sizeof(double));
      if (pix[i] == NULL) {
          printf("Memory allocation error\n");
          return(1);
      }
  }

  work = (double *) malloc(nfiles * sizeof(double));
  var = (double *) malloc(naxes[0]*naxes[1]*naxes[2] * sizeof(double));

  // create the new empty output file if
  if (!fits_create_file(&output_cube_fptr, cube_filename, &status) && !fits_create_file(&output_expmap_fptr, expmap_filename, &status)) 
  {
     long one = 1;
     fits_create_img(output_cube_fptr, 8 ,0, &one, &status);
     fits_copy_header(fptr[0], output_cube_fptr, &status);
     fits_create_img(output_expmap_fptr, 8 ,0, &one, &status);
     fits_copy_header(fptr[0], output_expmap_fptr, &status);

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
			  work[n] = pix[i][ii];
			  files_id[n] = i;
			  n = n + 1;
			}
                    }
		    if (n==0)
		    {
		      pix[0][ii] = NAN; //mean value
		      pix[1][ii] = 0; //exp map
		      var[(firstpix[2]-1)*naxes[0]*naxes[1]+(firstpix[1]-1)*naxes[0]+ii] = NAN;//var
                    } 
		    else if (n==1)
		    {
		      pix[0][ii] = work[0]; //mean value
		      pix[1][ii] = 1; //exp map
		      var[(firstpix[2]-1)*naxes[0]*naxes[1]+(firstpix[1]-1)*naxes[0]+ii] = NAN;//var
		    }
		    else
		    {
                      sigma_clip(work, n, x, nmax, nclip, nstop, files_id);
                      pix[0][ii] = x[0];//mean value
		      pix[1][ii] = x[2];//exp map
		      if (x[2]>1)
		      {
			  var[(firstpix[2]-1)*naxes[0]*naxes[1]+(firstpix[1]-1)*naxes[0]+ii] = (x[1]*x[1]);//var
			  if (var_mean==1)
		 	  {
			      var[(firstpix[2]-1)*naxes[0]*naxes[1]+(firstpix[1]-1)*naxes[0]+ii] /= (x[2]-1);
			  }
		      } else {
			  var[(firstpix[2]-1)*naxes[0]*naxes[1]+(firstpix[1]-1)*naxes[0]+ii] = NAN;//var
		      }
		      for (i=0; i<x[2]; i++)
		      {
			valid_pix[files_id[i]] += 1;
		      }
       		    }
	       }
             
	       fits_write_pix(output_cube_fptr, TDOUBLE, firstpix, npixels,
			      pix[0], &status); // write new values to output image
	       fits_write_pix(output_expmap_fptr, TDOUBLE, firstpix, npixels,
                       pix[1], &status);
               if (status)
	       {
                    fits_report_error(stderr, status);
                    return(status);
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

      //novalidpix file
      fpix = fopen(novalid_filename,"w");
      char int_str[15];
      for (i=0; i<nfiles; i++)
      {
	strcpy(buff, filenames[i]);
	sprintf(int_str, "%d", (int)(naxes[0]*naxes[1]*naxes[2]-valid_pix[i]));
	strcat(buff, "\t");
	strcat(buff, int_str);
	strcat(buff, "\n");
	fputs(buff, fpix); 
	//printf("%s %i \n",filenames[i], (int)(naxes[0]*naxes[1]*naxes[2]-valid_pix[i]));
      }
      fclose(fpix);
      printf("%s created\n", novalid_filename);

      printf("write variance extension\n");
      fits_insert_img(output_cube_fptr, DOUBLE_IMG,  naxis, naxes, &status);
      fits_write_key(output_cube_fptr, TSTRING, "EXTNAME", "STAT", "Extension name", &status);
      firstpix[0] = 1;
      firstpix[1] = 1;
      firstpix[2] = 1;
      npixels = naxes[0]*naxes[1]*naxes[2];
      fits_write_img(output_cube_fptr, TDOUBLE, 1, npixels, var, &status);
      free(var);

     fits_close_file(output_cube_fptr, &status);
     printf("%s created\n", cube_filename);
     free(work);
     printf("%s created\n", expmap_filename);
     fits_close_file(output_expmap_fptr, &status);

     for (i=0; i<nfiles; i++)
     {
         free(pix[i]);
         fits_close_file(fptr[i], &status);
     } 
   
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





