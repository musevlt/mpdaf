/*
tools.h
-------

different C methods used by several functions
*/

// Compute the arithmetic mean
void mpdaf_mean(double* data, int n, double x[3]);
// Compute the median
double mpdaf_median(double* data, int n);
// Iterative sigma-clipping of array elements
void mpdaf_mean_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop, int* id);
void mpdaf_median_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop);
// Linear interpolation
int mpdaf_locate(double* data, int n, double x);
double mpdaf_linear_interpolation(double* xx, double* yy, int n, double x);
