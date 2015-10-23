import numpy as np
cimport numpy as np
cimport cython
# from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

cdef extern from "../../../src/tools.h":
    double mpdaf_sum(double* data, int n, int* indx) nogil
    void mpdaf_mean_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop, int* indx) nogil

cdef extern from "numpy/npy_math.h" nogil:
    long double NAN "NPY_NAN"
    bint isnan "npy_isnan"(long double)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sigma_clip(double[:,:,:] data, double[:,:,:] stat, double[:,:,:] cube,
               double[:,:,:] var, int[:,:,:] expmap, int[:,:,:] rejmap,
               int[:] valid_pix, int[:] select_pix, int l, int nmax,
               double nclip_low, double nclip_up, int nstop, int vartype):
    cdef unsigned int i, x, y, n, nuse
    cdef unsigned int ymax = data.shape[0]
    cdef unsigned int xmax = data.shape[1]
    cdef unsigned int nfiles = data.shape[2]
    cdef double res[3]

    cdef int *ind = <int *>malloc(nfiles * sizeof(int))
    cdef unsigned int *files_id = <unsigned int *>malloc(nfiles * sizeof(unsigned int))
    cdef double *wdata = <double *>malloc(nfiles * sizeof(double))
    cdef double *wstat = <double *>malloc(nfiles * sizeof(double))

    # with nogil:
    for y in range(ymax):
        for x in range(xmax):
            n = 0
            for i in range(nfiles):
                if not isnan(data[y, x, i]):
                    wdata[n] = data[y, x, i]
                    if vartype == 0:
                        wstat[n] = stat[y, x, i]
                    files_id[n] = i
                    ind[n] = n
                    valid_pix[i] = valid_pix[i] + 1
                    n = n + 1
            if n>0:
                # print '-->', y, x
                # print 'data: ',
                # for i in range(n): print wdata[i], ' / ',
                # print '\n'
                # if vartype == 0:
                #     print 'stat: ',
                #     for i in range(n): print wstat[i], ' / ',
                #     print '\n'

                mpdaf_mean_sigma_clip(&wdata[0], n, res, nmax, nclip_low,
                                      nclip_up, nstop, &ind[0])
                nuse = <int>res[2]
                cube[l, y, x] = res[0]
                expmap[l, y, x] = nuse
                rejmap[l, y, x] = n - nuse
                if nuse > 0:
                    if vartype == 0:
                        var[l, y, x] = mpdaf_sum(&wstat[0], nuse, &ind[0]) / (<double>nuse * <double>nuse)
                    elif nuse > 1:
                        var[l, y, x] = res[1] * res[1]
                        if vartype == 1:
                            var[l, y, x] /= (nuse - 1)
                    else:
                        var[l, y, x] = NAN
                # print cube[l, y, x], var[l, y, x]
                for i in range(nuse):
                    select_pix[files_id[ind[i]]] += 1
            else:
                cube[l, y, x] = NAN
                var[l, y, x] = NAN
