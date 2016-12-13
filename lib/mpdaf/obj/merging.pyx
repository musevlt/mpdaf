"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2015 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2015 Aurelien Jarno <aurelien.jarno@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
cimport numpy as np
cimport cython
# from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

cdef extern from "../../../src/tools.h":
    double mpdaf_sum(double* data, int n, int* indx) nogil
    void mpdaf_mean_sigma_clip(double* data, int n, double x[3], int nmax,
                               double nclip_low, double nclip_up, int nstop,
                               int* indx) nogil
    void mpdaf_mean_madsigma_clip(double* data, int n, double x[3], int nmax,
                                  double nclip_low, double nclip_up, int nstop,
                                  int* indx) nogil

cdef extern from "numpy/npy_math.h" nogil:
    long double NAN "NPY_NAN"
    bint isnan "npy_isnan"(long double)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sigma_clip(double[:,:,:] data, double[:,:,:] stat, double[:,:,:] cube,
               double[:,:,:] var, int[:,:,:] expmap, int[:,:,:] rejmap,
               int[:] valid_pix, int[:] select_pix, int l, int nmax,
               double nclip_low, double nclip_up, int nstop, int vartype,
               int mad):
    cdef unsigned int i, x, y, n, nuse
    cdef unsigned int ymax = data.shape[0]
    cdef unsigned int xmax = data.shape[1]
    cdef unsigned int nfiles = data.shape[2]
    cdef double res[3]

    cdef int *ind = <int *>malloc(nfiles * sizeof(int))
    cdef unsigned int *files_id = <unsigned int *>malloc(nfiles * sizeof(unsigned int))
    cdef double *wdata = <double *>malloc(nfiles * sizeof(double))
    cdef double *wstat = <double *>malloc(nfiles * sizeof(double))

    if mad == 0:
        merge_func = mpdaf_mean_sigma_clip
    else:
        merge_func = mpdaf_mean_madsigma_clip

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

                merge_func(&wdata[0], n, res, nmax, nclip_low, nclip_up,
                           nstop, &ind[0])
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
