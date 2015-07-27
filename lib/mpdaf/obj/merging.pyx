import numpy as np
cimport numpy as np
cimport cython
# from cython.parallel cimport prange

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "../../../src/tools.h":
    void mpdaf_mean_sigma_clip(double* data, int n, double x[3], int nmax, double nclip_low, double nclip_up, int nstop, int* indx) nogil

cdef extern from "numpy/npy_math.h" nogil:
    long double NAN "NPY_NAN"
    bint isnan "npy_isnan"(long double)

@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_clip(double[:,:,:] data, double[:,:,:] cube, double[:,:,:] var,
               int[:,:,:] expmap, int[:,:,:] rejmap, int[:] valid_pix,
               int[:] select_pix, int l, int nmax, double nclip_low,
               double nclip_up, int nstop):
    cdef unsigned int i, x, y, n, nuse
    cdef unsigned int nfiles = data.shape[0]
    cdef unsigned int ymax = data.shape[1]
    cdef unsigned int xmax = data.shape[2]
    cdef double res[3]

    cdef int[:] ind = np.empty([nfiles], dtype=np.int32)
    cdef unsigned int[:] files_id = np.empty([nfiles], dtype=np.uint32)
    cdef double[:] wdata = np.empty([nfiles], dtype=DTYPE)

    # cdef int[:] valid_pix = np.zeros([nfiles], dtype=np.int32)
    # cdef int[:] select_pix = np.zeros([nfiles], dtype=np.int32)

    # with nogil:
    for y in range(ymax):
        for x in range(xmax):
            n = 0
            # print '-->', y, x
            for i in range(nfiles):
                if not isnan(data[i, y, x]):
                    wdata[n] = data[i, y, x]
                    files_id[n] = i
                    ind[n] = n
                    valid_pix[i] = valid_pix[i] + 1
                    n = n + 1
            # print n, wdata[:n], ind[:n]

            if n>0:
                mpdaf_mean_sigma_clip(&wdata[0], n, res, nmax, nclip_low,
                                      nclip_up, nstop, &ind[0])
                nuse = <int>res[2]
                # print res, ind[:n], '\n'
                cube[l, y, x] = res[0]
                expmap[l, y, x] = nuse
                rejmap[l, y, x] = n - nuse
                if nuse > 1:
                    var[l, y, x] = res[1] * res[1]
                    # if (typ_var==1)
                    #     var[index] /= (x[2]-1)
                for i in range(nuse):
                    select_pix[files_id[ind[i]]] += 1
            else:
                cube[l, y, x] = NAN
                var[l, y, x] = NAN

    # return valid_pix, select_pix
