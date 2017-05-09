# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

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


from __future__ import absolute_import, print_function

from mpdaf.obj import Cube
from mpdaf.MUSE import FieldsMap
from mpdaf.tools import copy_header, write_hdulist_to

from astropy.io import fits
import argparse
from datetime import datetime
import sys
from time import time
import warnings

def extract_cube_fieldsMap(inputfile, outputfile, item, verbose=False):
    t0 = time()
    if outputfile == inputfile:
        sys.exit('Input and output files are identical')

    print('Creating subcube {}'.format(outputfile))
    cube = Cube(inputfile)
    if verbose:
        cube.info()
    fmap = FieldsMap(inputfile, extname='FIELDMAP')

    if len(item) == 6:
        subcub = cube[item[0]:item[1], item[2]:item[3], item[4]:item[5]]
        submap = fmap[item[2]:item[3], item[4]:item[5]]
    elif len(item) == 4:
        subcub = cube[:, item[0]:item[1], item[2]:item[3]]
        submap = fmap[item[0]:item[1], item[2]:item[3]]
    else:
        raise IOError('item not valid')
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        header = copy_header(subcub.primary_header)

    header['date'] = (str(datetime.now()), 'creation date')
    header['author'] = ('MPDAF', 'origin of the file')
    header['NFIELDS'] = submap.nfields
    hdulist = fits.HDUList([fits.PrimaryHDU(header=header)])

    # create cube DATA extension
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        datahdu = subcub.get_data_hdu(savemask='nan')
    hdulist.append(datahdu)

    # create spectrum STAT extension
    if subcub._var is not None:
        hdulist.append(subcub.get_stat_hdu(header=datahdu.header.copy()))

    # create FIELDMAP extension
    hdulist.append(fits.ImageHDU(name='FIELDMAP', data=submap.data))

    write_hdulist_to(hdulist, outputfile, overwrite=True,
                     output_verify='silentfix')
    
    if verbose:
        print('Execution time {:.3f} seconds.'.format(time() - t0))


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Extract a sub-cube, by including the '
                    'FIELDMAP extension.')
    parser.add_argument('input_cube', help='Input cube (FITS file).')
    parser.add_argument('output_cube', help='Output cube (FITS file).')
    parser.add_argument('item', nargs='+', type=int,
                        help=' k1 k2 p1 p2 q1 q2 Evaluation of Cube[k1:k2, p1:p2, q1:q2] '
                                     'p1 p2 q1 q2 Evaluation of Cube[:, p1:p2, q1:q2]')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose flag')
    args = parser.parse_args(args=args)
    extract_cube_fieldsMap(args.input_cube, args.output_cube, args.item, verbose=args.verbose)


if __name__ == '__main__':
    main()
