# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
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

import argparse
import sys
from mpdaf.obj import Cube
from time import time


def make_white_image(inputfile, outputfile, verbose=False):
    t0 = time()
    if outputfile is None:
        outputfile = inputfile.replace('DATACUBE', 'IMAGE')
    if outputfile == inputfile:
        sys.exit('Input and output files are identical')

    print('Creating white light image {}'.format(outputfile))
    cube = Cube(inputfile, convert_float64=False)
    if verbose:
        cube.info()

    im = cube.mean(axis=0)
    im.write(outputfile, savemask='nan')
    if verbose:
        print('Execution time {:.3f} seconds.'.format(time() - t0))


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Make a white-light image from a cube, by computing the '
                    'mean for each pixel along the spectral axis.')
    parser.add_argument('input_cube', help='Input cube (FITS file).')
    parser.add_argument('output_image', nargs='?',
                        help='Output image (FITS file).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose flag')
    args = parser.parse_args(args=args)
    make_white_image(args.input_cube, args.output_image, verbose=args.verbose)


if __name__ == '__main__':
    main()
