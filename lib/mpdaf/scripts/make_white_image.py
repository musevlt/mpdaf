# -*- coding: utf-8 -*-
"""Copyright 2010-2016 CNRS/CRAL

This file is part of MPDAF.

MPDAF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version

MPDAF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MPDAF.  If not, see <http://www.gnu.org/licenses/>.
"""


from __future__ import absolute_import, print_function

import argparse
from mpdaf.obj import Cube
from time import time


def make_white_image(inputfile, outputfile, verbose=False):
    t0 = time()
    print('Creating white light image {}'.format(outputfile))
    cube = Cube(inputfile)
    if verbose:
        cube.info()

    im = cube.mean(axis=0)
    im.write(outputfile)
    if verbose:
        print('Execution time {:.3f} seconds.'.format(time() - t0))


def main():
    parser = argparse.ArgumentParser(
        description='Make a white-light image from a cube, by computing the '
                    'mean for each pixel along the spectral axis.')
    parser.add_argument('input_cube', help='Input cube (FITS file).')
    parser.add_argument('output_image', help='Output image (FITS file).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose flag')
    args = parser.parse_args()
    make_white_image(args.input_cube, args.output_image, verbose=args.verbose)


if __name__ == '__main__':
    main()
